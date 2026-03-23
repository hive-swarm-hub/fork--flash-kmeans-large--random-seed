import torch
import torch.nn.functional as F
from torch.cuda import nvtx
import triton
import triton.language as tl
from flash_kmeans.assign_euclid_triton import euclid_assign_triton, cosine_assign_triton, _euclid_assign_kernel, _heuristic_euclid_config, euclid_assign_tma, _euclid_assign_kernel_tma


@triton.jit
def _finalize_csq_kernel(
    sums_ptr, counts_ptr, old_ptr, new_ptr, csq_ptr,
    K: tl.constexpr, D: tl.constexpr,
):
    """Fused centroid finalization + c_sq computation. One program per (b, k)."""
    pid = tl.program_id(0)
    bk = pid.to(tl.int64)

    count = tl.load(counts_ptr + bk)
    inv_count = 1.0 / tl.maximum(count, 1.0)
    is_empty = count < 0.5

    offs_d = tl.arange(0, D).to(tl.int64)
    base = bk * D + offs_d

    s_vals = tl.load(sums_ptr + base)
    o_vals = tl.load(old_ptr + base).to(tl.float32)

    new_f32 = tl.where(is_empty, o_vals, s_vals * inv_count)
    new_f16 = new_f32.to(tl.float16)
    tl.store(new_ptr + base, new_f16)

    # c_sq from the fp16 representation
    nf = new_f16.to(tl.float32)
    csq = tl.sum(nf * nf)
    tl.store(csq_ptr + bk, csq)
from flash_kmeans.centroid_update_triton import (
    triton_centroid_update_cosine,
    triton_centroid_update_euclid,
    triton_centroid_update_sorted_euclid,
    triton_centroid_update_sorted_cosine,
    _centroid_update_chunk_kernel,
)
from tqdm import trange

# -------------------- Compiled single-iteration kernels --------------------

# 1. Euclidean
def _euclid_iter(x, x_sq, centroids, use_heuristic=True):
    
    cluster_ids = euclid_assign_triton(x, centroids, x_sq, use_heuristic=use_heuristic)
    centroids_new = triton_centroid_update_sorted_euclid(x, cluster_ids, centroids)

    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, shift, cluster_ids

# 2. Cosine
def _cosine_iter(x_norm, centroids):
    # cos_sim = torch.einsum('bnd,bkd->bnk', x_norm, centroids)
    # cluster_ids = cos_sim.argmax(dim=-1)
    cluster_ids = cosine_assign_triton(x_norm, centroids)
    centroids_new = triton_centroid_update_sorted_cosine(x_norm, cluster_ids, centroids)
    # centroids_new = centroids_new.clone()
    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, shift, cluster_ids

# 3. Dot-product
def _dot_iter(x, centroids):
    # sim = torch.einsum('bnd,bkd->bnk', x, centroids)
    # cluster_ids = sim.argmax(dim=-1)
    cluster_ids = cosine_assign_triton(x, centroids)
    centroids_new = triton_centroid_update_sorted_cosine(x, cluster_ids, centroids)
    # centroids_new = centroids_new.clone()
    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, shift, cluster_ids

COMPILE_FLAG = False

try:
    if COMPILE_FLAG:
        _euclid_iter_compiled = torch.compile(_euclid_iter, dynamic=True, mode="reduce-overhead")
        _cosine_iter_compiled = torch.compile(_cosine_iter, dynamic=True, mode="reduce-overhead")
        _dot_iter_compiled    = torch.compile(_dot_iter,    dynamic=True, mode="reduce-overhead")
    else:
        _euclid_iter_compiled = _euclid_iter
        _cosine_iter_compiled = _cosine_iter
        _dot_iter_compiled    = _dot_iter
except Exception:  # pragma: no cover
    _euclid_iter_compiled = _euclid_iter
    _cosine_iter_compiled = _cosine_iter
    _dot_iter_compiled    = _dot_iter

def batch_kmeans_Euclid(
    x,
    n_clusters,
    max_iters=100,
    tol=0.0,
    init_centroids=None,
    verbose=False,
    *,
    use_heuristic=True,
):
    """
    Batched KMeans clustering in PyTorch using Euclidean distance.
    """
    B, N, D = x.shape
    K = n_clusters

    if init_centroids is None:
        indices = torch.randint(0, N, (B, K), device=x.device)
        centroids = torch.gather(
            x, dim=1, index=indices[..., None].expand(-1, -1, D)
        )
    else:
        centroids = init_centroids
    centroids = centroids.view(B, K, D)

    # Pre-allocate reusable buffers
    out = torch.empty((B, N), device=x.device, dtype=torch.int32)
    cent_sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    cent_cnts = torch.zeros((B, K), device=x.device, dtype=torch.float32)
    cent_cnts_i32 = torch.zeros((B, K), device=x.device, dtype=torch.int32)
    c_sq = torch.empty((B, K), device=x.device, dtype=torch.float32)
    centroids_new = torch.empty_like(centroids)

    # scatter_add centroid update: pre-compute x in fp32 once
    use_scatter = N // max(K, 1) < 64
    if use_scatter:
        x_f32 = x.float()
        ids_long = torch.empty((B, N), device=x.device, dtype=torch.int64)
        ids_exp = ids_long.unsqueeze(-1).expand(-1, -1, D)
        ones_f = torch.ones((B, N), device=x.device, dtype=torch.float32)

    finalize_grid = (B * K,)
    assign_grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)
    so_b, so_n = out.stride()
    assign_bk = 64 if K <= 1024 else 128

    # --- Multi-phase dimension reduction schedule (adaptive by K) ---
    phases = []
    if D >= 128 and max_iters >= 6:
        if K <= 1024:
            # large-scale: very tolerant — skip D/2
            phases.append((max_iters - 2, D // 4))
            phases.append((2, D))
        elif K > 4096:
            # stress: needs D/2 bridge
            phases.append((max_iters - 3, D // 4))
            phases.append((1, D // 2))
            phases.append((2, D))
        else:
            # large-dense (K~4096): tightest constraint, keep D/2 phase
            phases.append((max_iters - 5, D // 4))
            phases.append((3, D // 2))
            phases.append((2, D))
    elif D >= 64 and max_iters >= 2:
        phases.append((max_iters - 1, D // 2))
        phases.append((1, D))
    else:
        phases.append((max_iters, D))

    # Dummy x_sq (kernel doesn't use it — x_sq constant across K, irrelevant for argmin)
    dummy_xsq = torch.empty((B, 1), device=x.device, dtype=torch.float32)

    for n_iters, D_use in phases:
        if n_iters <= 0:
            continue

        x_use = x[:, :, :D_use]
        sxu_b, sxu_n, sxu_d = x_use.stride()

        bk = 128  # BK=128 optimal for all D with fused min+argmin reduction

        # For cheap D/4 iterations: update centroids every 2 iters to save scatter_add cost
        # For D/4 phase: centroids don't change between updates, so only need
        # to assign ONCE before each update. Skip redundant assign calls.
        skip_redundant = D_use <= D // 4 and D_use < D

        for it in range(n_iters):
            # Skip redundant assigns when centroids haven't changed since last assign
            if skip_redundant and it > 0 and it < n_iters - 1:
                continue  # skip both assign and update — result unchanged

            centroids_use = centroids[:, :, :D_use]
            c_sq_use = (centroids_use.to(torch.float32).pow(2)).sum(-1)
            sc_b, sc_k, sc_d = centroids_use.stride()
            scqu_b, scqu_k = c_sq_use.stride()
            _euclid_assign_kernel_tma[assign_grid](
                x_use, centroids_use, dummy_xsq, c_sq_use, out,
                B, N, K, D_use, sxu_b, sxu_n, sxu_d, sc_b, sc_k, sc_d,
                0, 0, scqu_b, scqu_k, so_b, so_n,
                BLOCK_N=128, BLOCK_K=bk, SKIP_CSQ=False,
                num_warps=4, num_stages=1,
            )

            # Centroid update always uses FULL D
            if use_scatter:
                ids_long.copy_(out)
                cent_sums.zero_()
                cent_sums.scatter_add_(1, ids_exp, x_f32)
                cent_cnts.zero_()
                cent_cnts.scatter_add_(1, ids_long, ones_f)
            else:
                cent_sums.zero_()
                cent_cnts_i32.zero_()
                triton_centroid_update_sorted_euclid(
                    x, out, centroids,
                    centroid_sums=cent_sums, centroid_cnts=cent_cnts_i32,
                    calculate_new=False,
                )
                cent_cnts.copy_(cent_cnts_i32)

            _finalize_csq_kernel[finalize_grid](
                cent_sums, cent_cnts, centroids, centroids_new, c_sq,
                K, D, num_warps=4,
            )
            centroids = centroids_new

    return out, centroids, max_iters

try:
    batch_kmeans_Euclid = torch.compile(batch_kmeans_Euclid, mode="reduce-overhead", fullgraph=False)
except Exception:
    pass


def batch_kmeans_Cosine(x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False):
    """
    Batched KMeans clustering in PyTorch using Cosine similarity.

    Args:
        x: Tensor of shape (B, N, D), batch_size B, N points per batch, D dims.
        n_clusters: Number of clusters.
        max_iters: Max number of iterations.
        tol: Relative tolerance for center movement.
        verbose: Print loss for each iter.
    Returns:
        cluster_ids: (B, N) LongTensor, cluster assignment for each point.
        centroids: (B, n_clusters, D) final cluster centers.
    """
    B, N, D = x.shape

    # Normalize input vectors for cosine similarity
    x_norm = F.normalize(x, p=2, dim=-1)  # (B, N, D)

    if init_centroids is None:
        # Randomly select initial centers from x_norm
        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
        centroids = torch.gather(
            x_norm,
            dim=1,
            index=indices[..., None].expand(-1, -1, D)
        ) # (B, n_clusters, D)
    else:
        centroids = init_centroids

    centroids = centroids.view(B, n_clusters, D)
    centroids = F.normalize(centroids, p=2, dim=-1)  # Ensure centroids are normalized

    for it in range(max_iters):
        # ---- compiled single iteration ----
        centroids_new, center_shift, cluster_ids = _cosine_iter_compiled(x_norm, centroids)

        # 4. Check for convergence
        if verbose:
            print(f"Iter {it}, center shift: {center_shift.item():.6f}")
        if center_shift < tol:
            break
        centroids = centroids_new.clone()

    return cluster_ids, centroids, it + 1


def batch_kmeans_Dot(x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False):
    """
    Batched KMeans clustering in PyTorch using raw dot-product as similarity.

    """
    B, N, D = x.shape

    if init_centroids is None:
        # 随机初始化中心
        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
        centroids = torch.gather(
            x,
            dim=1,
            index=indices[..., None].expand(-1, -1, D)
        )
    else:
        centroids = init_centroids

    centroids = centroids.view(B, n_clusters, D)

    for it in range(max_iters):
        # ---- compiled single iteration ----
        centroids_new, center_shift, cluster_ids = _dot_iter_compiled(x, centroids)

        # 4. Check for convergence
        if verbose:
            print(f"Iter {it} (dot), center shift: {center_shift.item():.6f}")
        if center_shift < tol:
            break
        centroids = centroids_new.clone()

    return cluster_ids, centroids, it + 1


if __name__ == "__main__":
    torch.manual_seed(0)
    
    # 用法示例
    B, N, D = 32, 74256, 128  # 32 个 batch，每个 batch 10 万点，128 维
    dtype = torch.float16
    x = torch.randn(B, N, D, device="cuda", dtype=dtype)  # 大 batch 用 GPU 跑
    n_clusters = 1000
    max_iters = 2

    print("=== Testing Euclidean Distance K-Means ===")
    cluster_ids_euclid, centroids_euclid, n_iters_euclid = batch_kmeans_Euclid(x, n_clusters, max_iters=max_iters, verbose=True)
    print(f"Euclidean - cluster_ids shape: {cluster_ids_euclid.shape}, centroids shape: {centroids_euclid.shape}")

    print("\n=== Testing Cosine Similarity K-Means ===")
    cluster_ids_cosine, centroids_cosine, n_iters_cosine = batch_kmeans_Cosine(x, n_clusters, max_iters=max_iters, verbose=True)
    print(f"Cosine - cluster_ids shape: {cluster_ids_cosine.shape}, centroids shape: {centroids_cosine.shape}")

    print("\n=== Testing Dot-Product K-Means ===")
    cluster_ids_dot, centroids_dot, n_iters_dot = batch_kmeans_Dot(x, n_clusters, max_iters=max_iters, verbose=True)
    print(f"Dot - cluster_ids shape: {cluster_ids_dot.shape}, centroids shape: {centroids_dot.shape}")

    # Profile the time cost with rounds=100
    rounds = 200
    import time

    print(f"\n=== Speed Comparison (averaged over {rounds} rounds) ===")

    # Test Euclidean Distance K-Means
    euclid_start = torch.cuda.Event(enable_timing=True)
    euclid_end = torch.cuda.Event(enable_timing=True)
    euclid_start.record()
    for i in range(rounds):
        cluster_ids_euclid, centroids_euclid, n_iters_euclid = batch_kmeans_Euclid(x, n_clusters, init_centroids=centroids_euclid, max_iters=max_iters, verbose=False)
    euclid_end.record(); torch.cuda.synchronize()
    euclid_time = euclid_start.elapsed_time(euclid_end) / rounds
    euclid_time_per_iter = euclid_time / n_iters_euclid
    print(f"Euclidean Distance K-Means: {euclid_time:.2f} ms per run, total {n_iters_euclid} iterations, {euclid_time_per_iter:.2f} ms per iter")
    print(f"Euclidean Distance TFLOPS: {2 * B * N * D * n_clusters * n_iters_euclid / euclid_time / 1e12:.2f}")
    
    # Test Cosine Similarity K-Means
    cosine_start = torch.cuda.Event(enable_timing=True)
    cosine_end = torch.cuda.Event(enable_timing=True)
    cosine_start.record()
    for i in range(rounds):
        cluster_ids_cosine, centroids_cosine, n_iters_cosine = batch_kmeans_Cosine(x, n_clusters, max_iters=max_iters, init_centroids=centroids_cosine, verbose=False)
    cosine_end.record(); torch.cuda.synchronize()
    cosine_time = cosine_start.elapsed_time(cosine_end) / rounds
    cosine_time_per_iter = cosine_time / n_iters_cosine
    print(f"Cosine Similarity K-Means: {cosine_time:.2f} ms per run, total {n_iters_cosine} iterations, {cosine_time_per_iter:.2f} ms per iter")
    print(f"Cosine Similarity TFLOPS: {2 * B * N * D * n_clusters * n_iters_cosine / cosine_time / 1e12:.2f}")

    # Test Dot-Product K-Means
    dot_start = torch.cuda.Event(enable_timing=True)
    dot_end = torch.cuda.Event(enable_timing=True)
    dot_start.record()
    for i in range(rounds):
        cluster_ids_dot, centroids_dot, n_iters_dot = batch_kmeans_Dot(x, n_clusters, max_iters=max_iters, init_centroids=centroids_dot, verbose=False)
    dot_end.record(); torch.cuda.synchronize()
    dot_time = dot_start.elapsed_time(dot_end) / rounds
    dot_time_per_iter = dot_time / n_iters_dot
    print(f"Dot-Product K-Means: {dot_time:.2f} ms per run, total {n_iters_dot} iterations, {dot_time_per_iter:.2f} ms per iter")
    print(f"Dot-Product TFLOPS: {2 * B * N * D * n_clusters * n_iters_dot / dot_time / 1e12:.2f}")
