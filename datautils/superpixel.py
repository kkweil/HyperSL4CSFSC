import numpy as np
import heapq
from typing import Tuple, Dict, Any


def _pca_reduce(X: np.ndarray, d: int) -> np.ndarray:
    """
    X: (N, C) float
    return: (N, d)
    """
    N, C = X.shape
    d = int(min(d, C))
    if d <= 0:
        return X
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD PCA: Xc = U S V^T, take V[:d]
    # For small patches, this is fast enough.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:d].T  # (C, d)
    return Xc @ W  # (N, d)


def _neighbors_4(h: int, w: int, idx: int):
    r = idx // w
    c = idx % w
    if r > 0:
        yield idx - w
    if r < h - 1:
        yield idx + w
    if c > 0:
        yield idx - 1
    if c < w - 1:
        yield idx + 1


def spectral_region_growing_aggregate(
    patch_hwc: np.ndarray,
    K_max: int = 32,
    pca_dim: int = 12,
    A_min: int = 16,
    seed_top_ratio: float = 2.0,
    sigma_f: float = None,
    use_spatial: bool = False,
    sigma_x: float = 1.5,
    return_stats: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Spectral Similarity Region Growing (SSRG) for small patches.

    Args:
        patch_hwc: (H, W, C) hyperspectral patch
        K_max: upper bound of regions
        pca_dim: PCA dim for region growing similarity (use 8~16)
        A_min: minimum stable area; small patches should set 9/16/25
        seed_top_ratio: number of initial seeds M = int(K * seed_top_ratio)
        sigma_f: feature scale; if None, estimated from neighbor distances
        use_spatial: whether to include a weak spatial term
        sigma_x: spatial scale if use_spatial
        return_stats: whether to return per-region std/range

    Returns:
        labels_hw: (H, W) int32 region id in [0, K-1]
        region_mean: (K, C) float32 mean spectrum per region
        info: dict with optional stats and diagnostics
    """
    patch = np.asarray(patch_hwc)
    assert patch.ndim == 3, "patch_hwc should be (H,W,C)"
    H, W, C = patch.shape
    N = H * W

    X = patch.reshape(N, C).astype(np.float32)

    # Determine adaptive K to avoid over-segmentation on small patches
    A_min_eff = max(int(A_min), 4)  # safe
    K = int(min(K_max, max(1, (N // A_min_eff))))
    # If patch is extremely small, force at least 1 region
    K = max(1, K)

    # Feature for growing
    F = _pca_reduce(X, pca_dim).astype(np.float32)  # (N,d)
    d = F.shape[1]

    # Estimate sigma_f from neighbor feature distances (robust)
    if sigma_f is None:
        dists = []
        for idx in range(N):
            for nb in _neighbors_4(H, W, idx):
                if nb > idx:
                    diff = F[idx] - F[nb]
                    dists.append(float(diff @ diff))
        if len(dists) == 0:
            sigma_f = 1.0
        else:
            # Use median to be robust
            med = np.median(dists)
            sigma_f = float(np.sqrt(max(med, 1e-6)))
    sigma_f = max(float(sigma_f), 1e-6)

    # Centrality score: sum of neighbor similarities
    # c_i = sum exp(-||fi-fj||^2 / sigma_f^2) * exp(-||xi-xj||^2 / sigma_x^2)
    # For 4-neighbors, spatial term is constant-ish; keep optional.
    c = np.zeros((N,), dtype=np.float32)
    if use_spatial:
        spatial_w = np.exp(-1.0 / (sigma_x * sigma_x)).astype(np.float32)
    else:
        spatial_w = 1.0

    inv_sf2 = 1.0 / (sigma_f * sigma_f)
    for idx in range(N):
        s = 0.0
        fi = F[idx]
        for nb in _neighbors_4(H, W, idx):
            diff = fi - F[nb]
            s += np.exp(-float(diff @ diff) * inv_sf2) * float(spatial_w)
        c[idx] = s

    # Pick seeds: top-M by centrality, then non-maximum suppression (NMS) in 1-hop
    M = int(min(N, max(K, int(np.ceil(K * seed_top_ratio)))))
    seed_candidates = np.argpartition(-c, M - 1)[:M]
    seed_candidates = seed_candidates[np.argsort(-c[seed_candidates])]

    chosen = []
    chosen_mask = np.zeros((N,), dtype=bool)
    for idx in seed_candidates:
        if chosen_mask[idx]:
            continue
        chosen.append(int(idx))
        chosen_mask[idx] = True
        # 1-hop suppression
        for nb in _neighbors_4(H, W, idx):
            chosen_mask[nb] = True
        if len(chosen) >= K:
            break

    # If still not enough seeds, fill randomly from remaining
    if len(chosen) < K:
        remaining = np.where(~chosen_mask)[0]
        if remaining.size > 0:
            extra = remaining[: (K - len(chosen))]
            chosen.extend([int(x) for x in extra])
    seeds = chosen[:K]
    K = len(seeds)

    # Initialize region stats (in feature space for assignment; spectrum for output)
    labels = -np.ones((N,), dtype=np.int32)

    region_count = np.zeros((K,), dtype=np.int32)
    region_fsum = np.zeros((K, d), dtype=np.float32)
    region_xsum = np.zeros((K, 2), dtype=np.float32)  # centroid accum
    region_specsum = np.zeros((K, C), dtype=np.float32)

    # Assign seeds
    for rid, idx in enumerate(seeds):
        labels[idx] = rid
        region_count[rid] = 1
        region_fsum[rid] = F[idx]
        r = idx // W
        c0 = idx % W
        region_xsum[rid] = np.array([r, c0], dtype=np.float32)
        region_specsum[rid] = X[idx]

    # Priority queue: max-heap by score (use negative for heapq)
    # score = exp(-||f - mean_f||^2/sf^2) * exp(-||x - centroid||^2/sx^2) optional
    heap = []

    def push_neighbors(idx: int, rid: int):
        mf = region_fsum[rid] / max(region_count[rid], 1)
        if use_spatial:
            cx = region_xsum[rid] / max(region_count[rid], 1)
        for nb in _neighbors_4(H, W, idx):
            if labels[nb] != -1:
                continue
            diff = F[nb] - mf
            sim = np.exp(-float(diff @ diff) * inv_sf2)
            if use_spatial:
                rr = nb // W
                cc = nb % W
                dx = (rr - cx[0])
                dy = (cc - cx[1])
                sim *= np.exp(-(dx * dx + dy * dy) / (sigma_x * sigma_x))
            heapq.heappush(heap, (-sim, nb, rid))

    # Start from seeds
    for rid, idx in enumerate(seeds):
        push_neighbors(idx, rid)

    # Region growing
    while heap:
        negsim, idx, rid = heapq.heappop(heap)
        if labels[idx] != -1:
            continue

        # Recompute similarity against current region mean (lazy heap correction)
        mf = region_fsum[rid] / max(region_count[rid], 1)
        diff = F[idx] - mf
        sim = np.exp(-float(diff @ diff) * inv_sf2)
        if use_spatial:
            cx = region_xsum[rid] / max(region_count[rid], 1)
            rr = idx // W
            cc = idx % W
            dx = (rr - cx[0])
            dy = (cc - cx[1])
            sim *= np.exp(-(dx * dx + dy * dy) / (sigma_x * sigma_x))

        # If the popped score is too stale, reinsert with updated score
        if sim < (-negsim) * 0.90:
            heapq.heappush(heap, (-sim, idx, rid))
            continue

        # Assign
        labels[idx] = rid
        region_count[rid] += 1
        region_fsum[rid] += F[idx]
        r = idx // W
        c0 = idx % W
        region_xsum[rid] += np.array([r, c0], dtype=np.float32)
        region_specsum[rid] += X[idx]

        # Expand
        push_neighbors(idx, rid)

    # Any unassigned pixels (rare): assign to nearest seed region by feature
    unassigned = np.where(labels == -1)[0]
    if unassigned.size > 0:
        region_fmean = region_fsum / np.maximum(region_count[:, None], 1)
        for idx in unassigned:
            diff = region_fmean - F[idx][None, :]
            dd = np.sum(diff * diff, axis=1)
            rid = int(np.argmin(dd))
            labels[idx] = rid
            region_count[rid] += 1
            region_fsum[rid] += F[idx]
            r = idx // W
            c0 = idx % W
            region_xsum[rid] += np.array([r, c0], dtype=np.float32)
            region_specsum[rid] += X[idx]

    # Post-merge small regions to enforce A_min_eff
    # Merge each small region into its most similar neighboring region.
    labels_hw = labels.reshape(H, W)
    region_fmean = region_fsum / np.maximum(region_count[:, None], 1)

    # Build adjacency via boundary scanning
    # For each small region, find neighbor regions and choose best by mean feature similarity.
    changed = True
    max_iters = 5
    it = 0
    while changed and it < max_iters:
        it += 1
        changed = False
        region_count = np.bincount(labels, minlength=K).astype(np.int32)
        small = np.where(region_count < A_min_eff)[0]
        if small.size == 0:
            break

        for rid in small:
            # Collect neighboring region ids
            neigh = set()
            coords = np.argwhere(labels_hw == rid)
            if coords.size == 0:
                continue
            for (rr, cc) in coords:
                idx = rr * W + cc
                for nb in _neighbors_4(H, W, idx):
                    nb_rid = labels[nb]
                    if nb_rid != rid:
                        neigh.add(int(nb_rid))
            if not neigh:
                continue

            # Choose best neighbor by feature distance between region means
            cand = np.array(list(neigh), dtype=np.int32)
            diff = region_fmean[cand] - region_fmean[rid][None, :]
            dd = np.sum(diff * diff, axis=1)
            best = int(cand[np.argmin(dd)])

            # Merge rid -> best
            labels[labels == rid] = best
            labels_hw = labels.reshape(H, W)
            changed = True

        # Re-index labels to compact [0..K'-1]
        uniq = np.unique(labels)
        remap = {int(old): i for i, old in enumerate(uniq)}
        labels = np.vectorize(remap.get)(labels).astype(np.int32)
        labels_hw = labels.reshape(H, W)
        K = len(uniq)

        # Recompute region stats (spectrum mean etc.)
        region_specsum = np.zeros((K, C), dtype=np.float32)
        region_fsum = np.zeros((K, d), dtype=np.float32)
        region_count = np.zeros((K,), dtype=np.int32)
        for idx in range(N):
            rid = labels[idx]
            region_count[rid] += 1
            region_specsum[rid] += X[idx]
            region_fsum[rid] += F[idx]
        region_fmean = region_fsum / np.maximum(region_count[:, None], 1)

    region_mean = region_specsum / np.maximum(region_count[:, None], 1)

    info = {
        "K_final": int(K),
        "K_initial": int(len(seeds)),
        "sigma_f": float(sigma_f),
        "A_min": int(A_min_eff),
        "counts": region_count.copy(),
    }

    if return_stats:
        # per-region std (spectrum)
        region_var = np.zeros((K, C), dtype=np.float32)
        # second pass
        for idx in range(N):
            rid = labels[idx]
            diff = X[idx] - region_mean[rid]
            region_var[rid] += diff * diff
        region_var /= np.maximum(region_count[:, None], 1)
        region_std = np.sqrt(np.maximum(region_var, 0.0))
        info["region_std"] = region_std

    return labels_hw.astype(np.int32), region_mean.astype(np.float32), info



import numpy as np
from typing import Tuple, Dict, Any


def _kmeanspp_init(X: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """
    k-means++ init (vectorized enough for small N).
    X: (N, D)
    return centers: (K, D)
    """
    N, D = X.shape
    centers = np.empty((K, D), dtype=X.dtype)

    # pick first center: closest to mean (stable)
    mean = X.mean(axis=0, keepdims=True)
    dist2 = np.sum((X - mean) ** 2, axis=1)
    first = int(np.argmin(dist2))
    centers[0] = X[first]

    # min dist to chosen centers
    min_dist2 = np.sum((X - centers[0:1]) ** 2, axis=1)

    for k in range(1, K):
        # probability proportional to min_dist2
        probs = min_dist2 / (min_dist2.sum() + 1e-12)
        idx = int(rng.choice(N, p=probs))
        centers[k] = X[idx]
        dist2_new = np.sum((X - centers[k:k+1]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2_new)

    return centers


def fast_spectral_kmeans_aggregate_numpy(
    patch_hwc: np.ndarray,
    K_max: int = 32,
    A_min: int = 9,
    iters: int = 5,
    feat_dim: int = 12,
    init: str = "kmeans++",     # "kmeans++" or "random"
    seed: int = 0,
    return_std: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fast Spectral K-means Aggregation (NumPy), designed for small patches (e.g., 15x15).

    Args:
        patch_hwc: (H, W, C) float/uint
        K_max: max clusters
        A_min: min stable area; K = min(K_max, floor(HW/A_min)) (avoid overseg on tiny patches)
        iters: k-means iterations (3~8). 5 is a good default.
        feat_dim: PCA dim used ONLY for clustering distance (output mean is in original C)
        init: "kmeans++" (recommended) or "random"
        seed: RNG seed
        return_std: whether to compute per-cluster std spectrum (adds one pass)

    Returns:
        labels_hw: (H,W) int32
        mean_kc: (K,C) float32
        info: dict with diagnostics, counts, (optional) std_kc
    """
    patch = np.asarray(patch_hwc)
    assert patch.ndim == 3, "patch_hwc must be (H,W,C)"
    H, W, C = patch.shape
    N = H * W
    X_spec = patch.reshape(N, C).astype(np.float32)

    # Choose K adaptively (critical for 15x15)
    A_min_eff = max(int(A_min), 1)
    K = int(min(K_max, max(1, N // A_min_eff)))

    rng = np.random.default_rng(seed)

    # --- PCA for clustering feature (fast SVD; N is tiny) ---
    # clustering feature: (N, D)
    D = min(int(feat_dim), C)
    if D <= 0 or D >= C:
        X_feat = X_spec
    else:
        Xc = X_spec - X_spec.mean(axis=0, keepdims=True)
        # SVD for PCA
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Wp = Vt[:D].T  # (C,D)
        X_feat = Xc @ Wp  # (N,D)

    # --- init centers in feature space ---
    if init == "kmeans++":
        centers = _kmeanspp_init(X_feat, K, rng)  # (K,D)
    elif init == "random":
        idx = rng.choice(N, size=K, replace=False)
        centers = X_feat[idx]
    else:
        raise ValueError("init must be 'kmeans++' or 'random'")

    labels = np.zeros((N,), dtype=np.int32)

    # --- Lloyd iterations (vectorized) ---
    # Precompute X_feat norms if you want speed; using (x-c)^2 = x^2 + c^2 -2xc
    X2 = np.sum(X_feat * X_feat, axis=1, keepdims=True)  # (N,1)

    for t in range(int(iters)):
        C2 = np.sum(centers * centers, axis=1, keepdims=True).T  # (1,K)
        # dist2: (N,K)
        dist2 = X2 + C2 - 2.0 * (X_feat @ centers.T)
        new_labels = np.argmin(dist2, axis=1).astype(np.int32)

        if t > 0 and np.all(new_labels == labels):
            labels = new_labels
            break
        labels = new_labels

        # update centers: sum over clusters (vectorized via bincount per dim)
        counts = np.bincount(labels, minlength=K).astype(np.float32)  # (K,)
        # avoid empty clusters: re-seed empties with farthest points
        empty = np.where(counts < 0.5)[0]
        if empty.size > 0:
            # pick farthest points from their assigned center (use current dist2)
            # dist_to_assigned = dist2[np.arange(N), labels]
            d_assigned = dist2[np.arange(N), labels]
            far_idx = np.argsort(-d_assigned)  # descending
            used = set()
            ptr = 0
            for k0 in empty:
                while ptr < N and int(far_idx[ptr]) in used:
                    ptr += 1
                if ptr >= N:
                    break
                i = int(far_idx[ptr])
                used.add(i)
                centers[k0] = X_feat[i]
                counts[k0] = 1.0
                labels[i] = k0  # force assign seed point
            counts = np.bincount(labels, minlength=K).astype(np.float32)

        # compute sums
        sums = np.zeros((K, X_feat.shape[1]), dtype=np.float32)
        np.add.at(sums, labels, X_feat)
        centers = sums / np.maximum(counts[:, None], 1.0)

    # --- compute output mean spectrum per cluster ---
    counts = np.bincount(labels, minlength=K).astype(np.int32)
    sum_spec = np.zeros((K, C), dtype=np.float32)
    np.add.at(sum_spec, labels, X_spec)
    mean_kc = sum_spec / np.maximum(counts[:, None], 1)

    info: Dict[str, Any] = {
        "K": int(K),
        "iters": int(t + 1),
        "counts": counts,
    }

    if return_std:
        # one more pass for std
        var = np.zeros((K, C), dtype=np.float32)
        diff = X_spec - mean_kc[labels]
        np.add.at(var, labels, diff * diff)
        var = var / np.maximum(counts[:, None], 1)
        std_kc = np.sqrt(np.maximum(var, 0.0))
        info["std_kc"] = std_kc

    labels_hw = labels.reshape(H, W)
    return labels_hw.astype(np.int32), mean_kc.astype(np.float32), info


# # ---- quick test ----
# if __name__ == "__main__":
#     H, W, C = 15, 15, 128
#     patch = np.random.rand(H, W, C).astype(np.float32)
#     labels, mean, info = fast_spectral_kmeans_aggregate_numpy(
#         patch, K_max=64, A_min=9, iters=5, feat_dim=12, init="kmeans++", return_std=True
#     )
#     print("labels:", labels.shape, "mean:", mean.shape, "K:", info["K"], "min/max area:", info["counts"].min(), info["counts"].max())



# ---------- Quick sanity test ----------
if __name__ == "__main__":
    H, W, C = 15, 15, 128
    patch = np.random.rand(H, W, C).astype(np.float32)
    labels, mean_spec, info = spectral_region_growing_aggregate(
        patch, K_max=50, pca_dim=8, A_min=16, use_spatial=False
    )
    print(labels.shape, mean_spec.shape, info["K_final"], info["sigma_f"])
