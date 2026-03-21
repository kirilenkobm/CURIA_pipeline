#!/usr/bin/env python3
"""
Shared MMD (Maximum Mean Discrepancy) computation utilities.

Provides efficient MMD calculation with level-1 and level-2 optimizations
(mean-distance thresholding and precomputed self-kernels).

Level-3 optimisation (added later): callers that compare many island pairs
sharing the same embedding space should use the *_fast path:

    gamma = estimate_gamma_global(all_windows)
    ref_xx = precompute_self_kernels_batch(ref_wins, gamma)
    qry_yy = precompute_self_kernels_batch(qry_wins, gamma)
    mat, nc, ns = compute_mmd_matrix_fast(
        ref_wins, qry_wins, gamma, ref_xx, qry_yy, ...
    )

This avoids re-estimating gamma and re-computing XX/YY for every pair,
and replaces the Python double loop over (i, j) window pairs with
batched numpy operations.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _pairwise_sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distances between X and Y."""
    x_norm = (X ** 2).sum(axis=1)[:, None]
    y_norm = (Y ** 2).sum(axis=1)[None, :]
    return np.maximum(x_norm + y_norm - 2.0 * (X @ Y.T), 0.0)


def _estimate_gamma(windows_a: List[np.ndarray], windows_b: List[np.ndarray],
                    max_sample: int = 8) -> float:
    """Estimate RBF kernel gamma parameter from sampled windows."""
    sample = []
    for step, wins in [(max(1, len(windows_a) // max_sample), windows_a),
                       (max(1, len(windows_b) // max_sample), windows_b)]:
        for i in range(0, len(wins), step):
            sample.append(wins[i])

    if not sample:
        return 1.0

    all_pts = np.vstack(sample)
    dists = np.sqrt(_pairwise_sq_dists(all_pts, all_pts))
    median = np.median(dists[dists > 0])
    if median <= 0:
        median = 1.0
    return 1.0 / (2.0 * median * median + 1e-10)


def _precompute_self_kernel(windows: List[np.ndarray], gamma: float) -> List[float]:
    """Precompute self-kernel terms for each window (scalar loop fallback)."""
    terms = []
    for X in windows:
        n = X.shape[0]
        if n < 2:
            terms.append(0.0)
            continue
        K = np.exp(-gamma * _pairwise_sq_dists(X, X))
        terms.append((K.sum() - np.trace(K)) / (n * (n - 1)))
    return terms


# ---------------------------------------------------------------------------
# Level-3: shared-gamma helpers for the island-alignment hot path
# ---------------------------------------------------------------------------

def estimate_gamma_global(all_windows: List[np.ndarray],
                          max_sample: int = 16) -> float:
    """Estimate a single RBF gamma from a combined pool of windows.

    Intended to be called once per gene with *all* ref + query windows
    so that gamma is shared across island pairs.
    """
    if not all_windows:
        return 1.0
    step = max(1, len(all_windows) // max_sample)
    sample = [all_windows[i] for i in range(0, len(all_windows), step)]
    all_pts = np.vstack(sample)
    dists = np.sqrt(_pairwise_sq_dists(all_pts, all_pts))
    median = np.median(dists[dists > 0])
    if median <= 0:
        median = 1.0
    return 1.0 / (2.0 * median * median + 1e-10)


def precompute_self_kernels_batch(windows: List[np.ndarray],
                                  gamma: float) -> np.ndarray:
    """Vectorised self-kernel computation for a list of same-shape windows.

    Falls back to the scalar loop when window sizes are mixed.
    Returns a (n_windows,) float32 array.
    """
    if not windows:
        return np.array([], dtype=np.float32)

    n_pts_first = windows[0].shape[0]
    uniform = all(w.shape[0] == n_pts_first for w in windows)

    if uniform and n_pts_first >= 2:
        W = np.asarray(np.stack(windows), dtype=np.float32)  # (n_wins, n_pts, d)
        norms = (W ** 2).sum(axis=2)                   # (n_wins, n_pts)
        dots = W @ W.transpose(0, 2, 1)                # (n_wins, n_pts, n_pts)
        sq = np.maximum(norms[:, :, None] + norms[:, None, :] - 2 * dots, 0.0)
        K = np.exp(-gamma * sq)
        K_sum = K.sum(axis=(1, 2))
        K_trace = np.trace(K, axis1=1, axis2=2)
        return (K_sum - K_trace) / (n_pts_first * (n_pts_first - 1))

    return np.asarray(_precompute_self_kernel(windows, gamma), dtype=np.float32)


def compute_mmd_matrix_fast(
    ref_wins: List[np.ndarray],
    query_wins: List[np.ndarray],
    gamma: float,
    ref_xx: np.ndarray,
    query_yy: np.ndarray,
    mmd_skip: float = 1.0,
    mean_dist_threshold: float = 3.0,
) -> Tuple[np.ndarray, int, int]:
    """Compute the MMD matrix using precomputed gamma and self-kernels.

    The cross-kernel XY terms are computed with batched matrix ops instead
    of a Python double loop, which is the dominant cost in the original
    ``compute_mmd_matrix``.

    Args:
        ref_wins / query_wins: window embeddings (each [n_pts, d]).
        gamma: RBF bandwidth (call ``estimate_gamma_global`` once upstream).
        ref_xx / query_yy: self-kernel arrays from
            ``precompute_self_kernels_batch``.
        mmd_skip / mean_dist_threshold: same semantics as the original.

    Returns:
        (mmd_matrix, n_computed, n_skipped) — same contract as
        ``compute_mmd_matrix``.
    """
    nr, nq = len(ref_wins), len(query_wins)
    _f32 = np.float32
    if nr == 0 or nq == 0:
        return np.full((nr, nq), np.inf, dtype=_f32), 0, 0

    # Mean-distance thresholding
    ref_means = np.array([w.mean(axis=0) for w in ref_wins], dtype=_f32)
    query_means = np.array([w.mean(axis=0) for w in query_wins], dtype=_f32)
    mean_dist = np.sqrt(_pairwise_sq_dists(ref_means, query_means))
    compute_mask = mean_dist <= mean_dist_threshold

    n_computed = int(compute_mask.sum())
    n_skipped = nr * nq - n_computed

    if n_computed == 0:
        return np.full((nr, nq), mmd_skip, dtype=_f32), 0, n_skipped

    # --- Vectorised cross-kernel computation --------------------------------
    n_r = ref_wins[0].shape[0]
    n_q = query_wins[0].shape[0]
    d = ref_wins[0].shape[1]

    uniform = (
        n_r >= 2 and n_q >= 2
        and all(w.shape[0] == n_r for w in ref_wins)
        and all(w.shape[0] == n_q for w in query_wins)
    )

    if uniform:
        R = np.asarray(np.stack(ref_wins), dtype=_f32)
        Q = np.asarray(np.stack(query_wins), dtype=_f32)

        R_norms = (R ** 2).sum(axis=2)  # (nr, n_r)
        Q_norms = (Q ** 2).sum(axis=2)  # (nq, n_q)
        Q_flat = Q.reshape(-1, d)       # (nq*n_q, d)
        Q_norms_flat = Q_norms.reshape(1, -1)

        cross_means = np.empty((nr, nq), dtype=_f32)

        # Process ref windows in chunks to cap memory at ~256 MB per temp
        max_entries = 256 * 1024 * 1024 // 4  # 256 MB of float32
        chunk = max(1, max_entries // (n_r * nq * n_q))
        chunk = min(chunk, nr)

        for i0 in range(0, nr, chunk):
            i1 = min(i0 + chunk, nr)
            nc = i1 - i0

            R_chunk = R[i0:i1].reshape(-1, d)       # (nc*n_r, d)
            R_n = R_norms[i0:i1].reshape(-1, 1)     # (nc*n_r, 1)

            sq = np.maximum(R_n + Q_norms_flat - 2.0 * (R_chunk @ Q_flat.T), 0.0)
            np.multiply(-gamma, sq, out=sq)
            np.exp(sq, out=sq)
            cross_means[i0:i1] = sq.reshape(nc, n_r, nq, n_q).mean(axis=(1, 3))

        mmd_sq = ref_xx[:, None] + query_yy[None, :] - 2.0 * cross_means
        mat = np.sqrt(np.maximum(mmd_sq, 0.0))
        mat[~compute_mask] = mmd_skip
    else:
        # Rare fallback for mixed window sizes
        mat = np.full((nr, nq), mmd_skip, dtype=_f32)
        for i in range(nr):
            X = ref_wins[i]
            n = X.shape[0]
            for j in range(nq):
                if not compute_mask[i, j]:
                    continue
                Y = query_wins[j]
                m = Y.shape[0]
                if n < 2 or m < 2:
                    continue
                K_XY = np.exp(-gamma * _pairwise_sq_dists(X, Y))
                mmd_sq_val = ref_xx[i] + query_yy[j] - 2.0 * K_XY.mean()
                mat[i, j] = math.sqrt(max(mmd_sq_val, 0.0))

    return mat, n_computed, n_skipped


# ---------------------------------------------------------------------------
# Original API (kept for backward compatibility / non-hot-path callers)
# ---------------------------------------------------------------------------

def compute_mmd_matrix(
    ref_wins: List[np.ndarray],
    query_wins: List[np.ndarray],
    mmd_skip: float = 1.0,
    mean_dist_threshold: float = 3.0,
) -> Tuple[np.ndarray, int, int]:
    """
    Compute MMD matrix between reference and query windows.

    Args:
        ref_wins: List of reference window embeddings (each shape [n_points, n_dims])
        query_wins: List of query window embeddings (each shape [n_points, n_dims])
        mmd_skip: Value to use for skipped pairs (default: 1.0)
        mean_dist_threshold: Distance threshold for mean-based skipping (default: 3.0)

    Returns:
        Tuple of (mmd_matrix, n_computed, n_skipped)
        - mmd_matrix: (nr, nq) array of MMD values
        - n_computed: number of MMD computations performed
        - n_skipped: number of pairs skipped due to mean distance threshold
    """
    nr, nq = len(ref_wins), len(query_wins)
    if nr == 0 or nq == 0:
        return np.full((nr, nq), np.inf, dtype=np.float32), 0, 0

    gamma = _estimate_gamma(ref_wins, query_wins)
    ref_xx = precompute_self_kernels_batch(ref_wins, gamma)
    query_yy = precompute_self_kernels_batch(query_wins, gamma)
    return compute_mmd_matrix_fast(ref_wins, query_wins, gamma,
                                   ref_xx, query_yy, mmd_skip,
                                   mean_dist_threshold)
