#!/usr/bin/env python3
"""
Shared MMD (Maximum Mean Discrepancy) computation utilities.

Provides efficient MMD calculation with level-1 and level-2 optimizations
(mean-distance thresholding and precomputed self-kernels).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


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
    """Precompute self-kernel terms for each window."""
    terms = []
    for X in windows:
        n = X.shape[0]
        if n < 2:
            terms.append(0.0)
            continue
        K = np.exp(-gamma * _pairwise_sq_dists(X, X))
        terms.append((K.sum() - np.trace(K)) / (n * (n - 1)))
    return terms


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
        return np.full((nr, nq), np.inf), 0, 0

    gamma = _estimate_gamma(ref_wins, query_wins)
    ref_xx = _precompute_self_kernel(ref_wins, gamma)
    query_yy = _precompute_self_kernel(query_wins, gamma)

    ref_means = np.array([w.mean(axis=0) for w in ref_wins])
    query_means = np.array([w.mean(axis=0) for w in query_wins])
    mean_dist = np.sqrt(_pairwise_sq_dists(ref_means, query_means))

    mat = np.full((nr, nq), mmd_skip)
    n_computed = 0
    n_skipped = 0

    for i in range(nr):
        X = ref_wins[i]
        n = X.shape[0]
        for j in range(nq):
            if mean_dist[i, j] > mean_dist_threshold:
                n_skipped += 1
                continue
            Y = query_wins[j]
            m = Y.shape[0]
            if n < 2 or m < 2:
                continue
            K_XY = np.exp(-gamma * _pairwise_sq_dists(X, Y))
            mmd_sq = ref_xx[i] + query_yy[j] - 2.0 * K_XY.mean()
            mat[i, j] = math.sqrt(max(mmd_sq, 0.0))
            n_computed += 1

    return mat, n_computed, n_skipped
