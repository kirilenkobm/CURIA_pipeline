"""RNA-FM island matcher: windowed embeddings -> per-window MMD matrix -> SW.

This is the legacy, context-dependent path (RNA-FM drifts under flanking
context, so every sliding window is re-embedded in its own context). The
scoring functions here are moved VERBATIM from island_alignment.py so RNA-FM
output stays byte-identical; only the wrapping into MatchResult/Chain is new.
RiNALMo uses the cheaper, more accurate dotplot matcher (see rinalmo.py).
"""

from __future__ import annotations

import asyncio
from typing import List, Tuple

import numpy as np

from modules.utils.mmd_utils import (
    compute_mmd_matrix_fast,
    estimate_gamma_global,
    precompute_self_kernels_batch,
)
from modules.pipeline.matchers.base import Chain, MatchResult, EMPTY_MATCH


# ===========================================================================
# Moved verbatim from island_alignment.py (behavior must not change)
# ===========================================================================

def best_diagonal_run(mat: np.ndarray, min_run: int = 3
                      ) -> Tuple[float, int, int, int]:
    """Find best mean diagonal run in MMD matrix."""
    nr, nq = mat.shape
    best_mean = float("inf")
    best_len = 0
    best_r = best_q = 0

    for q_off in range(-nr + 1, nq):
        r_start = max(0, -q_off)
        q_start = max(0, q_off)
        diag_len = min(nr - r_start, nq - q_start)
        if diag_len < min_run:
            continue
        vals = [mat[r_start + k, q_start + k] for k in range(diag_len)]

        for start in range(len(vals) - min_run + 1):
            cum = sum(vals[start:start + min_run])
            run = min_run
            mean = cum / run
            if mean < best_mean:
                best_mean, best_len = mean, run
                best_r, best_q = r_start + start, q_start + start
            for end in range(start + min_run, len(vals)):
                cum += vals[end]
                run += 1
                m = cum / run
                if m < best_mean:
                    best_mean, best_len = m, run
                    best_r, best_q = r_start + start, q_start + start

    return best_mean, best_len, best_r, best_q


def _sw_single(S, nr, nq, max_drift, gap_open, gap_extend, mask=None):
    """Single SW alignment."""
    H = np.zeros((nr + 1, nq + 1))
    tb_di = np.zeros((nr + 1, nq + 1), dtype=np.int32)
    tb_dj = np.zeros((nr + 1, nq + 1), dtype=np.int32)
    best_score = 0.0
    best_pos = (0, 0)

    for i in range(1, nr + 1):
        for j in range(1, nq + 1):
            if mask is not None and mask[i - 1, j - 1]:
                continue
            sij = S[i - 1, j - 1]
            best_val = 0.0
            best_di, best_dj = 0, 0

            v = H[i - 1, j - 1] + sij
            if v > best_val:
                best_val, best_di, best_dj = v, 1, 1
            for d in range(2, min(max_drift + 1, j + 1)):
                cost = gap_open + gap_extend * (d - 2)
                v = H[i - 1, j - d] + sij - cost
                if v > best_val:
                    best_val, best_di, best_dj = v, 1, d
            for d in range(2, min(max_drift + 1, i + 1)):
                cost = gap_open + gap_extend * (d - 2)
                v = H[i - d, j - 1] + sij - cost
                if v > best_val:
                    best_val, best_di, best_dj = v, d, 1

            H[i, j] = best_val
            tb_di[i, j] = best_di
            tb_dj[i, j] = best_dj
            if best_val > best_score:
                best_score = best_val
                best_pos = (i, j)

    path = []
    i, j = best_pos
    while H[i, j] > 0:
        path.append((i - 1, j - 1))
        di, dj = int(tb_di[i, j]), int(tb_dj[i, j])
        if di == 0 and dj == 0:
            break
        i -= di
        j -= dj
    path.reverse()
    return best_score, path


def island_match_score_sw(mmd_matrix: np.ndarray, config):
    """Multi-chain local alignment on the MMD matrix."""
    nr, nq = mmd_matrix.shape
    if nr == 0 or nq == 0:
        return 0.0, 0, float("inf"), []

    S = config.sw_tau - mmd_matrix
    mask = np.zeros((nr, nq), dtype=bool)
    overlap = config.stride / config.window_size

    all_paths: List[List[Tuple[int, int]]] = []
    total_score = 0.0
    all_mmds: List[float] = []
    total_eff_nt = 0
    best_first = None

    for _ in range(config.sw_max_chains):
        raw, path = _sw_single(S, nr, nq, config.sw_max_drift,
                               config.sw_gap_open, config.sw_gap_extend, mask)
        if not path or raw <= 0:
            break
        if best_first is None:
            best_first = raw
        elif raw < config.sw_min_score_frac * best_first:
            break

        all_paths.append(path)
        total_score += raw * overlap
        total_eff_nt += (config.window_size - config.stride) + len(path) * config.stride
        all_mmds.extend(float(mmd_matrix[pi, pj]) for pi, pj in path)

        for pi, pj in path:
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = pi + di, pj + dj
                    if 0 <= ni < nr and 0 <= nj < nq:
                        mask[ni, nj] = True

    if not all_paths:
        return 0.0, 0, float("inf"), []
    return total_score, total_eff_nt, float(np.mean(all_mmds)), all_paths


def get_matched_region_nt(path, side: int, config) -> Tuple[int, int]:
    """Get matched region in nucleotide coordinates from alignment path."""
    wins = [p[side] for p in path]
    return min(wins) * config.stride, max(wins) * config.stride + config.window_size


# ===========================================================================
# Matcher wrapper
# ===========================================================================

class RnafmMatcher:
    """windowed re-embedding -> window-MMD matrix -> multi-chain SW."""

    representation = "windows"     # prepare_island returns List[(window_size, k)]

    async def prepare_island(self, seq, gpu, job_id, island_id, config):
        seq_len = len(seq)
        if seq_len < config.window_size:
            emb = await gpu.embed(job_id, f"{island_id}:full", seq, mean_pool=False)
            return [emb]
        starts = list(range(0, seq_len - config.window_size + 1, config.stride))
        windows = [seq[s:s + config.window_size] for s in starts]
        tasks = [gpu.embed(job_id, f"{island_id}:w{i}", w, mean_pool=False)
                 for i, w in enumerate(windows)]
        results = await asyncio.gather(*tasks)
        return list(results)

    def dist_ceiling(self, config):
        return config.max_match_mmd        # reject matches with mean MMD above this

    def gene_precompute(self, ref_reprs, q_reprs, valid_pairs, config):
        # Estimate gamma once from all active windows, precompute per-island
        # self-kernels reused across all pair comparisons (as before).
        all_windows = [w for embs in ref_reprs if embs for w in embs]
        all_windows.extend(w for embs in q_reprs if embs for w in embs)
        gamma = estimate_gamma_global(all_windows)
        ref_xx = [None] * len(ref_reprs)
        query_yy = [None] * len(q_reprs)
        for ri in {ri for ri, _ in valid_pairs}:
            ref_xx[ri] = precompute_self_kernels_batch(ref_reprs[ri], gamma)
        for qi in {qi for _, qi in valid_pairs}:
            query_yy[qi] = precompute_self_kernels_batch(q_reprs[qi], gamma)
        return gamma, ref_xx, query_yy

    def score_pair(self, ri, qi, ref_reprs, q_reprs, ctx, config):
        gamma, ref_xx, query_yy = ctx
        mat, _nc, _ns = compute_mmd_matrix_fast(
            ref_reprs[ri], q_reprs[qi], gamma,
            ref_xx[ri], query_yy[qi],
            config.mmd_skip, config.mean_dist_threshold,
        )
        sc, eff, mm, paths = island_match_score_sw(mat, config)
        if not paths or sc <= 0:
            return EMPTY_MATCH
        chains = []
        for path in paths:
            if not path:
                continue
            rs, re = get_matched_region_nt(path, side=0, config=config)
            qs, qe = get_matched_region_nt(path, side=1, config=config)
            pmmd = float(np.mean([mat[p[0], p[1]] for p in path]))
            chains.append(Chain(int(rs), int(re), int(qs), int(qe), pmmd))
        return MatchResult(score=float(sc), dist=float(mm), eff_nt=int(eff), chains=chains)
