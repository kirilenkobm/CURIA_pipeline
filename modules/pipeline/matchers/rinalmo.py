"""RiNALMo island matcher: embed-once -> per-token cosine dotplot -> nt-SW.

RiNALMo embeddings are context-stable, so each island is embedded ONCE and
matched at nucleotide resolution by Smith-Waterman local alignment on the
per-token cosine dotplot. This beats the RNA-FM window-MMD approach on both
accuracy and cost in the flank-diluted regime (see
notebooks/matching_benchmark.ipynb: AUC 0.993 vs 0.651, 100% core localization,
~0.9 ms/pair compiled). Representation is the deployed k=16 matching PCA
(the GPU executor's per-token / matching path is unchanged).

The SW DP + traceback is JIT-compiled with numba; a pure-Python fallback with
identical comparison order is used if numba is unavailable.
"""

from __future__ import annotations

import numpy as np

from modules.pipeline.matchers.base import Chain, MatchResult, EMPTY_MATCH

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:                       # pragma: no cover - numba optional
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap


def _sw_dp(S, tau, gap):
    """Smith-Waterman local alignment on a (La,Lb) score matrix S = cos - tau.

    Returns (best_score, r0, r1, q0, q1, mean_cos) with island-relative nt
    ranges [r0,r1) / [q0,q1) and the mean cosine along matched (diagonal) cells.
    Comparison / tie-break order is kept identical between the numba and
    pure-Python paths so tracebacks match bit-for-bit.
    """
    la = S.shape[0]
    lb = S.shape[1]
    H = np.zeros((la + 1, lb + 1))
    ptr = np.zeros((la + 1, lb + 1), dtype=np.int8)   # 1=diag, 2=up, 3=left
    best = 0.0
    bi = 0
    bj = 0
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            d = H[i - 1, j - 1] + S[i - 1, j - 1]
            u = H[i - 1, j] - gap
            l = H[i, j - 1] - gap
            v = 0.0
            p = 0
            if d > v:
                v = d
                p = 1
            if u > v:
                v = u
                p = 2
            if l > v:
                v = l
                p = 3
            H[i, j] = v
            ptr[i, j] = p
            if v > best:
                best = v
                bi = i
                bj = j
    # traceback
    i = bi
    j = bj
    r_lo = la
    r_hi = -1
    q_lo = lb
    q_hi = -1
    sum_cos = 0.0
    n = 0
    while i > 0 and j > 0 and H[i, j] > 0.0:
        p = ptr[i, j]
        if p == 1:
            ri = i - 1
            qj = j - 1
            if ri < r_lo:
                r_lo = ri
            if ri > r_hi:
                r_hi = ri
            if qj < q_lo:
                q_lo = qj
            if qj > q_hi:
                q_hi = qj
            sum_cos += S[i - 1, j - 1] + tau     # S = cos - tau -> add tau back
            n += 1
            i -= 1
            j -= 1
        elif p == 2:
            i -= 1
        elif p == 3:
            j -= 1
        else:
            break
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return (float(best), float(r_lo), float(r_hi + 1),
            float(q_lo), float(q_hi + 1), float(sum_cos / n))


_sw_dp_jit = njit(cache=True)(_sw_dp)
_SW = _sw_dp_jit if _HAS_NUMBA else _sw_dp


def _dotplot_sw(A, B, tau, gap):
    """L2-normalize rows, build the cosine dotplot, run SW. Returns
    (score, r0, r1, q0, q1, mean_cos) with integer nt ranges."""
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    S = np.ascontiguousarray((An @ Bn.T).astype(np.float64)) - tau
    if S.shape[0] == 0 or S.shape[1] == 0:
        return 0.0, 0, 0, 0, 0, 0.0
    best, r0, r1, q0, q1, mean_cos = _SW(S, float(tau), float(gap))
    return best, int(r0), int(r1), int(q0), int(q1), mean_cos


class RinalmoMatcher:
    """embed-once -> cosine dotplot -> nt-Smith-Waterman."""

    representation = "full"        # prepare_island returns the full (L, k) array

    async def prepare_island(self, seq, gpu, job_id, island_id, config):
        emb = await gpu.embed(job_id, f"{island_id}:full", seq, mean_pool=False)
        return np.ascontiguousarray(emb, dtype=np.float32)

    def dist_ceiling(self, config):
        return config.max_match_dist       # reject matches with (1 - mean cos) above this

    def gene_precompute(self, ref_reprs, q_reprs, valid_pairs, config):
        return None                # dotplot needs no shared per-gene state

    def score_pair(self, ri, qi, ref_reprs, q_reprs, ctx, config):
        ref_repr = ref_reprs[ri]
        q_repr = q_reprs[qi]
        if ref_repr is None or q_repr is None or len(ref_repr) == 0 or len(q_repr) == 0:
            return EMPTY_MATCH
        score, r0, r1, q0, q1, mean_cos = _dotplot_sw(
            ref_repr, q_repr, config.sw_tau_cos, config.sw_gap)
        if score <= 0.0 or r1 <= r0 or q1 <= q0:
            return EMPTY_MATCH
        # Distance = 1/(1+score) (lower = better). The SW *score* integrates
        # similarity over the aligned band LENGTH and is the real discriminator
        # (AUC 0.993 on Rfam same/cross-family); mean-cos alone is weak (0.79)
        # because a spurious short high-cos band inflates it. This monotonic map
        # keeps the same ranking while giving the shared layer a bounded, positive,
        # lower-is-better quality comparable to a max_match_dist ceiling.
        dist = 1.0 / (1.0 + score)
        eff_nt = ((r1 - r0) + (q1 - q0)) // 2
        return MatchResult(score=float(score), dist=float(dist), eff_nt=int(eff_nt),
                           chains=[Chain(int(r0), int(r1), int(q0), int(q1), float(dist))])


# JIT warmup at import so the first real pair isn't charged the ~1-2 s compile
# (cache=True persists the compiled kernel across processes).
if _HAS_NUMBA:
    try:
        _sw_dp_jit(np.zeros((2, 2), dtype=np.float64), 0.5, 0.3)
    except Exception:                   # pragma: no cover
        pass
