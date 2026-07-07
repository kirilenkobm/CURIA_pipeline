#!/usr/bin/env python3
"""Standalone tests for the RiNALMo island matcher (CPU-only, no GPU/model).

Run:  .venv/bin/python tests/test_rinalmo_matcher.py
Exits non-zero on failure. Covers:
  1. numba SW == pure-Python SW (bit-identical traceback/score).
  2. Core localization: the aligned band overlaps a planted core >=50% both sides.
  3. Direction: a shared-core pair has LOWER dist (better) than a different-core pair,
     and passes the deployed filter while the different-core pair does not.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.pipeline.matchers import rinalmo as rm
from modules.pipeline.matchers.base import MatchResult

RNG = np.random.RandomState(0)
CFG = SimpleNamespace(sw_tau_cos=0.5, sw_gap=0.3, max_match_dist=0.1, min_match_eff_nt=40)
D = 16


def _emb(n):
    return RNG.randn(n, D).astype(np.float32)


def _island(core, flank=100, noise=0.0):
    """Plant `core` (Lc,D) between random flanks; optional per-row noise on core."""
    c = core + noise * RNG.randn(*core.shape).astype(np.float32) if noise else core
    left, right = _emb(flank), _emb(flank)
    seq = np.vstack([left, c, right]).astype(np.float32)
    return seq, (flank, flank + len(core))


def _overlap(lo, hi, span):
    return max(0, min(hi, span[1]) - max(lo, span[0])) / max(1, span[1] - span[0])


def test_numba_parity():
    assert rm._HAS_NUMBA, "numba not available — parity test would be vacuous"
    bad = 0
    for _ in range(10):
        la, lb = RNG.randint(20, 120), RNG.randint(20, 120)
        S = (RNG.randn(la, lb) * 0.6).astype(np.float64)
        a = rm._sw_dp(S, 0.5, 0.3)
        b = rm._sw_dp_jit(S, 0.5, 0.3)
        if abs(a[0] - b[0]) > 1e-9 or a[1:5] != b[1:5] or abs(a[5] - b[5]) > 1e-9:
            bad += 1
    assert bad == 0, f"numba vs pure-Python mismatch on {bad}/10 matrices"
    print("  [ok] numba == pure-Python on 10 random matrices")


def test_localization():
    core = _emb(80)
    n = 30
    hits = 0
    for _ in range(n):
        ref, rs = _island(core, flank=RNG.randint(60, 140))
        q, qs = _island(core, flank=RNG.randint(60, 140), noise=0.15)
        sc, r0, r1, q0, q1, _mc = rm._dotplot_sw(ref, q, 0.5, 0.3)
        if sc > 0 and _overlap(r0, r1, rs) >= 0.5 and _overlap(q0, q1, qs) >= 0.5:
            hits += 1
    assert hits >= int(0.9 * n), f"localization only {hits}/{n}"
    print(f"  [ok] core localized >=50% both sides in {hits}/{n} planted pairs")


def test_direction_and_filter():
    m = rm.RinalmoMatcher()
    core_a = _emb(80)
    n = 30
    pos_pass = 0
    neg_pass = 0
    lower = 0
    for _ in range(n):
        ref, _ = _island(core_a, flank=100)
        q_same, _ = _island(core_a, flank=100, noise=0.15)
        q_diff, _ = _island(_emb(80), flank=100)     # unrelated core
        rp = m.score_pair(0, 0, [ref], [q_same], None, CFG)
        rn = m.score_pair(0, 0, [ref], [q_diff], None, CFG)
        if rp.dist < rn.dist:
            lower += 1
        if rp.score > 0 and rp.dist <= CFG.max_match_dist and rp.eff_nt >= CFG.min_match_eff_nt:
            pos_pass += 1
        if rn.score > 0 and rn.dist <= CFG.max_match_dist and rn.eff_nt >= CFG.min_match_eff_nt:
            neg_pass += 1
    assert lower >= int(0.9 * n), f"same-core dist not lower in {lower}/{n}"
    assert pos_pass >= int(0.9 * n), f"same-core passed filter only {pos_pass}/{n}"
    assert neg_pass <= int(0.1 * n), f"different-core wrongly passed filter {neg_pass}/{n}"
    print(f"  [ok] same-core dist<diff-core in {lower}/{n}; "
          f"filter pass same={pos_pass}/{n} diff={neg_pass}/{n}")


def test_empty_and_types():
    m = rm.RinalmoMatcher()
    r = m.score_pair(0, 0, [np.zeros((0, D), np.float32)], [_emb(50)], None, CFG)
    assert isinstance(r, MatchResult) and r.score == 0.0 and not r.chains
    print("  [ok] empty island -> empty MatchResult")


def test_orchestrator_both_matchers():
    """Drive the refactored _compute_island_alignments (shared layer) with each
    matcher on synthetic islands — no GPU. Confirms wiring + well-formed rows."""
    from modules.pipeline.island_alignment import (
        _compute_island_alignments, IslandAlignmentConfig,
    )
    from modules.pipeline.matchers.rinalmo import RinalmoMatcher
    from modules.pipeline.matchers.rnafm import RnafmMatcher

    def isl(start, end):
        return {"chrom": "chr1", "start": start, "end": end, "strand": 1}

    # --- RiNALMo path: full (L,16) reprs with a shared core in island 0<->0 ---
    core = _emb(80)
    ref0, _ = _island(core, flank=100)          # ref island 0 contains the core
    q0, _ = _island(core, flank=100, noise=0.15)  # query island 0: same core
    ref1 = _emb(200)                             # unrelated islands
    q1 = _emb(200)
    ref_islands = [isl(0, len(ref0)), isl(500, 500 + len(ref1))]
    q_islands = [isl(0, len(q0)), isl(500, 500 + len(q1))]
    ref_seqs = ["A" * len(ref0), "A" * len(ref1)]
    q_seqs = ["A" * len(q0), "A" * len(q1)]
    cfg_ri = IslandAlignmentConfig.for_model("rinalmo")
    rows = _compute_island_alignments(
        "gene1", ref_islands, q_islands, ref_seqs, q_seqs,
        [ref0, ref1], [q0, q1], cfg_ri,
        {(0, 0), (0, 1), (1, 0), (1, 1)}, RinalmoMatcher(),
    )
    matched = {(r["ref_island"], r["query_island"]) for r in rows}
    assert ("R0", "Q0") in matched, f"RiNALMo did not match the planted core pair; rows={matched}"
    for r in rows:
        import json as _json
        for ch in _json.loads(r["chains_json"]):
            ri = int(r["ref_island"][1:])
            assert ch["ref_to"] <= len(ref_seqs[ri]), "chain ref_to exceeds island length"
            float(ch["mmd"])                # parseable
        float(r["diag_mmd"])
    print(f"  [ok] RiNALMo orchestrator: {len(rows)} row(s), planted pair matched")

    # --- RNA-FM path: window reprs (list of (window_size,16)); just runs ---
    cfg_fm = IslandAlignmentConfig.for_model("rnafm")
    W = cfg_fm.window_size

    def wins(n_windows):
        return [_emb(W) for _ in range(n_windows)]

    rows_fm = _compute_island_alignments(
        "gene2", ref_islands, q_islands, ref_seqs, q_seqs,
        [wins(6), wins(6)], [wins(6), wins(6)], cfg_fm,
        {(0, 0), (0, 1), (1, 0), (1, 1)}, RnafmMatcher(),
    )
    assert isinstance(rows_fm, list)
    print(f"  [ok] RNA-FM orchestrator: ran cleanly, {len(rows_fm)} row(s)")


if __name__ == "__main__":
    print("RiNALMo matcher tests:")
    test_numba_parity()
    test_localization()
    test_direction_and_filter()
    test_empty_and_types()
    test_orchestrator_both_matchers()
    print("ALL PASS")
