#!/usr/bin/env python3
"""
Island alignment via windowed MMD on RNA-FM PCA embeddings (Step 4).

For each gene, compares reference stability islands against query islands using:
  1) Sliding-window RNA-FM embeddings projected to PCA space
  2) MMD distance between per-window point clouds
  3) Multi-chain Smith-Waterman alignment on the MMD matrix
  4) Hybrid monotonic chain: strict SW anchors + permissive fill

Outputs a TSV summary of matched regions per island pair.
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pyrion import TwoBitAccessor

# Import shared modules
import sys
from pathlib import Path as PathLib
script_dir = PathLib(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.global_PCA.apply_pca import apply_pca, load_pca
from modules.utils.mmd_utils import compute_mmd_matrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE = 80
STRIDE = 4
BATCH_SIZE = 64
MIN_ISLAND_LEN = 48

# MMD computation
MMD_SKIP = 1.0
MEAN_DIST_THRESHOLD = 3.0

# Smith-Waterman
SW_TAU = 0.10
SW_MAX_DRIFT = 3
SW_GAP_OPEN = 0.03
SW_GAP_EXTEND = 0.01
SW_MIN_SCORE_FRAC = 0.3
SW_MAX_CHAINS = 5

# Hook-based pair selection (level-3 optimisation)
PROBE_HW = 2
HOOK_BUFFER = 1
HOOK_MMD_THRESHOLD = 0.15


# ===========================================================================
# Joblist creation
# ===========================================================================

def write_island_alignment_joblist(
    ref_islands_json_path: str,
    u2q_map_path: str,
    query_islands_json_path: str,
    out_joblist_path: str,
) -> int:
    """
    Create joblist for island alignment jobs.

    Format: gene_id (one line per gene with both ref and query islands)

    Returns number of jobs created.
    """
    with open(ref_islands_json_path, "r") as f:
        ref_data = json.load(f)

    with open(u2q_map_path, "r") as f:
        u_to_query = json.load(f)

    with open(query_islands_json_path, "r") as f:
        query_islands_data = json.load(f)

    jobs_created = 0
    genes_with_both = 0
    genes_no_ref = 0
    genes_no_query = 0

    with open(out_joblist_path, "w") as dst:
        dst.write("gene_id\tn_ref_islands\tn_query_islands\n")

        for gene_id in sorted(ref_data.keys()):
            # Check if gene has reference islands
            ref_islands = [
                i for i in ref_data[gene_id].get("islands", [])
                if i["end"] - i["start"] >= MIN_ISLAND_LEN
            ]

            if not ref_islands:
                genes_no_ref += 1
                continue

            # Check if gene has query islands
            qr_ids = list(set(u_to_query.get(gene_id, [])))
            q_islands = [
                isl for qr in qr_ids
                for isl in query_islands_data.get(qr, [])
                if isl["end"] - isl["start"] >= MIN_ISLAND_LEN
            ]

            if not q_islands:
                genes_no_query += 1
                continue

            dst.write(f"{gene_id}\t{len(ref_islands)}\t{len(q_islands)}\n")
            jobs_created += 1
            genes_with_both += 1

    print(f"# Island alignment joblist summary:")
    print(f"#   Total genes in reference: {len(ref_data)}")
    print(f"#   Genes with reference islands: {len(ref_data) - genes_no_ref}")
    print(f"#   Genes with both ref & query islands: {genes_with_both}")
    print(f"#   Jobs created: {jobs_created}")

    return jobs_created


# ===========================================================================
# Job definition
# ===========================================================================

@dataclass(frozen=True)
class IslandAlignmentJob:
    gene_id: str
    n_ref_islands: int
    n_query_islands: int


def _load_joblist(joblist_path: str) -> List[IslandAlignmentJob]:
    """Load island alignment joblist."""
    jobs: List[IslandAlignmentJob] = []
    with open(joblist_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            jobs.append(IslandAlignmentJob(
                gene_id=row["gene_id"],
                n_ref_islands=int(row["n_ref_islands"]),
                n_query_islands=int(row["n_query_islands"]),
            ))
    return jobs


# ===========================================================================
# Sequence extraction
# ===========================================================================

def _fetch_seq(accessor: TwoBitAccessor, chrom: str, start: int, end: int,
               strand: int) -> str:
    """Extract sequence from 2bit, convert to RNA, handle strand."""
    seq = str(accessor.fetch(chrom, int(start), int(end))).upper()
    if strand == -1 or strand == "-":
        comp = {"A": "U", "T": "A", "G": "C", "C": "G", "N": "N"}
        seq = "".join(comp.get(b, "N") for b in reversed(seq))
    else:
        seq = seq.replace("T", "U")
    return seq


# ===========================================================================
# RNA-FM + PCA windowed embeddings
# ===========================================================================

def get_window_embeddings(
    seq: str,
    rna_fm_model,
    batch_converter,
    padding_idx: int,
    device: str,
    pca_model: Dict,
) -> List[np.ndarray]:
    """Slide windows over sequence, embed each, apply PCA."""
    import torch

    seq_len = len(seq)
    if seq_len < WINDOW_SIZE:
        # Embed full sequence
        data = [("seq", seq.upper().replace("T", "U"))]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            reps = rna_fm_model(tokens, repr_layers=[12])["representations"][12]
            length = int((tokens[0] != padding_idx).sum().item())
            if length <= 2:
                raw_emb = np.zeros((1, reps.shape[-1]), dtype=np.float32)
            else:
                raw_emb = reps[0, 1:length - 1, :].cpu().numpy().astype(np.float32)
        return [apply_pca(raw_emb, pca_model)]

    # Slide windows
    windows = [seq[s:s + WINDOW_SIZE]
               for s in range(0, seq_len - WINDOW_SIZE + 1, STRIDE)]

    all_pca = []
    for i in range(0, len(windows), BATCH_SIZE):
        batch = windows[i:i + BATCH_SIZE]
        data = [(f"s{j}", s.upper().replace("T", "U")) for j, s in enumerate(batch)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            reps = rna_fm_model(tokens, repr_layers=[12])["representations"][12]
            for b in range(reps.shape[0]):
                length = int((tokens[b] != padding_idx).sum().item())
                if length <= 2:
                    raw_emb = np.zeros((1, reps.shape[-1]), dtype=np.float32)
                else:
                    raw_emb = reps[b, 1:length - 1, :].cpu().numpy().astype(np.float32)
                all_pca.append(apply_pca(raw_emb, pca_model))

    return all_pca


# ===========================================================================
# Diagonal-run heuristic
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


# ===========================================================================
# Multi-chain Smith-Waterman on MMD matrix
# ===========================================================================

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


def island_match_score_sw(mmd_matrix: np.ndarray):
    """Multi-chain local alignment on the MMD matrix."""
    nr, nq = mmd_matrix.shape
    if nr == 0 or nq == 0:
        return 0.0, 0, float("inf"), []

    S = SW_TAU - mmd_matrix
    mask = np.zeros((nr, nq), dtype=bool)
    overlap = STRIDE / WINDOW_SIZE

    all_paths: List[List[Tuple[int, int]]] = []
    total_score = 0.0
    all_mmds: List[float] = []
    total_eff_nt = 0
    best_first = None

    for _ in range(SW_MAX_CHAINS):
        raw, path = _sw_single(S, nr, nq, SW_MAX_DRIFT,
                               SW_GAP_OPEN, SW_GAP_EXTEND, mask)
        if not path or raw <= 0:
            break
        if best_first is None:
            best_first = raw
        elif raw < SW_MIN_SCORE_FRAC * best_first:
            break

        all_paths.append(path)
        total_score += raw * overlap
        total_eff_nt += (WINDOW_SIZE - STRIDE) + len(path) * STRIDE
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


# ===========================================================================
# Matched-region coordinate helpers
# ===========================================================================

def get_matched_region_nt(path, side: int) -> Tuple[int, int]:
    """Get matched region in nucleotide coordinates from alignment path."""
    wins = [p[side] for p in path]
    return min(wins) * STRIDE, max(wins) * STRIDE + WINDOW_SIZE


# ===========================================================================
# Hook-based island pair selection (level-3 optimisation)
# ===========================================================================

def _get_probe_pairs(n_ref: int, n_q: int) -> List[Tuple[int, int]]:
    """Generate positional probe pairs."""
    pairs = set()
    for ri in range(n_ref):
        q_exp = round(ri * max(n_q - 1, 0) / max(n_ref - 1, 1))
        for off in range(-PROBE_HW, PROBE_HW + 1):
            qi = q_exp + off
            if 0 <= qi < n_q:
                pairs.add((ri, qi))
    return sorted(pairs)


def select_pairs_to_compute(n_ref, n_q, pair_results, mmd_matrices):
    """Phase 2-3: find hooks from probes, determine fill pairs."""
    best_per_ref: Dict[int, Tuple[int, float, float]] = {}
    for pr in pair_results:
        ri, qi = pr["ri"], pr["qi"]
        mmd_val = pr["diag_mean_mmd"]
        run_len = pr["diag_run_len"]
        if mmd_val > HOOK_MMD_THRESHOLD or run_len < 2:
            continue
        score = -mmd_val + 0.001 * run_len
        if ri not in best_per_ref or score > best_per_ref[ri][1]:
            best_per_ref[ri] = (qi, score, mmd_val)

    mono_hooks = []
    last_q = -1
    for ri in sorted(best_per_ref):
        qi, _, _ = best_per_ref[ri]
        if qi > last_q:
            mono_hooks.append((ri, qi))
            last_q = qi

    bounded = [(-1, -1)] + mono_hooks + [(n_ref, n_q)]
    fill_pairs = set()
    for k in range(len(bounded) - 1):
        r_lo = bounded[k][0] + 1
        q_lo = max(0, bounded[k][1] + 1 - HOOK_BUFFER)
        r_hi = bounded[k + 1][0] - 1
        q_hi = min(n_q - 1, bounded[k + 1][1] - 1 + HOOK_BUFFER)
        for ri in range(max(0, r_lo), min(n_ref, r_hi + 1)):
            for qi in range(max(0, q_lo), min(n_q, q_hi + 1)):
                if (ri, qi) not in mmd_matrices:
                    fill_pairs.add((ri, qi))

    return mono_hooks, sorted(fill_pairs)


# ===========================================================================
# Monotonic chain DP
# ===========================================================================

def _monotonic_chain_dp(candidates):
    """Generic monotonic chain DP.  candidates = [(ri, qi, score, ...)]"""
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda x: (x[0], x[1]))
    nc = len(candidates)
    dp = [c[2] for c in candidates]
    parent = [-1] * nc
    for k in range(nc):
        for p in range(k):
            if candidates[p][0] < candidates[k][0] and \
               candidates[p][1] < candidates[k][1]:
                val = dp[p] + candidates[k][2]
                if val > dp[k]:
                    dp[k] = val
                    parent[k] = p
    best_end = int(np.argmax(dp))
    chain = []
    idx = best_end
    while idx >= 0:
        chain.append(candidates[idx])
        idx = parent[idx]
    chain.reverse()
    return chain


# ===========================================================================
# Hybrid chain: strict anchors + permissive fill
# ===========================================================================

def build_hybrid_chain(sw_chain, sw_results, pair_results, n_ref, n_q):
    """Build hybrid chain combining SW anchors with permissive fill."""
    anchors = [(ri, qi) for ri, qi, *_ in sw_chain]

    max_mmd_val = max(
        (pr["diag_mean_mmd"] for pr in pair_results
         if np.isfinite(pr["diag_mean_mmd"])),
        default=1.0,
    )
    perm_pool = {}
    for pr in pair_results:
        ri, qi = pr["ri"], pr["qi"]
        mmd_val = pr["diag_mean_mmd"]
        if not np.isfinite(mmd_val):
            continue
        perm_pool[(ri, qi)] = {
            "score": max_mmd_val - mmd_val,
            "mmd": mmd_val,
            "run_len": pr["diag_run_len"],
        }

    anchors_set = set(anchors)

    def _fill_gap(r_lo, r_hi, q_lo, q_hi):
        cands = []
        for (ri, qi), info in perm_pool.items():
            if ri < r_lo or ri > r_hi or qi < q_lo or qi > q_hi:
                continue
            if (ri, qi) in anchors_set or info["score"] <= 0:
                continue
            cands.append((ri, qi, info["score"], info["mmd"], info["run_len"]))
        return _monotonic_chain_dp(cands) if cands else []

    anchors_sorted = sorted(anchors)
    boundaries = [(-1, -1)] + anchors_sorted + [(n_ref, n_q)]
    hybrid = []

    for k in range(len(boundaries) - 1):
        lo_r, lo_q = boundaries[k]
        hi_r, hi_q = boundaries[k + 1]

        if (lo_r, lo_q) in anchors_set:
            info = perm_pool.get((lo_r, lo_q))
            sw = next((r for r in sw_results
                       if r["ri"] == lo_r and r["qi"] == lo_q), None)
            mmd = info["mmd"] if info else (sw["sw_mean_mmd"] if sw else 0)
            rlen = info["run_len"] if info else 0
            hybrid.append((lo_r, lo_q, "anchor", mmd, rlen))

        for ri, qi, sc, mmd, rlen in _fill_gap(
                lo_r + 1, hi_r - 1, lo_q + 1, hi_q - 1):
            hybrid.append((ri, qi, "fill", mmd, rlen))

    if anchors_sorted:
        lr, lq = anchors_sorted[-1]
        if not hybrid or hybrid[-1][:2] != (lr, lq):
            info = perm_pool.get((lr, lq))
            sw = next((r for r in sw_results
                       if r["ri"] == lr and r["qi"] == lq), None)
            mmd = info["mmd"] if info else (sw["sw_mean_mmd"] if sw else 0)
            rlen = info["run_len"] if info else 0
            hybrid.append((lr, lq, "anchor", mmd, rlen))

    return hybrid


# ===========================================================================
# Async job processor
# ===========================================================================

async def _process_job(
    job: IslandAlignmentJob,
    ref_data: Dict,
    u_to_query: Dict,
    query_islands_data: Dict,
    ref_acc: TwoBitAccessor,
    query_acc: TwoBitAccessor,
    rna_fm_model,
    batch_converter,
    padding_idx: int,
    device: str,
    pca_model: Dict,
) -> List[Dict]:
    """Process a single island alignment job."""
    gene_id = job.gene_id

    # Select islands
    ref_islands = sorted(
        [i for i in ref_data[gene_id]["islands"]
         if i["end"] - i["start"] >= MIN_ISLAND_LEN],
        key=lambda x: x["start"],
    )

    qr_ids = list(set(u_to_query.get(gene_id, [])))
    q_islands = sorted(
        [isl for qr in qr_ids
         for isl in query_islands_data.get(qr, [])
         if isl["end"] - isl["start"] >= MIN_ISLAND_LEN],
        key=lambda x: x["start"],
    )

    n_ref = len(ref_islands)
    n_q = len(q_islands)

    if n_ref == 0 or n_q == 0:
        return []

    # Extract sequences
    ref_seqs = [_fetch_seq(ref_acc, i["chrom"], i["start"], i["end"],
                           i["strand"]) for i in ref_islands]
    q_seqs = [_fetch_seq(query_acc, i["chrom"], i["start"], i["end"],
                         i["strand"]) for i in q_islands]

    # Compute window embeddings
    ref_win_embs = [get_window_embeddings(s, rna_fm_model, batch_converter,
                                          padding_idx, device, pca_model)
                    for s in ref_seqs]
    q_win_embs = [get_window_embeddings(s, rna_fm_model, batch_converter,
                                        padding_idx, device, pca_model)
                  for s in q_seqs]

    # Phase 1: Positional probes
    probe_pairs = _get_probe_pairs(n_ref, n_q)
    pair_results = []
    mmd_matrices: Dict[Tuple[int, int], np.ndarray] = {}

    def _do_pair(ri, qi):
        mat, nc, ns = compute_mmd_matrix(ref_win_embs[ri], q_win_embs[qi],
                                         MMD_SKIP, MEAN_DIST_THRESHOLD)
        mmd_matrices[(ri, qi)] = mat
        mr = max(2, min(len(ref_win_embs[ri]), len(q_win_embs[qi])) // 3)
        mm, rl, rs, qs = best_diagonal_run(mat, min_run=mr)
        pair_results.append({
            "ri": ri, "qi": qi,
            "ref_len": len(ref_seqs[ri]), "query_len": len(q_seqs[qi]),
            "n_ref_win": len(ref_win_embs[ri]),
            "n_q_win": len(q_win_embs[qi]),
            "diag_mean_mmd": mm, "diag_run_len": rl,
        })

    for ri, qi in probe_pairs:
        _do_pair(ri, qi)

    # Phase 2-3: Hooks + fill
    hooks, fill_pairs = select_pairs_to_compute(
        n_ref, n_q, pair_results, mmd_matrices)

    for ri, qi in fill_pairs:
        _do_pair(ri, qi)

    # SW scoring
    sw_results = []
    sw_paths = {}
    for (ri, qi), mat in sorted(mmd_matrices.items()):
        sc, eff, mm, paths = island_match_score_sw(mat)
        sw_paths[(ri, qi)] = paths
        old = next((r for r in pair_results
                    if r["ri"] == ri and r["qi"] == qi), None)
        sw_results.append({
            "ri": ri, "qi": qi,
            "sw_score": sc, "sw_eff_nt": eff, "sw_mean_mmd": mm,
            "sw_n_chains": len(paths),
            "sw_path_len": sum(len(p) for p in paths),
            "diag_mean_mmd": old["diag_mean_mmd"] if old else float("inf"),
            "diag_run_len": old["diag_run_len"] if old else 0,
        })

    # Strict monotonic chain (SW)
    sw_cands = [(r["ri"], r["qi"], r["sw_score"], r["sw_mean_mmd"],
                 r["sw_eff_nt"], r["sw_n_chains"])
                for r in sw_results if r["sw_score"] > 0]
    sw_chain = _monotonic_chain_dp(sw_cands)

    # Hybrid chain
    hybrid = build_hybrid_chain(sw_chain, sw_results, pair_results,
                                n_ref, n_q)

    # Build output rows
    rows = []
    max_ch = max((len(sw_paths.get((h[0], h[1]), [])) for h in hybrid),
                 default=1)

    for ri, qi, typ, mmd_val, run_len in hybrid:
        ri_isl = ref_islands[ri]
        qi_isl = q_islands[qi]

        paths = sw_paths.get((ri, qi), [])
        if not paths or all(len(p) == 0 for p in paths):
            mat = mmd_matrices.get((ri, qi))
            if mat is not None:
                mr = max(2, min(mat.shape[0], mat.shape[1]) // 3)
                _, dl, dr, dq = best_diagonal_run(mat, min_run=mr)
                if dl >= 2:
                    paths = [[(dr + k, dq + k) for k in range(dl)]]

        row = {
            "gene_id": gene_id,
            "ref_island": f"R{ri}",
            "query_island": f"Q{qi}",
            "type": typ,
            "ref_chrom": ri_isl["chrom"],
            "ref_start": ri_isl["start"],
            "ref_end": ri_isl["end"],
            "ref_len": ri_isl["end"] - ri_isl["start"],
            "query_chrom": qi_isl["chrom"],
            "query_start": qi_isl["start"],
            "query_end": qi_isl["end"],
            "query_len": qi_isl["end"] - qi_isl["start"],
            "n_chains": len(paths),
            "diag_mmd": f"{mmd_val:.4f}",
        }

        for ci in range(max_ch):
            if ci < len(paths) and paths[ci]:
                rs, re = get_matched_region_nt(paths[ci], side=0)
                re = min(re, len(ref_seqs[ri]))
                qs, qe = get_matched_region_nt(paths[ci], side=1)
                qe = min(qe, len(q_seqs[qi]))
                pmmd = np.mean([mmd_matrices[(ri, qi)][p[0], p[1]]
                                for p in paths[ci]])
                row[f"chain{ci + 1}_ref_from"] = rs
                row[f"chain{ci + 1}_ref_to"] = re
                row[f"chain{ci + 1}_q_from"] = qs
                row[f"chain{ci + 1}_q_to"] = qe
                row[f"chain{ci + 1}_mmd"] = f"{pmmd:.4f}"
            else:
                row[f"chain{ci + 1}_ref_from"] = ""
                row[f"chain{ci + 1}_ref_to"] = ""
                row[f"chain{ci + 1}_q_from"] = ""
                row[f"chain{ci + 1}_q_to"] = ""
                row[f"chain{ci + 1}_mmd"] = ""

        rows.append(row)

    return rows


# ===========================================================================
# Main scheduler
# ===========================================================================

def run_island_alignment_scheduler(
    joblist_path: str,
    ref_2bit_path: str,
    query_2bit_path: str,
    ref_islands_json_path: str,
    u2q_map_path: str,
    query_islands_json_path: str,
    input_q,
    output_q,
    output_tsv_path: str,
    max_concurrent: int = 128,
    test_cap_jobs: Optional[int] = None,
) -> None:
    """
    Run island alignment scheduler with GPU executor integration.

    Args:
        joblist_path: Path to island alignment joblist
        ref_2bit_path: Path to reference 2bit file
        query_2bit_path: Path to query 2bit file
        ref_islands_json_path: Path to reference islands JSON
        u2q_map_path: Path to ultimate-to-query mapping JSON
        query_islands_json_path: Path to query islands JSON
        input_q: GPU executor input queue
        output_q: GPU executor output queue
        output_tsv_path: Path to output TSV file
        max_concurrent: Maximum concurrent jobs
        test_cap_jobs: Optional cap on number of jobs for testing
    """
    import sys
    import os

    # Load RNA-FM model
    import torch

    # Locate RNA-FM
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    rna_fm_path = str(project_root.parent / "RNA-FM")
    if not os.path.exists(rna_fm_path):
        rna_fm_path = "/Users/Bogdan.Kirilenko/PycharmProjects/RNA-FM"
    if rna_fm_path not in sys.path:
        sys.path.insert(0, rna_fm_path)

    import fm

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    model, alphabet = fm.pretrained.rna_fm_t12()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    padding_idx = alphabet.padding_idx
    pca_model = load_pca()

    print(f"# RNA-FM loaded on {device}, PCA components={pca_model['n_components']}")

    # Load data
    jobs = _load_joblist(joblist_path)
    if test_cap_jobs:
        jobs = jobs[:test_cap_jobs]

    with open(ref_islands_json_path, "r") as f:
        ref_data = json.load(f)
    with open(u2q_map_path, "r") as f:
        u_to_query = json.load(f)
    with open(query_islands_json_path, "r") as f:
        query_islands_data = json.load(f)

    ref_acc = TwoBitAccessor(ref_2bit_path)
    query_acc = TwoBitAccessor(query_2bit_path)

    print(f"# Processing {len(jobs)} island alignment jobs...")

    # Determine max chains for TSV header
    max_chains_global = 5  # Use SW_MAX_CHAINS as default

    # Write TSV header
    with open(output_tsv_path, "w") as f:
        header = [
            "gene_id", "ref_island", "query_island", "type",
            "ref_chrom", "ref_start", "ref_end", "ref_len",
            "query_chrom", "query_start", "query_end", "query_len",
            "n_chains", "diag_mmd",
        ]
        for ci in range(max_chains_global):
            header.extend([
                f"chain{ci + 1}_ref_from", f"chain{ci + 1}_ref_to",
                f"chain{ci + 1}_q_from", f"chain{ci + 1}_q_to",
                f"chain{ci + 1}_mmd"
            ])
        f.write("\t".join(header) + "\n")

    # Process jobs
    async def _run():
        semaphore = asyncio.Semaphore(max_concurrent)
        lock = threading.Lock()
        completed = 0
        total = len(jobs)

        async def _worker(job):
            nonlocal completed
            async with semaphore:
                rows = await _process_job(
                    job, ref_data, u_to_query, query_islands_data,
                    ref_acc, query_acc, model, batch_converter,
                    padding_idx, device, pca_model,
                )

                with lock:
                    with open(output_tsv_path, "a") as f:
                        for row in rows:
                            # Ensure all columns exist
                            row_vals = [str(row.get(h, "")) for h in header]
                            f.write("\t".join(row_vals) + "\n")

                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        print(f"# Island alignment progress: {completed}/{total} jobs completed")

        tasks = [_worker(job) for job in jobs]
        await asyncio.gather(*tasks)

    asyncio.run(_run())

    print(f"# Island alignment completed. Results written to {output_tsv_path}")
