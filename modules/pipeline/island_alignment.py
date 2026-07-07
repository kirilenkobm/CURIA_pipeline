#!/usr/bin/env python3
"""
Island alignment (Step 4): match reference structured-RNA islands to query islands.

This module is the model-agnostic orchestrator. Per-pair scoring is delegated to
a matcher (modules/pipeline/matchers/): RiNALMo embeds each island once and aligns
the per-token cosine dotplot at nucleotide resolution (Smith-Waterman); RNA-FM
re-embeds sliding windows and scores an MMD matrix (legacy). Both emit a common
MatchResult, so the shared assignment/collinearity/output layer below is identical.

Only provenance-valid pairs (from liftover) are computed.  Matching
uses a three-stage pipeline:
  a) Quality filter — positive SW score, mean MMD ≤ threshold,
     minimum aligned nucleotides.
  b) Greedy pruning — each query island is assigned to the single
     best ref island (by lowest mean MMD).  One ref island may
     claim multiple query islands (1-to-many).
  c) Collinearity — the largest non-crossing subset is kept via
     LIS (same strand) or LDS (opposite strand), removing
     "spaghetti" connections.

Outputs a TSV summary of matched regions per island pair.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import csv
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from pyrion import TwoBitAccessor

# Import shared modules
import sys
from pathlib import Path as PathLib
script_dir = PathLib(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.pipeline.matchers.base import MatchResult, get_matcher

# ---------------------------------------------------------------------------
# Island alignment configuration
# ---------------------------------------------------------------------------

@dataclass
class IslandAlignmentConfig:
    """All tunable hyperparameters for the island-alignment step.

    Grouped into four blocks:
      1. Windowed embedding – how sequences are sliced before RNA-FM
      2. MMD matrix – kernel-MMD computation controls
      3. Smith-Waterman – local alignment on the MMD matrix
      4. Match quality – MMD ceiling, min aligned length
    """

    # -- windowed embedding ---------------------------------------------------
    window_size: int = 96
    stride: int = 4
    batch_size: int = 64
    min_island_len: int = 72

    # -- MMD matrix -----------------------------------------------------------
    mmd_skip: float = 1.0           # fill value for skipped pairs
    mean_dist_threshold: float = 3.0  # mean-centroid distance cutoff

    # -- Smith-Waterman -------------------------------------------------------
    sw_tau: float = 0.15            # score = tau − mmd  (match bonus)
    sw_max_drift: int = 3           # max off-diagonal gap in SW
    sw_gap_open: float = 0.03
    sw_gap_extend: float = 0.01
    sw_min_score_frac: float = 0.3  # secondary chain threshold vs. best
    sw_max_chains: int = 1          # max SW chains extracted per pair

    # -- match quality ---------------------------------------------------------
    max_match_mmd: float = 0.15      # reject matches with mean MMD above tau (RNA-FM)
    min_match_eff_nt: int = 40       # minimum aligned nucleotides to keep
    max_queries_per_ref: int = 2     # max query islands one ref island can claim

    # -- RiNALMo dotplot matcher (SW on the per-token cosine dotplot) ----------
    sw_tau_cos: float = 0.5          # cosine offset in the dotplot (score = cos − tau)
    sw_gap: float = 0.3              # SW gap penalty (up / left moves)
    max_match_dist: float = 0.5      # reject matches with (1 − mean cos) above this

    # -- embedding strategy ----------------------------------------------------
    # "windowed"   : re-embed every sliding window in its own context (RNA-FM,
    #                which drifts under flanking context).
    # "embed_once" : embed the whole island once and slice the per-token
    #                embeddings into windows locally (RiNALMo, context-stable).
    embed_strategy: str = "windowed"

    @classmethod
    def for_model(cls, model_name: Optional[str]) -> "IslandAlignmentConfig":
        """Build a config with per-model tuned params from the model registry.

        Falls back to the RNA-FM-tuned defaults if the registry has no entry.
        """
        try:
            from modules.model_registry import (
                get_island_align_params, get_embed_strategy,
            )
        except Exception:
            return cls()
        p = get_island_align_params(model_name)
        return cls(
            window_size=p.get("window_size", cls.window_size),
            stride=p.get("stride", cls.stride),
            min_island_len=p.get("min_island_len", cls.min_island_len),
            mean_dist_threshold=p.get("mean_dist_threshold", cls.mean_dist_threshold),
            sw_tau=p.get("sw_tau", cls.sw_tau),
            max_match_mmd=p.get("max_match_mmd", cls.max_match_mmd),
            min_match_eff_nt=p.get("min_match_eff_nt", cls.min_match_eff_nt),
            # RiNALMo dotplot matcher params (ignored by RNA-FM window-MMD path)
            sw_tau_cos=p.get("sw_tau_cos", cls.sw_tau_cos),
            sw_gap=p.get("sw_gap", cls.sw_gap),
            max_match_dist=p.get("max_match_dist", cls.max_match_dist),
            embed_strategy=get_embed_strategy(model_name),
        )


DEFAULT_CONFIG = IslandAlignmentConfig()


# ===========================================================================
# GPU client (shared GPU executor pattern)
# ===========================================================================

class GPUClient:
    """Async GPU client for embeddings via the shared GPU executor process."""
    def __init__(self, input_queue, output_queue, loop: asyncio.AbstractEventLoop):
        self._input = input_queue
        self._output = output_queue
        self._loop = loop
        self._pending: Dict[Tuple[str, str], asyncio.Future] = {}
        self._lock = threading.Lock()
        self._stopping = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    async def embed(self, worker_id: str, sequence_id: str, sequence: str, mean_pool: bool = False) -> np.ndarray:
        future = self._loop.create_future()
        key = (worker_id, sequence_id)
        with self._lock:
            self._pending[key] = future
        self._input.put((worker_id, sequence_id, sequence, {"mean_pool": mean_pool}))
        return await future

    def stop(self) -> None:
        self._stopping = True
        self._thread.join(timeout=2.0)

    def _reader(self) -> None:
        while not self._stopping:
            try:
                payload = self._output.get(timeout=0.5)
                for worker_id, sequence_id, emb in payload:
                    key = (worker_id, sequence_id)
                    if self._loop.is_closed():
                        return
                    self._loop.call_soon_threadsafe(self._resolve_future, key, emb)
            except Exception:
                if self._stopping or self._loop.is_closed():
                    return
                continue

    def _resolve_future(self, key: Tuple[str, str], emb: np.ndarray) -> None:
        with self._lock:
            future = self._pending.pop(key, None)
        if future is not None and not future.done():
            future.set_result(emb)


# ===========================================================================
# Joblist creation
# ===========================================================================

def write_island_alignment_joblist(
    ref_islands_json_path: str,
    u2q_map_path: str,
    query_islands_json_path: str,
    out_joblist_path: str,
    config: IslandAlignmentConfig = DEFAULT_CONFIG,
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
                if i["end"] - i["start"] >= config.min_island_len
            ]

            if not ref_islands:
                genes_no_ref += 1
                continue

            # Check if gene has query islands
            qr_ids = list(set(u_to_query.get(gene_id, [])))
            q_islands = [
                isl for qr in qr_ids
                for isl in query_islands_data.get(qr, [])
                if isl["end"] - isl["start"] >= config.min_island_len
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
# Per-pair scoring (moved to modules/pipeline/matchers/): RNA-FM window-MMD +
# SW lives in matchers/rnafm.py; RiNALMo dotplot + nt-SW in matchers/rinalmo.py.
# The orchestrator below dispatches to a matcher via get_matcher(model_name).
# ===========================================================================


# ===========================================================================
# Collinearity helpers (LIS / LDS)
# ===========================================================================

def _longest_increasing_subsequence(seq: List[int]) -> List[int]:
    """Return indices of the longest non-decreasing subsequence.

    Used for same-strand collinearity: Q indices should increase (or stay
    equal for 1-to-many from a single R) as R indices increase.
    O(n log n) via patience sorting.
    """
    import bisect
    if not seq:
        return []
    tails = []      # smallest tail value for IS of each length
    tail_idx = []   # index in seq of that tail
    parents = [-1] * len(seq)

    for i, val in enumerate(seq):
        pos = bisect.bisect_right(tails, val)
        if pos == len(tails):
            tails.append(val)
            tail_idx.append(i)
        else:
            tails[pos] = val
            tail_idx[pos] = i
        parents[i] = tail_idx[pos - 1] if pos > 0 else -1

    result = []
    idx = tail_idx[-1]
    while idx >= 0:
        result.append(idx)
        idx = parents[idx]
    result.reverse()
    return result


def _longest_decreasing_subsequence(seq: List[int]) -> List[int]:
    """Return indices of the longest non-increasing subsequence.

    Used for opposite-strand collinearity: Q indices should decrease as
    R indices increase.  Implemented by negating and running LIS.
    """
    negated = [-v for v in seq]
    return _longest_increasing_subsequence(negated)


# ===========================================================================
# CPU-bound alignment phase (runs in thread pool)
# ===========================================================================

def _compute_island_alignments(
    gene_id: str,
    ref_islands: List[Dict],
    q_islands: List[Dict],
    ref_seqs: List[Optional[str]],
    q_seqs: List[Optional[str]],
    ref_reprs: List[Optional[object]],
    q_reprs: List[Optional[object]],
    config: IslandAlignmentConfig,
    valid_pairs: Set[Tuple[int, int]],
    matcher,
) -> List[Dict]:
    """CPU-bound phase: per-pair scoring (via the model matcher) + assignment.

    Runs in a thread pool so the event loop stays responsive. The matcher owns
    per-pair scoring (RNA-FM window-MMD or RiNALMo dotplot + nt-SW) and returns
    a MatchResult; everything below the scoring loop — quality filter, balanced
    greedy assignment, collinearity, and row/chain serialization — is
    model-agnostic. Only (ri, qi) in valid_pairs (liftover provenance) are scored.
    """
    # Optional per-gene shared state (RNA-FM: MMD gamma + self-kernels; RiNALMo: None)
    ctx = matcher.gene_precompute(ref_reprs, q_reprs, valid_pairs, config)

    results: Dict[Tuple[int, int], MatchResult] = {}
    for ri, qi in sorted(valid_pairs):
        results[(ri, qi)] = matcher.score_pair(ri, qi, ref_reprs, q_reprs, ctx, config)

    # ---- Quality filter + balanced greedy + collinearity ---------------------
    # 1. Filter: positive score, distance <= per-model ceiling, min aligned nt.
    # 2. Balanced greedy: lowest-distance first; each Q once, each R capped.
    # 3. Collinearity: largest non-crossing subset (LIS/LDS) to avoid spaghetti.
    ceiling = matcher.dist_ceiling(config)
    candidates = [
        {"ri": ri, "qi": qi, "res": res}
        for (ri, qi), res in sorted(results.items())
        if (res.score > 0
            and res.dist <= ceiling
            and res.eff_nt >= config.min_match_eff_nt)
    ]

    candidates.sort(key=lambda c: c["res"].dist)
    assigned_qi: Set[int] = set()
    ri_count: Dict[int, int] = defaultdict(int)
    pruned = []
    for c in candidates:
        ri, qi = c["ri"], c["qi"]
        if qi in assigned_qi:
            continue
        if ri_count[ri] >= config.max_queries_per_ref:
            continue
        assigned_qi.add(qi)
        ri_count[ri] += 1
        pruned.append((ri, qi))

    # Collinearity: strand relationship decides increasing vs decreasing order.
    ref_strand = ref_islands[0].get("strand", 1) if ref_islands else 1
    q_strand = q_islands[0].get("strand", 1) if q_islands else 1
    same_strand = (ref_strand == q_strand)

    pruned.sort(key=lambda x: (x[0], x[1]))
    if len(pruned) > 1:
        q_seq = [qi for _, qi in pruned]
        if same_strand:
            keep_idx = _longest_increasing_subsequence(q_seq)
        else:
            keep_idx = _longest_decreasing_subsequence(q_seq)
        pruned = [pruned[i] for i in keep_idx]

    # Intra-island collinearity: sub-regions within one ref island must be
    # ordered consistently. Use each pair's first chain's ref start position
    # (nt); monotonicity is identical to window-index order (stride > 0).
    ri_groups: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for idx, (ri, qi) in enumerate(pruned):
        chains = results[(ri, qi)].chains
        ref_pos = chains[0].ref_from if chains else 0
        ri_groups[ri].append((idx, qi, ref_pos))

    drop_indices: Set[int] = set()
    for ri, group in ri_groups.items():
        if len(group) <= 1:
            continue
        group.sort(key=lambda x: x[1])  # sort by qi
        ref_positions = [ref_pos for _, _, ref_pos in group]
        kept_set = set(_longest_increasing_subsequence(ref_positions))
        for i, (orig_idx, _, _) in enumerate(group):
            if i not in kept_set:
                drop_indices.add(orig_idx)

    if drop_indices:
        pruned = [p for i, p in enumerate(pruned) if i not in drop_indices]

    accepted = set(pruned)

    # Build output rows (schema/columns unchanged; diag_mmd/chain "mmd" now hold
    # the matcher's distance — MMD for RNA-FM, 1-cos for RiNALMo).
    rows = []
    for c in candidates:
        ri, qi = c["ri"], c["qi"]
        if (ri, qi) not in accepted:
            continue
        res = c["res"]
        ri_isl = ref_islands[ri]
        qi_isl = q_islands[qi]

        row = {
            "gene_id": gene_id,
            "ref_island": f"R{ri}",
            "query_island": f"Q{qi}",
            "type": "match",
            "ref_chrom": ri_isl["chrom"],
            "ref_start": ri_isl["start"],
            "ref_end": ri_isl["end"],
            "ref_len": ri_isl["end"] - ri_isl["start"],
            "query_chrom": qi_isl["chrom"],
            "query_start": qi_isl["start"],
            "query_end": qi_isl["end"],
            "query_len": qi_isl["end"] - qi_isl["start"],
            "n_chains": len(res.chains),
            "diag_mmd": f"{res.dist:.4f}",
        }

        chains = []
        for ch in res.chains:
            chains.append({
                "ref_from": int(ch.ref_from),
                "ref_to": int(min(ch.ref_to, len(ref_seqs[ri]))),
                "q_from": int(ch.q_from),
                "q_to": int(min(ch.q_to, len(q_seqs[qi]))),
                "mmd": f"{ch.dist:.4f}",
            })

        row["chains_json"] = json.dumps(chains)
        rows.append(row)

    return rows


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
    gpu: GPUClient,
    config: IslandAlignmentConfig = DEFAULT_CONFIG,
    cpu_pool: concurrent.futures.ThreadPoolExecutor = None,
    clusters_data: Dict = None,
    matcher=None,
) -> List[Dict]:
    """Process a single island alignment job.

    The async phase (embedding) runs on the event loop via GPUClient.
    The CPU-bound phase (MMD, SW, chain DP) is offloaded to a thread pool
    so the event loop stays responsive for other workers' embedding requests.
    """
    gene_id = job.gene_id

    # Select ref islands, tracking each island's original index in the
    # unfiltered list so we can cross-reference liftover provenance.
    all_ref_islands = ref_data[gene_id]["islands"]
    ref_with_orig = sorted(
        [(orig_idx, isl) for orig_idx, isl in enumerate(all_ref_islands)
         if isl["end"] - isl["start"] >= config.min_island_len],
        key=lambda x: x[1]["start"],
    )
    orig_indices = [orig_idx for orig_idx, _ in ref_with_orig]
    ref_islands = [isl for _, isl in ref_with_orig]

    # Collect query islands, tagging each with its merged_query_id
    qr_ids = list(set(u_to_query.get(gene_id, [])))
    q_with_region = sorted(
        [(qr, isl) for qr in qr_ids
         for isl in query_islands_data.get(qr, [])
         if isl["end"] - isl["start"] >= config.min_island_len],
        key=lambda x: x[1]["start"],
    )
    q_islands = [isl for _, isl in q_with_region]

    n_ref = len(ref_islands)
    n_q = len(q_islands)

    if n_ref == 0 or n_q == 0:
        return []

    # Build provenance-based valid_pairs using the *core* projected
    # interval (before query-side flanking).  This reflects where the
    # reference island honestly maps in the query, not the wider scan
    # region.  Falls back to the flanked start/end if core is absent.
    ref_idx_to_intervals: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for merged_id in qr_ids:
        cluster_info = clusters_data.get(merged_id, {})
        for entry in cluster_info.get("merged_transcripts", []):
            if entry.get("transcript_id") == gene_id and entry.get("island_idx") is not None:
                proj_start = entry.get("core_start") or entry.get("start")
                proj_end = entry.get("core_end") or entry.get("end")
                if proj_start is not None and proj_end is not None:
                    ref_idx_to_intervals[entry["island_idx"]].append(
                        (proj_start, proj_end)
                    )

    def _overlaps(q_start: int, q_end: int, intervals: List[Tuple[int, int]]) -> bool:
        return any(q_end > iv_s and q_start < iv_e for iv_s, iv_e in intervals)

    valid_pairs: Set[Tuple[int, int]] = set()
    for ri, orig_idx in enumerate(orig_indices):
        intervals = ref_idx_to_intervals.get(orig_idx)
        if not intervals:
            for qi in range(n_q):
                valid_pairs.add((ri, qi))
        else:
            for qi in range(n_q):
                q_isl = q_islands[qi]
                if _overlaps(q_isl["start"], q_isl["end"], intervals):
                    valid_pairs.add((ri, qi))

    if not valid_pairs:
        return []

    # Only extract sequences and embed islands that participate in at
    # least one valid pair — avoids wasted GPU work for islands excluded
    # by provenance filtering.
    active_ri = sorted({ri for ri, _ in valid_pairs})
    active_qi = sorted({qi for _, qi in valid_pairs})

    ref_seqs: List[Optional[str]] = [None] * n_ref
    q_seqs: List[Optional[str]] = [None] * n_q
    for ri in active_ri:
        i = ref_islands[ri]
        ref_seqs[ri] = _fetch_seq(ref_acc, i["chrom"], i["start"], i["end"],
                                  i["strand"])
    for qi in active_qi:
        i = q_islands[qi]
        q_seqs[qi] = _fetch_seq(query_acc, i["chrom"], i["start"], i["end"],
                                i["strand"])

    # Async phase: fetch each active island's representation via the matcher
    # (RiNALMo = one embed-once per island; RNA-FM = per-window re-embedding).
    ref_embed_tasks = {
        ri: matcher.prepare_island(ref_seqs[ri], gpu, gene_id, f"R{ri}", config)
        for ri in active_ri
    }
    q_embed_tasks = {
        qi: matcher.prepare_island(q_seqs[qi], gpu, gene_id, f"Q{qi}", config)
        for qi in active_qi
    }
    all_coros = list(ref_embed_tasks.values()) + list(q_embed_tasks.values())
    all_embs = await asyncio.gather(*all_coros)

    ref_reprs: List[Optional[object]] = [None] * n_ref
    q_reprs: List[Optional[object]] = [None] * n_q
    idx = 0
    for ri in active_ri:
        ref_reprs[ri] = all_embs[idx]
        idx += 1
    for qi in active_qi:
        q_reprs[qi] = all_embs[idx]
        idx += 1

    # CPU-bound phase: offload to thread pool (numpy releases the GIL)
    loop = asyncio.get_running_loop()
    rows = await loop.run_in_executor(
        cpu_pool,
        _compute_island_alignments,
        gene_id, ref_islands, q_islands, ref_seqs, q_seqs,
        ref_reprs, q_reprs, config, valid_pairs, matcher,
    )
    return rows


# ===========================================================================
# SQLite intermediate storage
# ===========================================================================

def _ensure_table(conn: sqlite3.Connection, table: str, columns: Dict[str, str]) -> None:
    cols_sql = ", ".join(f"{name} {ctype}" for name, ctype in columns.items())
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({cols_sql})")

    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, ctype in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ctype}")


async def _sqlite_writer(queue: asyncio.Queue, sqlite_path: str) -> None:
    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    columns = {
        "gene_id": "TEXT NOT NULL",
        "ref_island": "TEXT NOT NULL",
        "query_island": "TEXT NOT NULL",
        "type": "TEXT",
        "ref_chrom": "TEXT",
        "ref_start": "INTEGER",
        "ref_end": "INTEGER",
        "ref_len": "INTEGER",
        "query_chrom": "TEXT",
        "query_start": "INTEGER",
        "query_end": "INTEGER",
        "query_len": "INTEGER",
        "n_chains": "INTEGER",
        "diag_mmd": "TEXT",
        "chains_json": "TEXT",
    }
    _ensure_table(conn, "island_alignment_results", columns)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_island_align_key "
        "ON island_alignment_results(gene_id, ref_island, query_island)"
    )

    while True:
        item = await queue.get()
        if item is None:
            break

        data = item.copy()
        placeholders = ", ".join(["?"] * len(data))
        columns_sql = ", ".join(data.keys())
        updates = ", ".join(f"{k}=excluded.{k}" for k in data.keys())
        sql = (
            f"INSERT INTO island_alignment_results ({columns_sql}) VALUES ({placeholders}) "
            f"ON CONFLICT(gene_id, ref_island, query_island) DO UPDATE SET {updates}"
        )
        conn.execute(sql, tuple(data.values()))
        conn.commit()

    conn.close()


def _export_sqlite_to_tsv(
    sqlite_path: str,
    tsv_path: str,
    config: IslandAlignmentConfig,
) -> None:
    """Export island alignment results from SQLite to the canonical TSV format."""
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.execute(
        "SELECT gene_id, ref_island, query_island, type, "
        "ref_chrom, ref_start, ref_end, ref_len, "
        "query_chrom, query_start, query_end, query_len, "
        "n_chains, diag_mmd, chains_json "
        "FROM island_alignment_results ORDER BY gene_id, ref_island, query_island"
    )

    max_chains = config.sw_max_chains
    header = [
        "gene_id", "ref_island", "query_island", "type",
        "ref_chrom", "ref_start", "ref_end", "ref_len",
        "query_chrom", "query_start", "query_end", "query_len",
        "n_chains", "diag_mmd",
    ]
    for ci in range(max_chains):
        header.extend([
            f"chain{ci + 1}_ref_from", f"chain{ci + 1}_ref_to",
            f"chain{ci + 1}_q_from", f"chain{ci + 1}_q_to",
            f"chain{ci + 1}_mmd",
        ])

    with open(tsv_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for row in cursor:
            core = [str(v) if v is not None else "" for v in row[:14]]
            chains_json_str = row[14]
            chains = json.loads(chains_json_str) if chains_json_str else []
            for ci in range(max_chains):
                if ci < len(chains):
                    c = chains[ci]
                    core.extend([
                        str(c["ref_from"]), str(c["ref_to"]),
                        str(c["q_from"]), str(c["q_to"]),
                        str(c["mmd"]),
                    ])
                else:
                    core.extend(["", "", "", "", ""])
            f.write("\t".join(core) + "\n")

    total = conn.execute("SELECT COUNT(*) FROM island_alignment_results").fetchone()[0]
    genes = conn.execute(
        "SELECT COUNT(DISTINCT gene_id) FROM island_alignment_results"
    ).fetchone()[0]
    conn.close()
    print(f"# Exported {total} island alignment results ({genes} genes) to {tsv_path}")


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
    sqlite_path: str,
    output_tsv_path: str,
    max_concurrent: int = 128,
    test_cap_jobs: Optional[int] = None,
    config: Optional[IslandAlignmentConfig] = None,
    clusters_json_path: str = None,
    model_name: Optional[str] = None,
) -> None:
    """
    Run island alignment scheduler with GPU executor integration.

    Uses the shared GPU executor process for RNA-FM inference (per-token PCA
    embeddings, NOT mean-pooled) and writes results to an intermediate SQLite
    database, then exports to TSV.

    clusters_json_path (query_regions_clusters.json from the liftover step)
    provides provenance: each reference island is only matched against query
    islands found in the regions it was projected into.
    """
    if config is None:
        config = IslandAlignmentConfig.for_model(model_name)
    matcher = get_matcher(model_name)
    if matcher.representation == "full":
        print(f"# Island alignment: model={model_name or 'default'}, "
              f"matcher={type(matcher).__name__} (embed-once dotplot + nt-SW), "
              f"tau_cos={config.sw_tau_cos}, gap={config.sw_gap}, "
              f"max_dist={config.max_match_dist}, min_eff_nt={config.min_match_eff_nt}")
    else:
        print(f"# Island alignment: model={model_name or 'default'}, "
              f"matcher={type(matcher).__name__} (windowed MMD + SW), "
              f"window={config.window_size}, stride={config.stride}, "
              f"sw_tau={config.sw_tau}, max_match_mmd={config.max_match_mmd}")

    # Pin numpy/BLAS to single-threaded: we handle parallelism ourselves via
    # the thread pool, so internal threading would cause oversubscription.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    n_cpu_workers = max(1, os.cpu_count() or 4)
    cpu_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=n_cpu_workers,
        thread_name_prefix="island_mmd",
    )

    # Limit async workers so GPU and CPU phases overlap as a pipeline.
    # With too many async workers (e.g. 128), all GPU work is front-loaded
    # and the GPU sits idle while 128 jobs queue for 12 CPU slots.
    # With ~3x CPU threads, there's always a mix of workers embedding (GPU)
    # and computing MMD/SW (CPU) simultaneously.
    n_async_workers = min(max_concurrent, n_cpu_workers * 3)

    async def _run() -> None:
        t0 = time.monotonic()

        jobs = _load_joblist(joblist_path)
        if test_cap_jobs:
            jobs = jobs[:test_cap_jobs]
            print(f"# [TEST MODE] Capped to {len(jobs)} jobs (--test-cap-jobs={test_cap_jobs})")
        print(f"# Loaded {len(jobs)} island alignment jobs.")
        from modules.utils.mmd_utils import _HAS_NUMBA
        mmd_backend = "numba (JIT)" if _HAS_NUMBA else "numpy (vectorised)"
        print(f"# Island alignment: {n_async_workers} async workers, {n_cpu_workers} CPU threads, MMD backend: {mmd_backend}")

        with open(ref_islands_json_path, "r") as f:
            ref_data = json.load(f)
        with open(u2q_map_path, "r") as f:
            u_to_query = json.load(f)
        with open(query_islands_json_path, "r") as f:
            query_islands_data = json.load(f)

        with open(clusters_json_path, "r") as f:
            clusters_data = json.load(f)
        print(f"# Provenance-aware matching: {len(clusters_data)} query regions")

        ref_acc = TwoBitAccessor(ref_2bit_path)
        query_acc = TwoBitAccessor(query_2bit_path)

        loop = asyncio.get_running_loop()
        gpu = GPUClient(input_q, output_q, loop)

        result_queue: asyncio.Queue = asyncio.Queue()
        writer_task = asyncio.create_task(_sqlite_writer(result_queue, sqlite_path))

        job_queue: asyncio.Queue = asyncio.Queue()
        total_jobs = len(jobs)
        for job in jobs:
            await job_queue.put(job)
        for _ in range(n_async_workers):
            await job_queue.put(None)

        completed_counter: Dict = {"count": 0, "last_log_time": t0}
        counter_lock = asyncio.Lock()

        async def _worker() -> None:
            while True:
                job = await job_queue.get()
                if job is None:
                    break

                try:
                    rows = await _process_job(
                        job, ref_data, u_to_query, query_islands_data,
                        ref_acc, query_acc, gpu, config,
                        cpu_pool, clusters_data, matcher,
                    )
                    for row in rows:
                        await result_queue.put(row)
                except Exception as exc:
                    print(f"# Error processing gene {job.gene_id}: {exc}")

                async with counter_lock:
                    completed_counter["count"] += 1
                    completed = completed_counter["count"]
                    current_time = time.monotonic()
                    remaining = total_jobs - completed
                    should_log = (
                        completed % 50 == 0
                        or (current_time - completed_counter["last_log_time"]) >= 30.0
                        or completed == total_jobs
                    )
                    if should_log:
                        elapsed = current_time - t0
                        pct = (completed / total_jobs * 100) if total_jobs > 0 else 0
                        print(
                            f"# Island alignment: {completed}/{total_jobs} completed "
                            f"({pct:.1f}%), {remaining} remaining (elapsed {elapsed:.1f}s)"
                        )
                        completed_counter["last_log_time"] = current_time

        workers = [asyncio.create_task(_worker()) for _ in range(n_async_workers)]
        await asyncio.gather(*workers)

        await result_queue.put(None)
        await writer_task

        gpu.stop()

        elapsed_total = time.monotonic() - t0
        print(f"# Island alignment scheduler finished in {elapsed_total:.1f}s.")

        # Print summary
        conn = sqlite3.connect(sqlite_path)
        total_rows = conn.execute(
            "SELECT COUNT(*) FROM island_alignment_results"
        ).fetchone()[0]
        gene_count = conn.execute(
            "SELECT COUNT(DISTINCT gene_id) FROM island_alignment_results"
        ).fetchone()[0]
        conn.close()
        print(
            f"# Island alignment summary: {total_rows} island-pair results "
            f"across {gene_count} genes"
        )

        # Export to TSV
        _export_sqlite_to_tsv(sqlite_path, output_tsv_path, config)

    asyncio.run(_run())
    cpu_pool.shutdown(wait=False)

    print(f"# Island alignment completed. Results written to {output_tsv_path}")
