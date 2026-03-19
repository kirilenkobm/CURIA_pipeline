#!/usr/bin/env python3
"""
Reference islands liftover (Step 2.5).

Instead of using the full TOGA transcript projections as query regions,
this step lifts over reference stability islands (with flanks) through
alignment chains to create targeted query regions.

This dramatically reduces the amount of query genome that needs to be scanned
for functional islands, cutting GPU time from ~20 h to ~2 h on hg38→mm39.

Outputs are format-compatible with the legacy merge_query_regions step,
so downstream pipeline stages (query island scanning, island alignment)
work without modification.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from pyrion import read_chain_file
from pyrion.core.intervals import GenomicInterval, GenomicIntervalsCollection
from pyrion.core.strand import Strand
from pyrion.ops.chains import project_intervals_through_chain_strict
from pyrion.ops.interval_collection_ops import group_intervals_by_proximity


def _parse_region(region: str) -> Tuple[str, int, int]:
    chrom, rest = region.split(":")
    start_s, end_s = rest.split("-")
    start, end = int(start_s), int(end_s)
    if start > end:
        start, end = end, start
    return chrom, start, end


def _load_short_pairs(short_joblist_path: str | Path) -> set:
    short_pairs = set()
    with open(short_joblist_path, "r") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            tid_idx = header.index("transcript_id")
            chain_idx = header.index("chain_id")
        except ValueError:
            raise ValueError("short_ncRNA_joblist.txt header missing required columns")
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            short_pairs.add((parts[tid_idx], parts[chain_idx]))
    return short_pairs


def _load_rna_regions(
    rna_regions_path: str | Path,
    short_pairs: set,
    transcripts_with_islands: set,
    max_chains_per_gene: int = 5,
) -> Dict[int, List[dict]]:
    """Load (transcript_id, chain_id) pairs grouped by chain_id (int).

    If a gene maps to more than max_chains_per_gene orthologous chains,
    only the top-K are kept (lowest chain_id = highest alignment score).
    """
    pairs_by_transcript: Dict[str, List[dict]] = defaultdict(list)

    with open(rna_regions_path, "r") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            tid_idx = header.index("transcript_id")
            chain_idx = header.index("chain_id")
            tstrand_idx = header.index("transcript_strand")
            cstrand_idx = header.index("chain_strand")
            biotype_idx = header.index("biotype")
        except ValueError:
            raise ValueError("rna_orthologous_regions.tsv header missing required columns")

        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            transcript_id = parts[tid_idx]
            chain_id_str = parts[chain_idx]

            if (transcript_id, chain_id_str) in short_pairs:
                continue
            if transcript_id not in transcripts_with_islands:
                continue

            pairs_by_transcript[transcript_id].append({
                "transcript_id": transcript_id,
                "chain_id_str": chain_id_str,
                "transcript_strand": int(parts[tstrand_idx]),
                "chain_strand": int(parts[cstrand_idx]),
                "biotype": parts[biotype_idx],
            })

    # Top-K chain filter: keep only the highest-scoring chains per gene
    n_before = sum(len(v) for v in pairs_by_transcript.values())
    genes_capped = 0
    for transcript_id, pairs in pairs_by_transcript.items():
        if len(pairs) > max_chains_per_gene:
            pairs.sort(key=lambda x: int(x["chain_id_str"]))
            pairs_by_transcript[transcript_id] = pairs[:max_chains_per_gene]
            genes_capped += 1
    n_after = sum(len(v) for v in pairs_by_transcript.values())

    if n_before > n_after:
        print(
            f"# Top-{max_chains_per_gene} chain filter: {n_before} → {n_after} pairs "
            f"({n_before - n_after} dropped from {genes_capped} genes)"
        )

    # Regroup by chain_id for efficient batched projection
    pairs_by_chain: Dict[int, List[dict]] = defaultdict(list)
    for pairs in pairs_by_transcript.values():
        for pair in pairs:
            pairs_by_chain[int(pair["chain_id_str"])].append(pair)

    return pairs_by_chain


SCAN_WINDOW_SIZE = 72


def _project_flanked_islands(
    pairs_by_chain: Dict[int, List[dict]],
    ref_islands_data: dict,
    chains,
    ref_flank: int = SCAN_WINDOW_SIZE,
    max_gap_ratio: float = 25.0,
    max_expansion_ratio: float = 25.0,
    query_flank_ratio: float = 5.0,
) -> List[dict]:
    """
    Project reference islands through alignment chains with minimal flanks.

    Reference-side: adds ±ref_flank (scan window size) around each island before
    projecting through the chain.

    Query-side: after projection, if the projected region is shorter than
    query_flank_ratio × island_length, adds ±island_length flanks to ensure
    coverage.  Otherwise the projection is already wide enough.

    Expansion filter: projections exceeding max_expansion_ratio × flanked interval
    length are rejected (gap-spanning explosions).
    """
    projected = []
    stats = {"total_pairs": 0, "total_islands": 0, "projected_ok": 0,
             "chain_not_found": 0, "projection_failed": 0, "too_expanded": 0}

    for chain_id_int, tc_list in pairs_by_chain.items():
        try:
            chain = chains.get_by_chain_id(chain_id_int)
        except Exception:
            stats["chain_not_found"] += len(tc_list)
            continue
        if chain is None:
            stats["chain_not_found"] += len(tc_list)
            continue

        all_intervals: List[List[int]] = []
        all_island_lens: List[int] = []
        all_metadata: List[dict] = []

        for tc in tc_list:
            stats["total_pairs"] += 1
            transcript_id = tc["transcript_id"]
            islands = ref_islands_data[transcript_id].get("islands", [])

            for idx, island in enumerate(islands):
                stats["total_islands"] += 1
                island_len = island["end"] - island["start"]

                flanked_start = max(0, island["start"] - ref_flank)
                flanked_end = island["end"] + ref_flank

                all_intervals.append([flanked_start, flanked_end])
                all_island_lens.append(island_len)
                all_metadata.append({
                    "transcript_id": transcript_id,
                    "chain_id_str": tc["chain_id_str"],
                    "transcript_strand": tc["transcript_strand"],
                    "chain_strand": tc["chain_strand"],
                    "biotype": tc["biotype"],
                    "island_idx": idx,
                })

        if not all_intervals:
            continue

        try:
            intervals_arr = np.array(all_intervals, dtype=np.int64)
            proj_list = project_intervals_through_chain_strict(
                intervals_arr, chain.blocks, chain.q_strand,
                max_gap_ratio=max_gap_ratio,
            )
        except Exception:
            stats["projection_failed"] += len(all_intervals)
            continue

        for iv_pair, island_len, meta, elem in zip(
            all_intervals, all_island_lens, all_metadata, proj_list
        ):
            q_start, q_end = int(elem[0][0]), int(elem[0][1])

            if q_start == 0 and q_end == 0:
                stats["projection_failed"] += 1
                continue
            if q_start > q_end:
                q_start, q_end = q_end, q_start
            if q_start >= q_end:
                stats["projection_failed"] += 1
                continue

            flanked_len = iv_pair[1] - iv_pair[0]
            projected_len = q_end - q_start
            if projected_len > flanked_len * max_expansion_ratio:
                stats["too_expanded"] += 1
                continue

            if projected_len < island_len * query_flank_ratio:
                q_start = max(0, q_start - island_len)
                q_end = q_end + island_len

            query_strand = 1 if meta["transcript_strand"] == meta["chain_strand"] else -1
            stats["projected_ok"] += 1

            projected.append({
                "chrom": chain.q_chrom,
                "start": q_start,
                "end": q_end,
                "strand": query_strand,
                "transcript_id": meta["transcript_id"],
                "chain_id": meta["chain_id_str"],
                "biotype": meta["biotype"],
                "island_idx": meta["island_idx"],
            })

    print(f"# Islands liftover stats:")
    print(f"#   (transcript, chain) pairs processed: {stats['total_pairs']}")
    print(f"#   Total flanked islands attempted:     {stats['total_islands']}")
    print(f"#   Successfully projected:              {stats['projected_ok']}")
    if stats["chain_not_found"]:
        print(f"#   Skipped (chain not found):           {stats['chain_not_found']}")
    if stats["projection_failed"]:
        print(f"#   Skipped (projection failed):         {stats['projection_failed']}")
    if stats["too_expanded"]:
        print(f"#   Skipped (>{max_expansion_ratio:.0f}x expansion):       {stats['too_expanded']}")

    return projected


def _merge_projected_intervals(
    projected: List[dict],
) -> List[Tuple[str, int, int, int, List[dict]]]:
    """Merge overlapping projected intervals using pyrion's proximity clustering."""
    intervals_by_group: Dict[Tuple[str, int], List[Tuple[GenomicInterval, dict]]] = defaultdict(list)

    for global_idx, iv in enumerate(projected):
        if iv["start"] >= iv["end"]:
            continue
        strand = Strand.PLUS if iv["strand"] == 1 else Strand.MINUS
        gi = GenomicInterval(
            chrom=iv["chrom"],
            start=iv["start"],
            end=iv["end"],
            strand=strand,
            id=f"proj_{global_idx}",
        )
        key = (iv["chrom"], iv["strand"])
        intervals_by_group[key].append((gi, iv))

    merged_regions: List[Tuple[str, int, int, int, List[dict]]] = []

    for (chrom, strand_int), group_data in intervals_by_group.items():
        intervals = [gi for gi, _ in group_data]
        collection = GenomicIntervalsCollection.from_intervals(intervals)
        clustered_collections = group_intervals_by_proximity(collection, max_gap=0)

        for cluster_coll in clustered_collections:
            cluster_ivs = cluster_coll.to_intervals_list()
            cluster_ids = {iv.id for iv in cluster_ivs}

            cluster_metadata = [meta for gi, meta in group_data if gi.id in cluster_ids]
            if not cluster_metadata:
                continue

            c_start = min(m["start"] for m in cluster_metadata)
            c_end = max(m["end"] for m in cluster_metadata)
            merged_regions.append((chrom, c_start, c_end, strand_int, cluster_metadata))

    return merged_regions


def _write_outputs(
    merged_regions: List[Tuple[str, int, int, int, List[dict]]],
    clusters_json_path: str | Path,
    union_to_query_path: str | Path,
) -> None:
    """Write all output files in formats compatible with downstream pipeline steps."""
    clusters_output = {}
    union_to_query: Dict[str, List[str]] = {}

    for idx, (chrom, start, end, strand_int, cluster) in enumerate(merged_regions, start=1):
        merged_id = f"query_merged_region_{idx}"

        clusters_output[merged_id] = {
            "merged_region": {
                "chrom": chrom,
                "start": start,
                "end": end,
                "strand": strand_int,
            },
            "merged_transcripts": [
                {
                    "transcript_id": e["transcript_id"],
                    "chain_id": e["chain_id"],
                    "chrom": e["chrom"],
                    "start": e["start"],
                    "end": e["end"],
                    "strand": e["strand"],
                }
                for e in cluster
            ],
        }

        for e in cluster:
            uid = e["transcript_id"]
            union_to_query.setdefault(uid, [])
            if merged_id not in union_to_query[uid]:
                union_to_query[uid].append(merged_id)

    with open(clusters_json_path, "w") as f:
        json.dump(clusters_output, f, indent=2)

    with open(union_to_query_path, "w") as f:
        json.dump(union_to_query, f, indent=2)


def liftover_reference_islands(
    chain_path: str,
    ref_islands_json_path: str,
    rna_regions_path: str | Path,
    short_joblist_path: str | Path,
    clusters_json_path: str | Path,
    union_to_query_path: str | Path,
    max_chains_per_gene: int = 5,
    chains=None,
    chain_min_score: int = 25_000,
) -> None:
    """
    Liftover reference islands through alignment chains to create targeted query regions.

    Replaces the legacy merge_query_regions step. Instead of projecting full
    transcripts, only flanked reference islands are projected — dramatically
    reducing the query region to scan.

    Reference-side flanking: ±SCAN_WINDOW_SIZE (72 nt) around each island.
    After projection, if the query region is < 5× island length, ±island_length
    flanks are added on the query side for safety.
    Projections exceeding 25× the flanked interval are rejected (gap explosions).

    Args:
        chain_path: Path to alignment chain file (can be .gz); ignored if chains provided
        ref_islands_json_path: Path to preprocessed_reference_data.json (step 2 output)
        rna_regions_path: Path to rna_orthologous_regions.tsv
        short_joblist_path: Path to short_ncRNA_joblist.txt (pairs to exclude)
        clusters_json_path: Output path for query_regions_clusters.json
        union_to_query_path: Output path for union_to_query.json
        max_chains_per_gene: Keep only top-K chains per gene (default 5, lowest id = best)
        chains: Pre-loaded GenomeAlignmentsCollection (optional, avoids reloading)
        chain_min_score: Minimum chain score for loading chains (default 25_000)
    """
    # 1. Load reference islands
    print("# Loading reference islands data...")
    with open(ref_islands_json_path, "r") as f:
        ref_islands_data = json.load(f)

    transcripts_with_islands = {
        tid for tid, data in ref_islands_data.items()
        if data.get("islands")
    }
    total_islands = sum(
        len(data.get("islands", []))
        for data in ref_islands_data.values()
        if data.get("islands")
    )
    print(f"# {len(transcripts_with_islands)} transcripts with {total_islands} reference islands")

    # 2. Load short ncRNA pairs to exclude
    short_pairs = _load_short_pairs(short_joblist_path)

    # 3. Load RNA regions → (transcript_id, chain_id) pairs grouped by chain
    pairs_by_chain = _load_rna_regions(
        rna_regions_path, short_pairs, transcripts_with_islands,
        max_chains_per_gene=max_chains_per_gene,
    )
    n_total_pairs = sum(len(v) for v in pairs_by_chain.values())
    print(f"# {n_total_pairs} (transcript, chain) pairs across {len(pairs_by_chain)} chains")

    if n_total_pairs == 0:
        print("# No pairs to process — writing empty outputs")
        _write_empty_outputs(clusters_json_path, union_to_query_path)
        return

    # 4. Use pre-loaded chains or load from file
    if chains is None:
        print(f"# Loading chain file (min_score={chain_min_score})...")
        chains = read_chain_file(chain_path, chain_min_score)
        print(f"# Loaded {len(chains)} chains")
    else:
        print(f"# Using pre-loaded chains ({len(chains)} chains)")

    # 5. Project flanked islands through chains
    print(f"# Projecting flanked islands (ref flank=±{SCAN_WINDOW_SIZE} nt)...")
    projected = _project_flanked_islands(
        pairs_by_chain, ref_islands_data, chains,
    )

    if not projected:
        print("# No intervals projected — writing empty outputs")
        _write_empty_outputs(clusters_json_path, union_to_query_path)
        return

    # 6. Merge overlapping projected intervals
    print("# Merging overlapping query regions...")
    merged_regions = _merge_projected_intervals(projected)
    total_bp = sum(end - start for _, start, end, _, _ in merged_regions)
    print(f"# Merged into {len(merged_regions)} query regions ({total_bp:,} bp total)")

    # 7. Write outputs
    _write_outputs(merged_regions, clusters_json_path, union_to_query_path)
    print(f"# Written: {clusters_json_path}")
    print(f"# Written: {union_to_query_path}")


def _write_empty_outputs(clusters_json_path, union_to_query_path):
    with open(clusters_json_path, "w") as f:
        json.dump({}, f, indent=2)
    with open(union_to_query_path, "w") as f:
        json.dump({}, f, indent=2)


__all__ = ["liftover_reference_islands"]
