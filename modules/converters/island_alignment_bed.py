#!/usr/bin/env python3
"""
Generate BED12 files from island alignment results using pyrion.

Creates two BED12 files:
- Reference BED: aligned reference islands as "exons" per gene
- Query BED: aligned query islands as "exons" per gene

Each transcript ID = gene_id + "_aligned"
Each aligned island segment becomes an "exon" block in the transcript.
"""

from __future__ import annotations

import csv
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

from pyrion.core.genes import Transcript, TranscriptsCollection
from pyrion.core.strand import Strand


def write_island_alignment_beds(
    alignment_tsv_path: str,
    ref_islands_json_path: str,
    query_islands_json_path: str,
    ref_bed_path: str,
    query_bed_path: str,
) -> Tuple[int, int]:
    """
    Write BED12 files for aligned reference and query islands using pyrion.

    Args:
        alignment_tsv_path: Path to island alignment TSV results
        ref_islands_json_path: Path to reference islands JSON
        query_islands_json_path: Path to query islands JSON
        ref_bed_path: Output path for reference BED12
        query_bed_path: Output path for query BED12

    Returns:
        Tuple of (n_ref_transcripts, n_query_transcripts) written
    """
    # Load island coordinate data
    with open(ref_islands_json_path, "r") as f:
        ref_islands_data = json.load(f)

    with open(query_islands_json_path, "r") as f:
        query_islands_data = json.load(f)

    # Parse alignment results and group by gene_id
    alignments_by_gene: Dict[str, List[Dict]] = defaultdict(list)

    with open(alignment_tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene_id = row["gene_id"]
            alignments_by_gene[gene_id].append(row)

    # Build transcripts
    ref_transcripts = []
    query_transcripts = []

    for gene_id, rows in sorted(alignments_by_gene.items()):
        # Get reference gene info
        if gene_id not in ref_islands_data:
            continue

        ref_gene_data = ref_islands_data[gene_id]
        ref_islands = {i: isl for i, isl in enumerate(ref_gene_data.get("islands", []))}

        # Collect all aligned segments (chain data) from all island pairs
        ref_segments = []
        query_segments = []

        for row in rows:
            ref_island_idx = int(row["ref_island"].replace("R", ""))
            query_island_idx = int(row["query_island"].replace("Q", ""))

            n_chains = int(row.get("n_chains", 0))
            if n_chains == 0:
                continue

            # Get island base coordinates
            ref_chrom = row["ref_chrom"]
            ref_island_start = int(row["ref_start"])
            ref_strand_val = ref_islands.get(ref_island_idx, {}).get("strand", 1)

            query_chrom = row["query_chrom"]
            query_island_start = int(row["query_start"])
            # Infer query strand from query_islands_data if available
            # For now assume same as ref or use a default
            query_strand_val = 1

            # Process each chain within this island pair
            for chain_idx in range(n_chains):
                chain_num = chain_idx + 1
                ref_from_key = f"chain{chain_num}_ref_from"
                ref_to_key = f"chain{chain_num}_ref_to"
                q_from_key = f"chain{chain_num}_q_from"
                q_to_key = f"chain{chain_num}_q_to"

                if ref_from_key not in row or not row[ref_from_key]:
                    continue

                # Get relative coordinates
                ref_from_rel = int(row[ref_from_key])
                ref_to_rel = int(row[ref_to_key])
                q_from_rel = int(row[q_from_key])
                q_to_rel = int(row[q_to_key])

                # Convert to genomic coordinates
                ref_chain_start = ref_island_start + ref_from_rel
                ref_chain_end = ref_island_start + ref_to_rel
                query_chain_start = query_island_start + q_from_rel
                query_chain_end = query_island_start + q_to_rel

                ref_segments.append({
                    "chrom": ref_chrom,
                    "start": ref_chain_start,
                    "end": ref_chain_end,
                    "strand": ref_strand_val,
                })

                query_segments.append({
                    "chrom": query_chrom,
                    "start": query_chain_start,
                    "end": query_chain_end,
                    "strand": query_strand_val,
                })

        if not ref_segments or not query_segments:
            continue

        # Sort segments by genomic position
        ref_segments.sort(key=lambda x: (x["chrom"], x["start"]))
        query_segments.sort(key=lambda x: (x["chrom"], x["start"]))

        # Merge overlapping/duplicate segments
        def merge_segments(segments):
            if not segments:
                return []
            merged = [segments[0].copy()]
            for seg in segments[1:]:
                last = merged[-1]
                # Check if same chrom and overlapping or adjacent
                if seg["chrom"] == last["chrom"] and seg["start"] <= last["end"]:
                    # Merge by extending the end
                    last["end"] = max(last["end"], seg["end"])
                else:
                    merged.append(seg.copy())
            return merged

        ref_segments = merge_segments(ref_segments)
        query_segments = merge_segments(query_segments)

        # Build reference transcript using pyrion
        ref_transcript_id = f"{gene_id}_aligned"
        ref_chrom = ref_segments[0]["chrom"]
        ref_strand_val = ref_segments[0]["strand"]
        ref_strand = Strand.PLUS if ref_strand_val == 1 or ref_strand_val == "+" else Strand.MINUS

        # Create blocks array: [[start, end], [start, end], ...]
        ref_blocks = np.array([[s["start"], s["end"]] for s in ref_segments], dtype=np.int64)

        ref_transcript = Transcript(
            blocks=ref_blocks,
            strand=ref_strand,
            chrom=ref_chrom,
            id=ref_transcript_id,
        )
        ref_transcripts.append(ref_transcript)

        # Build query transcript
        query_transcript_id = f"{gene_id}_aligned"
        query_chrom = query_segments[0]["chrom"]
        query_strand_val = query_segments[0]["strand"]
        query_strand = Strand.PLUS if query_strand_val == 1 or query_strand_val == "+" else Strand.MINUS

        query_blocks = np.array([[s["start"], s["end"]] for s in query_segments], dtype=np.int64)

        query_transcript = Transcript(
            blocks=query_blocks,
            strand=query_strand,
            chrom=query_chrom,
            id=query_transcript_id,
        )
        query_transcripts.append(query_transcript)

    # Create TranscriptsCollection and save to BED12
    ref_collection = TranscriptsCollection(transcripts=ref_transcripts)
    query_collection = TranscriptsCollection(transcripts=query_transcripts)

    ref_collection.save_to_bed12(ref_bed_path)
    query_collection.save_to_bed12(query_bed_path)

    print(f"# Wrote {len(ref_transcripts)} reference aligned island transcripts to {ref_bed_path}")
    print(f"# Wrote {len(query_transcripts)} query aligned island transcripts to {query_bed_path}")

    return len(ref_transcripts), len(query_transcripts)
