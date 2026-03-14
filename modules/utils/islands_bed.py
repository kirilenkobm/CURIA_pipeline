#!/usr/bin/env python3
"""
Generate BED12 files for detected islands (steps 2 and 3).

Creates BED12 files showing functional islands detected by RNA-FM + LogReg:
- Reference islands: Islands detected in reference transcripts (step 2)
- Query islands: Islands detected in query regions (step 3)

Each island becomes a single-exon "transcript" in BED12 format.
"""

from __future__ import annotations

import json
import numpy as np

from pyrion.core.genes import Transcript, TranscriptsCollection
from pyrion.core.strand import Strand


def write_reference_islands_bed(
    ref_islands_json_path: str,
    output_bed_path: str,
) -> int:
    """
    Write BED12 file for reference islands.

    Args:
        ref_islands_json_path: Path to reference islands JSON (step 2 output)
        output_bed_path: Output BED12 path

    Returns:
        Number of islands written
    """
    with open(ref_islands_json_path, "r") as f:
        ref_islands_data = json.load(f)

    transcripts = []
    total_islands = 0

    for gene_id, gene_data in sorted(ref_islands_data.items()):
        islands = gene_data.get("islands", [])
        if not islands:
            continue

        for idx, island in enumerate(islands):
            chrom = island["chrom"]
            start = island["start"]
            end = island["end"]
            strand_val = island.get("strand", 1)
            strand = Strand.PLUS if strand_val == 1 or strand_val == "+" else Strand.MINUS

            # Create transcript ID: gene_id + island index
            transcript_id = f"{gene_id}_island_{idx}"

            # Single block for this island
            blocks = np.array([[start, end]], dtype=np.int64)

            transcript = Transcript(
                blocks=blocks,
                strand=strand,
                chrom=chrom,
                id=transcript_id,
            )
            transcripts.append(transcript)
            total_islands += 1

    # Create collection and save
    collection = TranscriptsCollection(transcripts=transcripts)
    collection.save_to_bed12(output_bed_path)

    print(f"# Wrote {total_islands} reference islands to {output_bed_path}")
    return total_islands


def write_query_islands_bed(
    query_islands_json_path: str,
    ultimate_to_query_map_path: str,
    output_bed_path: str,
) -> int:
    """
    Write BED12 file for query islands.

    Args:
        query_islands_json_path: Path to query islands JSON (step 3 output)
        ultimate_to_query_map_path: Path to ultimate-to-query mapping JSON
        output_bed_path: Output BED12 path

    Returns:
        Number of islands written
    """
    with open(query_islands_json_path, "r") as f:
        query_islands_data = json.load(f)

    with open(ultimate_to_query_map_path, "r") as f:
        u_to_query = json.load(f)

    # Build reverse map: merged_query_id -> list of gene_ids
    query_to_genes = {}
    for gene_id, query_ids in u_to_query.items():
        for qid in query_ids:
            if qid not in query_to_genes:
                query_to_genes[qid] = []
            query_to_genes[qid].append(gene_id)

    transcripts = []
    total_islands = 0

    for merged_query_id, islands in sorted(query_islands_data.items()):
        if not islands:
            continue

        # Get corresponding gene IDs
        gene_ids = query_to_genes.get(merged_query_id, [])
        if not gene_ids:
            # Use merged_query_id as fallback
            base_name = merged_query_id
        else:
            # Use first gene ID as base name
            base_name = gene_ids[0]

        for idx, island in enumerate(islands):
            chrom = island["chrom"]
            start = island["start"]
            end = island["end"]
            strand_val = island.get("strand", 1)
            strand = Strand.PLUS if strand_val == 1 or strand_val == "+" else Strand.MINUS

            # Create transcript ID: base_name.chain + island index
            transcript_id = f"{base_name}.{merged_query_id.split('_')[-1]}_island_{idx}"

            # Single block for this island
            blocks = np.array([[start, end]], dtype=np.int64)

            transcript = Transcript(
                blocks=blocks,
                strand=strand,
                chrom=chrom,
                id=transcript_id,
            )
            transcripts.append(transcript)
            total_islands += 1

    # Create collection and save
    collection = TranscriptsCollection(transcripts=transcripts)
    collection.save_to_bed12(output_bed_path)

    print(f"# Wrote {total_islands} query islands to {output_bed_path}")
    return total_islands
