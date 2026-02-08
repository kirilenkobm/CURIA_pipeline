from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from pyrion import read_bed12_file, read_gene_data
from pyrion.core.genes import Transcript, TranscriptsCollection
from pyrion.ops.transcript_serialization import save_transcripts_collection_to_bed12


def _merge_intervals(blocks: np.ndarray) -> np.ndarray:
    if blocks.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    blocks = np.asarray(blocks, dtype=np.int64)
    order = np.argsort(blocks[:, 0], kind="mergesort")
    blocks = blocks[order]

    merged = []
    cur_start = int(blocks[0, 0])
    cur_end = int(blocks[0, 1])
    for start, end in blocks[1:]:
        start = int(start)
        end = int(end)
        if start <= cur_end:
            if end > cur_end:
                cur_end = end
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    return np.array(merged, dtype=np.int32)


def _pick_biotype(biotypes: List[str]) -> str:
    for b in biotypes:
        if b is None:
            continue
        b_norm = b.strip().lower().replace("-", "_").replace(" ", "_")
        if b_norm == "protein_coding":
            return "protein_coding"
    for b in biotypes:
        if b:
            return b
    return "unknown"


def collapse_to_ultimate_isoforms(
    bed12_path: str,
    metadata_tsv_path: str,
    out_bed12_path: str,
    out_metadata_tsv_path: str,
    id_prefix: str = "U_",
) -> Tuple[str, str]:
    transcripts = read_bed12_file(bed12_path)
    gene_data = read_gene_data(
        metadata_tsv_path,
        gene_column="gene_id",
        transcript_id_column="transcript_id",
        transcript_type_column="biotype",
    )
    transcripts.bind_gene_data(gene_data)

    ultimate_transcripts: List[Transcript] = []
    metadata_rows: List[Tuple[str, str, str]] = []

    for gene_id in gene_data.gene_ids:
        gene = transcripts.get_gene_by_id(gene_id)
        if gene is None:
            continue

        ts = gene.transcripts
        biotype = _pick_biotype([t.biotype for t in ts])
        has_coding = any(t.is_coding for t in ts)

        if has_coding:
            cds_blocks = [t.cds_blocks for t in ts if t.is_coding and t.cds_blocks.size > 0]
            utr_blocks = [t.utr_blocks for t in ts if t.is_coding and t.utr_blocks.size > 0]

            cds_union = _merge_intervals(np.vstack(cds_blocks)) if cds_blocks else np.empty((0, 2), dtype=np.int32)
            utr_union = _merge_intervals(np.vstack(utr_blocks)) if utr_blocks else np.empty((0, 2), dtype=np.int32)

            if cds_union.size == 0 and utr_union.size == 0:
                continue

            if cds_union.size == 0:
                merged = utr_union
                cds_start = None
                cds_end = None
            elif utr_union.size == 0:
                merged = cds_union
                cds_start = int(cds_union[:, 0].min())
                cds_end = int(cds_union[:, 1].max())
            else:
                merged = _merge_intervals(np.vstack([cds_union, utr_union]))
                cds_start = int(cds_union[:, 0].min())
                cds_end = int(cds_union[:, 1].max())
        else:
            blocks = [t.blocks for t in ts if t.blocks.size > 0]
            if not blocks:
                continue
            merged = _merge_intervals(np.vstack(blocks))
            cds_start = None
            cds_end = None

        transcript_id = f"{id_prefix}{gene_id}"
        ultimate_transcripts.append(
            Transcript(
                blocks=merged,
                strand=gene.strand,
                chrom=gene.chrom,
                id=transcript_id,
                cds_start=cds_start,
                cds_end=cds_end,
                biotype=biotype,
            )
        )
        metadata_rows.append((transcript_id, gene_id, biotype))

    out_bed12_path = str(out_bed12_path)
    out_metadata_tsv_path = str(out_metadata_tsv_path)
    Path(out_bed12_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_metadata_tsv_path).parent.mkdir(parents=True, exist_ok=True)

    collection = TranscriptsCollection(ultimate_transcripts)
    save_transcripts_collection_to_bed12(collection, out_bed12_path)

    with open(out_metadata_tsv_path, "w") as f:
        f.write("transcript_id\tgene_id\tbiotype\n")
        for transcript_id, gene_id, biotype in metadata_rows:
            f.write(f"{transcript_id}\t{gene_id}\t{biotype}\n")

    return out_bed12_path, out_metadata_tsv_path
