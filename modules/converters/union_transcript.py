from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from pyrion import read_bed12_file, read_gene_data
from pyrion.core.genes import Transcript, TranscriptsCollection
from pyrion.ops.transcript_serialization import save_transcripts_collection_to_bed12

LNCRNA_BIOTYPE = "lncRNA"


def collapse_to_union_transcripts(
    bed12_path: str,
    metadata_tsv_path: str,
    out_bed12_path: str,
    out_metadata_tsv_path: str,
    union_to_isoforms_path: str | None = None,
    id_prefix: str = "U_",
) -> Tuple[str, str]:
    transcripts = read_bed12_file(bed12_path)
    gene_data = read_gene_data(
        metadata_tsv_path,
        gene_column="gene_id",
        transcript_id_column="transcript_id",
        transcript_type_column="transcript_biotype",
    )
    transcripts.bind_gene_data(gene_data)

    union_transcripts: List[Transcript] = []
    metadata_rows: List[Tuple[str, str, str]] = []
    union_to_isoforms = {}

    for gene_id in gene_data.gene_ids:
        gene = transcripts.get_gene_by_id(gene_id)
        if gene is None:
            continue

        union_transcript = gene.to_union_transcript(id_prefix=id_prefix)

        original_biotypes = [t.biotype for t in gene.transcripts if t.biotype]
        biotype = "unknown"
        if original_biotypes:
            if any(b and "protein_coding" in b.lower().replace("-", "_").replace(" ", "_") for b in original_biotypes):
                biotype = "protein_coding"
            elif any(b == LNCRNA_BIOTYPE for b in original_biotypes):
                biotype = LNCRNA_BIOTYPE
            else:
                biotype = original_biotypes[0]

        union_transcripts.append(union_transcript)
        metadata_rows.append((union_transcript.id, gene_id, biotype))
        union_to_isoforms[union_transcript.id] = {
            "isoforms": [t.id for t in gene.transcripts],
            "biotype": biotype,
        }

    out_bed12_path = str(out_bed12_path)
    out_metadata_tsv_path = str(out_metadata_tsv_path)
    Path(out_bed12_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_metadata_tsv_path).parent.mkdir(parents=True, exist_ok=True)

    collection = TranscriptsCollection(union_transcripts)
    save_transcripts_collection_to_bed12(collection, out_bed12_path)

    with open(out_metadata_tsv_path, "w") as f:
        f.write("transcript_id\tgene_id\tbiotype\n")
        for transcript_id, gene_id, biotype in metadata_rows:
            f.write(f"{transcript_id}\t{gene_id}\t{biotype}\n")

    if union_to_isoforms_path:
        with open(union_to_isoforms_path, "w") as f:
            json.dump(union_to_isoforms, f, indent=2)

    return out_bed12_path, out_metadata_tsv_path
