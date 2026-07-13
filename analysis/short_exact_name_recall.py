#!/usr/bin/env python3
"""Conservative short-branch recall proxy using exact human/mouse gene names.

The denominator is geometry-eligible, non-protein-coding human union loci whose
gene name occurs exactly (case-insensitive) in the mouse GENCODE annotation. A
locus is recovered when any final CURIA prediction overlaps an exon of that
same-name mouse gene. This favors well-named conserved families and is therefore
a rough named-counterpart recall, not genome-wide biological recall.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path,
                    default=REPO / "preprint_results/hg38_vs_mm39")
    ap.add_argument("--mouse-annotation", type=Path,
                    default=REPO / "input_data/mm39_annotation_validation")
    ap.add_argument("--output", type=Path,
                    default=REPO / "analysis/data/short_exact_name_recall.json")
    args = ap.parse_args()

    meta = pd.read_csv(args.results / "reference_union_transcripts_metadata.tsv", sep="\t")
    names = pd.read_csv(REPO / "input_data/reference_annotation/hg38_gene_names.txt",
                        sep="\t", usecols=["Gene stable ID", "Gene name"])
    names = names.drop_duplicates("Gene stable ID").set_index("Gene stable ID")["Gene name"]
    meta["gene_bare"] = meta.gene_id.str.split(".").str[0]
    meta["gene_name"] = meta.gene_bare.map(names)

    bed = pd.read_csv(args.results / "reference_union_transcripts.bed", sep="\t", header=None)
    bed["transcript_id"] = bed[3]
    bed["exonic_len"] = (bed[10].astype(str).str.rstrip(",")
                         .map(lambda x: sum(map(int, x.split(",")))))
    eligible = set(bed.loc[(bed[9] == 1) & (bed.exonic_len <= 256), "transcript_id"])
    eligible -= set(meta.loc[meta.biotype == "protein_coding", "transcript_id"])
    human_name = {
        r.transcript_id: str(r.gene_name).upper()
        for r in meta[meta.transcript_id.isin(eligible) & meta.gene_name.notna()].itertuples()
        if str(r.gene_name).strip()
    }

    mm_meta = pd.read_csv(args.mouse_annotation / "mm39_gencode_metadata.tsv", sep="\t",
                          usecols=["transcript_id", "gene_name"])
    transcript_name = dict(zip(mm_meta.transcript_id,
                               mm_meta.gene_name.astype(str).str.upper()))
    mm_bed = pd.read_csv(args.mouse_annotation / "mm39_gencode_all_transcripts.bed",
                         sep="\t", header=None)
    exons = defaultdict(lambda: defaultdict(list))
    for row in mm_bed.itertuples(index=False, name=None):
        chrom, tx_start, transcript_id = str(row[0]), int(row[1]), str(row[3])
        gene_name = transcript_name.get(transcript_id)
        if not gene_name or gene_name == "NAN":
            continue
        sizes = [int(x) for x in str(row[10]).rstrip(",").split(",")]
        offsets = [int(x) for x in str(row[11]).rstrip(",").split(",")]
        exons[gene_name][chrom].extend(
            (tx_start + offset, tx_start + offset + size)
            for offset, size in zip(offsets, sizes)
        )

    denominator = {tx for tx, name in human_name.items() if name in exons}
    predictions = pd.read_csv(args.results / "query_annotation/short_ncRNA_details.tsv",
                              sep="\t")
    recovered = set()
    for r in predictions.itertuples(index=False):
        if r.transcript_id not in denominator:
            continue
        for start, end in exons[human_name[r.transcript_id]].get(str(r.chrom), []):
            if int(r.start) < end and start < int(r.end):
                recovered.add(r.transcript_id)
                break

    out = {
        "criterion": "exact case-insensitive gene name; prediction overlaps same-name mouse exon",
        "n_geometry_eligible_non_protein_coding": len(eligible),
        "n_with_named_mouse_counterpart": len(denominator),
        "n_recovered": len(recovered),
        "rough_named_counterpart_recall": len(recovered) / len(denominator),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
