#!/usr/bin/env python3
"""Figure-5 case studies: per-prediction sequence-vs-embedding numbers for the three
loci shown in the CURIA case-study figure (human->mouse):
  SNORD57, RNU6-7, and the vault RNA ENSG00000199990.

For each prediction of these genes it reports (from the existing short_ncrna_metrics.tsv,
so metric definitions are identical to Analysis A):
  1. reference + query identifiers
  2. reference + predicted query coordinates
  3. reference + query lengths
  4. edit-distance identity (ident_levenshtein)
  5. normalized nucleotide Smith-Waterman (sw_norm)
  6. Smith-Waterman aligned identity (sw_aligned_ident)
  7. pipeline MMD score
  8. query-annotation gene(s) overlapped by the prediction (mm39 GENCODE + tRNA)

Output: analysis/embedding_vs_sequence/case_studies_fig5.tsv

Run:
  .venv/bin/python analysis/embedding_vs_sequence/scripts/case_studies_fig5.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "analysis/embedding_vs_sequence"
RESULTS = REPO / "preprint_results/hg38_vs_mm39"
MM_ANNO = REPO / "input_data/mm39_annotation_validation"

# human genes shown in Fig. 5 (by symbol, and the vault RNA by stable id)
BY_NAME = {"SNORD57", "RNU6-7"}
BY_GENEID = {"ENSG00000199990"}   # vault RNA -> mouse Vaultrc5
NAME_OVERRIDE = {"ENSG00000199990": "VTRNA(vaultRNA)"}   # not symbol-named in hg38 export


def load_human_names():
    n = pd.read_csv(REPO / "input_data/reference_annotation/hg38_gene_names.txt", sep="\t",
                    usecols=["Gene stable ID", "Gene name"])
    return n.drop_duplicates("Gene stable ID").set_index("Gene stable ID")["Gene name"]


def load_ref_coords():
    bed = pd.read_csv(RESULTS / "reference_union_transcripts.bed", sep="\t", header=None)
    return {r[3]: (str(r[0]), int(r[1]), int(r[2]), "+" if r[5] == "+" else "-")
            for r in bed.itertuples(index=False, name=None)}


def load_mouse_exons():
    meta = pd.read_csv(MM_ANNO / "mm39_gencode_metadata.tsv", sep="\t",
                       usecols=["transcript_id", "gene_name"])
    tx_name = dict(zip(meta.transcript_id, meta.gene_name.astype(str)))
    beds = [(MM_ANNO / "mm39_gencode_all_transcripts.bed", True)]
    trna = MM_ANNO / "mm39-tRNAs.bed"
    if trna.exists():
        beds.append((trna, False))
    # exons[chrom] -> list of (start, end, gene_label)
    exons = defaultdict(list)
    for path, use_name in beds:
        b = pd.read_csv(path, sep="\t", header=None)
        for row in b.itertuples(index=False, name=None):
            chrom, tx_start, tid = str(row[0]), int(row[1]), str(row[3])
            label = tx_name.get(tid) if use_name else tid
            if not label or label == "nan":
                label = tid
            try:
                sizes = [int(x) for x in str(row[10]).rstrip(",").split(",")]
                offs = [int(x) for x in str(row[11]).rstrip(",").split(",")]
            except Exception:
                continue
            for off, sz in zip(offs, sizes):
                exons[chrom].append((tx_start + off, tx_start + off + sz, label))
    return exons


def overlapped_genes(exons, chrom, start, end):
    hits = {}
    for s, e, label in exons.get(str(chrom), []):
        if start < e and s < end:
            ov = min(end, e) - max(start, s)
            hits[label] = max(hits.get(label, 0), ov)
    return sorted(hits.items(), key=lambda kv: -kv[1])


def main():
    names = load_human_names()
    ref_coords = load_ref_coords()
    exons = load_mouse_exons()
    m = pd.read_csv(BASE / "short_ncrna_metrics.tsv", sep="\t")
    m["gene_bare"] = m.gene_id.str.split(".").str[0]
    m["human_name"] = m.gene_bare.map(names)
    for gid, nm in NAME_OVERRIDE.items():
        m.loc[m.gene_bare == gid, "human_name"] = nm

    sel = m[m.human_name.isin(BY_NAME) | m.gene_bare.isin(BY_GENEID)].copy()
    rows = []
    for r in sel.itertuples(index=False):
        rc = ref_coords.get(r.transcript_id, (None, None, None, None))
        og = overlapped_genes(exons, r.chrom, int(r.start), int(r.end))
        og_str = "; ".join(f"{g}({bp}bp)" for g, bp in og[:4]) if og else "(none)"
        rows.append(dict(
            fig5_label=f"{r.human_name or r.gene_bare}.{r.chain_id}",
            human_gene=r.human_name or "", ref_gene_id=r.gene_id, biotype=r.biotype,
            ref_transcript_id=r.transcript_id,
            ref_chrom=rc[0], ref_start=rc[1], ref_end=rc[2], ref_strand=rc[3],
            query_chrom=r.chrom, query_start=int(r.start), query_end=int(r.end),
            query_strand=r.strand, chain_id=r.chain_id,
            ref_len_spliced=int(r.len_ref), query_len=int(r.len_query),
            ident_levenshtein=round(float(r.ident_levenshtein), 4),
            sw_norm=round(float(r.sw_norm), 4),
            sw_aligned_ident=round(float(r.sw_aligned_ident), 4),
            mmd=round(float(r.mmd), 4),
            overlap_any=bool(r.overlap_any), overlap_bp=int(r.overlap_bp),
            query_overlap_genes=og_str,
        ))
    out = pd.DataFrame(rows).sort_values(["human_gene", "chain_id"])
    out.to_csv(BASE / "case_studies_fig5.tsv", sep="\t", index=False)

    pd.set_option("display.max_columns", None, "display.width", 240)
    show = ["fig5_label", "ref_gene_id", "ref_transcript_id",
            "ref_chrom", "ref_start", "ref_end", "ref_strand",
            "query_chrom", "query_start", "query_end", "query_strand",
            "ref_len_spliced", "query_len", "ident_levenshtein", "sw_norm",
            "sw_aligned_ident", "mmd", "query_overlap_genes"]
    for _, g in out.groupby("human_gene"):
        print(f"\n########## {g.iloc[0].human_gene or g.iloc[0].ref_gene_id} "
              f"({g.iloc[0].biotype}, {len(g)} prediction(s)) ##########")
        for _, r in g.iterrows():
            for k in show:
                print(f"  {k:20s}: {r[k]}")
            print("  " + "-" * 40)
    print(f"\n# wrote {BASE/'case_studies_fig5.tsv'}")


if __name__ == "__main__":
    main()
