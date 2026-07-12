#!/usr/bin/env python3
"""Regenerate UCSC-valid aligned-island BEDs from island_alignment_results.tsv.

The grouped BED12 writer (before the chrom-grouping fix) collapsed a gene's
matched islands into one BED12 record even when they mapped to multiple
chromosomes, writing blocks from one chrom under another's coordinate frame ->
chromEnd past the chromosome end -> UCSC rejects the track. This rebuilds a
clean PER-ISLAND BED6 straight from the TSV's true per-island coordinates, so
every feature is a single contiguous island on one chromosome.

    .venv/bin/python analysis/clean_aligned_islands_bed.py --pair preprint_results/hg38_vs_mm39
    # optional validation against the query genome:
    #   --query-2bit ../2bits/mm39.2bit

Writes <pair>/query_annotation/aligned_{query,reference}_islands_clean.bed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _write(df, chrom_c, start_c, end_c, name_fn, path, sizes=None):
    n = skipped = 0
    with open(path, "w") as f:
        for _, r in df.iterrows():
            c = r[chrom_c]; s = int(r[start_c]); e = int(r[end_c])
            if s < 0 or s >= e or (sizes is not None and (c not in sizes or e > sizes[c])):
                skipped += 1
                continue
            score = max(0, min(1000, int(1000 * (1 - min(float(r["diag_mmd"]), 1)))))
            f.write(f"{c}\t{s}\t{e}\t{name_fn(r)}\t{score}\t.\n")
            n += 1
    return n, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, help="run dir, e.g. preprint_results/hg38_vs_mm39")
    ap.add_argument("--query-2bit", default=None, help="optional: validate query coords vs this 2bit")
    ap.add_argument("--ref-2bit", default=None, help="optional: validate reference coords vs this 2bit")
    args = ap.parse_args()

    pair = Path(args.pair)
    m = pd.read_csv(pair / "island_alignment_results.tsv", sep="\t")
    m = m[m["type"] == "match"].reset_index(drop=True)
    qa = pair / "query_annotation"

    def sizes_of(p):
        if not p:
            return None
        from pyrion import TwoBitAccessor
        return dict(TwoBitAccessor(p).chrom_sizes())

    qn, qs = _write(m, "query_chrom", "query_start", "query_end",
                    lambda r: f"{r.gene_id}:{r.query_island}",
                    qa / "aligned_query_islands_clean.bed", sizes_of(args.query_2bit))
    rn, rs = _write(m, "ref_chrom", "ref_start", "ref_end",
                    lambda r: f"{r.gene_id}:{r.ref_island}",
                    qa / "aligned_reference_islands_clean.bed", sizes_of(args.ref_2bit))
    print(f"# query: {qn} islands ({qs} skipped) -> {qa/'aligned_query_islands_clean.bed'}")
    print(f"# ref:   {rn} islands ({rs} skipped) -> {qa/'aligned_reference_islands_clean.bed'}")


if __name__ == "__main__":
    main()
