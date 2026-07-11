#!/usr/bin/env python3
"""Compute reportable numbers for one reference-vs-query CURIA run.

Reads a results directory (e.g. preprint_results/hg38_vs_mm39) and prints
short-ncRNA, island, and sequence-identity-vs-MMD statistics that can go straight
into the paper. Parametrized by pair so the same script aggregates over the
10-mammal panel once those runs land.

Usage:
    .venv/bin/python analysis/pair_numbers.py \
        --results preprint_results/hg38_vs_mm39 \
        --ref-2bit input_data/2bit/hg38.2bit --query-2bit input_data/2bit/mm39.2bit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for sub in ("pipeline", "RiNALMo"):
    sys.path.insert(0, str(REPO / "modules" / sub))
from short_ncrna import _get_spliced_sequence, _extract_sequence  # noqa: E402
from pyrion import TwoBitAccessor                                 # noqa: E402
from pyrion.io.bed import read_bed12_file                         # noqa: E402

from numba import njit  # noqa: E402


@njit(cache=True)
def _edit_ident(a, b):
    """Levenshtein-based identity of two int-encoded seqs: 1 - dist/max(len)."""
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    prev = np.arange(lb + 1)
    cur = np.zeros(lb + 1, dtype=np.int64)
    for i in range(1, la + 1):
        cur[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d = prev[j] + 1
            l = cur[j - 1] + 1
            diag = prev[j - 1] + cost
            m = d if d < l else l
            m = m if m < diag else diag
            cur[j] = m
        prev, cur = cur, prev
    return 1.0 - prev[lb] / max(la, lb)


_MAP = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4}


def _enc(s):
    return np.array([_MAP.get(c, 4) for c in s.upper()], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--ref-2bit", type=Path, default=REPO / "input_data/2bit/hg38.2bit")
    ap.add_argument("--query-2bit", type=Path, default=REPO / "input_data/2bit/mm39.2bit")
    ap.add_argument("--max-seqid", type=int, default=1500, help="cap loci for seq-id (speed)")
    args = ap.parse_args()
    out = {}

    # ---- short ncRNA ----
    sh = pd.read_csv(args.results / "query_annotation/short_ncRNA_details.tsv", sep="\t")
    bt = sh.groupby("biotype").mmd_score.agg(["count", "median"]).sort_values("median")
    out["short"] = {
        "n": int(len(sh)),
        "mmd_median": round(float(sh.mmd_score.median()), 3),
        "frac_mmd_lt_0.05": round(float((sh.mmd_score < 0.05).mean()), 3),
        "biotype_median_mmd": {k: round(float(v), 3) for k, v in bt["median"].items()},
    }

    # ---- islands ----
    isl = pd.read_csv(args.results / "island_alignment_results.tsv", sep="\t")
    per_gene = isl.groupby("gene_id").size()
    out["islands"] = {
        "n_matches": int(len(isl)),
        "n_ref_genes_matched": int(isl.gene_id.nunique()),
        "matches_per_gene_median": float(per_gene.median()),
        "dist_median": round(float(isl.diag_mmd.median()), 3),
        "frac_dist_lt_0.05": round(float((isl.diag_mmd < 0.05).mean()), 3),
    }

    # ---- sequence identity vs MMD (answers the 45-60 vs 50-60 question) ----
    ref_acc = TwoBitAccessor(str(args.ref_2bit))
    qry_acc = TwoBitAccessor(str(args.query_2bit))
    ref_bed = {t.id: t for t in read_bed12_file(str(args.results / "reference_union_transcripts.bed"))}
    _edit_ident(_enc("ACGU"), _enc("ACGU"))  # warm up numba

    ids, mmds = [], []
    sub = sh.sample(min(len(sh), args.max_seqid), random_state=0)
    for _, r in sub.iterrows():
        t = ref_bed.get(r["transcript_id"])
        if t is None:
            continue
        try:
            rseq = _get_spliced_sequence(t, ref_acc)
            qseq = _extract_sequence(qry_acc, r["chrom"], int(r["start"]), int(r["end"]),
                                     -1 if r["strand"] == "-" else 1)
        except Exception:
            continue
        if not rseq or not qseq or "N" in (rseq + qseq).upper():
            continue
        ids.append(_edit_ident(_enc(rseq), _enc(qseq)) * 100)
        mmds.append(float(r["mmd_score"]))
    ids, mmds = np.array(ids), np.array(mmds)
    from scipy.stats import pearsonr, spearmanr
    pr = pearsonr(ids, mmds); sp = spearmanr(ids, mmds)
    # MMD spread (IQR) in 5%-identity bins -> where is the spread largest?
    bins = np.arange(0, 101, 10)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (ids >= lo) & (ids < hi)
        if m.sum() >= 15:
            q = np.percentile(mmds[m], [25, 75])
            rows.append((f"{lo}-{hi}%", int(m.sum()), round(float(np.median(mmds[m])), 3),
                         round(float(q[1] - q[0]), 3)))
    out["seqid_vs_mmd"] = {
        "n": int(len(ids)),
        "pearson_r": round(float(pr[0]), 3),
        "spearman_rho": round(float(sp[0]), 3),
        "identity_bins": [{"bin": b, "n": n, "mmd_median": md, "mmd_iqr": iqr}
                          for b, n, md, iqr in rows],
        "max_spread_bin": max(rows, key=lambda x: x[3])[0] if rows else None,
    }

    print(json.dumps(out, indent=2))
    (REPO / "analysis" / "data" / f"pair_numbers_{args.results.name}.json").write_text(
        json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
