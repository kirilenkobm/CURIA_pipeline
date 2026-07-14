#!/usr/bin/env python3
"""Analysis B table builder (GPU-free).

For every scored (ref_island -> query_island) candidate in island_alignment_results.tsv,
attach the stored embedding-SW distance (diag_mmd) and every conventional nucleotide
metric computed from the 2bit-fetched sequences. Query metrics use the best of
{fwd, revcomp} to mirror the embedding scorer's min-over-orientation.

Marks the assigned candidate (min diag_mmd within a gene/ref-island) vs within-locus
alternatives. This produces the paired (embedding, sequence) table for the assigned +
detected-alternative candidates; cross-locus and shuffle rows come from the MPS dump.

Output: analysis/embedding_vs_sequence/island_pair_metrics_<tag>.tsv

Run:
  .venv/bin/python analysis/embedding_vs_sequence/scripts/build_island_metrics.py \
      --pair preprint_results/hg38_vs_mm39 \
      --ref-2bit input_data/2bit/hg38.2bit --query-2bit input_data/2bit/mm39.2bit --tag mouse
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import seqmetrics as sm  # noqa: E402
from pyrion import TwoBitAccessor  # noqa: E402
from modules.utils.twobit_alias import AliasedTwoBitAccessor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--ref-2bit", required=True)
    ap.add_argument("--query-2bit", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--min-len", type=int, default=40)
    args = ap.parse_args()

    pair = Path(args.pair)
    df = pd.read_csv(pair / "island_alignment_results.tsv", sep="\t")
    df = df[df["type"] == "match"].reset_index(drop=True)
    # assigned = min diag_mmd within (gene, ref island); n_cand = #candidates
    grp = df.groupby(["gene_id", "ref_island"])
    df["n_cand"] = grp["diag_mmd"].transform("size")
    assigned_idx = grp["diag_mmd"].idxmin()
    df["assigned"] = False
    df.loc[assigned_idx, "assigned"] = True

    ref_acc = TwoBitAccessor(args.ref_2bit)
    q_acc = AliasedTwoBitAccessor(args.query_2bit)

    ref_cache: dict = {}
    rows = []
    drops = 0
    for i, r in df.iterrows():
        rk = (r.ref_chrom, int(r.ref_start), int(r.ref_end))
        rs = ref_cache.get(rk)
        if rs is None:
            try:
                rs = str(ref_acc.fetch(*rk)).upper()
            except Exception:
                rs = ""
            ref_cache[rk] = rs
        try:
            qs = str(q_acc.fetch(r.query_chrom, int(r.query_start), int(r.query_end))).upper()
        except Exception:
            qs = ""
        if len(rs) < args.min_len or len(qs) < args.min_len:
            drops += 1
            continue
        m = sm.all_metrics(rs, qs, best_orientation=True)
        diag = float(r.diag_mmd)
        rows.append(dict(
            gene_id=r.gene_id, ref_island=r.ref_island, query_island=r.query_island,
            ref_chrom=r.ref_chrom, ref_start=int(r.ref_start), ref_end=int(r.ref_end),
            query_chrom=r.query_chrom, query_start=int(r.query_start), query_end=int(r.query_end),
            n_cand=int(r.n_cand), assigned=bool(r.assigned),
            pair_type="assigned" if r.assigned else "within_locus_alt",
            diag_mmd=diag, emb_sw=(1.0 / diag - 1.0) if diag > 0 else np.inf,
            **m,
        ))
        if (i + 1) % 2000 == 0:
            print(f"# [{args.tag}] {i+1}/{len(df)} rows", flush=True)

    out_df = pd.DataFrame(rows)
    out = REPO / f"analysis/embedding_vs_sequence/island_pair_metrics_{args.tag}.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, sep="\t", index=False)
    print(f"# [{args.tag}] rows={len(out_df)} drops={drops} "
          f"assigned={int(out_df.assigned.sum())} alts={int((~out_df.assigned).sum())}")
    print(f"# [{args.tag}] emb_sw vs sw_norm Pearson (assigned) = "
          f"{np.corrcoef(out_df[out_df.assigned].emb_sw.clip(upper=1e6), out_df[out_df.assigned].sw_norm)[0,1]:.3f}")
    print(f"# wrote {out}")


if __name__ == "__main__":
    main()
