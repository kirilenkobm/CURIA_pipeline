#!/usr/bin/env python3
"""Bounded RiNALMo pass (MPS/CUDA/CPU auto) that PERSISTS the per-pair island
embedding-SW scores the null scripts only ever computed in memory. Everything
here reuses the exact pipeline scorer (_dotplot_sw via island_null_test._dist);
we only save the arrays so analyze_island.py can put every embedding score next
to its nucleotide-metric counterpart on IDENTICAL pairs.

Two passes per species, both on the SAME deterministically-sampled islands as
analysis/island_null_test.py (np.linspace, so results are comparable):
  1. NxN specificity + dinucleotide-shuffle:  D[i,j] = embed-SW dist(ref_i, query_j),
     diag = assigned pairs, off-diag = cross-locus negatives, d_shuf = ref_i vs
     dinuc-shuffle(query_i).  (N islands)
  2. within-locus tiled windows: for each ref island, embed-SW dist of its assigned
     query window vs same-length windows tiled across the projected syntenic locus.
     (n_win islands, <= max_windows candidates each)

Output: analysis/embedding_vs_sequence/data/island_embed_<tag>.npz

Run (mouse then cow):
  KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python \
    analysis/embedding_vs_sequence/scripts/dump_island_embeddings.py \
    --pair preprint_results/hg38_vs_mm39 \
    --ref-2bit input_data/2bit/hg38.2bit --query-2bit input_data/2bit/mm39.2bit \
    --tag mouse --n 200 --n-win 100 --max-windows 25
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

from pyrion import TwoBitAccessor  # noqa: E402
from modules.pipeline.island_alignment import IslandAlignmentConfig  # noqa: E402
from modules.utils.twobit_alias import AliasedTwoBitAccessor  # noqa: E402
from island_null_test import start_executor, revcomp, dinuc_shuffle, _dist  # noqa: E402
from island_searchspace_null import _load_locus_map, _pick_locus  # noqa: E402
from modules.pipeline.island_alignment import GPUClient  # noqa: E402

OUTDIR = REPO / "analysis/embedding_vs_sequence/data"


async def _amain(args):
    t0 = time.monotonic()
    cfg = IslandAlignmentConfig.for_model(args.model)
    pair = Path(args.pair)
    df = pd.read_csv(pair / "island_alignment_results.tsv", sep="\t")
    df = df[df["type"] == "match"].reset_index(drop=True)
    # assigned = best (min diag_mmd) candidate per (gene, ref island)
    df = df.loc[df.groupby(["gene_id", "ref_island"])["diag_mmd"].idxmin()].reset_index(drop=True)

    take = np.linspace(0, len(df) - 1, min(args.n, len(df))).astype(int)
    sub = df.iloc[take].reset_index(drop=True)

    ref_acc = TwoBitAccessor(args.ref_2bit)
    q_acc = AliasedTwoBitAccessor(args.query_2bit)
    rng = np.random.default_rng(0)

    proc, inq, outq = start_executor(args.model, args.gpu_max_batch, 32, args.gpu_max_tokens)
    gpu = GPUClient(inq, outq, asyncio.get_running_loop())

    async def emb(tag, seq):
        return np.ascontiguousarray(await gpu.embed("dump", tag, seq, mean_pool=False),
                                    dtype=np.float32)

    # ---------- pass 1: NxN specificity + shuffle ----------
    recs = []
    for i, r in sub.iterrows():
        try:
            rs = str(ref_acc.fetch(r.ref_chrom, int(r.ref_start), int(r.ref_end))).upper()
            qs = str(q_acc.fetch(r.query_chrom, int(r.query_start), int(r.query_end))).upper()
        except Exception:
            continue
        if len(rs) < 40 or len(qs) < 40:
            continue
        recs.append((i, r, rs, qs))
    print(f"# [{args.tag}] usable islands: {len(recs)}", flush=True)

    E_ref, E_qf, E_qr, E_shf, E_shr = [], [], [], [], []
    meta = []
    for k, (i, r, rs, qs) in enumerate(recs, 1):
        sh = dinuc_shuffle(qs, rng)
        E_ref.append(await emb(f"r{i}", rs))
        E_qf.append(await emb(f"qf{i}", qs))
        E_qr.append(await emb(f"qr{i}", revcomp(qs)))
        E_shf.append(await emb(f"shf{i}", sh))
        E_shr.append(await emb(f"shr{i}", revcomp(sh)))
        meta.append(dict(gene_id=r.gene_id, ref_island=r.ref_island,
                         ref_chrom=r.ref_chrom, ref_start=int(r.ref_start), ref_end=int(r.ref_end),
                         query_chrom=r.query_chrom, query_start=int(r.query_start),
                         query_end=int(r.query_end), diag_mmd=float(r.diag_mmd),
                         ref_seq=rs, query_seq=qs, shuf_seq=sh))
        if k % max(1, len(recs) // 20) == 0 or k == len(recs):
            el = (time.monotonic() - t0) / 60
            print(f"# [{args.tag}] embed {k}/{len(recs)} ({100*k/len(recs):.0f}%) {el:.1f}m", flush=True)
    n = len(E_ref)

    def dpair(i, j):
        return min(_dist(E_ref[i], E_qf[j], cfg), _dist(E_ref[i], E_qr[j], cfg))

    D = np.array([[dpair(i, j) for j in range(n)] for i in range(n)])
    d_shuf = np.array([min(_dist(E_ref[i], E_shf[i], cfg), _dist(E_ref[i], E_shr[i], cfg))
                       for i in range(n)])
    diag = np.diag(D).copy()
    stored = np.array([m["diag_mmd"] for m in meta])
    r_sanity = float(np.corrcoef(diag, stored)[0, 1])
    print(f"# [{args.tag}] NxN done n={n}  sanity diag-vs-stored r={r_sanity:.3f}", flush=True)

    # ---------- pass 2: within-locus tiled windows ----------
    locus_map = _load_locus_map(pair)
    win_take = np.linspace(0, len(df) - 1, min(args.n_win, len(df))).astype(int)
    wsub = df.iloc[win_take].reset_index(drop=True)
    win_records = []
    for wi, r in wsub.iterrows():
        regs = locus_map.get(r.gene_id)
        if regs is None:
            continue
        locus = _pick_locus(regs, r.query_chrom, int(r.query_start), int(r.query_end))
        if locus is None:
            continue
        lc, ls, le = locus
        L = int(r.query_end) - int(r.query_start)
        if L < 40 or le - ls < L:
            continue
        try:
            ref_seq = str(ref_acc.fetch(r.ref_chrom, int(r.ref_start), int(r.ref_end))).upper()
        except Exception:
            continue
        if len(ref_seq) < 40:
            continue
        A_f = await emb(f"wrf{wi}", ref_seq)
        A_r = await emb(f"wrr{wi}", revcomp(ref_seq))
        stride = max(L // 2, 25)
        starts = list(range(ls, le - L + 1, stride))
        if len(starts) > args.max_windows:
            step = len(starts) / args.max_windows
            starts = [starts[int(t * step)] for t in range(args.max_windows)]
        cand = [(s, s + L) for s in starts]
        assigned = (int(r.query_start), int(r.query_end))
        if assigned not in cand:
            cand.append(assigned)
        assigned_idx = cand.index(assigned)
        dists, wseqs = [], []
        for (s, e) in cand:
            try:
                w = str(q_acc.fetch(lc, s, e)).upper()
            except Exception:
                dists.append(1.0); wseqs.append(""); continue
            if len(w) < 40:
                dists.append(1.0); wseqs.append(""); continue
            B = await emb(f"w{wi}_{s}", w)
            dists.append(min(_dist(A_f, B, cfg), _dist(A_r, B, cfg)))
            wseqs.append(w)
        win_records.append(dict(
            gene_id=r.gene_id, ref_island=r.ref_island, ref_seq=ref_seq,
            locus_chrom=lc, L=int(L), assigned_idx=int(assigned_idx),
            cand_starts=np.array([c[0] for c in cand]),
            cand_ends=np.array([c[1] for c in cand]),
            emb_dists=np.array(dists), win_seqs=np.array(wseqs, dtype=object),
        ))
        if len(win_records) % 10 == 0:
            print(f"# [{args.tag}] windows {len(win_records)}/{len(wsub)}", flush=True)

    gpu.stop()
    try:
        proc.terminate(); proc.join(timeout=5)
    except Exception:
        pass

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"island_embed_{args.tag}.npz"
    np.savez(
        out,
        D=D, diag=diag, d_shuf=d_shuf, stored_diag_mmd=stored, sanity_r=r_sanity,
        meta=np.array(meta, dtype=object),
        win_records=np.array(win_records, dtype=object),
        pair=str(pair), tag=args.tag,
    )
    print(f"# [{args.tag}] wrote {out}  (N={n}, windows={len(win_records)}, "
          f"sanity_r={r_sanity:.3f}) elapsed={(time.monotonic()-t0)/60:.1f}m", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--ref-2bit", required=True)
    ap.add_argument("--query-2bit", required=True)
    ap.add_argument("--tag", required=True, help="mouse | cow")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-win", type=int, default=100)
    ap.add_argument("--max-windows", type=int, default=25)
    ap.add_argument("--model", default="rinalmo")
    ap.add_argument("--gpu-max-batch", type=int, default=256)
    ap.add_argument("--gpu-max-tokens", type=int, default=32768)
    args = ap.parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
