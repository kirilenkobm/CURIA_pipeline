#!/usr/bin/env python3
"""Within-locus positional null (winner's-curse test) for island matching.

The other-gene specificity test (island_null_test.py) uses too-easy negatives:
beating OTHER genes' islands doesn't prove the assigned position isn't just the
best-of-many maxima inside its own big syntenic window. This test uses the sharp
negative: for every reference island, rank its ASSIGNED query counterpart against
same-length candidate positions WITHIN THE SAME PROJECTED SYNTENIC LOCUS —
the detected query islands of that gene AND sliding windows tiled across the
projected merged region.

Per pair it reports:
  - assigned is #1 / top-3 (fraction of ref islands)
  - empirical percentile of the assigned position among in-locus candidates
  - best-vs-second-best margin (how sharp the score peak is)
  - displacement (bp) between the assigned match and the absolute best position
  - the LARGEST sanity discrepancies (recomputed assigned dist vs stored diag_mmd),
    listed individually rather than summarized by the median.

Reuses the exact pipeline scorer (_dotplot_sw). Needs GPU -> run on the pod:
  .venv/bin/python analysis/island_searchspace_null.py \
      --pair preprint_results/hg38_vs_mm39 \
      --ref-2bit ../2bits/hg38.2bit --query-2bit ../2bits/mm39.2bit --n 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyrion import TwoBitAccessor  # noqa: E402
from modules.pipeline.island_alignment import GPUClient, IslandAlignmentConfig  # noqa: E402
from modules.utils.twobit_alias import AliasedTwoBitAccessor  # noqa: E402
# single source of truth for the executor bootstrap + orientation-robust scorer
from island_null_test import start_executor, revcomp, _dist  # noqa: E402


def _load_locus_map(pair: Path):
    """union_id -> (chrom, start, end) merged syntenic region(s)."""
    u2q = json.load(open(pair / "mappings" / "union_to_query.json"))
    clusters = json.load(open(pair / "mappings" / "query_regions_clusters.json"))
    out = {}
    for uid, region_ids in u2q.items():
        regs = []
        for rid in region_ids:
            c = clusters.get(rid)
            if c and "merged_region" in c:
                m = c["merged_region"]
                regs.append((m["chrom"], int(m["start"]), int(m["end"])))
        if regs:
            out[uid] = regs
    return out


def _pick_locus(regs, qchrom, qstart, qend):
    """The merged region on the assigned query chrom that contains the match."""
    for c, s, e in regs:
        if c == qchrom and s <= qstart and qend <= e:
            return (c, s, e)
    same = [(c, s, e) for (c, s, e) in regs if c == qchrom]
    return same[0] if same else None


async def _amain(args):
    started = time.monotonic()
    cfg = IslandAlignmentConfig.for_model(args.model)
    pair = Path(args.pair)
    df = pd.read_csv(pair / "island_alignment_results.tsv", sep="\t")
    df = df[df["type"] == "match"].reset_index(drop=True)
    take = np.linspace(0, len(df) - 1, min(args.n, len(df))).astype(int)
    df = df.iloc[take].reset_index(drop=True)
    locus_map = _load_locus_map(pair)

    ref_acc = TwoBitAccessor(args.ref_2bit)
    q_acc = AliasedTwoBitAccessor(args.query_2bit)

    proc, inq, outq = start_executor(args.model, args.gpu_max_batch, 32, args.gpu_max_tokens)
    gpu = GPUClient(inq, outq, asyncio.get_running_loop())
    _uid = [0]

    async def emb(seq):
        _uid[0] += 1
        e = await gpu.embed("ssnull", f"e{_uid[0]}", seq, mean_pool=False)
        return np.ascontiguousarray(e, dtype=np.float32)

    recs = []          # per ref-island results
    sanity = []        # (|delta|, stored, recomputed, gene, ref, query)
    skipped = 0
    total = len(df)
    progress_every = max(1, total // 20)
    for row_no, (_, r) in enumerate(df.iterrows(), 1):
        gid = r.gene_id
        regs = locus_map.get(gid)
        if regs is None:
            skipped += 1
            continue
        locus = _pick_locus(regs, r.query_chrom, int(r.query_start), int(r.query_end))
        if locus is None:
            skipped += 1
            continue
        lc, ls, le = locus
        L = int(r.query_end) - int(r.query_start)          # assigned window length
        if L < 40 or le - ls < L:
            skipped += 1
            continue
        try:
            ref_seq = str(ref_acc.fetch(r.ref_chrom, int(r.ref_start), int(r.ref_end))).upper()
        except Exception:
            skipped += 1
            continue
        if len(ref_seq) < 40:
            skipped += 1
            continue
        A_f = await emb(ref_seq)
        A_r = await emb(revcomp(ref_seq))

        # candidate windows: tile the locus at stride L//2 (cap count), always
        # include the assigned window at its true coordinates.
        stride = max(L // 2, 25)
        starts = list(range(ls, le - L + 1, stride))
        if len(starts) > args.max_windows:
            step = len(starts) / args.max_windows
            starts = [starts[int(i * step)] for i in range(args.max_windows)]
        cand = [(s, s + L) for s in starts]
        assigned = (int(r.query_start), int(r.query_end))
        if assigned not in cand:
            cand.append(assigned)
        assigned_idx = cand.index(assigned)

        dists = []
        for (s, e) in cand:
            try:
                w = str(q_acc.fetch(lc, s, e)).upper()
            except Exception:
                dists.append(1.0); continue
            if len(w) < 40:
                dists.append(1.0); continue
            B = await emb(w)
            dists.append(min(_dist(A_f, B, cfg), _dist(A_r, B, cfg)))
        dists = np.array(dists)

        d_assigned = dists[assigned_idx]
        mid = lambda a, b: (a + b) / 2.0
        asg_mid = mid(*assigned)
        mids = np.array([mid(*c) for c in cand])
        # displacement to the nearest globally-best position (ties resolved toward
        # the assigned position, so if assigned is tied-best -> 0, not a grid neighbor)
        gmin = float(dists.min())
        best_mask = dists <= gmin + 1e-9
        displacement = float(np.abs(mids[best_mask] - asg_mid).min())
        # winner's-curse core: consider only NON-OVERLAPPING positions (a tiled
        # window coinciding with the assigned one is not an independent negative).
        far = np.abs(mids - asg_mid) > L
        d_far = dists[far]
        if d_far.size:
            d_best_far = float(d_far.min())
            rank_far = int((d_far < d_assigned).sum())
            pct_far = rank_far / d_far.size
            beats_far = bool(d_assigned <= d_best_far)
            margin_far = d_best_far - d_assigned          # >0 => assigned strictly better
        else:
            d_best_far, rank_far, pct_far, beats_far, margin_far = np.nan, 0, 0.0, True, np.nan
        recs.append(dict(
            ncand=len(cand), n_far=int(d_far.size),
            is_top1=bool(np.argmin(dists) == assigned_idx),
            beats_far=beats_far, pct_far=pct_far, margin_far=margin_far,
            displacement=displacement, d_assigned=float(d_assigned), L=L,
            locus_len=int(le - ls),
        ))
        stored = float(r.diag_mmd)
        sanity.append((abs(d_assigned - stored), stored, d_assigned, gid,
                       f"{r.ref_chrom}:{int(r.ref_start)}-{int(r.ref_end)}",
                       f"{lc}:{assigned[0]}-{assigned[1]}"))
        if row_no == total or row_no % progress_every == 0:
            elapsed = time.monotonic() - started
            rate = row_no / elapsed if elapsed else 0.0
            eta = (total - row_no) / rate if rate else 0.0
            print(f"# within-locus: {row_no}/{total} rows ({100*row_no/total:.0f}%), "
                  f"usable={len(recs)} skipped={skipped} elapsed={elapsed/60:.1f}m "
                  f"eta={eta/60:.1f}m", flush=True)

    gpu.stop()
    try:
        proc.terminate(); proc.join(timeout=5)
    except Exception:
        pass

    R_all = pd.DataFrame(recs)
    n = len(R_all)
    print("\n================ WITHIN-LOCUS POSITIONAL NULL =================")
    print(f"pair: {args.pair}   ref-islands tested: {n}   skipped: {skipped}   model={args.model}")
    if n == 0:
        print("no usable islands (locus map / coords)."); return
    print(f"median candidates / locus : {int(R_all.ncand.median())}   "
          f"median non-overlapping: {int(R_all.n_far.median())}")
    print("--- Does the ASSIGNED position win inside its own locus? ---")
    print(f"  assigned is #1 of all candidates      : {R_all.is_top1.mean()*100:.0f}%")
    # Loci without a non-overlapping alternative pass `beats_far` by definition,
    # so exclude them from every winner's-curse summary.
    R = R_all[R_all["n_far"] >= 1].copy()
    R["chance"] = 1.0 / (R["n_far"] + 1)
    print(f"  loci with independent alternatives    : {len(R)}/{n}")
    print(f"  assigned beats ALL non-overlapping pos: {R.beats_far.mean()*100:.0f}%   <-- winner's-curse pass")
    print(f"  mean chance for beating all positions : {R.chance.mean()*100:.0f}%")
    print(f"  median pctile among non-overlapping   : {R.pct_far.median():.3f}   (0=best, 0.5=random)")
    print(f"  median displacement to abs-best (bp)  : {R.displacement.median():.0f}   (0 = peak at true position)")
    print(f"  median margin (best_far - assigned)   : {R.margin_far.median():.3f}   (>0 => assigned better than any far pos)")
    print("--- VERDICT GUIDE ---")
    print("  REAL     : beats-far high, pctile_far << 0.5, displacement ~ 0, margin > 0")
    print("  ARTIFACT : beats-far ~ chance, pctile_far ~ 0.5, big displacement, margin ~ 0")

    # --- STRATIFIED BY SEARCH-SPACE SIZE (n_far): the key question is whether
    # specificity stays ABOVE CHANCE as the number of same-locus alternatives grows.
    # chance for beats-all with k non-overlapping alternatives = 1/(k+1).
    bins = [(1, 1), (2, 4), (5, 9), (10, 19), (20, 10**9)]
    print("--- STRATIFIED BY n_far (independent same-locus alternatives) ---")
    print(f"  {'n_far':>7} {'n':>5} {'beats':>6} {'chance':>7} {'enrich':>6} "
          f"{'pctile':>6} {'margin':>7} {'displ':>6}")
    for lo, hi in bins:
        b = R[(R.n_far >= lo) & (R.n_far <= hi)]
        if len(b) == 0:
            continue
        beats = b.beats_far.mean(); ch = b.chance.mean()
        lab = f"{lo}" if lo == hi else (f"{lo}+" if hi > 10**8 else f"{lo}-{hi}")
        print(f"  {lab:>7} {len(b):>5} {beats:>6.2f} {ch:>7.2f} "
              f"{beats/ch if ch else float('nan'):>6.1f}x {b.pct_far.median():>6.3f} "
              f"{b.margin_far.median():>7.3f} {b.displacement.median():>6.0f}")
    print("  (REAL: enrich stays >> 1x and pctile ~0 as n_far grows; "
          "ARTIFACT: enrich -> 1x, pctile -> 0.5)")

    print("--- LARGEST SANITY DISCREPANCIES (recomputed vs stored diag_mmd) ---")
    sanity.sort(reverse=True)
    print(f"  {'|delta|':>7}  {'stored':>6}  {'recomp':>6}  gene / ref / query")
    for dlt, st, rc, gid, refc, qc in sanity[:10]:
        print(f"  {dlt:7.3f}  {st:6.3f}  {rc:6.3f}  {gid}  {refc}  {qc}")
    good = np.mean([s[0] < 0.02 for s in sanity])
    print(f"  fraction reproduced within 0.02: {good*100:.0f}%")
    print("===============================================================")

    outdir = REPO / "analysis" / "scratch"; outdir.mkdir(parents=True, exist_ok=True)
    tag = pair.name
    summary = dict(pair=str(pair), n=n, n_with_independent_alternatives=len(R),
                   skipped=skipped,
                   assigned_top1=float(R_all.is_top1.mean()),
                   beats_all_far=float(R.beats_far.mean()),
                   mean_chance_beats_all=float(R.chance.mean()),
                   median_pctile_far=float(R.pct_far.median()),
                   median_displacement=float(R.displacement.median()),
                   median_margin_far=float(R.margin_far.median()),
                   sanity_within_0p02=float(good))
    (outdir / f"searchspace_null_{tag}.json").write_text(json.dumps(summary, indent=2))
    # per-island CSV so the plots can be rebuilt WITHOUT re-running the GPU
    # Preserve all records; the plotting code performs its own n_far >= 1 filter.
    R_all.assign(pair=tag).to_csv(outdir / f"searchspace_null_{tag}.csv", index=False)
    print(f"# wrote {outdir/f'searchspace_null_{tag}.json'} and .csv "
          f"(plot with: analysis/plot_searchspace_null.py)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--ref-2bit", required=True)
    ap.add_argument("--query-2bit", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-windows", type=int, default=100)
    ap.add_argument("--model", default="rinalmo")
    ap.add_argument("--gpu-max-batch", type=int, default=1024)
    ap.add_argument("--gpu-max-tokens", type=int, default=65536)
    args = ap.parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
