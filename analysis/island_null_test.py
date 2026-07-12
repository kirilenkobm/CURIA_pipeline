#!/usr/bin/env python3
"""Island-matching null test: is a ref<->query island MATCH specific (genuine
shared signal) or a plausible artifact the matcher would produce against
anything sitting in the syntenic window?

Premises (granted, NOT under test): synteny is a valid locus-level prior, and
RiNALMo embeddings encode structure. The open question is the MATCH itself. This
tests it with two controls that reuse the EXACT pipeline scorer (_dotplot_sw),
after a single embedding pass:

  1. SPECIFICITY (true-partner rank). Embed N ref islands + N true-query islands,
     score the full NxN distance matrix. For each ref island, rank its TRUE
     syntenic partner among all N query islands.
       - true partner near rank 0 (top) consistently  -> SPECIFIC / real
       - true-partner rank ~ uniform (median ~0.5)     -> GENERIC / artifact
  2. ABOVE-COMPOSITION (dinucleotide shuffle). Score ref_i vs a dinucleotide-
     preserving shuffle of its own true query (same locus, same composition and
     local autocorrelation, homology/structure destroyed).
       - d_real << d_shuffle -> specific structural signal
       - d_real ~= d_shuffle -> composition/autocorrelation artifact

Sanity: the recomputed diagonal distance should track the stored diag_mmd.

Needs the GPU/model -> run on the pod. Example:
  .venv/bin/python analysis/island_null_test.py \
      --pair preprint_results/hg38_vs_mm39 \
      --ref-2bit ../2bits/hg38.2bit --query-2bit ../2bits/mm39.2bit \
      --n 250 --model rinalmo
Run it for a close (rheMac10), mid (mm39) and deep (a marsupial) pair: a real
matcher's specificity should weaken with divergence but stay above chance; an
artifact shows no specificity at any distance.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pyrion import TwoBitAccessor  # noqa: E402
from modules.GPU_executor.gpu_executor import ExecutorConfig, run_gpu_executor  # noqa: E402
from modules.pipeline.island_alignment import GPUClient, IslandAlignmentConfig  # noqa: E402
from modules.pipeline.matchers.rinalmo import _dotplot_sw  # noqa: E402
from modules.utils.twobit_alias import AliasedTwoBitAccessor  # noqa: E402

_COMP = str.maketrans("ACGTUN", "TGCAAN")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def dinuc_shuffle(seq: str, rng: np.random.Generator) -> str:
    """Altschul-Erikson dinucleotide-preserving shuffle: a random sequence with
    EXACTLY the same dinucleotide (and hence mononucleotide) composition."""
    seq = seq.upper()
    if len(seq) < 3:
        return seq
    # edges of the de Bruijn graph over nucleotides
    edges: dict[str, list[str]] = {}
    for a, b in zip(seq[:-1], seq[1:]):
        edges.setdefault(a, []).append(b)
    last = seq[-1]
    # For each node != last, pick a random outgoing edge to be the "last" edge;
    # these last-edges must form an arborescence toward `last` (no cycle) — retry.
    nodes = list(edges.keys())
    for _ in range(50):
        last_edge = {}
        ok = True
        for n in nodes:
            if n == last:
                continue
            outs = edges[n]
            last_edge[n] = outs[rng.integers(len(outs))]
        # check arborescence: following last_edge from each node reaches `last`
        for n in nodes:
            if n == last:
                continue
            seen = set()
            cur = n
            while cur != last:
                if cur in seen or cur not in last_edge:
                    ok = False
                    break
                seen.add(cur)
                cur = last_edge[cur]
            if not ok:
                break
        if ok:
            break
    else:
        return seq  # give up -> return original (rare)
    # shuffle remaining out-edges, append the reserved last-edge at the end
    order: dict[str, list[str]] = {}
    for n in nodes:
        outs = list(edges[n])
        if n != last:
            outs.remove(last_edge[n])
        rng.shuffle(outs)
        if n != last:
            outs.append(last_edge[n])
        order[n] = outs
    # walk the Eulerian path from seq[0]
    out = [seq[0]]
    cur = seq[0]
    ptr = {n: 0 for n in nodes}
    for _ in range(len(seq) - 1):
        nxt = order[cur][ptr[cur]]
        ptr[cur] += 1
        out.append(nxt)
        cur = nxt
    return "".join(out)


def _dist(A, B, cfg) -> float:
    """Pipeline distance for two per-token embedding matrices (best orientation
    handled by caller). d = 1/(1+SW score); returns 1.0 (worst) if no path."""
    score, r0, r1, q0, q1, _ = _dotplot_sw(A, B, cfg.sw_tau_cos, cfg.sw_gap)
    if score <= 0.0 or r1 <= r0 or q1 <= q0:
        return 1.0
    return 1.0 / (1.0 + score)


def start_executor(model, max_batch, min_batch, max_tokens):
    ctx = mp.get_context("spawn")
    inq, outq = ctx.Queue(), ctx.Queue()
    cfg = ExecutorConfig(max_batch=max_batch, min_batch=min_batch,
                         max_tokens=max_tokens, enable_logging=False, model_name=model)
    proc = ctx.Process(target=run_gpu_executor, args=(inq, outq, cfg),
                       name="gpu_executor", daemon=True)
    proc.start()
    return proc, inq, outq


async def _amain(args):
    cfg = IslandAlignmentConfig.for_model(args.model)
    df = pd.read_csv(Path(args.pair) / "island_alignment_results.tsv", sep="\t")
    df = df[df["type"] == "match"].reset_index(drop=True)
    rng = np.random.default_rng(0)
    take = np.linspace(0, len(df) - 1, min(args.n, len(df))).astype(int)
    df = df.iloc[take].reset_index(drop=True)

    ref_acc = TwoBitAccessor(args.ref_2bit)
    q_acc = AliasedTwoBitAccessor(args.query_2bit)

    proc, inq, outq = start_executor(args.model, args.gpu_max_batch, 32, args.gpu_max_tokens)
    gpu = GPUClient(inq, outq, asyncio.get_running_loop())

    async def emb(tag, seq):
        return np.ascontiguousarray(await gpu.embed("null", tag, seq, mean_pool=False),
                                    dtype=np.float32)

    rows = []
    for i, r in df.iterrows():
        try:
            rs = str(ref_acc.fetch(r.ref_chrom, int(r.ref_start), int(r.ref_end))).upper()
            qs = str(q_acc.fetch(r.query_chrom, int(r.query_start), int(r.query_end))).upper()
        except Exception:
            continue
        if len(rs) < 40 or len(qs) < 40:
            continue
        rows.append((i, rs, qs, float(r.diag_mmd)))
    print(f"# usable islands: {len(rows)}")

    # one embedding pass: ref, query (fwd+rc), dinuc-shuffle(query) (fwd+rc).
    # Both orientations everywhere so the diagonal gets no unfair 2-orientation
    # advantage over the shuffle/off-diagonal controls.
    E_ref, E_qf, E_qr, E_shf, E_shr, stored = [], [], [], [], [], []
    for i, rs, qs, dm in rows:
        sh = dinuc_shuffle(qs, rng)
        E_ref.append(await emb(f"r{i}", rs))
        E_qf.append(await emb(f"qf{i}", qs))
        E_qr.append(await emb(f"qr{i}", revcomp(qs)))
        E_shf.append(await emb(f"shf{i}", sh))
        E_shr.append(await emb(f"shr{i}", revcomp(sh)))
        stored.append(dm)
    n = len(E_ref)
    gpu.stop()
    try:
        proc.terminate(); proc.join(timeout=5)
    except Exception:
        pass

    # best-orientation distance for pair (i ref, j query): min over query fwd/rc.
    def dpair(i, j):
        return min(_dist(E_ref[i], E_qf[j], cfg), _dist(E_ref[i], E_qr[j], cfg))

    D = np.array([[dpair(i, j) for j in range(n)] for i in range(n)])
    diag = np.diag(D).copy()
    d_shuf = np.array([min(_dist(E_ref[i], E_shf[i], cfg),
                           _dist(E_ref[i], E_shr[i], cfg)) for i in range(n)])
    stored = np.array(stored)

    # SPECIFICITY: rank of the true partner (col i) within row i, ascending dist
    rank_pct = np.array([(D[i] < diag[i]).sum() / (n - 1) for i in range(n)])
    top1 = float((rank_pct == 0).mean())
    top5 = float((rank_pct <= 0.05).mean())
    ranks = np.array([1 + int((D[i] < diag[i]).sum()) for i in range(n)])  # 1-based
    mrr = float(np.mean(1.0 / ranks))
    # reciprocal top-1: true query best in its ref row AND ref best in its query col
    recip = float(np.mean([(D[i].argmin() == i) and (D[:, i].argmin() == i) for i in range(n)]))
    # AUC: diagonal (true pairs) vs off-diagonal (false pairs); lower dist = better.
    # Mann-Whitney with score = -dist (higher = more "true"). 0.5=chance, 1=perfect.
    pos = -diag
    neg = -D[~np.eye(n, dtype=bool)]
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    rr = np.empty(len(allv)); rr[order] = np.arange(1, len(allv) + 1)
    U = rr[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    auc = float(U / (len(pos) * len(neg)))

    def med(x):
        return float(np.median(x))

    print("\n================ ISLAND MATCH NULL TEST ================")
    print(f"pair: {args.pair}   n={n}   model={args.model}")
    print("--- SANITY: recomputed diagonal vs stored diag_mmd ---")
    print(f"  Pearson r = {np.corrcoef(diag, stored)[0,1]:.3f}   median|Δ| = {med(np.abs(diag-stored)):.3f}")
    print("--- SPECIFICITY (true-partner rank; 0=best, 0.5=random) ---")
    print(f"  median rank-percentile = {med(rank_pct):.3f}   (real << 0.5)")
    print(f"  true partner is #1        : {top1*100:.0f}%")
    print(f"  true partner in top 5%    : {top5*100:.0f}%")
    print(f"  MRR                       : {mrr:.3f}   (real -> 1; chance -> ~{1/n:.3f})")
    print(f"  reciprocal top-1          : {recip*100:.0f}%")
    print(f"  AUC true-vs-false pairs    : {auc:.3f}   (0.5=chance, 1=perfect)")
    print("--- ABOVE-COMPOSITION (dinucleotide shuffle) ---")
    print(f"  d_real  median = {med(diag):.3f}")
    print(f"  d_shuf  median = {med(d_shuf):.3f}")
    print(f"  frac d_real < d_shuf = {float((diag < d_shuf).mean()):.2f}")
    denom = np.sqrt((diag.var() + d_shuf.var()) / 2) or 1.0
    print(f"  Cohen's d (shuf-real)  = {(d_shuf.mean()-diag.mean())/denom:.2f}")
    print("--- VERDICT GUIDE ---")
    print("  REAL     : median rank-pct << 0.5, top-5% high, d_real << d_shuf")
    print("  ARTIFACT : rank-pct ~ 0.5,        top-5% ~ 5%,   d_real ~ d_shuf")
    print("========================================================")

    out = {
        "pair": str(args.pair), "n": n,
        "sanity_r": float(np.corrcoef(diag, stored)[0, 1]),
        "median_rank_pct": med(rank_pct), "top1": top1, "top5": top5,
        "mrr": mrr, "reciprocal_top1": recip, "auc_true_vs_false": auc,
        "d_real_median": med(diag), "d_shuf_median": med(d_shuf),
        "frac_real_lt_shuf": float((diag < d_shuf).mean()),
    }
    outdir = REPO / "analysis" / "scratch"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.pair).name
    (outdir / f"island_null_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"# wrote {outdir/f'island_null_{tag}.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, help="e.g. preprint_results/hg38_vs_mm39")
    ap.add_argument("--ref-2bit", required=True)
    ap.add_argument("--query-2bit", required=True)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--model", default="rinalmo")
    ap.add_argument("--gpu-max-batch", type=int, default=1024)
    ap.add_argument("--gpu-max-tokens", type=int, default=65536)
    args = ap.parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
