#!/usr/bin/env python3
"""Rebuild the within-locus positional-null plots from the per-island CSVs that
island_searchspace_null.py writes (analysis/scratch/searchspace_null_*.csv).
GPU-free — re-runnable any time without touching the model.

    .venv/bin/python analysis/plot_searchspace_null.py
-> analysis/scratch/searchspace_null_bins.png (+ .pdf)

Key question: does specificity (assigned island beats ALL same-locus
alternatives) stay ABOVE CHANCE as the number of alternatives (n_far) and the
projected locus length grow? chance for beating k independent alternatives =
1/(k+1); if the curve tracks that dashed line, the matches are noise-maxima.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
SCR = REPO / "analysis" / "scratch"
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import figstyle as fs
    fs.set_style()
except Exception:
    pass

NBINS = [(1, 1), (2, 4), (5, 9), (10, 19), (20, 10**9)]
NLAB = ["1", "2-4", "5-9", "10-19", "20+"]
LBINS = [(0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 10**9)]
LLAB = ["<0.5k", "0.5-1k", "1-2k", "2-5k", ">5k"]


def load() -> pd.DataFrame:
    fs_ = sorted(glob.glob(str(SCR / "searchspace_null_*.csv")))
    if not fs_:
        sys.exit("no searchspace_null_*.csv in analysis/scratch — "
                 "run analysis/island_searchspace_null.py first")
    return pd.concat([pd.read_csv(f) for f in fs_], ignore_index=True)


def agg(df: pd.DataFrame, bins, col: str) -> pd.DataFrame:
    df = df[df.n_far >= 1].copy()
    df["chance"] = 1.0 / (df.n_far + 1)
    rows = []
    for lo, hi in bins:
        b = df[(df[col] >= lo) & (df[col] <= hi)]
        rows.append(dict(
            n=len(b),
            beats=b.beats_far.mean() if len(b) else np.nan,
            chance=b.chance.mean() if len(b) else np.nan,
            pct=b.pct_far.median() if len(b) else np.nan,
        ))
    return pd.DataFrame(rows)


def main():
    df = load()
    pairs = sorted(df.pair.unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axA, axB, axC, axD = axes.ravel()
    x = np.arange(len(NLAB))

    for p in pairs:
        a = agg(df[df.pair == p], NBINS, "n_far")
        axA.plot(x, a.beats, "o-", label=p)
        axB.plot(x, a.beats / a.chance, "o-", label=p)
        axD.plot(x, a.pct, "o-", label=p)
    axA.plot(x, agg(df, NBINS, "n_far").chance, "k--", lw=1.4, label="chance 1/(k+1)")

    axA.set(xticks=x, xlabel="n_far (independent same-locus positions)",
            ylabel="beats ALL non-overlapping", ylim=(0, 1.05),
            title="Specificity vs search-space size")
    axA.set_xticklabels(NLAB); axA.legend(fontsize=7)

    axB.axhline(1, ls="--", c="k", lw=1.4)
    axB.set(xticks=x, xlabel="n_far", ylabel="enrichment over chance",
            title="Enrichment vs search-space size")
    axB.set_xticklabels(NLAB); axB.legend(fontsize=7)

    xl = np.arange(len(LLAB))
    for p in pairs:
        al = agg(df[df.pair == p], LBINS, "locus_len")
        axC.plot(xl, al.beats, "o-", label=p)
    axC.plot(xl, agg(df, LBINS, "locus_len").chance, "k--", lw=1.4, label="chance")
    axC.set(xticks=xl, xlabel="projected locus length (bp)",
            ylabel="beats ALL non-overlapping", ylim=(0, 1.05),
            title="Specificity vs locus length")
    axC.set_xticklabels(LLAB); axC.legend(fontsize=7)

    axD.axhline(0.5, ls="--", c="k", lw=1.4, label="random")
    axD.set(xticks=x, xlabel="n_far", ylabel="median pctile among far (0=best)",
            ylim=(0, 0.55), title="Rank of true position vs search-space size")
    axD.set_xticklabels(NLAB); axD.legend(fontsize=7)

    fig.suptitle("Within-locus positional null — does specificity survive a growing search space?",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(SCR / f"searchspace_null_bins.{ext}", dpi=140, bbox_inches="tight")
    print(f"wrote {SCR/'searchspace_null_bins.png'} ({len(pairs)} pairs: {', '.join(pairs)})")


if __name__ == "__main__":
    main()
