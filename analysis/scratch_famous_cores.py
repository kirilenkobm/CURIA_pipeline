#!/usr/bin/env python3
"""SCRATCH (not a paper figure): core x species heatmaps for a panel of famous
lncRNAs, laid out in a grid, so we can eyeball conservation patterns and spot
anomalies (missing genes, weird cores). Reuses Figure 6's core machinery.

    python analysis/scratch_famous_cores.py [--results-dir rinalmo_version_outputs]
-> analysis/scratch/famous_cores.png (+ .pdf)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import figstyle as fs
import make_figures as mf

REPO = Path(__file__).resolve().parents[1]

# The full "famous RNAs" panel (CASE_STUDY_GENES from island_phylo_conservation.ipynb)
FAMOUS = [
    ("MALAT1", "ENSG00000251562"), ("NEAT1", "ENSG00000245532"),
    ("XIST", "ENSG00000229807"),   ("XACT", "ENSG00000241743"),
    ("MIAT", "ENSG00000225783"),   ("NORAD", "ENSG00000260032"),
    ("HOTAIR", "ENSG00000228630"), ("H19", "ENSG00000130600"),
    ("TUG1", "ENSG00000253352"),   ("RMST", "ENSG00000255794"),
    ("LINC-PINT", "ENSG00000231721"), ("FIRRE", "ENSG00000213468"),
    ("PVT1", "ENSG00000249859"),   ("DDX25-AS1", "ENSG00000255027"),
    ("HYOU1-AS1", "ENSG00000255114"), ("KCNQ1OT1", "ENSG00000269821"),
    ("MEG3", "ENSG00000214548"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=REPO / "rinalmo_version_outputs")
    ap.add_argument("--outdir", type=Path, default=REPO / "analysis" / "scratch")
    args = ap.parse_args()

    fs.set_style()
    df, present = mf._load_islands(args.results_dir)
    if df is None:
        raise SystemExit(f"no island results under {args.results_dir}")
    df = mf._cluster_cores(df)
    best = mf._best_per_core_species(df)
    cmap = plt.cm.viridis_r

    ncol = 4
    nrow = (len(FAMOUS) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.2 * nrow + 0.6))
    fig.subplots_adjust(hspace=0.55, wspace=0.25)
    axes = axes.flatten()

    im = None
    for i, (sym, ens) in enumerate(FAMOUS):
        ax = axes[i]
        r = mf._fig6a_heatmap(ax, best, ens, sym, present, cmap,
                              show_ylabels=(i % ncol == 0))
        im = im or r
    for j in range(len(FAMOUS), len(axes)):
        axes[j].set_axis_off()

    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), shrink=0.5, pad=0.02,
                     label=mf._DIST_LABEL)
    fig.suptitle(f"Famous lncRNA conserved cores  —  {len(present)} species: "
                 f"{', '.join(present)}", fontsize=10, y=0.995)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "famous_cores.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # quick text summary: cores/species per gene (spot the anomalies)
    print(f"\n{'gene':12} {'cores':>5} {'species':>7}  ENSG")
    for sym, ens in FAMOUS:
        g = best[best["gene_bare"] == ens]
        print(f"{sym:12} {g['core_id'].nunique():>5} {g['species'].nunique():>7}  {ens}"
              + ("" if not g.empty else "   <-- no cores"))


if __name__ == "__main__":
    main()
