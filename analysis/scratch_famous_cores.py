#!/usr/bin/env python3
"""SCRATCH (not a paper figure): core x species heatmaps for a panel of famous
lncRNAs, laid out in a grid, so we can eyeball conservation patterns and spot
anomalies (missing genes, weird cores). Reuses Figure 6's core machinery.

    python analysis/scratch_famous_cores.py [--results-dir preprint_results]
-> analysis/scratch/famous_cores.png (+ .pdf)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import figstyle as fs
import make_figures as mf

REPO = Path(__file__).resolve().parents[1]

# "Famous RNAs" panel as gene SYMBOLS; resolved to ENSG at runtime against the
# hg38 annotation (avoids hand-typed-ID mistakes). Grouped by regime so it's easy
# to extend. Unresolvable symbols (not in this GENCODE) are warned + skipped.
FAMOUS_SYMBOLS = [
    # original case studies
    "MALAT1", "NEAT1", "XIST", "XACT", "MIAT", "NORAD", "HOTAIR", "H19", "TUG1",
    "RMST", "LINC-PINT", "FIRRE", "PVT1", "DDX25-AS1", "HYOU1-AS1", "KCNQ1OT1", "MEG3",
    # ancient / very conserved (SNHG = snoRNA hosts; RMRP/RPPH1 = structured ribozymes)
    "GAS5", "SNHG12", "DANCR", "RMRP", "RPPH1",
    # nuclear architecture / chromatin / imprinting
    "JPX", "FTX", "AIRN", "CDKN2B-AS1", "GNAS-AS1", "MEG8",
    # developmental / morphogenesis
    "FENDRR", "HOTTIP", "HOTAIRM1", "LINC00261",
    # neuronal / other diagnostic
    "BCYRN1", "SOX2-OT", "HAR1A", "BDNF-AS",
]
# aliases the annotation does not carry under the common name -> official symbol
SYMBOL_ALIASES = {"ANRIL": "CDKN2B-AS1", "GOMAFU": "MIAT", "NESPAS": "GNAS-AS1",
                  "DEANR1": "LINC00261"}

# Verified ENSG for the curated panel. hg38_gene_names.txt is an INCOMPLETE export
# (missing e.g. XIST/XACT/TUG1), so these overrides are authoritative; name lookup
# is only a fallback for symbols added later. All confirmed present in the union.
SYMBOL_OVERRIDES = {
    "MALAT1": "ENSG00000251562", "NEAT1": "ENSG00000245532", "XIST": "ENSG00000229807",
    "XACT": "ENSG00000241743", "MIAT": "ENSG00000225783", "NORAD": "ENSG00000260032",
    "HOTAIR": "ENSG00000228630", "H19": "ENSG00000130600", "TUG1": "ENSG00000253352",
    "RMST": "ENSG00000255794", "LINC-PINT": "ENSG00000231721", "FIRRE": "ENSG00000213468",
    "PVT1": "ENSG00000249859", "DDX25-AS1": "ENSG00000255027", "HYOU1-AS1": "ENSG00000255114",
    "KCNQ1OT1": "ENSG00000269821", "MEG3": "ENSG00000214548", "GAS5": "ENSG00000234741",
    "SNHG12": "ENSG00000197989", "DANCR": "ENSG00000226950", "RMRP": "ENSG00000277027",
    "RPPH1": "ENSG00000277209", "JPX": "ENSG00000225470", "FTX": "ENSG00000230590",
    "AIRN": "ENSG00000268257", "CDKN2B-AS1": "ENSG00000240498", "GNAS-AS1": "ENSG00000235590",
    "MEG8": "ENSG00000225746", "FENDRR": "ENSG00000268388", "HOTTIP": "ENSG00000243766",
    "HOTAIRM1": "ENSG00000233429", "LINC00261": "ENSG00000259974", "BCYRN1": "ENSG00000236824",
    "SOX2-OT": "ENSG00000242808", "HAR1A": "ENSG00000225978", "BDNF-AS": "ENSG00000245573",
}


def _resolve_symbols(symbols):
    """Map gene symbols -> ENSG. Uses verified overrides first, then the (partial)
    annotation name table as a fallback for symbols added later. Returns
    [(symbol, ensg), ...] for resolved genes and warns on misses."""
    import pandas as pd
    gn = pd.read_csv(REPO / "input_data/reference_annotation/hg38_gene_names.txt", sep="\t")
    name2ensg = (gn.dropna(subset=["Gene name"]).drop_duplicates("Gene name")
                 .set_index("Gene name")["Gene stable ID"].to_dict())
    resolved, missing = [], []
    for s in symbols:
        key = SYMBOL_ALIASES.get(s, s)
        ensg = SYMBOL_OVERRIDES.get(s) or SYMBOL_OVERRIDES.get(key) or name2ensg.get(key)
        if ensg:
            resolved.append((s, ensg))
        else:
            missing.append(s)
    if missing:
        print(f"# WARNING: {len(missing)} symbols not found (add to SYMBOL_OVERRIDES "
              f"or check the name): {', '.join(missing)}")
    return resolved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=REPO / "preprint_results")
    ap.add_argument("--outdir", type=Path, default=REPO / "analysis" / "scratch")
    args = ap.parse_args()

    fs.set_style()
    famous = _resolve_symbols(FAMOUS_SYMBOLS)
    df, present = mf._load_islands(args.results_dir)
    if df is None:
        raise SystemExit(f"no island results under {args.results_dir}")
    df = mf._cluster_cores(df)
    best = mf._best_per_core_species(df)
    cmap = plt.cm.viridis_r

    ncol = 4
    nrow = (len(famous) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.2 * nrow + 0.6))
    fig.subplots_adjust(hspace=0.6, wspace=0.25)
    axes = axes.flatten()

    im = None
    for i, (sym, ens) in enumerate(famous):
        ax = axes[i]
        r = mf._fig6a_heatmap(ax, best, ens, sym, present, cmap,
                              show_ylabels=(i % ncol == 0))
        im = im or r
    for j in range(len(famous), len(axes)):
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
    for sym, ens in famous:
        g = best[best["gene_bare"] == ens]
        print(f"{sym:12} {g['core_id'].nunique():>5} {g['species'].nunique():>7}  {ens}"
              + ("" if not g.empty else "   <-- no cores"))


if __name__ == "__main__":
    main()
