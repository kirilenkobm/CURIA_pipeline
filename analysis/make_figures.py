#!/usr/bin/env python3
"""Build CURIA paper figures into paper/figures/.

Design (see paper/README.md "Figure pipeline"):
  * SCRIPTED figures (1, 3B, 4, 6) are composed *entirely* in matplotlib via
    figstyle.mosaic --- panel letters, text-width sizing, embedded fonts --- so
    no hand-composition is needed. They emit vector PDF (+ PNG preview).
  * SCHEMATIC figures (2, 3A) and SCREENSHOTS (5) are hand-made assets dropped
    into paper/figures/; this script does not generate them (listed in MANUAL).

Each scripted figure is one builder function registered with @figure(name).
Builders receive (outdir, results_dir); data-dependent panels load from
results_dir so the 10-mammal RiNALMo run plugs in without code changes. Where
data isn't wired yet, a panel draws a labelled TODO stub referencing the source
(old notebook / raw_plots) so the layout is reviewable now.

Usage:
    python analysis/make_figures.py --outdir paper/figures
    python analysis/make_figures.py --outdir paper/figures --only fig1_embeddings
    python analysis/make_figures.py --outdir paper/figures --results-dir preprint_results
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

import figstyle as fs

FIGURES: dict[str, Callable] = {}

# Hand-made assets this script intentionally does NOT generate.
MANUAL = {
    "fig2_pipeline":  "schematic (Affinity/draw.io) --- pipeline concept diagram",
    "fig3_islands":   "panel A is a schematic; see fig3b_dotplot for the scripted panel B",
    "fig5_cases":     "UCSC genome-browser screenshots, hand-arranged",
}


def figure(name: str):
    def deco(fn: Callable) -> Callable:
        FIGURES[name] = fn
        return fn
    return deco


def _todo(ax, text: str) -> None:
    """Draw a placeholder inside an axes so the composition is reviewable."""
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=7, color="#888888", wrap=True)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linestyle((0, (3, 3))); s.set_color("#cccccc")


# =====================================================================
# Figure 1 --- Embedding representations (fully scripted, 4 panels)
# Source of panel code: preprint__deprecated/create_figure_1.ipynb
# =====================================================================
@figure("fig1_embeddings")
def fig1(outdir: Path, results_dir: Optional[Path]) -> None:
    fs.set_style()
    fig, axd = fs.mosaic("""
        AB
        CD
    """, width=fs.FULL_WIDTH, height=5.4)

    # (A) per-token RiNALMo embeddings (PCA): tRNA copies + miRNA
    # TODO: swap RNA-FM->RiNALMo in create_figure_1.ipynb get_token_embeddings();
    # PCA is now 1280->16 (modules/global_PCA/rinalmo_pca_k16.npz).
    _todo(axd["A"], "A: per-token PCA scatter\n(tRNA x2 vs miRNA)\nRiNALMo — same code, swap model")
    axd["A"].set_title("Per-nucleotide embeddings", loc="left")

    # (B) mean-pooled signal vs background
    _todo(axd["B"], "B: mean-pooled PCA\nncRNA (signal) vs intergenic/shuffled\nsame code, swap model")
    axd["B"].set_title("Signal vs background", loc="left")

    # (C) reframed: short-ncRNA MMD sliding-window search profile
    _todo(axd["C"], "C: short-ncRNA MMD search\n(MMD vs window offset; min at best hit)")
    axd["C"].set_title("Local MMD search", loc="left")

    # (D) OPTIONAL (best-hit vs distant control). Drop if not shown.
    _todo(axd["D"], "D (optional): best-hit vs distant control\n— drop if not used")
    axd["D"].set_title("Best hit vs control", loc="left")

    for k in "ABCD":
        fs.panel_label(axd[k], k)
    fs.save(fig, outdir / "fig1_embeddings.pdf")
    print(f"wrote {outdir/'fig1_embeddings.pdf'}")


# =====================================================================
# Figure 3B --- island matching (scripted panel; A is a schematic)
# Redraw for RiNALMo: per-token COSINE dotplot + nt-Smith-Waterman path,
# NOT the old window-MMD matrix. Source: modules/pipeline/matchers/rinalmo.py
# =====================================================================
@figure("fig3b_dotplot")
def fig3b(outdir: Path, results_dir: Optional[Path]) -> None:
    fs.set_style()
    fig, axd = fs.mosaic("A", width=fs.HALF_WIDTH * 1.3, height=3.0)
    _todo(axd["A"], "Cosine dotplot (ref x query per-token)\n+ nt-Smith-Waterman traceback\n(replaces RNA-FM window-MMD matrix)")
    axd["A"].set_title("Island cosine dotplot + SW", loc="left")
    fs.panel_label(axd["A"], "B")  # this is panel B of Figure 3
    fs.save(fig, outdir / "fig3b_dotplot.pdf")
    print(f"wrote {outdir/'fig3b_dotplot.pdf'}")


# =====================================================================
# Figure 4 --- MMD behaviour across short ncRNA loci (3 panels)
# Source: figure_mmd_validation_3panel.pdf / MMD_vs_seqID_short_ncRNA.pdf
# Short-ncRNA metric is still MMD; swap model, same claims, new numbers.
# =====================================================================
@figure("fig4_mmd")
def fig4(outdir: Path, results_dir: Optional[Path]) -> None:
    fs.set_style()
    fig, axd = fs.mosaic("ABC", width=fs.FULL_WIDTH, height=2.4)
    _todo(axd["A"], "A: seq identity vs MMD\n(anticorrelated; spread at 45-60%)")
    _todo(axd["B"], "B: annotation agreement vs MMD")
    _todo(axd["C"], "C: MMD by biotype (violin/box)")
    for k in "ABC":
        fs.panel_label(axd[k], k, dx=-0.18)
    fs.save(fig, outdir / "fig4_mmd.pdf")
    print(f"wrote {outdir/'fig4_mmd.pdf'}")


# =====================================================================
# Figure 6 --- lncRNA cores (3 panels); redo with results tomorrow.
# color scale becomes cosine-SW distance d=1/(1+score), NOT MMD.
# Source: island_phylo_conservation.ipynb, core_reproducibility_cumulative.pdf,
# MMD_per_species_phylo.pdf
# =====================================================================
@figure("fig6_cores")
def fig6(outdir: Path, results_dir: Optional[Path]) -> None:
    fs.set_style()
    fig, axd = fs.mosaic("""
        AA
        BC
    """, width=fs.FULL_WIDTH, height=5.0)
    _todo(axd["A"], "A: per-gene core x species heatmaps (NORAD, PVT1, FIRRE)\ncolor = cosine-SW distance")
    _todo(axd["B"], "B: core reproducibility (fraction in >=k species)")
    _todo(axd["C"], "C: median island distance vs phylo distance")
    for k, lab in [("A", "A"), ("B", "B"), ("C", "C")]:
        fs.panel_label(axd[k], lab)
    fs.save(fig, outdir / "fig6_cores.pdf")
    print(f"wrote {outdir/'fig6_cores.pdf'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", type=Path, default=Path("paper/figures"))
    p.add_argument("--results-dir", type=Path, default=None,
                   help="directory with pipeline outputs to plot (e.g. preprint_results)")
    p.add_argument("--only", nargs="*", help="only build these figure names")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    names = args.only or list(FIGURES)
    for name in names:
        if name in MANUAL:
            print(f"skip {name}: {MANUAL[name]} (hand-made asset)")
            continue
        if name not in FIGURES:
            raise SystemExit(f"unknown figure: {name} "
                             f"(scripted: {', '.join(FIGURES)}; manual: {', '.join(MANUAL)})")
        FIGURES[name](args.outdir, args.results_dir)


if __name__ == "__main__":
    main()
