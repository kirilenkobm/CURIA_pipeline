"""Shared publication style for CURIA figures.

Import this in every figure builder so all panels share fonts, sizes, colours,
and export settings. The goal is that matplotlib composes each multi-panel figure
publication-ready (panel letters, correct text width, editable embedded fonts) so
that hand-composition in Affinity is only needed for true schematics and
screenshots.

Usage:
    from figstyle import set_style, panel_label, PALETTE, FULL_WIDTH, mosaic
    set_style()
    fig, axd = mosaic('''
        AB
        CD
    ''', width=FULL_WIDTH, height=5.2)
    panel_label(axd['A'], 'A')
    ...
    fig.savefig('figures/fig1_embeddings.pdf')
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

# --- paper geometry -------------------------------------------------------
# article, a4paper, margin=1in  ->  text width ~= 6.27 in.
FULL_WIDTH = 6.3   # inches; use for \includegraphics[width=\linewidth]
HALF_WIDTH = 3.05  # inches; two-up panels

# --- palette (kept small and consistent across figures) -------------------
PALETTE = {
    "trna":       "#2b6cb0",  # blue
    "mirna":      "#dd6b20",  # orange
    "signal":     "#6b46c1",  # purple  (annotated ncRNA)
    "background": "#a0aec0",  # gray    (intergenic / shuffled)
    "accent":     "#1a3a5c",  # dark navy (lines, ROC)
    "muted":      "#cbd5e0",  # light gray (diagonals, grids)
}
# ordered list for categorical (e.g. biotypes)
CYCLE = ["#2b6cb0", "#dd6b20", "#6b46c1", "#38a169", "#d53f8c", "#718096"]


def set_style() -> None:
    """Apply global rcParams. Call once at the top of each figure build."""
    matplotlib.use("Agg")  # headless
    plt.rcParams.update({
        # fonts: sans, embedded as TrueType (editable text in the PDF)
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # sizes tuned for ~6.3in-wide figures printed at column scale
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        # lines / spines
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=CYCLE),
        # export
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def mosaic(layout: str, width: float = FULL_WIDTH, height: float = 4.0, **kw):
    """Thin wrapper over plt.subplot_mosaic with our default figure size."""
    fig, axd = plt.subplot_mosaic(layout, figsize=(width, height), **kw)
    return fig, axd


def panel_label(ax, letter: str, dx: float = -0.11, dy: float = 1.06) -> None:
    """Bold panel letter at the top-left of an axes, in axes fraction coords."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")


def save(fig, path, also_png: bool = True) -> None:
    """Save a figure as PDF (vector, for the paper) and optionally a PNG preview."""
    fig.savefig(path)
    if also_png and str(path).endswith(".pdf"):
        fig.savefig(str(path)[:-4] + ".png")
    plt.close(fig)
