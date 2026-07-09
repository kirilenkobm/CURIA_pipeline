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
    python analysis/make_figures.py --outdir paper/figures --results-dir preprint_results__deprecated
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import figstyle as fs

FIGURES: dict[str, Callable] = {}

# Hand-made assets this script intentionally does NOT generate.
MANUAL = {
    "fig5_cases":     "UCSC genome-browser screenshots, hand-arranged",
}

# schematic palette (shared by Fig 2 and Fig 3A)
_GREEN, _BLUE, _ISLAND, _GRAY, _SHADE = "#62c45a", "#3b5ba5", "#2e8b57", "#b6b6b6", "#eeeeee"


def _exon_track(ax, y, blocks, color, h=4.0, lw=1.3):
    """Draw an exon-block track: thin connecting line + filled boxes."""
    from matplotlib.patches import Rectangle
    xs = [b[0] for b in blocks]; xe = [b[0] + b[1] for b in blocks]
    ax.plot([min(xs), max(xe)], [y + h / 2, y + h / 2], color=color, lw=lw, zorder=2)
    for x, w in blocks:
        ax.add_patch(Rectangle((x, y), w, h, color=color, zorder=3))


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
_FIG1_CACHE = Path(__file__).resolve().parent / "data" / "fig1_embeddings.npz"


def _short_label(lbl: str) -> str:
    """'tRNA-Asn (copy 1)' -> 'tRNA-Asn 1'; 'miRNA (MIR103A1)' -> 'miRNA'."""
    import re
    m = re.search(r"copy (\d)", lbl)
    base = lbl.split(" (")[0]
    return f"{base} {m.group(1)}" if m else base


def _scatter_pca(ax, coords, group, meta, noise_first=True, legend="dots"):
    """Scatter PC1 vs PC2, one colour per group; draw non-signal groups behind.

    legend: "dots" (default matplotlib legend), "squares" (small square swatches,
    top-left, so the key is not confused with data points), or False."""
    import matplotlib.patches as mpatches
    order = sorted(meta, key=lambda g: (meta[g].get("signal", True) if noise_first else True))
    for g in order:
        m = coords[group == int(g)]
        if len(m) == 0:
            continue
        is_sig = meta[g].get("signal", True)
        ax.scatter(m[:, 0], m[:, 1], c=meta[g]["color"], label=meta[g]["label"],
                   s=14 if is_sig else 10, alpha=0.75 if is_sig else 0.4,
                   linewidths=0, zorder=3 if is_sig else 1)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    if legend == "squares":
        handles = [mpatches.Patch(color=meta[g]["color"], label=_short_label(meta[g]["label"]))
                   for g in order if (group == int(g)).any()]
        ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=6,
                  handlelength=0.9, handleheight=0.9, handletextpad=0.4,
                  labelspacing=0.3, borderaxespad=0.2)
    elif legend:
        ax.legend(frameon=False, loc="best", handletextpad=0.2, borderaxespad=0.2)


def _signal_separation(ax, coords, group, meta):
    """Show that the embeddings separate ncRNA from background.

    The separation lives in the full feature space, not PC1/PC2 (which capture
    only ~48% of variance). We quantify it honestly with a 5-fold cross-validated
    logistic classifier on the embeddings and plot its out-of-fold P(signal) for
    ncRNA vs background."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    noise_code = max(int(k) for k in meta)
    y = (group != noise_code).astype(int)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    p = cross_val_predict(clf, coords, y, cv=5, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    bins = np.linspace(0, 1, 26)
    ax.hist(p[y == 0], bins=bins, color=fs.PALETTE["background"], alpha=0.75,
            label="Background", density=True)
    ax.hist(p[y == 1], bins=bins, color=fs.PALETTE["signal"], alpha=0.6,
            label="Annotated ncRNA", density=True)
    ax.set_xlabel("Cross-validated P(ncRNA)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper center", handletextpad=0.4)
    # AUC in the central valley (between the two peaks), clear of the bars/legend
    ax.text(0.5, 0.62, f"AUC = {auc:.2f}", transform=ax.transAxes,
            va="top", ha="center", fontsize=8, fontweight="bold")


def _search_profile(ax, offsets, curves):
    """Aggregate MMD vs window offset. Each locus is min-max normalized to its own
    scale (RiNALMo's absolute MMD varies by locus), so the shared U-shape --- the
    dip at the true position (offset 0) --- is visible instead of buried in spread."""
    lo = curves.min(axis=1, keepdims=True)
    rng = curves.max(axis=1, keepdims=True) - lo
    norm = (curves - lo) / np.where(rng > 0, rng, 1.0)
    med = np.median(norm, axis=0)
    q25, q75 = np.percentile(norm, [25, 75], axis=0)
    ax.fill_between(offsets, q25, q75, color=fs.PALETTE["accent"], alpha=0.20, linewidth=0)
    ax.plot(offsets, med, color=fs.PALETTE["accent"], zorder=3)
    ax.axvline(0, color=fs.PALETTE["muted"], lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("Window offset from locus (nt)")
    ax.set_ylabel("MMD (per-locus normalized)")
    ax.text(0.5, 0.92, f"n = {len(curves)}", transform=ax.transAxes, ha="center", fontsize=7)


def _best_vs_control(fig, subplotspec, best, ctrl, label="D"):
    """Jointplot: best-hit MMD vs distant-control MMD, with marginal histograms.

    The marginals make the point visible that many loci have best-hit MMD ~ 0
    (spike at 0) while the distant control almost never is --- otherwise those
    points pile up invisibly on the y-axis."""
    hi = float(max(best.max(), ctrl.max())) * 1.05
    gs = subplotspec.subgridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                                 wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    axtop = fig.add_subplot(gs[0, 0], sharex=ax)
    axright = fig.add_subplot(gs[1, 1], sharey=ax)
    bins = np.linspace(0, hi, 26)

    ax.plot([0, hi], [0, hi], color=fs.PALETTE["muted"], lw=0.8, ls="--", zorder=1)
    ax.scatter(best, ctrl, s=12, alpha=0.6, linewidths=0,
               color=fs.PALETTE["accent"], zorder=3)
    frac = float((ctrl > best).mean()) * 100
    n0 = int((best < 0.005).sum())
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("Best-hit MMD"); ax.set_ylabel("Distant-control MMD")
    ax.text(0.95, 0.06, f"{frac:.0f}% above diagonal", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7)

    axtop.hist(best, bins=bins, color=fs.PALETTE["accent"], alpha=0.8)
    axtop.axvline(0, color=fs.PALETTE["muted"], lw=0.8)
    axtop.text(0.5, 0.7, f"{n0} loci best-hit $\\approx$ 0", transform=axtop.transAxes,
               ha="center", fontsize=6.5)
    axright.hist(ctrl, bins=bins, orientation="horizontal",
                 color=fs.PALETTE["accent"], alpha=0.8)
    for a in (axtop, axright):
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    axtop.set_title("Best hit vs distant control", loc="left")
    fs.panel_label(axtop, label, dx=-0.18, dy=1.15)
    return ax


@figure("fig1_embeddings")
def fig1(outdir: Path, results_dir: Optional[Path]) -> None:
    fs.set_style()
    fig, axd = fs.mosaic("""
        AB
        CD
    """, width=fs.FULL_WIDTH, height=5.4)

    have_cache = _FIG1_CACHE.exists()
    if have_cache:
        d = np.load(_FIG1_CACHE)
        meta = json.loads((_FIG1_CACHE.with_suffix(".json")).read_text())
        # (A) per-token embeddings (matching PCA 1280->16): tRNA copies + miRNA
        #     legend off --- markers overlapped the data; colours are in the caption
        _scatter_pca(axd["A"], d["A_coords"], d["A_group"], meta["A"],
                     noise_first=False, legend="squares")
        # (B) cross-validated classifier separation on the embeddings
        _signal_separation(axd["B"], d["B_coords"], d["B_group"], meta["B"])
        # (C) aggregate MMD sliding-window search across short ncRNAs
        _search_profile(axd["C"], d["C_offsets"], d["C_curves"])
        # (D) best-hit vs distant control --- jointplot (replaces its own cell)
        ss = axd["D"].get_subplotspec(); axd["D"].remove()
        _best_vs_control(fig, ss, d["D_best"], d["D_ctrl"], label="D")
    else:
        for k, msg in [("A", "per-token PCA: tRNA x2 vs miRNA"),
                       ("B", "classifier separation ncRNA vs background"),
                       ("C", "MMD search vs offset"),
                       ("D", "best-hit vs distant control")]:
            _todo(axd[k], f"run analysis/compute_fig1_embeddings.py\n({msg})")
    axd["A"].set_title("Per-nucleotide embeddings", loc="left")
    axd["B"].set_title("Signal vs background", loc="left")
    axd["C"].set_title("Local MMD search", loc="left")
    fs.panel_label(axd["A"], "A"); fs.panel_label(axd["B"], "B"); fs.panel_label(axd["C"], "C")
    if not have_cache:
        axd["D"].set_title("Best hit vs distant control", loc="left")
        fs.panel_label(axd["D"], "D")
    fs.save(fig, outdir / "fig1_embeddings.pdf")
    print(f"wrote {outdir/'fig1_embeddings.pdf'}")


# =====================================================================
# Figure 2 --- pipeline overview (fully scripted schematic, Fig 3A style)
# =====================================================================
def _fig2a(ax):
    """Panel A: alignment chains restrict the search space."""
    ax.set_xlim(0, 100); ax.set_ylim(40, 96); ax.set_axis_off()
    # reference ncRNAs
    _exon_track(ax, 86, [(20, 12), (38, 7), (49, 11)], _GREEN)
    _exon_track(ax, 86, [(70, 17)], _GREEN)
    ax.text(19, 88, "lncRNA-1", color=_GREEN, fontsize=8, fontweight="bold", ha="right", va="center")
    ax.text(69, 88, "ncRNA-2", color=_GREEN, fontsize=8, fontweight="bold", ha="right", va="center")
    ax.text(55, 78, "alignment chains", fontsize=8, ha="center")
    # chains: chain-1 (syntenic, blue) covers both loci; chains 2-4 partial (gray)
    _exon_track(ax, 68, [(16, 6), (24, 5), (32, 6), (40, 6), (49, 6), (57, 6),
                         (66, 6), (74, 5), (81, 5), (88, 6)], _BLUE, lw=1.1)
    _exon_track(ax, 60, [(19, 5), (41, 6), (60, 4), (67, 4)], _GRAY, lw=1.0)
    _exon_track(ax, 52, [(20, 4), (27, 4), (39, 5), (46, 4)], _GRAY, lw=1.0)
    _exon_track(ax, 52, [(73, 4), (81, 5)], _GRAY, lw=1.0)
    ax.text(14, 70, "chain-1", color=_BLUE, fontsize=8, fontweight="bold", ha="right", va="center")
    for y, lab in [(62, "chain-2"), (54, "chain-3")]:
        ax.text(14, y, lab, color=_GRAY, fontsize=8, ha="right", va="center")
    ax.text(70, 54, "chain-4", color=_GRAY, fontsize=8, ha="right", va="center")


def _fig2b(ax):
    """Panel B: long ncRNA --- islands detected and matched across species."""
    from matplotlib.patches import Rectangle, ConnectionPatch
    ax.set_xlim(0, 100); ax.set_ylim(-2, 100); ax.set_axis_off()
    ax.text(55, 96, "long ncRNA pipeline", fontsize=8, fontweight="bold", ha="center")
    _exon_track(ax, 80, [(25, 15), (46, 10), (60, 14)], _GREEN)
    ax.text(24, 82, "lncRNA-1", color=_GREEN, fontsize=7.5, fontweight="bold", ha="right", va="center")
    ref = [27, 58, 72]; qry = [27, 42, 58, 74]
    for x in ref:
        ax.add_patch(Rectangle((x, 60), 5, 5, color=_ISLAND))
    for x in qry:
        ax.add_patch(Rectangle((x, 18), 5, 5, color=_ISLAND))
    _exon_track(ax, 6, [(24, 7), (34, 6), (44, 8), (56, 9), (70, 7)], _BLUE, lw=1.1)
    ax.text(20, 62.5, "reference islands", fontsize=7.5, fontweight="bold", ha="right", va="center")
    ax.text(20, 20.5, "query islands", fontsize=7.5, fontweight="bold", ha="right", va="center")
    ax.text(20, 8.5, "chain-1", color=_BLUE, fontsize=7.5, fontweight="bold", ha="right", va="center")
    # matches: two good (low distance), one rejected (high, dashed gray)
    matches = [(27, 27, "0.02", "0.4", False), (72, 74, "0.03", "0.4", False),
               (58, 58, "0.5", "0.75", True)]
    for rx, qx, lab, col, rej in matches:
        ax.add_artist(ConnectionPatch(
            xyA=(rx + 2.5, 60), coordsA=ax.transData,
            xyB=(qx + 2.5, 23), coordsB=ax.transData,
            arrowstyle="-", color=col, lw=1.3,
            linestyle="--" if rej else "-", connectionstyle="arc3,rad=0.12"))
        ax.text((rx + qx) / 2 + (7 if not rej else -6), 41, lab, fontsize=7,
                color=col, ha="center", va="center")


def _fig2c(ax):
    """Panel C: short ncRNA --- sliding-window MMD search in the projected region."""
    from matplotlib.patches import Rectangle, FancyArrow
    ax.set_xlim(0, 100); ax.set_ylim(-2, 100); ax.set_axis_off()
    ax.text(52, 96, "short ncRNA pipeline", fontsize=8, fontweight="bold", ha="center")
    _exon_track(ax, 80, [(28, 50)], _GREEN)
    ax.text(26, 82, "ncRNA-2", color=_GREEN, fontsize=7.5, fontweight="bold", ha="right", va="center")
    _exon_track(ax, 70, [(30, 10), (44, 12), (60, 12)], _BLUE, lw=1.1)
    ax.text(26, 72, "chain-1", color=_BLUE, fontsize=7.5, fontweight="bold", ha="right", va="center")
    ax.text(52, 58, "projected region", fontsize=8, ha="center")
    ax.text(50, 44, "AUGUGACAACAGGUAGACAAUCUAUCGG...", fontsize=6.5, family="monospace",
            ha="center", va="center")
    # tiled sliding windows; the best one highlighted
    for i, (x, a) in enumerate([(18, 0.25), (30, 0.25), (42, 0.7), (54, 0.25)]):
        ax.add_patch(Rectangle((x, 38), 20, 12, color=_GREEN, alpha=a, linewidth=0, zorder=1))
    ax.add_patch(FancyArrow(52, 34, 0, -10, width=0.4, head_width=3, head_length=3,
                            color=_GREEN, length_includes_head=True))
    ax.text(52, 16, "best match:\nlow MMD", fontsize=7.5, ha="center", va="center")


@figure("fig2_pipeline")
def fig2(outdir: Path, results_dir: Optional[Path]) -> None:
    fs.set_style()
    fig, axd = fs.mosaic("""
        AAA
        BBC
    """, width=fs.FULL_WIDTH, height=4.6, gridspec_kw={"height_ratios": [1.0, 1.5]})
    _fig2a(axd["A"]); _fig2b(axd["B"]); _fig2c(axd["C"])
    fs.panel_label(axd["A"], "A", dx=-0.03, dy=1.04)
    fs.panel_label(axd["B"], "B", dx=-0.04, dy=1.04)
    fs.panel_label(axd["C"], "C", dx=-0.08, dy=1.04)
    fs.save(fig, outdir / "fig2_pipeline.pdf")
    print(f"wrote {outdir/'fig2_pipeline.pdf'}")


# =====================================================================
# Figure 3B --- island matching (scripted). Panel A stays a schematic.
# Real MALAT1 core: cosine dotplot + SW band (match) vs an unrelated island.
# Data: analysis/compute_fig3b_dotplot.py (rinalmo_version_outputs/hg38_vs_mm39)
# =====================================================================
_FIG3B_CACHE = Path(__file__).resolve().parent / "data" / "fig3b_dotplot.npz"
_SW_TAU, _SW_GAP = 0.5, 0.3   # deployed island_align params


def _sw_accum(cos, tau=_SW_TAU, gap=_SW_GAP):
    """Smith-Waterman local-alignment accumulation matrix H on S = cos - tau.

    Only coherent diagonals accumulate, so H cuts through the anisotropic cosine
    'carpet' (high baseline) that buries the diagonal in the raw dotplot."""
    S = cos - tau
    la, lb = S.shape
    H = np.zeros((la + 1, lb + 1))
    for i in range(1, la + 1):
        pr, row, Si = H[i - 1], H[i], S[i - 1]
        for j in range(1, lb + 1):
            v = pr[j - 1] + Si[j - 1]
            u = pr[j] - gap
            l = row[j - 1] - gap
            if u > v: v = u
            if l > v: v = l
            row[j] = v if v > 0 else 0.0
    return H[1:, 1:]


def _sw_panel(ax, H, title, vmax, score):
    from matplotlib.colors import PowerNorm
    im = ax.imshow(H, origin="lower", aspect="auto", cmap="magma",
                   norm=PowerNorm(gamma=0.6, vmin=0, vmax=vmax),
                   interpolation="nearest",
                   extent=[0, H.shape[1], 0, H.shape[0]])
    ax.set_xlabel("Query island position (nt)")
    ax.set_ylabel("Reference island position (nt)")
    ax.set_title(title, loc="left", fontsize=8)
    ax.text(0.04, 0.96, f"SW = {score:.0f}", transform=ax.transAxes, va="top",
            ha="left", fontsize=7.5, color="white", fontweight="bold")
    return im


def _fig3a_schematic(ax):
    """Panel A schematic: RNA-like score along a reference transcript -> islands
    (thresholded) -> chain-guided projection -> query islands. Islands are derived
    from where the score exceeds threshold, so they align by construction."""
    from matplotlib.patches import Rectangle, ConnectionPatch
    green, blue, orange, shade = "#62c45a", "#9bb6d8", "#ff5a1f", "#eeeeee"
    ax.set_xlim(16, 100); ax.set_ylim(-16, 98); ax.set_axis_off()
    LBL = 25.5

    # reference exons
    for x, w in [(30, 24), (58, 10), (74, 22)]:
        ax.add_patch(Rectangle((x, 84), w, 5, color=green))
    ax.text(LBL, 86.5, "reference exons", color=green, fontsize=8,
            fontweight="bold", ha="right", va="center")
    ax.text(63, 94, r"reference genomic position $\rightarrow$", fontsize=8, ha="center")

    # RNA-like score: gaussians whose peaks define the islands
    thr, base = 56.0, 46.0
    peaks = [(35, 21, 2.6), (50, 18, 2.3), (83, 22, 3.1)]
    xg = np.linspace(28, 96, 600)
    yg = base + sum(h * np.exp(-0.5 * ((xg - c) / w) ** 2) for c, h, w in peaks)
    # a little wiggle so it reads as a real trace
    yg = yg + 1.1 * np.sin(xg * 1.3)
    ax.plot([28, 96], [thr, thr], color="0.7", lw=0.9, zorder=1)
    ax.text(96.5, thr, "threshold", fontsize=8, ha="left", va="center")
    ax.text(29.5, 68, "0.5", fontsize=7, ha="right", va="center", color="0.5")
    ax.plot(xg, yg, color=orange, lw=1.4, zorder=3)
    ax.text(LBL, 58, "RNA-like score\n(embeddings-based)", fontsize=8,
            fontweight="bold", ha="right", va="center")

    # supra-threshold spans -> islands (aligned with the peaks by construction)
    above = yg > thr
    spans, i = [], 0
    while i < len(xg):
        if above[i]:
            j = i
            while j < len(xg) and above[j]:
                j += 1
            spans.append((xg[i], xg[j - 1]))
            i = j
        else:
            i += 1
    for x0, x1 in spans:
        ax.add_patch(Rectangle((x0, 42), x1 - x0, 30, color=shade, zorder=0))
        ax.add_patch(Rectangle((x0, 28), x1 - x0, 5, color=blue))
    ax.text(LBL, 30.5, "structured islands", fontsize=8, fontweight="bold",
            ha="right", va="center")

    # candidate query regions (broad, group nearby islands) + projection arrows
    cand = [(29, 24), (74, 15)]
    for x, w in cand:
        ax.add_patch(Rectangle((x, 8), w, 5, color=shade))
    ax.text(LBL, 10.5, "candidate regions", fontsize=8, fontweight="bold",
            ha="right", va="center")
    for x0, x1 in spans:
        xc = (x0 + x1) / 2
        ax.annotate("", xy=(xc, 13.2), xytext=(xc, 27.8),
                    arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.4))
    ax.text(63, 20, "chain-guided projection\nto query", fontsize=7.5,
            ha="center", va="center")

    # query islands (within candidate regions) + faint projection arcs
    q = [(31, 3.6), (39.5, 3.4), (47.5, 2.6), (79.5, 4.2)]
    for x, w in q:
        ax.add_patch(Rectangle((x, -9), w, 5, color=blue))
    ax.text(LBL, -6.5, "query islands", fontsize=8, fontweight="bold",
            ha="right", va="center")
    for (rx0, rx1), (qx, qw) in [(spans[0], q[0]), (spans[-1], q[-1])]:
        ax.add_artist(ConnectionPatch(
            xyA=((rx0 + rx1) / 2, 28), coordsA=ax.transData,
            xyB=(qx + qw / 2, -4), coordsB=ax.transData,
            arrowstyle="-", color="0.82", lw=1.1,
            connectionstyle="arc3,rad=-0.12"))


@figure("fig3_islands")
def fig3(outdir: Path, results_dir: Optional[Path]) -> None:
    fs.set_style()
    fig, axd = fs.mosaic("""
        AA
        MC
    """, width=fs.FULL_WIDTH, height=4.4, gridspec_kw={"height_ratios": [0.8, 1.3]})
    _fig3a_schematic(axd["A"])
    fs.panel_label(axd["A"], "A", dx=-0.08, dy=1.02)

    if _FIG3B_CACHE.exists():
        d = np.load(_FIG3B_CACHE)
        meta = json.loads(_FIG3B_CACHE.with_suffix(".json").read_text())
        Hm, Hc = _sw_accum(d["match_cos"]), _sw_accum(d["ctrl_cos"])
        vmax = float(Hm.max())
        im = _sw_panel(axd["M"], Hm, f"{meta['gene']} core: human × mouse",
                       vmax, meta["match_score"])
        _sw_panel(axd["C"], Hc, "vs unrelated island", vmax, meta["ctrl_score"])
        fig.colorbar(im, ax=[axd["M"], axd["C"]],
                     label="Smith–Waterman accumulated score", shrink=0.8, pad=0.02)
    else:
        _todo(axd["M"], "run analysis/compute_fig3b_dotplot.py")
        _todo(axd["C"], "(control dotplot)")
    fs.panel_label(axd["M"], "B", dx=-0.22)
    fs.save(fig, outdir / "fig3_islands.pdf")
    print(f"wrote {outdir/'fig3_islands.pdf'}")


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
                   help="directory with pipeline outputs to plot (e.g. preprint_results__deprecated)")
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
