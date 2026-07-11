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
    "fig2_pipeline":  "pipeline overview schematic (hand-made PNG)",
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
# Figure 3B --- island matching (scripted). Panel A stays a schematic.
# Real MALAT1 core: cosine dotplot + SW band (match) vs an unrelated island.
# Data: analysis/compute_fig3b_dotplot.py (preprint_results/hg38_vs_mm39)
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
# Short-ncRNA metric is still MMD (short_ncrna._compute_mmd_with_ref); swap
# model, same claims, new numbers. Data: analysis/compute_fig4_data.py
# (preprint_results/hg38_vs_mm39) -> analysis/data/fig4_mmd.npz.
# =====================================================================
_FIG4_CACHE = Path(__file__).resolve().parent / "data" / "fig4_mmd.npz"

# Short biotype labels + display order (rare biotypes fold into "other").
_BIOTYPE_LABEL = {
    "snoRNA": "snoRNA", "tRNA": "tRNA", "miRNA": "miRNA", "lncRNA": "lncRNA",
    "snRNA": "snRNA", "misc_RNA": "misc_RNA", "rRNA": "rRNA",
    "scaRNA": "scaRNA", "vault_RNA": "vaultRNA",
}


def _fig4_seqid(ax, seqid, mmd):
    """Panel A: sequence identity (%) vs MMD scatter."""
    ax.scatter(seqid, mmd, s=7, alpha=0.28, linewidths=0, color="#7a9cc0")
    ax.set_xlabel("Sequence identity (%)")
    ax.set_ylabel("MMD")
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)


def _fig4_overlap(ax, mmd, any_ov, over50):
    """Panel B: MMD distributions for loci that do vs do not overlap annotation.

    The paper claim (Results 3.3) is that annotation-overlapping loci are enriched
    at low MMD while non-overlapping ones shift higher --- shown directly as two
    overlaid, density-normalized histograms."""
    hi = float(min(mmd.max(), 0.6))
    bins = np.linspace(0, hi, 31)
    ov, no = mmd[any_ov], mmd[~any_ov]
    ax.hist(ov, bins=bins, density=True, color=fs.PALETTE["accent"], alpha=0.65,
            label="overlaps annotation")
    ax.hist(no, bins=bins, density=True, color=fs.PALETTE["mirna"], alpha=0.55,
            label="no overlap")
    ax.set_xlabel("MMD")
    ax.set_ylabel("Density")
    ax.set_xlim(0, hi)
    ax.legend(frameon=False, loc="upper right", handletextpad=0.4, fontsize=6.5)


def _fig4_biotype(ax, mmd, biotype):
    """Panel C: MMD distribution by biotype, boxes ordered by median MMD."""
    lab = np.array([_BIOTYPE_LABEL.get(b, "other") for b in biotype])
    groups = {}
    for l, v in zip(lab, mmd):
        groups.setdefault(l, []).append(v)
    # keep reasonably-sized groups; order by median (low = well conserved)
    items = [(k, np.asarray(v)) for k, v in groups.items() if len(v) >= 8]
    items.sort(key=lambda kv: np.median(kv[1]))
    names = [k for k, _ in items]
    data = [v for _, v in items]
    pos = np.arange(1, len(data) + 1)
    bp = ax.boxplot(data, positions=pos, vert=False, widths=0.6, showfliers=False,
                    patch_artist=True, medianprops=dict(color="black", lw=1.2),
                    zorder=3)
    for patch in bp["boxes"]:
        patch.set(facecolor=fs.PALETTE["signal"], alpha=0.55, linewidth=0.6, zorder=3)
    for whisk in bp["whiskers"] + bp["caps"]:
        whisk.set(color=fs.PALETTE["muted"], linewidth=0.8)
    # A median marker per biotype keeps degenerate distributions (e.g. tRNA,
    # median ~ 0, box collapsed at the axis) visibly represented.
    med = [float(np.median(d)) for d in data]
    ax.scatter(med, pos, s=16, color=fs.PALETTE["accent"], zorder=5,
               edgecolor="white", linewidths=0.5)
    ax.set_yticks(pos)
    ax.set_yticklabels([f"{n} ({len(d)})" for n, d in zip(names, data)], fontsize=6.5)
    ax.set_ylim(0.4, len(data) + 0.6)
    ax.set_xlabel("MMD")
    ax.set_xlim(left=-0.01)


@figure("fig4_mmd")
def fig4(outdir: Path, results_dir: Optional[Path]) -> None:
    fs.set_style()
    fig, axd = fs.mosaic("ABC", width=fs.FULL_WIDTH, height=2.7)
    if _FIG4_CACHE.exists():
        d = np.load(_FIG4_CACHE, allow_pickle=False)
        _fig4_seqid(axd["A"], d["A_seqid"], d["A_mmd"])
        _fig4_overlap(axd["B"], d["B_mmd"], d["B_any"], d["B_over50"])
        _fig4_biotype(axd["C"], d["C_mmd"], d["C_biotype"])
        axd["A"].set_title("Identity vs MMD", loc="left")
        axd["B"].set_title("Annotation agreement", loc="left")
        axd["C"].set_title("MMD by biotype", loc="left")
    else:
        _todo(axd["A"], "run analysis/compute_fig4_data.py\n(seq identity vs MMD)")
        _todo(axd["B"], "annotation agreement vs MMD")
        _todo(axd["C"], "MMD by biotype (box)")
    for k in "ABC":
        fs.panel_label(axd[k], k, dx=-0.18, dy=1.20)
    fs.save(fig, outdir / "fig4_mmd.pdf")
    print(f"wrote {outdir/'fig4_mmd.pdf'}")


# =====================================================================
# Figure 6 --- lncRNA cores (3 panels), multi-species.
# The island matcher reports cosine-SW distance d = 1/(1+score) (NOT MMD;
# see modules/pipeline/matchers/rinalmo.py), carried in the legacy-named
# `diag_mmd` column of island_alignment_results.tsv.
# Auto-detects available hg38_vs_* pairs so re-running once rheMac10/rn7/
# susScr11 (and dasNov3) land extends the figure to N=10 with no edit.
# Source: preprint__deprecated/island_phylo_conservation.ipynb.
# =====================================================================
# name / clade / divergence-from-human (Mya). Verbatim from the deprecated
# island_phylo_conservation notebook (cell 3).
# div_mya are clade-level approximations of divergence-from-human (My): great apes
# ~9, OWM ~25, Glires ~90, Laurasiatheria ~94, Atlantogenata (Xenarthra+Afrotheria)
# ~105, Marsupialia ~180. HL* = Hiller-lab/Zoonomia assemblies (HL<genus><species>).
_SPECIES_META = {
    # Primates
    "gorGor6":  {"name": "Gorilla",          "clade": "Euarchontoglires", "div_mya": 9},
    "rheMac10": {"name": "Rhesus",           "clade": "Euarchontoglires", "div_mya": 25},
    # Glires (rodents + rabbit)
    "mm39":     {"name": "Mouse",            "clade": "Euarchontoglires", "div_mya": 90},
    "rn7":      {"name": "Rat",              "clade": "Euarchontoglires", "div_mya": 90},
    "cavPor3":  {"name": "Guinea pig",       "clade": "Euarchontoglires", "div_mya": 90},
    "HLoryCun3":{"name": "Rabbit",           "clade": "Euarchontoglires", "div_mya": 90},  # Oryctolagus cuniculus
    # Laurasiatheria
    "bosTau9":  {"name": "Cow",              "clade": "Laurasiatheria",   "div_mya": 94},
    "susScr11": {"name": "Pig",              "clade": "Laurasiatheria",   "div_mya": 94},
    "HLhipAmp3":{"name": "Hippopotamus",     "clade": "Laurasiatheria",   "div_mya": 94},  # Hippopotamus amphibius
    "HLbalEde1":{"name": "Bryde's whale",    "clade": "Laurasiatheria",   "div_mya": 94},  # Balaenoptera edeni
    "HLturTru5":{"name": "Dolphin",          "clade": "Laurasiatheria",   "div_mya": 94},  # Tursiops truncatus
    "HLcamDro2":{"name": "Dromedary",        "clade": "Laurasiatheria",   "div_mya": 94},  # Camelus dromedarius
    "equCab3":  {"name": "Horse",            "clade": "Laurasiatheria",   "div_mya": 94},
    "felCat9":  {"name": "Cat",              "clade": "Laurasiatheria",   "div_mya": 94},
    "HLmanJav2":{"name": "Pangolin",         "clade": "Laurasiatheria",   "div_mya": 94},  # Manis javanica
    "HLmyoMyo6":{"name": "Mouse-eared bat",  "clade": "Laurasiatheria",   "div_mya": 94},  # Myotis myotis
    "HLmyoLuc1":{"name": "Little brown bat", "clade": "Laurasiatheria",   "div_mya": 94},  # Myotis lucifugus
    "HLpteVam2":{"name": "Flying fox",       "clade": "Laurasiatheria",   "div_mya": 94},  # Pteropus vampyrus
    "eriEur2":  {"name": "Hedgehog",         "clade": "Laurasiatheria",   "div_mya": 94},
    # Xenarthra
    "dasNov3":  {"name": "Armadillo",        "clade": "Xenarthra",        "div_mya": 105},
    "HLchoHof3":{"name": "Sloth",            "clade": "Xenarthra",        "div_mya": 105},  # Choloepus hoffmanni
    # Afrotheria
    "HLoryAfeAfe2":{"name": "Aardvark",      "clade": "Afrotheria",       "div_mya": 105},  # Orycteropus afer
    "HLeleMax1":{"name": "Asian elephant",   "clade": "Afrotheria",       "div_mya": 105},  # Elephas maximus
    "HLproCap4":{"name": "Rock hyrax",       "clade": "Afrotheria",       "div_mya": 105},  # Procavia capensis
    # Marsupials
    "monDom5":  {"name": "Gray opossum",     "clade": "Marsupialia",      "div_mya": 180},
    "HLdidVir1":{"name": "Virginia opossum", "clade": "Marsupialia",      "div_mya": 180},  # Didelphis virginiana
    "HLnotEug3":{"name": "Tammar wallaby",   "clade": "Marsupialia",      "div_mya": 180},  # Notamacropus eugenii
}
_PHYLO_ORDER = [
    "gorGor6", "rheMac10", "mm39", "rn7", "cavPor3", "HLoryCun3",
    "bosTau9", "susScr11", "HLhipAmp3", "HLbalEde1", "HLturTru5", "HLcamDro2",
    "equCab3", "felCat9", "HLmanJav2", "HLmyoMyo6", "HLmyoLuc1", "HLpteVam2", "eriEur2",
    "dasNov3", "HLchoHof3", "HLoryAfeAfe2", "HLeleMax1", "HLproCap4",
    "monDom5", "HLdidVir1", "HLnotEug3",
]


def _meta(sp: str) -> dict:
    """Species metadata, with a safe fallback for assemblies not yet curated in
    _SPECIES_META (new species land under preprint_results before we add them).
    Unknown species get div_mya=None so they still appear in the core heatmaps /
    counts but are dropped from the phylogenetic-distance panel."""
    return _SPECIES_META.get(sp, {"name": sp, "clade": "?", "div_mya": None})

# Fig 6A representative lncRNAs (bare ENSG, versionless). Famous nuclear/
# regulatory lncRNAs spanning a modularity gradient: NEAT1 (many modular cores),
# MIAT (broad, all species), MALAT1 (few, strongest matches), XIST (X-inactivation),
# NORAD (single conserved core).
_FIG6A_GENES = [("NEAT1",  "ENSG00000245532"),
                ("MIAT",   "ENSG00000225783"),
                ("MALAT1", "ENSG00000251562"),
                ("XIST",   "ENSG00000229807"),
                ("NORAD",  "ENSG00000260032")]

_DIST_LABEL = "cosine-SW distance  $d = 1/(1{+}s)$"
_DIST_VMIN, _DIST_VMAX = 0.0, 0.10   # diag_mmd is capped at the matcher ceiling


def _load_islands(results_dir: Path):
    """Concatenate island_alignment_results.tsv for every hg38_vs_<sp> pair that
    exists on disk (auto-detect), tagged with species + divergence time.

    Returns (dataframe, species_present_in_phylo_order)."""
    import pandas as pd
    # Auto-detect every hg38_vs_<sp> pair on disk. Known species come first in
    # curated phylo order; any species not yet in _SPECIES_META is appended
    # (sorted) with a loud warning so nothing is silently dropped as the panel grows.
    on_disk = sorted(p.name.replace("hg38_vs_", "")
                     for p in results_dir.glob("hg38_vs_*") if p.is_dir())
    known = [sp for sp in _PHYLO_ORDER if sp in on_disk]
    unknown = [sp for sp in on_disk if sp not in _SPECIES_META]
    if unknown:
        print(f"# WARNING: {len(unknown)} species not in _SPECIES_META (add "
              f"name/clade/div_mya): {', '.join(unknown)}")
    order = known + unknown

    frames, present = [], []
    for sp in order:
        tsv = results_dir / f"hg38_vs_{sp}" / "island_alignment_results.tsv"
        if not tsv.exists():
            continue
        df = pd.read_csv(tsv, sep="\t")
        if df.empty:
            continue
        df["species"] = sp
        df["div_mya"] = _meta(sp)["div_mya"]
        # bare, versionless gene id (U_ENSG00000260032.3 -> ENSG00000260032)
        df["gene_bare"] = df["gene_id"].str.extract(r"(ENSG\d+)", expand=False)
        frames.append(df)
        present.append(sp)
    if not frames:
        return None, []
    return pd.concat(frames, ignore_index=True), present


def _cluster_cores(df):
    """Cluster reference islands into cross-species 'cores' by genomic overlap.

    Reference island IDs (R0, R1, ...) are NOT consistent across species runs
    (the numbering depends on which islands that query recovered), but the
    reference coordinates are the shared anchor. Within each gene, merge
    overlapping (ref_start, ref_end) intervals across all species into a core and
    label each island row with its core_id and the core's reference start (for
    left-to-right column ordering)."""
    core_id = np.empty(len(df), dtype=object)
    core_start = np.zeros(len(df), dtype=np.int64)
    for _, idx in df.groupby("gene_id").groups.items():
        rows = df.loc[idx, ["ref_start", "ref_end"]].to_numpy()
        order = np.argsort(rows[:, 0])
        cur_end, cid, cstart = -1, -1, 0
        for pos in order:
            s, e = int(rows[pos, 0]), int(rows[pos, 1])
            if s >= cur_end:          # no overlap with the open cluster -> new core
                cid += 1
                cstart = s
                cur_end = e
            else:
                cur_end = max(cur_end, e)
            ridx = idx[pos]
            gene = df.at[ridx, "gene_id"]
            core_id[df.index.get_loc(ridx)] = f"{gene}_C{cid}"
            core_start[df.index.get_loc(ridx)] = cstart
    df = df.copy()
    df["core_id"] = core_id
    df["core_start"] = core_start
    return df


def _best_per_core_species(df):
    """Lowest (best) distance per (core, species)."""
    return (df.sort_values("diag_mmd")
              .drop_duplicates(subset=["core_id", "species"], keep="first"))


def _fig6a_heatmap(ax, best, gene_bare, gene_sym, present, cmap, show_ylabels):
    """One gene: rows = species (phylo order), cols = cores (ref order), color =
    cosine-SW distance; missing (core, species) left grey."""
    import pandas as pd
    g = best[best["gene_bare"] == gene_bare]
    if g.empty:
        _todo(ax, f"{gene_sym}\n(no islands)")
        return None
    cores = (g.drop_duplicates("core_id").sort_values("core_start")["core_id"].tolist())
    piv = (g.pivot_table(index="species", columns="core_id", values="diag_mmd",
                         aggfunc="min")
            .reindex(index=present, columns=cores))
    arr = np.ma.masked_invalid(piv.to_numpy(dtype=float))
    cmap = cmap.copy(); cmap.set_bad("#e6e6e6")
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=_DIST_VMIN, vmax=_DIST_VMAX)
    # per-core "C#" labels crowd once there are many cores -> label sparsely then.
    n = len(cores)
    if n <= 8:
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"C{c.rsplit('_C', 1)[-1]}" for c in cores], fontsize=6)
    else:
        step = int(np.ceil(n / 6))
        ticks = list(range(0, n, step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], fontsize=6)
    ax.set_xlabel("core", fontsize=6.5)
    ax.set_yticks(range(len(present)))
    if show_ylabels:
        ax.set_yticklabels([_meta(s)["name"] for s in present], fontsize=6.5)
    else:
        ax.set_yticklabels([])
    ax.set_title(gene_sym, loc="center", fontsize=8, fontweight="bold")
    ax.tick_params(length=0)
    return im


def _fig6b_reproducibility(ax, best, n_species):
    """Fraction of cores matched in >= k species (k = 1..N)."""
    per_core = best.groupby("core_id")["species"].nunique()
    total = len(per_core)
    ks = np.arange(1, n_species + 1)
    frac = np.array([(per_core >= k).mean() for k in ks])
    ax.bar(ks, frac, color=fs.PALETTE["signal"], edgecolor="white", width=0.75)
    for k, f in zip(ks, frac):
        n = int((per_core >= k).sum())
        ax.text(k, f + 0.015, f"{n}", ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(ks)
    ax.set_xticklabels([f"$\\geq${k}" for k in ks], fontsize=6.5)
    ax.set_xlabel(f"Species with a match (of {n_species})")
    ax.set_ylabel("Fraction of cores")
    ax.set_ylim(0, 1.12)
    ax.text(0.97, 0.95, f"{total:,} cores", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5)


def _fig6c_phylo(ax, df, present):
    """Median cosine-SW distance per species vs divergence time (median +/- IQR),
    in phylogenetic order. Species with unknown div_mya (not yet curated) are
    omitted from this panel since they have no x-position."""
    sp_ok = [s for s in present if _meta(s)["div_mya"] is not None]
    stat = (df.groupby("species")["diag_mmd"]
              .agg(median="median", q25=lambda x: x.quantile(0.25),
                   q75=lambda x: x.quantile(0.75))
              .reindex(sp_ok))
    x = np.arange(len(sp_ok))
    lo = (stat["median"] - stat["q25"]).to_numpy()
    hi = (stat["q75"] - stat["median"]).to_numpy()
    ax.errorbar(x, stat["median"].to_numpy(), yerr=[lo, hi], fmt="o-",
                color=fs.PALETTE["accent"], ecolor=fs.PALETTE["muted"],
                capsize=3, ms=5, lw=1.4, elinewidth=1.0,
                markerfacecolor=fs.PALETTE["accent"], markeredgecolor="white",
                markeredgewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{_meta(s)['name']} ({_meta(s)['div_mya']} My)"
                        for s in sp_ok], fontsize=6, rotation=40, ha="right")
    ax.set_ylabel("Median cosine-SW distance")
    ax.set_ylim(bottom=0)


@figure("fig6_cores")
def fig6(outdir: Path, results_dir: Optional[Path]) -> None:
    import matplotlib.pyplot as plt
    fs.set_style()
    rdir = results_dir or (Path(__file__).resolve().parents[1] / "preprint_results")
    df, present = _load_islands(rdir)

    # Explicit gridspec (not mosaic) so Panel A's inter-heatmap gap is tight and
    # the number of example genes can scale freely.
    ng = len(_FIG6A_GENES)
    fig = plt.figure(figsize=(fs.FULL_WIDTH, 5.4), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.14)
    gsA = gs[0].subgridspec(1, ng, wspace=0.06)
    axesA = [fig.add_subplot(gsA[0, i]) for i in range(ng)]
    gsBC = gs[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.28)
    axB = fig.add_subplot(gsBC[0, 0])
    axC = fig.add_subplot(gsBC[0, 1])

    if df is None:
        for ax in axesA:
            _todo(ax, f"no island results\nunder {rdir}")
        _todo(axB, "core reproducibility"); _todo(axC, "distance vs phylo")
        fs.panel_label(axesA[0], "A"); fs.panel_label(axB, "B"); fs.panel_label(axC, "C")
        fs.save(fig, outdir / "fig6_cores.pdf")
        print(f"wrote {outdir/'fig6_cores.pdf'} (stub: no data)")
        return

    df = _cluster_cores(df)
    best = _best_per_core_species(df)
    cmap = plt.cm.viridis_r

    # Panel A: per-gene heatmaps sharing one colour scale/bar; y labels only on
    # the leftmost so the panels sit close together as one strip.
    im = None
    for i, ((sym, ens), ax) in enumerate(zip(_FIG6A_GENES, axesA)):
        r = _fig6a_heatmap(ax, best, ens, sym, present, cmap, show_ylabels=(i == 0))
        im = im or r
    if im is not None:
        fig.colorbar(im, ax=axesA, shrink=0.85, pad=0.01, label=_DIST_LABEL)

    _fig6b_reproducibility(axB, best, len(present))
    _fig6c_phylo(axC, df, present)

    fs.panel_label(axesA[0], "A", dx=-0.55)
    fs.panel_label(axB, "B", dx=-0.14)
    fs.panel_label(axC, "C", dx=-0.12)
    fs.save(fig, outdir / "fig6_cores.pdf")
    print(f"wrote {outdir/'fig6_cores.pdf'} ({len(present)} species: {', '.join(present)})")


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
