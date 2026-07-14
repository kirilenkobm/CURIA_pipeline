#!/usr/bin/env python3
"""Helper functions for the exploratory ViennaRNA folding notebook
(analysis/vienna_structure_examples.ipynb).

Everything here is EXPLORATORY. The alignment-aware overlap numbers are ad-hoc
descriptive values for eyeballing structural concordance; they are NOT validated
structural metrics and do NOT validate CURIA, RNA function, or evolutionary
orthology.

Approach (v2 - alignment-aware, replaces the old 128x128 rescale + cosine):
  1. Sequences are recovered with the same accessors the pipeline uses
     (pyrion.TwoBitAccessor on hg38.2bit, AliasedTwoBitAccessor on mm39.2bit).
  2. Human and mouse nucleotides are aligned with an affine-gap Smith-Waterman
     (the CURIA-wired local aligner; encoding via pyrion.utils.encode_nucleotides).
  3. Both base-pair-probability matrices (BPPMs) are mapped into the shared gapped
     alignment coordinate system, so a few-nt insertion/deletion no longer shifts
     one matrix relative to the other.

The reference genes here (MALAT1, the three Fig.5 loci) are on the '+' strand, so
the forward genomic fetch already equals the sense RNA. Mouse Malat1 is on '-', so
its sense RNA is the reverse complement of the forward fetch (the pipeline scores
both orientations and keeps the best).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit

import RNA
from pyrion.utils import encode_nucleotides, UNKNOWN

REPO = Path(__file__).resolve().parents[2]

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


# --------------------------------------------------------------------------
# Sequence extraction (identical accessors to the CURIA pipeline)
# --------------------------------------------------------------------------
_REF_ACC = None
_Q_ACC = None


def _accessors():
    """Lazily open the 2bit accessors the pipeline uses."""
    global _REF_ACC, _Q_ACC
    if _REF_ACC is None:
        from pyrion import TwoBitAccessor
        from modules.utils.twobit_alias import AliasedTwoBitAccessor

        _REF_ACC = TwoBitAccessor(str(REPO / "input_data/2bit/hg38.2bit"))
        _Q_ACC = AliasedTwoBitAccessor(str(REPO / "input_data/2bit/mm39.2bit"))
    return _REF_ACC, _Q_ACC


def fetch(which: str, chrom: str, start: int, end: int, strand: str) -> str:
    """Return the sense-strand DNA sequence (uppercase) for a genomic interval.

    which in {'ref','query'}. Forward genomic fetch is reverse-complemented when
    strand == '-'. Coordinates are 0-based half-open (BED / pipeline convention).
    """
    ref_acc, q_acc = _accessors()
    acc = ref_acc if which == "ref" else q_acc
    start = max(0, int(start))
    seq = str(acc.fetch(chrom, int(start), int(end))).upper()
    if strand == "-":
        seq = revcomp(seq)
    return seq


def to_rna(seq: str) -> str:
    """DNA -> RNA alphabet for folding (T->U)."""
    return seq.replace("T", "U").replace("t", "u")


# --------------------------------------------------------------------------
# ViennaRNA folding
# --------------------------------------------------------------------------
def fold(seq: str) -> dict:
    """Fold one sequence and return MFE structure, partition-function summaries,
    the full base-pair-probability matrix (NxN, symmetric, 0-based), and the
    per-position pairing probability."""
    rna = to_rna(seq)
    n = len(rna)
    fc = RNA.fold_compound(rna)
    mfe_struct, mfe = fc.mfe()
    pf_struct, efe = fc.pf()                  # ensemble free energy
    ens_div = fc.mean_bp_distance()           # ensemble diversity (mean bp distance)
    bpp_list = fc.bpp()                        # 1-indexed list-of-lists, upper triangle
    bppm = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n + 1):
        row = bpp_list[i]
        for j in range(i + 1, n + 1):
            p = row[j]
            if p:
                bppm[i - 1, j - 1] = p
                bppm[j - 1, i - 1] = p
    per_pos = np.clip(bppm.sum(axis=1), 0.0, 1.0)   # P(position is paired)
    return dict(
        seq=rna,
        length=n,
        mfe_struct=mfe_struct,
        mfe=float(mfe),
        pf_struct=pf_struct,
        efe=float(efe),
        ensemble_diversity=float(ens_div),
        bppm=bppm,
        per_pos=per_pos,
    )


# --------------------------------------------------------------------------
# Pairwise alignment: affine-gap Smith-Waterman (CURIA-wired recurrence).
# The numba kernel fills the DP + pointer matrices; the O(L) traceback is Python.
# Encoding uses pyrion.utils.encode_nucleotides so it matches the rest of CURIA.
# --------------------------------------------------------------------------
@njit(cache=True)
def _sw_fill(a, b, match, mismatch, gap_open, gap_extend, unknown):
    la, lb = a.shape[0], b.shape[0]
    NEG = -1.0e18
    H = np.zeros((la + 1, lb + 1))
    E = np.full((la + 1, lb + 1), NEG)        # gap in a (consume b, horizontal)
    F = np.full((la + 1, lb + 1), NEG)        # gap in b (consume a, vertical)
    hp = np.zeros((la + 1, lb + 1), dtype=np.int8)   # 0 stop,1 diag,2 fromE,3 fromF
    ep = np.zeros((la + 1, lb + 1), dtype=np.int8)   # 0 open,1 extend
    fp = np.zeros((la + 1, lb + 1), dtype=np.int8)
    best = 0.0
    bi = 0
    bj = 0
    for i in range(1, la + 1):
        ai = a[i - 1]
        for j in range(1, lb + 1):
            e_open = H[i, j - 1] - gap_open
            e_ext = E[i, j - 1] - gap_extend
            if e_ext > e_open:
                E[i, j] = e_ext
                ep[i, j] = 1
            else:
                E[i, j] = e_open
                ep[i, j] = 0
            f_open = H[i - 1, j] - gap_open
            f_ext = F[i - 1, j] - gap_extend
            if f_ext > f_open:
                F[i, j] = f_ext
                fp[i, j] = 1
            else:
                F[i, j] = f_open
                fp[i, j] = 0
            s = match if (ai == b[j - 1] and ai != unknown) else mismatch
            diag = H[i - 1, j - 1] + s
            m = 0.0
            ptr = 0
            if diag > m:
                m = diag
                ptr = 1
            if E[i, j] > m:
                m = E[i, j]
                ptr = 2
            if F[i, j] > m:
                m = F[i, j]
                ptr = 3
            H[i, j] = m
            hp[i, j] = ptr
            if m > best:
                best = m
                bi = i
                bj = j
    return H, hp, ep, fp, best, bi, bj


def align_sequences(ref_seq: str, query_seq: str,
                    match: float = 2.0, mismatch: float = -1.0,
                    gap_open: float = 6.0, gap_extend: float = 1.0) -> dict:
    """Affine-gap local (Smith-Waterman) alignment of two nucleotide strings.

    Returns the gapped alignment as two column->sequence-index maps (value -1 =
    gap in that sequence) plus per-index inverse maps, aligned length and identity.
    Only the locally aligned block is returned - appropriate for CURIA islands,
    whose human/mouse fragments are often very different lengths.
    """
    a = encode_nucleotides(ref_seq.replace("U", "T")).astype(np.int64)
    b = encode_nucleotides(query_seq.replace("U", "T")).astype(np.int64)
    H, hp, ep, fp, best, bi, bj = _sw_fill(a, b, float(match), float(mismatch),
                                           float(gap_open), float(gap_extend),
                                           int(UNKNOWN))
    col_ref, col_query = [], []
    i, j, state = bi, bj, 0
    n_ident = 0
    while True:
        if state == 0:
            p = hp[i, j]
            if p == 0:
                break
            if p == 1:                        # diagonal: match/mismatch column
                col_ref.append(i - 1)
                col_query.append(j - 1)
                if a[i - 1] == b[j - 1] and a[i - 1] != UNKNOWN:
                    n_ident += 1
                i -= 1
                j -= 1
            elif p == 2:
                state = 1
            else:
                state = 2
        elif state == 1:                      # gap in ref, consume query
            col_ref.append(-1)
            col_query.append(j - 1)
            nxt = 0 if ep[i, j] == 0 else 1
            j -= 1
            state = nxt
        else:                                 # gap in query, consume ref
            col_ref.append(i - 1)
            col_query.append(-1)
            nxt = 0 if fp[i, j] == 0 else 2
            i -= 1
            state = nxt
    col_ref = np.array(col_ref[::-1], dtype=np.int64)
    col_query = np.array(col_query[::-1], dtype=np.int64)
    L = col_ref.shape[0]
    ref_to_col = np.full(len(ref_seq), -1, dtype=np.int64)
    query_to_col = np.full(len(query_seq), -1, dtype=np.int64)
    for k in range(L):
        if col_ref[k] >= 0:
            ref_to_col[col_ref[k]] = k
        if col_query[k] >= 0:
            query_to_col[col_query[k]] = k
    n_cols_both = int(((col_ref >= 0) & (col_query >= 0)).sum())
    return dict(
        L=L, col_ref=col_ref, col_query=col_query,
        ref_to_col=ref_to_col, query_to_col=query_to_col,
        score=float(best), aln_cols=n_cols_both,
        aln_ident=(n_ident / n_cols_both) if n_cols_both else 0.0,
    )


# --------------------------------------------------------------------------
# Map BPPMs / per-position profiles into alignment coordinates
# --------------------------------------------------------------------------
def map_bppm(bppm: np.ndarray, to_col: np.ndarray, L: int) -> np.ndarray:
    """Project an NxN BPPM into the LxL alignment coordinate system.
    Pairs with an endpoint outside the locally aligned block are dropped."""
    M = np.zeros((L, L), dtype=np.float64)
    idx = np.where(bppm > 0)
    for i, j in zip(*idx):
        if i >= j:
            continue
        a, b = to_col[i], to_col[j]
        if a >= 0 and b >= 0:
            M[a, b] = M[b, a] = bppm[i, j]
    return M


def map_profile(per_pos: np.ndarray, to_col: np.ndarray, L: int) -> np.ndarray:
    """Project a per-position pairing profile into alignment coords (NaN at gaps)."""
    out = np.full(L, np.nan)
    for i, c in enumerate(to_col):
        if c >= 0:
            out[c] = per_pos[i]
    return out


def alignment_aware_metrics(Mr: np.ndarray, Mq: np.ndarray, pmin: float = 0.1) -> dict:
    """Exploratory descriptive overlap of two alignment-mapped BPPMs (upper tri).

    NOT a validated structure-similarity score. Reports:
      * n_ref_pairs / n_query_pairs / n_shared : high-probability pairs (P>=pmin);
      * frac_shared_jaccard : |shared| / |union| over thresholded pairs;
      * weighted_jaccard    : sum min(Mr,Mq) / sum max(Mr,Mq);
      * weighted_overlap    : sum min(Mr,Mq) / min(sum Mr, sum Mq).
    """
    iu = np.triu_indices_from(Mr, k=1)
    r = Mr[iu]
    q = Mq[iu]
    hr = r >= pmin
    hq = q >= pmin
    shared = int((hr & hq).sum())
    union = int((hr | hq).sum())
    mn = np.minimum(r, q).sum()
    mx = np.maximum(r, q).sum()
    sr, sq = r.sum(), q.sum()
    return dict(
        n_ref_pairs=int(hr.sum()), n_query_pairs=int(hq.sum()), n_shared=shared,
        frac_shared_jaccard=(shared / union) if union else 0.0,
        weighted_jaccard=float(mn / mx) if mx > 0 else 0.0,
        weighted_overlap=float(mn / min(sr, sq)) if min(sr, sq) > 0 else 0.0,
    )


def _high_pairs(M: np.ndarray, pmin: float):
    """Upper-triangle high-probability pairs of a BPPM as (indices Nx2, probs N)."""
    iu = np.triu_indices_from(M, k=1)
    p = M[iu]
    m = p >= pmin
    ij = np.column_stack([iu[0][m], iu[1][m]]).astype(int)
    return ij, p[m]


def tolerant_overlap(Mr: np.ndarray, Mq: np.ndarray, pmin: float = 0.1, tol: int = 2) -> dict:
    """Tolerance-aware overlap of two alignment-mapped BPPMs (EXPLORATORY).

    A human pair (i,j) and mouse pair (k,l) are compatible iff |i-k|<=tol and
    |j-l|<=tol. Pairs are matched ONE-TO-ONE by maximum-weight bipartite matching
    (scipy.linear_sum_assignment), with match weight = min(P_human, P_mouse).
    Returns the tolerant shared-pair count and a tolerant weighted overlap
    (sum matched weights / min(sum P_human, sum P_mouse) over thresholded pairs).
    tol=0 reduces to a one-to-one column-exact match. NOT a validated metric."""
    from scipy.optimize import linear_sum_assignment

    hp, ph = _high_pairs(Mr, pmin)
    mp, pm = _high_pairs(Mq, pmin)
    if len(ph) == 0 or len(pm) == 0:
        return dict(n_shared=0, weighted_overlap=0.0)
    di = np.abs(hp[:, 0][:, None] - mp[:, 0][None, :])
    dj = np.abs(hp[:, 1][:, None] - mp[:, 1][None, :])
    W = np.where((di <= tol) & (dj <= tol),
                 np.minimum(ph[:, None], pm[None, :]), 0.0)
    r, c = linear_sum_assignment(-W)            # max-weight one-to-one matching
    w = W[r, c]
    sel = w > 0
    denom = min(ph.sum(), pm.sum())
    return dict(n_shared=int(sel.sum()),
                weighted_overlap=float(w[sel].sum() / denom) if denom > 0 else 0.0)


def overlap_metrics(Mr: np.ndarray, Mq: np.ndarray, pmin: float = 0.1) -> dict:
    """Bundle the column-exact metrics with the tolerant (+/-2, +/-3) metrics.
    All EXPLORATORY descriptive values, not validated structure-similarity scores."""
    m = alignment_aware_metrics(Mr, Mq, pmin)
    t2 = tolerant_overlap(Mr, Mq, pmin, tol=2)
    t3 = tolerant_overlap(Mr, Mq, pmin, tol=3)
    return dict(
        n_shared=m["n_shared"], weighted_jaccard=m["weighted_jaccard"],
        weighted_overlap=m["weighted_overlap"], frac_shared_jaccard=m["frac_shared_jaccard"],
        n_shared_tol2=t2["n_shared"], wov_tol2=t2["weighted_overlap"],
        n_shared_tol3=t3["n_shared"], wov_tol3=t3["weighted_overlap"],
    )


# --------------------------------------------------------------------------
# Plotting (publication-quality; PNG @300dpi + vector PDF)
# --------------------------------------------------------------------------
PALETTE = {
    "ref": "#2b6cb0",       # blue  (human / reference)
    "query": "#dd6b20",     # orange (mouse / query)
    "shared": "#6b46c1",    # purple (shared pairs)
    "accent": "#1a3a5c",
    "muted": "#cbd5e0",
}


def apply_style() -> None:
    """Publication rcParams WITHOUT switching the backend (keeps inline display)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _dotbracket_pairs(struct: str):
    """Parse a (pseudoknot-free) dot-bracket string into a list of (i, j) pairs."""
    stack, pairs = [], []
    for k, c in enumerate(struct):
        if c == "(":
            stack.append(k)
        elif c == ")":
            if stack:
                pairs.append((stack.pop(), k))
    return pairs


def _bppm_heatmap(ax, M, title):
    im = ax.imshow(M, cmap="magma", origin="upper", vmin=0.0, vmax=1.0,
                   interpolation="nearest", aspect="equal")
    ax.set_title(title, loc="left")
    ax.set_xlabel("alignment column j")
    ax.set_ylabel("alignment column i")
    return im


def _overlay(ax, Mr, Mq, L, pmin, title):
    """Categorical contact map: shared / human-only / mouse-only probable pairs."""
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    hr = Mr >= pmin
    hq = Mq >= pmin
    cat = np.zeros((L, L), dtype=np.int8)     # 0 none
    cat[hr & ~hq] = 1                         # human-only
    cat[hq & ~hr] = 2                         # mouse-only
    cat[hr & hq] = 3                          # shared
    cmap = ListedColormap(["#f7fafc", PALETTE["ref"], PALETTE["query"], PALETTE["shared"]])
    ax.imshow(cat, cmap=cmap, vmin=0, vmax=3, origin="upper",
              interpolation="nearest", aspect="equal")
    ax.set_title(title, loc="left")
    ax.set_xlabel("alignment column j")
    ax.set_ylabel("alignment column i")
    handles = [Line2D([0], [0], marker="s", ls="", mfc=PALETTE["shared"], mec="none", label="shared"),
               Line2D([0], [0], marker="s", ls="", mfc=PALETTE["ref"], mec="none", label="human-only"),
               Line2D([0], [0], marker="s", ls="", mfc=PALETTE["query"], mec="none", label="mouse-only")]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=6.5)


def _aligned_arcs(ax, fr, fq, aln):
    """Two-track MFE arc diagram on the shared alignment axis: human arcs above the
    baseline, mouse arcs below, so stems in corresponding regions line up."""
    from matplotlib.patches import Arc

    L = aln["L"]
    # shade gap columns lightly
    for k in range(L):
        if aln["col_ref"][k] < 0 or aln["col_query"][k] < 0:
            ax.axvspan(k - 0.5, k + 0.5, color=PALETTE["muted"], alpha=0.25, lw=0)
    ax.axhline(0, color="#4a5568", lw=0.8, zorder=3)

    def draw(pairs, to_col, sign, color):
        drawn = 0
        for i, j in pairs:
            a, b = to_col[i], to_col[j]
            if a < 0 or b < 0:
                continue
            lo, hi = (a, b) if a < b else (b, a)
            w = hi - lo
            # matplotlib Arc uses a positive bounding-box height; pick the upper
            # (0-180) or lower (180-360) semicircle to place arcs above/below.
            ax.add_patch(Arc(((lo + hi) / 2.0, 0), width=w, height=min(w, L * 0.5),
                             theta1=(0 if sign > 0 else 180), theta2=(180 if sign > 0 else 360),
                             color=color, lw=0.6, alpha=0.75))
            drawn += 1
        return drawn

    nr = draw(_dotbracket_pairs(fr["mfe_struct"]), aln["ref_to_col"], +1, PALETTE["ref"])
    nq = draw(_dotbracket_pairs(fq["mfe_struct"]), aln["query_to_col"], -1, PALETTE["query"])
    ax.set_xlim(-1, L + 1)
    ax.set_ylim(-0.55 * L, 0.55 * L)
    ax.set_yticks([])
    ax.set_xlabel("alignment column")
    ax.set_title(f"MFE arcs on aligned axis - human up ({nr} bp mapped), mouse down ({nq} bp mapped)",
                 loc="left")
    for s in ("top", "left", "right"):
        ax.spines[s].set_visible(False)


def _profile(ax, pr, pq, aln):
    """Per-position pairing probability of human and mouse along the aligned axis."""
    L = aln["L"]
    x = np.arange(L)
    for k in range(L):
        if aln["col_ref"][k] < 0 or aln["col_query"][k] < 0:
            ax.axvspan(k - 0.5, k + 0.5, color=PALETTE["muted"], alpha=0.25, lw=0)
    ax.plot(x, pr, color=PALETTE["ref"], lw=1.0, label="human")
    ax.plot(x, pq, color=PALETTE["query"], lw=1.0, label="mouse")
    ax.fill_between(x, 0, pr, color=PALETTE["ref"], alpha=0.15)
    ax.fill_between(x, 0, pq, color=PALETTE["query"], alpha=0.15)
    ax.set_xlim(-1, L + 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("alignment column")
    ax.set_ylabel("P(paired)")
    ax.set_title("per-position pairing probability (aligned)", loc="left")
    ax.legend(loc="upper right", ncol=2)


def _header_lines(meta, aln, extra, m) -> tuple:
    """Compact two-line manuscript header. Nucleotide identity / normalized\n    nucleotide-SW are the CURIA baseline-table values (extra); the branch distance\n    is labelled short-branch MMD (short) or island embedding-SW distance (island).\n    The local-alignment identity is NOT shown here (kept in summary.tsv)."""
    disp = meta["label"].replace("MALAT1.", "MALAT1 ")
    ai, ns, d = extra.get("table_sw_aligned"), extra.get("table_sw_norm"), extra.get("table_mmd")
    seq = (f"nucleotide identity {ai:.2f} | normalized nucleotide-SW {ns:.2f}"
           if ai is not None and ns is not None else "sequence metrics n/a")
    if meta.get("group") == "fig5_short":
        dist = f"short-branch MMD {d:.3f}" if d is not None else "short-branch MMD n/a"
    else:
        dist = f"island embedding-SW distance {d:.3f}" if d is not None else "island embedding-SW distance n/a"
    line1 = f"{disp} | aligned span {aln['aln_cols']} nt | {seq} | {dist}"
    line2 = (f"Structural overlap (exploratory): exact wJaccard {m['weighted_jaccard']:.2f} | "
             f"+/-2 tolerant overlap {m['wov_tol2']:.2f}")
    return line1, line2


def render_table_page(pdf, title, col_labels, cell_rows, caption=None, figsize=(11.0, 8.5),
                      col_widths=None):
    """Write a single table page (title + table + optional caption) to a PdfPages."""
    import matplotlib.pyplot as plt

    apply_style()
    fig = plt.figure(figsize=figsize)
    fig.text(0.05, 0.95, title, fontsize=13, weight="bold", va="top", ha="left")
    ax = fig.add_axes([0.03, 0.40, 0.94, 0.46]); ax.axis("off")
    tbl = ax.table(cellText=cell_rows, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cbd5e0")
        if col_widths is not None:
            cell.set_width(col_widths[c])
        if r == 0:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#edf2f7")
        if c == len(col_labels) - 1:               # left-align the last (text) column
            cell.set_text_props(ha="left"); cell._loc = "left"
    if caption:
        fig.text(0.05, 0.30, caption, fontsize=8, va="top", ha="left", wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def plot_pair_aligned(meta, fr, fq, aln, Mr, Mq, extra, out_stem, pmin: float = 0.1,
                      caption: str = None):
    """Alignment-aware comparison figure for one human->mouse pair."""
    import matplotlib.pyplot as plt

    apply_style()
    L = aln["L"]
    m = overlap_metrics(Mr, Mq, pmin)
    fig, axd = plt.subplot_mosaic("ABC\nDDD\nEEE",
                                  figsize=(11.0, 9.4 + (1.2 if caption else 0)),
                                  height_ratios=[1.55, 0.62, 0.9])
    _bppm_heatmap(axd["A"], Mr, f"human BPPM (aligned, {fr['length']} nt)")
    im = _bppm_heatmap(axd["B"], Mq, f"mouse BPPM (aligned, {fq['length']} nt)")
    fig.colorbar(im, ax=axd["B"], fraction=0.046, pad=0.04, label="P(pair)")
    _overlay(axd["C"], Mr, Mq, L, pmin, f"overlay contact map (P>={pmin})")
    _profile(axd["D"], map_profile(fr["per_pos"], aln["ref_to_col"], L),
             map_profile(fq["per_pos"], aln["query_to_col"], L), aln)
    _aligned_arcs(axd["E"], fr, fq, aln)

    line1, line2 = _header_lines(meta, aln, extra, m)
    fig.suptitle(line1 + "\n" + line2, fontsize=9.5, x=0.01, ha="left")
    bottom = 0.06 if caption else 0.0
    fig.tight_layout(rect=(0, bottom, 1, 0.94))
    if caption:
        fig.text(0.015, 0.012, caption, fontsize=7.6, va="bottom", ha="left", wrap=True)
    out_stem = str(out_stem)
    fig.savefig(out_stem + ".png")
    fig.savefig(out_stem + ".pdf")
    return fig


def plot_boundary(df, core_label: str, out_stem):
    """Boundary-sensitivity panel: MFE, ensemble diversity, and the alignment-aware
    weighted Jaccard vs boundary delta (nt)."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    d = df.sort_values("delta")
    x = d["delta"].values

    axes[0].plot(x, d["ref_mfe"], "o-", color=PALETTE["ref"], label="human")
    axes[0].plot(x, d["query_mfe"], "s-", color=PALETTE["query"], label="mouse")
    axes[0].set_ylabel("MFE (kcal/mol)"); axes[0].set_title("MFE", loc="left")
    axes[0].legend()

    axes[1].plot(x, d["ref_ediv"], "o-", color=PALETTE["ref"], label="human")
    axes[1].plot(x, d["query_ediv"], "s-", color=PALETTE["query"], label="mouse")
    axes[1].set_ylabel("ensemble diversity"); axes[1].set_title("ensemble diversity", loc="left")
    axes[1].legend()

    axes[2].plot(x, d["weighted_jaccard"], "D-", color=PALETTE["shared"])
    axes[2].set_ylabel("weighted Jaccard (exploratory)")
    axes[2].set_title("aligned BPPM weighted Jaccard", loc="left")
    axes[2].set_ylim(bottom=0)

    for ax in axes:
        ax.set_xlabel("boundary delta (nt, each side)")
        ax.axvline(0, color=PALETTE["muted"], lw=1.0, zorder=0)
        ax.grid(alpha=0.25)

    fig.suptitle(f"boundary sensitivity - {core_label}", fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_stem = str(out_stem)
    fig.savefig(out_stem + ".png")
    fig.savefig(out_stem + ".pdf")
    return fig


# --------------------------------------------------------------------------
# Matched-window analysis (MALAT1): fold comparable windows, not the raw
# highly-unequal intervals. Adds RNAplfold local pairing alongside global pf.
# --------------------------------------------------------------------------
def aligned_offsets(aln) -> tuple:
    """First/last sequence indices covered by the local alignment, as
    (ref_lo, ref_hi, query_lo, query_hi) half-open on each sequence."""
    cr, cq = aln["col_ref"], aln["col_query"]
    r = cr[cr >= 0]
    q = cq[cq >= 0]
    if r.size == 0 or q.size == 0:
        return 0, 0, 0, 0
    return int(r.min()), int(r.max()) + 1, int(q.min()), int(q.max()) + 1


def fold_plfold(seq: str, W: int = 150, L: int = 100, cutoff: float = 0.01) -> dict:
    """Local base-pair probabilities via RNAplfold (sliding window W, max span L).
    Returns a BPPM assembled from the plfold pair list + per-position profile.
    W and L are clamped to the sequence length and reported back."""
    rna = to_rna(seq)
    n = len(rna)
    w = min(W, n)
    span = min(L, w)
    bppm = np.zeros((n, n), dtype=np.float64)
    if n >= 2:
        for e in RNA.pfl_fold(rna, w, span, cutoff):
            i, j, p = e.i - 1, e.j - 1, e.p
            if 0 <= i < n and 0 <= j < n and p > 0:
                bppm[i, j] = bppm[j, i] = p
    per_pos = np.clip(bppm.sum(axis=1), 0.0, 1.0)
    return dict(seq=rna, length=n, bppm=bppm, per_pos=per_pos, plfold_W=w, plfold_L=span)


def _shared_dots(ax, Mr, Mq, maxL, pmin):
    """Scatter of SHARED probable pairs as large dots; human/mouse-only as very
    faint small dots. Uses a common axis scale (0..maxL) so windows are comparable."""
    hr = Mr >= pmin
    hq = Mq >= pmin
    shared = hr & hq
    for mask, color in ((hr & ~hq, "#cdd8e6"), (hq & ~hr, "#ecdcc9")):   # very faint context
        ii, jj = np.where(mask)
        if ii.size:
            ax.scatter(jj, ii, s=1.6, c=color, alpha=0.55, linewidths=0, zorder=1)
    ii, jj = np.where(shared)
    if ii.size:
        ax.scatter(jj, ii, s=15, c=PALETTE["shared"], alpha=0.9, linewidths=0, zorder=3)
    ax.set_xlim(0, maxL)
    ax.set_ylim(maxL, 0)                # contact-map convention (i downward)
    ax.set_aspect("equal")
    ax.set_xlabel("alignment column j")


def plot_window_stability(core_label, levels, out_stem, pmin: float = 0.1, caption: str = None):
    """Simplified boundary-stability figure for one MALAT1 core (global pf).

    Row 1: shared probable pairs (P>=pmin) as large dots per window (core / +-20 / +-40),
           human-only / mouse-only shown only as very faint context; equal axis scale.
    Row 2: mirrored MFE arc overlay per window (human up / mouse down), same x-scale.
    `levels` is an ordered list of dict(name, Mr, Mq, L, fr, fq, aln, metrics).
    """
    import matplotlib.pyplot as plt

    apply_style()
    maxL = max(d["L"] for d in levels)
    n = len(levels)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7.0 + (1.0 if caption else 0)),
                             squeeze=False, height_ratios=[1.35, 0.85])
    for ci, d in enumerate(levels):
        ax = axes[0][ci]
        _shared_dots(ax, d["Mr"], d["Mq"], maxL, pmin)
        if ci == 0:
            ax.set_ylabel("shared pairs\nalignment column i")
        m = d["metrics"]
        tol2 = m.get("n_shared_tol2", "-")
        ax.set_title(f"{d['name']}: exact {m['n_shared']} (+/-2: {tol2}), wJac {m['weighted_jaccard']:.2f}",
                     loc="left", fontsize=8.5)
        ax2 = axes[1][ci]
        _aligned_arcs(ax2, d["fr"], d["fq"], d["aln"])
        ax2.set_xlim(0, maxL)
        ax2.set_ylim(-0.55 * maxL, 0.55 * maxL)
        ax2.set_title(f"{d['name']}: MFE arcs (human up / mouse down)", loc="left", fontsize=8.5)
        if ci == 0:
            ax2.set_ylabel("MFE arc overlay")
    fig.suptitle(f"MALAT1 {core_label} - matched-window stability (global pf, shared pairs P>={pmin})",
                 fontsize=10, x=0.01, ha="left")
    bottom = 0.07 if caption else 0.0
    fig.tight_layout(rect=(0, bottom, 1, 0.95))
    if caption:
        fig.text(0.015, 0.012, caption, fontsize=7.6, va="bottom", ha="left", wrap=True)
    out_stem = str(out_stem)
    fig.savefig(out_stem + ".png")
    fig.savefig(out_stem + ".pdf")
    return fig


def render_text_page(pdf, title: str, paragraphs, figsize=(8.27, 11.69)):
    """Write a text-only page (title + wrapped paragraphs) to an open PdfPages."""
    import matplotlib.pyplot as plt

    apply_style()
    fig = plt.figure(figsize=figsize)
    fig.text(0.08, 0.94, title, fontsize=13, weight="bold", va="top", ha="left")
    y = 0.88
    for para in paragraphs:
        fig.text(0.08, y, para, fontsize=9, va="top", ha="left", wrap=True)
        y -= 0.045 + 0.012 * (len(para) // 95)
    pdf.savefig(fig)
    plt.close(fig)
    return fig
