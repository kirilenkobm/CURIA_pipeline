#!/usr/bin/env python3
"""Compute RiNALMo island-matching dotplots for Figure 3B (real MALAT1 core).

Illustrates the island matcher: embed each island once -> per-token cosine dotplot
-> nucleotide Smith-Waterman. We show a real conserved MALAT1 core (human chr11
island x mouse chr19 island) with the SW-aligned band, and a specificity control
(same human island x an unrelated mouse island).

Heavy step (torch + RiNALMo); caches arrays so plotting stays torch-free.

Outputs (analysis/data/):
    fig3b_dotplot.npz    match_cos, match_band, ctrl_cos, ctrl_band, dims
    fig3b_dotplot.json   labels / coordinates / scores

Run (project venv):
    .venv/bin/python analysis/compute_fig3b_dotplot.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("RiNALMo", "pipeline", "global_PCA"):
    sys.path.insert(0, str(REPO_ROOT / "modules" / sub))

from short_ncrna import _extract_sequence            # noqa: E402  strand-aware fetch
from apply_pca import load_pca, apply_pca             # noqa: E402
from pyrion import TwoBitAccessor                     # noqa: E402


def _sw_dp(S, tau, gap):
    """Local Smith-Waterman on score matrix S = cos - tau. Verbatim copy of
    modules/pipeline/matchers/rinalmo.py:_sw_dp (imported there behind numba and
    package-relative deps). Returns (score, r0, r1, q0, q1, mean_cos)."""
    la, lb = S.shape
    H = np.zeros((la + 1, lb + 1))
    ptr = np.zeros((la + 1, lb + 1), dtype=np.int8)
    best = 0.0; bi = bj = 0
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            d = H[i - 1, j - 1] + S[i - 1, j - 1]
            u = H[i - 1, j] - gap
            l = H[i, j - 1] - gap
            v = 0.0; p = 0
            if d > v: v, p = d, 1
            if u > v: v, p = u, 2
            if l > v: v, p = l, 3
            H[i, j] = v; ptr[i, j] = p
            if v > best: best, bi, bj = v, i, j
    i, j = bi, bj
    r_lo, r_hi, q_lo, q_hi = la, -1, lb, -1
    sum_cos = 0.0; n = 0
    while i > 0 and j > 0 and H[i, j] > 0.0:
        p = ptr[i, j]
        if p == 1:
            r_lo, r_hi = min(r_lo, i - 1), max(r_hi, i - 1)
            q_lo, q_hi = min(q_lo, j - 1), max(q_hi, j - 1)
            sum_cos += S[i - 1, j - 1] + tau; n += 1; i -= 1; j -= 1
        elif p == 2: i -= 1
        elif p == 3: j -= 1
        else: break
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return (float(best), float(r_lo), float(r_hi + 1),
            float(q_lo), float(q_hi + 1), float(sum_cos / n))


def _dotplot_sw(A, B, tau, gap):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    S = (An @ Bn.T).astype(np.float64) - tau
    return _sw_dp(S, float(tau), float(gap))

RESULTS = REPO_ROOT / "preprint_results" / "hg38_vs_mm39"
REF_2BIT = REPO_ROOT / "input_data" / "2bit" / "hg38.2bit"
QRY_2BIT = REPO_ROOT / "input_data" / "2bit" / "mm39.2bit"
PCA16    = REPO_ROOT / "modules" / "global_PCA" / "rinalmo_pca_k16.npz"
OUT_DIR  = REPO_ROOT / "analysis" / "data"

import argparse
# default: SNHG12, a conserved lncRNA with a long near-full-island match.
GENE_NAMES = {"ENSG00000197989": "SNHG12", "ENSG00000251562": "MALAT1"}
SW_TAU, SW_GAP = 0.5, 0.3  # deployed island_align params (model_registry)


def _islands_for(gene):
    """Rows of island_alignment_results.tsv for a gene: dicts of the key columns."""
    tsv = RESULTS / "island_alignment_results.tsv"
    hdr = None
    rows = []
    for i, line in enumerate(tsv.read_text().splitlines()):
        f = line.split("\t")
        if i == 0:
            hdr = f
            continue
        if gene in f[0]:
            rows.append(dict(zip(hdr, f)))
    return rows


def _strand(bed_path, gene):
    for line in Path(bed_path).read_text().splitlines():
        f = line.split("\t")
        if gene in f[3]:
            return -1 if f[5] == "-" else 1
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gene", default="ENSG00000197989",
                    help="reference gene id (ENSG...) whose best island core to plot")
    args = ap.parse_args()
    GENE = args.gene
    gene_name = GENE_NAMES.get(GENE, GENE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}  gene: {gene_name} ({GENE})")

    rows = _islands_for(GENE)
    rows.sort(key=lambda r: float(r["diag_mmd"]))
    best = rows[0]                                    # lowest-distance core = clearest match
    print(f"{gene_name} cores: {len(rows)}; best {best['ref_island']}<->{best['query_island']} "
          f"dist={best['diag_mmd']} ref_len={best['ref_len']} query_len={best['query_len']}")

    ref_strand = _strand(RESULTS / "query_annotation" / "aligned_reference_islands.bed", GENE)
    qry_strand = _strand(RESULTS / "query_annotation" / "aligned_query_islands.bed", GENE)
    ref_acc, qry_acc = TwoBitAccessor(str(REF_2BIT)), TwoBitAccessor(str(QRY_2BIT))

    ref_seq = _extract_sequence(ref_acc, best["ref_chrom"],
                                int(best["ref_start"]), int(best["ref_end"]), ref_strand)
    qry_seq = _extract_sequence(qry_acc, best["query_chrom"],
                                int(best["query_start"]), int(best["query_end"]), qry_strand)

    # unrelated mouse island (different gene, comparable length) for the control
    ctrl = None
    for line in (RESULTS / "query_annotation" / "aligned_query_islands.bed").read_text().splitlines():
        f = line.split("\t")
        if GENE in f[3]:
            continue
        length = int(f[2]) - int(f[1])
        if 250 <= length <= 500:
            ctrl = f
            break
    ctrl_seq = _extract_sequence(qry_acc, ctrl[0], int(ctrl[1]), int(ctrl[2]),
                                 -1 if ctrl[5] == "-" else 1)
    ctrl_gene = ctrl[3].replace("_aligned", "")
    print(f"control island: {ctrl_gene} {ctrl[0]}:{ctrl[1]}-{ctrl[2]} ({len(ctrl_seq)} nt)")

    # --- RiNALMo per-token embeddings -> k16 matching PCA ----------------
    from rinalmo.pretrained import get_pretrained_model
    print("loading RiNALMo giga-v1...")
    model, alpha = get_pretrained_model(model_name="giga-v1")
    model.eval().to(device)
    pca16 = load_pca(pca_path=PCA16)

    def emb(seq):
        rna = seq.upper().replace("T", "U")
        toks = torch.tensor(alpha.batch_tokenize([rna]), dtype=torch.int64, device=device)
        with torch.no_grad():
            rep = model(toks)["representation"].float()
        return apply_pca(rep[0, 1:1 + len(rna), :].cpu().numpy(), pca_model=pca16)

    A = emb(ref_seq); Bm = emb(qry_seq); Bc = emb(ctrl_seq)

    def cos_and_band(X, Y):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
        cos = Xn @ Yn.T
        score, r0, r1, q0, q1, mean_cos = _dotplot_sw(X, Y, SW_TAU, SW_GAP)
        return cos, np.array([r0, r1, q0, q1], float), float(score), float(mean_cos)

    match_cos, match_band, m_score, m_cos = cos_and_band(A, Bm)
    ctrl_cos, ctrl_band, c_score, c_cos = cos_and_band(A, Bc)
    print(f"match:  SW score={m_score:.1f} mean_cos={m_cos:.2f} band={match_band}")
    print(f"control: SW score={c_score:.1f} mean_cos={c_cos:.2f} band={ctrl_band}")

    np.savez(OUT_DIR / "fig3b_dotplot.npz",
             match_cos=match_cos, match_band=match_band,
             ctrl_cos=ctrl_cos, ctrl_band=ctrl_band)
    (OUT_DIR / "fig3b_dotplot.json").write_text(json.dumps({
        "gene": gene_name, "ref_island": best["ref_island"], "query_island": best["query_island"],
        "ref": f"{best['ref_chrom']}:{best['ref_start']}-{best['ref_end']} (hg38)",
        "query": f"{best['query_chrom']}:{best['query_start']}-{best['query_end']} (mm39)",
        "match_score": m_score, "match_mean_cos": m_cos,
        "ctrl_gene": ctrl_gene, "ctrl_score": c_score, "ctrl_mean_cos": c_cos,
        "tau": SW_TAU, "gap": SW_GAP}, indent=2))
    print(f"saved {OUT_DIR/'fig3b_dotplot.npz'}")


if __name__ == "__main__":
    main()
