#!/usr/bin/env python3
"""Compute RiNALMo embeddings for Figure 1 panels A and B, cache to an npz.

This is the one heavy (torch + 2.4 GB model) step. It is deliberately separate
from plotting so that make_figures.py / the paper build never import torch: run
this once, then make_figures.fig1_embeddings loads the cached arrays.

Panel A: per-token embeddings (matching PCA, 1280->16) for two tRNA copies and
         one miRNA --- each point is a nucleotide.
Panel B: mean-pooled embeddings (finding PCA, 1280->64) for annotated ncRNAs
         (tRNA / snoRNA / miRNA) vs background (intergenic + dinucleotide-shuffled).

Outputs (analysis/data/):
    fig1_embeddings.npz   arrays: A_coords,A_group / B_coords,B_group
    fig1_embeddings.json  group-code -> {label, color} for A and B

Run (project venv):
    .venv/bin/python analysis/compute_fig1_embeddings.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("RiNALMo", "pipeline", "global_PCA"):
    sys.path.insert(0, str(REPO_ROOT / "modules" / sub))

import short_ncrna as sn                       # noqa: E402
from short_ncrna import _compute_mmd, _extract_sequence  # noqa: E402
from apply_pca import load_pca, apply_pca       # noqa: E402
from pyrion import TwoBitAccessor               # noqa: E402
from pyrion.io.bed import read_bed12_file       # noqa: E402
import pandas as pd                             # noqa: E402

# --- config (mirrors preprint__deprecated/create_figure_1.ipynb) ----------
SEED = 42
BED_PATH   = REPO_ROOT / "input_data" / "reference_annotation" / "hg38.input.w.tRNA.bed"
TWOBIT     = REPO_ROOT / "input_data" / "2bit" / "hg38.2bit"
META_PATH  = REPO_ROOT / "input_data" / "reference_annotation" / "hg38.transcript_metadata.tsv"
PCA16      = REPO_ROOT / "modules" / "global_PCA" / "rinalmo_pca_k16.npz"   # matching (per-token)
PCA64      = REPO_ROOT / "modules" / "global_PCA" / "rinalmo_pca_find_k64.npz"  # finding (mean-pool)
LOGREG     = REPO_ROOT / "modules" / "logreg_signal_noise" / "logreg_noise_model_rinalmo.json"
OUT_DIR    = REPO_ROOT / "analysis" / "data"

PANEL_A = {  # per-token scatter
    "tRNA-Asn-GTT-chr1-140": {"color": "#2b6cb0", "label": "tRNA-Asn (copy 1)"},
    "tRNA-Asn-GTT-chr1-139": {"color": "#63b3ed", "label": "tRNA-Asn (copy 2)"},
    "ENST00000362168.1":     {"color": "#dd6b20", "label": "miRNA (MIR103A1)"},
}
BIOTYPES = {  # panel B signal classes
    "tRNA":   {"color": "#38a169", "n": 50},
    "snoRNA": {"color": "#dd6b20", "n": 39},
    "miRNA":  {"color": "#6b46c1", "n": 50},
}
NOISE_N, RANDOM_N, NOISE_LEN, MAX_SEQ_LEN = 150, 100, 100, 300
# Panels C/D: MMD sliding-window search across short ncRNAs.
N_SEARCH   = 100   # short ncRNAs to aggregate
FLANK_C    = 100   # slide window +/- this many nt around the true locus
CTRL_DIST  = 250   # distant control window offset
SHORT_MAX  = 160   # short ncRNA length cutoff


def main() -> None:
    random.seed(SEED); np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu")
    print(f"device: {device}")

    # --- RiNALMo ----------------------------------------------------------
    from rinalmo.pretrained import get_pretrained_model
    print("loading RiNALMo giga-v1 (2.4 GB)...")
    model, alpha = get_pretrained_model(model_name="giga-v1")
    model.eval().to(device)

    def per_token(seq: str) -> np.ndarray:
        """(L, 1280) per-token embeddings, special tokens stripped."""
        rna = seq.upper().replace("T", "U")
        toks = torch.tensor(alpha.batch_tokenize([rna]), dtype=torch.int64, device=device)
        with torch.no_grad():
            rep = model(toks)["representation"].float()
        return rep[0, 1:1 + len(rna), :].cpu().numpy()

    pca16 = load_pca(pca_path=PCA16)
    pca64 = load_pca(pca_path=PCA64)
    accessor = TwoBitAccessor(str(TWOBIT))
    bed = read_bed12_file(str(BED_PATH))
    by_id = {t.id: t for t in bed}

    # --- Panel A: per-token, matching PCA --------------------------------
    print("panel A: per-token embeddings")
    A_coords, A_group = [], []
    groups_a = {}
    for gi, (tid, meta) in enumerate(PANEL_A.items()):
        seq = sn._get_spliced_sequence(by_id[tid], accessor)
        coords = apply_pca(per_token(seq), pca_model=pca16)  # (L,16)
        A_coords.append(coords); A_group.append(np.full(len(coords), gi))
        groups_a[str(gi)] = meta
        print(f"  {tid}: {len(seq)} nt")
    A_coords = np.vstack(A_coords); A_group = np.concatenate(A_group)

    # --- Panel B: mean-pooled, finding PCA -------------------------------
    print("panel B: mean-pooled signal vs background")
    _meta = pd.read_csv(META_PATH, sep="\t")
    # key by full transcript_id AND version-stripped, so lookups match either way
    biotype_map = {}
    for tid, bt in zip(_meta["transcript_id"], _meta["transcript_biotype"]):
        biotype_map[tid] = bt
        biotype_map.setdefault(str(tid).split(".")[0], bt)

    def biotype_of(t):
        return biotype_map.get(t.id) or biotype_map.get(t.id.split(".")[0])

    pools = {bt: [] for bt in BIOTYPES}
    for t in bed:
        if t.id.startswith("tRNA-") and "Und-NNN" not in t.id:
            pools["tRNA"].append(t)
        elif biotype_of(t) in BIOTYPES:
            pools[biotype_of(t)].append(t)

    def mean_pooled(seq: str) -> np.ndarray:
        return apply_pca(per_token(seq).mean(axis=0), pca_model=pca64)  # (64,)

    B_coords, B_group = [], []
    groups_b, gi = {}, 0
    real_seqs = []
    for bt, cfg in BIOTYPES.items():
        random.shuffle(pools[bt])
        n = 0
        for t in pools[bt][: cfg["n"] * 3]:
            seq = sn._get_spliced_sequence(t, accessor)
            if seq and "N" not in seq.upper() and 20 <= len(seq) <= MAX_SEQ_LEN:
                B_coords.append(mean_pooled(seq)); B_group.append(gi)
                real_seqs.append(seq); n += 1
            if n >= cfg["n"]:
                break
        groups_b[str(gi)] = {"label": bt, "color": cfg["color"], "signal": True}
        print(f"  {bt}: {n}")
        gi += 1

    # background: intergenic
    chrom_sizes = accessor.chrom_sizes()
    main = [c for c in chrom_sizes if c.startswith("chr") and "_" not in c and c != "chrM"]
    noise_gi = gi
    n = 0; attempts = 0
    while n < NOISE_N and attempts < NOISE_N * 20:
        attempts += 1
        c = random.choice(main); ms = chrom_sizes[c] - NOISE_LEN
        if ms <= 0:
            continue
        s = random.randint(0, ms)
        so = accessor.fetch(c, s, s + NOISE_LEN)
        st = so.to_string()
        if "N" not in st.upper() and len(st) == NOISE_LEN:
            B_coords.append(mean_pooled(st)); B_group.append(noise_gi); n += 1
    print(f"  intergenic: {n}")
    # background: dinucleotide/random shuffle of real ncRNAs
    for _ in range(RANDOM_N):
        nts = list(random.choice(real_seqs)); random.shuffle(nts)
        B_coords.append(mean_pooled("".join(nts))); B_group.append(noise_gi)
    print(f"  shuffled: {RANDOM_N}")
    groups_b[str(noise_gi)] = {"label": "Background", "color": "#a0aec0", "signal": False}
    B_coords = np.vstack(B_coords); B_group = np.array(B_group)

    # deployed finding-classifier score P(signal) on the same PCA-64 features
    lg = json.loads(LOGREG.read_text())
    w = np.asarray(lg["coefficients"], float); b0 = float(lg["intercept"])
    B_score = 1.0 / (1.0 + np.exp(-(B_coords @ w + b0)))

    # --- Panels C/D: sliding-window MMD search across short ncRNAs ---------
    print("panels C/D: MMD search")

    def per_token_16(seq: str) -> np.ndarray:
        return apply_pca(per_token(seq), pca_model=pca16)

    offsets = np.arange(-FLANK_C, FLANK_C + 1)          # signed offset grid
    chrom_sizes = accessor.chrom_sizes()
    candidates = [t for bt in pools for t in pools[bt]]
    random.shuffle(candidates)

    C_curves, D_best, D_ctrl = [], [], []
    used = 0
    for t in candidates:
        if used >= N_SEARCH:
            break
        if len(t.blocks) != 1:                          # single-exon only
            continue
        ref_seq = sn._get_spliced_sequence(t, accessor)
        L = len(ref_seq)
        if not ref_seq or "N" in ref_seq.upper() or not (20 <= L <= SHORT_MAX):
            continue
        g_start, g_end = int(t.blocks[0][0]), int(t.blocks[0][1])
        csize = chrom_sizes.get(t.chrom, 0)
        # need room for flanks and a downstream control window
        if g_start < FLANK_C or g_end + FLANK_C + CTRL_DIST + L > csize:
            continue

        # extended region: reference sits at oriented offset FLANK_C..FLANK_C+L
        ext = _extract_sequence(accessor, t.chrom, g_start - FLANK_C,
                                g_end + FLANK_C, t.strand)
        if len(ext) != L + 2 * FLANK_C or "N" in ext.upper():
            continue
        E_ref = per_token_16(ref_seq)
        E_ext = per_token_16(ext)
        curve = []
        for o in range(0, 2 * FLANK_C + 1):
            win = E_ext[o:o + L]
            curve.append(_compute_mmd(E_ref, win) if len(win) == L else np.nan)
        curve = np.array(curve)
        if not np.isfinite(curve).all():
            continue

        # distant control window (same length, CTRL_DIST downstream)
        c0 = g_end + CTRL_DIST
        ctrl = _extract_sequence(accessor, t.chrom, c0, c0 + L, t.strand)
        if len(ctrl) != L or "N" in ctrl.upper():
            continue
        mmd_ctrl = _compute_mmd(E_ref, per_token_16(ctrl))

        C_curves.append(curve)
        D_best.append(float(curve.min()))   # best hit (dip, ~offset 0)
        D_ctrl.append(float(mmd_ctrl))
        used += 1
    print(f"  search loci used: {used}")
    C_curves = np.array(C_curves)
    D_best, D_ctrl = np.array(D_best), np.array(D_ctrl)

    # --- save -------------------------------------------------------------
    np.savez(OUT_DIR / "fig1_embeddings.npz",
             A_coords=A_coords, A_group=A_group,
             B_coords=B_coords, B_group=B_group, B_score=B_score,
             C_offsets=offsets, C_curves=C_curves,
             D_best=D_best, D_ctrl=D_ctrl)
    (OUT_DIR / "fig1_embeddings.json").write_text(
        json.dumps({"A": groups_a, "B": groups_b,
                    "provenance": {"model": "rinalmo giga-v1",
                                   "pca_A": "rinalmo_pca_k16 (per-token)",
                                   "pca_B": "rinalmo_pca_find_k64 (mean-pool)",
                                   "seed": SEED}}, indent=2))
    print(f"saved {OUT_DIR/'fig1_embeddings.npz'}  "
          f"(A: {A_coords.shape}, B: {B_coords.shape}, "
          f"C: {C_curves.shape}, D: {D_best.shape})")


if __name__ == "__main__":
    main()
