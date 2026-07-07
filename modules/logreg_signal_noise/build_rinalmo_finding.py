#!/usr/bin/env python3
"""
build_rinalmo_finding.py: Build the RiNALMo *island-finding* representation.

This is the deployed signal/noise scanner classifier. It is DISTINCT from the
matching path: it uses a finding-specific PCA-64 (not the shared k=16 matching
PCA), a larger window with overlap-labeling, and isolated-window embedding.

Recipe (from notebooks/island_scan_param_sweep.ipynb, the deployment-faithful
scan benchmark):
  - positives = tiled windows across single-exon structured ncRNA gene bodies in
    genomic context (+/-150 nt flank), a window is POSITIVE if it overlaps the
    gene body by >= OVERLAP_NT ("covers a piece") -- not just clean center cores;
  - negatives = intergenic windows;
  - each window embedded in ISOLATION (RiNALMo mean-pool over tokens, raw 1280-dim);
  - fit a finding-specific PCA-64 on the training window embeddings;
  - StandardScaler + balanced logreg, folded into a plain linear model so the
    deployed JSON stays coefficients/intercept in PCA-64 space (no scaler needed
    at inference);
  - calibrate the island-calling prob_threshold on HELD-OUT loci by running the
    real scan (pipeline smooth_signal + _get_islands) against a background
    false-island-rate budget.

Window size is read from the registry (island_scan.window_size) so training and
deployment cannot silently diverge.

Outputs:
  - modules/global_PCA/rinalmo_pca_find_k64.npz                    (finding PCA-64)
  - modules/logreg_signal_noise/logreg_noise_model_rinalmo.json    (signal/noise logreg, feature_dim=64)
  - modules/logreg_signal_noise/rinalmo_finding_trainset.npz       (PCA-64 features + labels used
                                                                    to fit the logreg; repro/inspection only,
                                                                    not read at inference)

Reproducibility: the full build is deterministic (seeded) and caches its training
features + calibrated threshold into rinalmo_finding_trainset.npz. `--from-cache`
re-fits the logreg from that cache (no embedding / PCA / scan) and reproduces the
deployed model bit-for-bit — the RiNALMo counterpart of train_logreg.py +
rnafm_finding_trainset.npz.

Usage:
    python modules/logreg_signal_noise/build_rinalmo_finding.py               # full build
    python modules/logreg_signal_noise/build_rinalmo_finding.py --from-cache  # fast reproducible re-fit
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import json
import random
import argparse
from datetime import date
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.model_registry import (
    load_model, get_logreg_path, get_finding_pca_path, get_island_scan_params,
)
from modules.pipeline import short_ncrna as sn
from modules.utils.signal_processing import smooth_signal
from modules.pipeline.reference_islands_scanner import _get_islands

from pyrion import TwoBitAccessor
from pyrion.io.bed import read_bed12_file

# --- config ---------------------------------------------------------------
MODEL_NAME = "rinalmo"
SEED = 42
_SCAN = get_island_scan_params(MODEL_NAME)
WINDOW = _SCAN["window_size"]        # single source of truth (registry)
DEPLOY_STRIDE = _SCAN["stride"]      # stride used at scan time (calibration)
TRAIN_STRIDE = 24                    # denser tiling for training windows
OVERLAP_NT = 20                      # window positive if it covers >= this many nt of gene
FLANK = 150                          # genomic flank each side of the gene body
SMOOTH_WINDOW = 5                    # matches the scanner's hardcoded default
PCA_K = 64
GENE_MIN, GENE_MAX = 40, 320         # single-exon structured ncRNA length bounds
SIGNAL_BIOTYPES = {"tRNA", "snoRNA", "miRNA", "snRNA", "misc_RNA", "scaRNA"}  # lncRNA excluded

N_HOLDOUT_PER_BT = 71                # loci reserved per biotype for calibration
N_TRAIN_PER_BT = 150                 # cap of train loci per biotype
FP_BUDGETS = (0.10, 0.20)            # background false-island-rate budgets to report

BED_PATH = REPO_ROOT / "input_data" / "reference_annotation" / "hg38.primary_only.bed"
TWOBIT_PATH = REPO_ROOT / "input_data" / "2bit" / "hg38.2bit"
META_PATH = REPO_ROOT / "input_data" / "reference_annotation" / "hg38.primary_only.transcript_metadata.tsv"

OUT_PCA = get_finding_pca_path(MODEL_NAME)
OUT_JSON = get_logreg_path(MODEL_NAME)
OUT_NPZ = Path(__file__).resolve().parent / "rinalmo_finding_trainset.npz"

_TR = str.maketrans("ACGU", "UGCA")


def revcomp(s):
    return s.translate(_TR)[::-1]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_biotype_map():
    biotype_map = {}
    with open(META_PATH) as f:
        header = f.readline().rstrip("\n").split("\t")
        ti, bi = header.index("transcript_id"), header.index("transcript_biotype")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > max(ti, bi):
                biotype_map[parts[ti]] = parts[bi]
    return biotype_map


def group_single_exon_ncrna(bed, biotype_map, rng):
    """Group single-exon structured ncRNA transcripts by biotype."""
    def get_bt(tid):
        return biotype_map.get(tid, biotype_map.get(tid.split(".")[0]))

    by_bt = {b: [] for b in SIGNAL_BIOTYPES}
    for t in bed:
        if t.blocks.shape[0] != 1:                       # single-exon (genomic body == gene)
            continue
        is_trna = t.id.startswith("tRNA-") and "Und-NNN" not in t.id
        bt = "tRNA" if is_trna else get_bt(t.id)
        if bt in SIGNAL_BIOTYPES and GENE_MIN <= (t.end - t.start) <= GENE_MAX:
            by_bt[bt].append(t)
    for b in by_bt:
        rng.shuffle(by_bt[b])
    return by_bt


def build_region(t, accessor, chrom_sizes):
    """Genomic locus +/- FLANK, strand-correct. Returns (seq, gene_span) or None."""
    gstart = max(0, t.start - FLANK)
    gend = min(chrom_sizes[t.chrom], t.end + FLANK)
    try:
        seq = str(accessor.fetch(t.chrom, gstart, gend)).upper().replace("T", "U")
    except Exception:
        return None
    genelen = t.end - t.start
    if "N" in seq or len(seq) < genelen + 40:
        return None
    left = t.start - gstart
    span = (left, left + genelen)
    if t.strand == -1:
        seq = revcomp(seq)
        span = (len(seq) - span[1], len(seq) - span[0])
    return seq, span


def sample_intergenic_regions(lengths, accessor, chrom_sizes, rng):
    """One intergenic region per requested length (RNA-cased, N-free)."""
    chroms = [c for c in chrom_sizes
              if c.startswith("chr") and "_" not in c and c != "chrM"]
    out = []
    for L in lengths:
        for _ in range(200):
            c = rng.choice(chroms)
            s = rng.randint(0, chrom_sizes[c] - L)
            try:
                q = str(accessor.fetch(c, s, s + L)).upper().replace("T", "U")
            except Exception:
                continue
            if "N" not in q and len(q) == L:
                out.append(q)
                break
    return out


def tile(seq, window, stride):
    """Yield (start, window_seq) for windows fully inside seq."""
    for p in range(0, len(seq) - window + 1, stride):
        yield p, seq[p:p + window]


def overlap_nt(p, window, span):
    return max(0, min(p + window, span[1]) - max(p, span[0]))


def mean_embed(seqs, model, tokenize_fn, extract_fn, device, batch=64, verbose=True):
    """Isolated-window RiNALMo embedding, mean-pooled over tokens (N, 1280)."""
    out = []
    for i in range(0, len(seqs), batch):
        chunk = [s.upper().replace("T", "U") for s in seqs[i:i + batch]]
        tokens = tokenize_fn(chunk)
        with torch.no_grad():
            reps = extract_fn(model, tokens)
        for j, s in enumerate(chunk):
            out.append(reps[j, 1:1 + len(s), :].mean(dim=0).cpu().float().numpy())
        if verbose and (i // batch) % 20 == 0:
            print(f"    embedded {i + len(chunk)}/{len(seqs)}", flush=True)
    return np.array(out)


def fold_scaler_into_logreg(scaler, clf):
    """Fold StandardScaler(z=(x-mu)/sigma) into the linear logreg so a plain
    coef/intercept in raw PCA space reproduces the pipeline's predict_proba."""
    mu = scaler.mean_
    sigma = scaler.scale_                     # handles zero-variance (=1.0) safely
    w = clf.coef_[0]
    b = float(clf.intercept_[0])
    new_coef = w / sigma
    new_intercept = b - float((w * mu / sigma).sum())
    return new_coef, new_intercept


def recovery(islands, span):
    """Best fraction of the gene body covered by any single island."""
    glen = span[1] - span[0]
    best = 0.0
    for isl in islands:
        inter = max(0, min(isl["end"], span[1]) - max(isl["start"], span[0]))
        best = max(best, inter / glen)
    return best


def max_overlap_nt(islands, span):
    return max((max(0, min(i["end"], span[1]) - max(i["start"], span[0]))
                for i in islands), default=0)


def scan_region(seq, model, tokenize_fn, extract_fn, device, coef, intercept):
    """Deployment-faithful scan of one region -> (positions, smoothed_probs)."""
    starts, windows = [], []
    for p, w in tile(seq, WINDOW, DEPLOY_STRIDE):
        starts.append(p)
        windows.append(w)
    if not windows:
        return np.array([]), np.array([])
    raw = mean_embed(windows, model, tokenize_fn, extract_fn, device, verbose=False)
    feats = apply_finding_pca(raw)
    logits = feats @ coef + intercept
    probs = 1.0 / (1.0 + np.exp(-logits))
    return np.array(starts), smooth_signal(probs, SMOOTH_WINDOW)


# Populated in main() once the finding PCA is fit.
_PCA_MEAN = None
_PCA_COMPONENTS = None


def apply_finding_pca(x):
    return (x - _PCA_MEAN) @ _PCA_COMPONENTS.T


def fit_logreg(X, y):
    """StandardScaler + balanced logreg on PCA-64 features, folded into a plain
    linear model. Returns (coef, intercept, classes, cv_auc). Deterministic in
    X/y, so a re-fit from the cached features reproduces the deployed model."""
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    clf.fit(scaler.transform(X), y)
    cv = cross_val_score(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        scaler.transform(X), y, cv=5, scoring="roc_auc",
    )
    coef, intercept = fold_scaler_into_logreg(scaler, clf)
    pipe = clf.predict_proba(scaler.transform(X))[:, 1]
    fold = 1.0 / (1.0 + np.exp(-(X @ coef + intercept)))
    assert np.allclose(pipe, fold, atol=1e-6), "scaler fold mismatch"
    return coef, intercept, clf.classes_.tolist(), float(cv.mean())


def write_model_json(coef, intercept, classes, feature_dim, prob_threshold, provenance):
    OUT_JSON.write_text(json.dumps({
        "coefficients": coef.tolist(),
        "intercept": float(intercept),
        "classes": classes,
        "feature_dim": int(feature_dim),
        "prob_threshold": float(prob_threshold),
        "provenance": provenance,
    }, indent=2))


def refit_from_cache():
    """Reproducible fast path: re-fit the logreg from the cached trainset features
    (no embedding / PCA / scan) and reuse the cached calibrated threshold. Mirrors
    the RNA-FM train_logreg.py + rnafm_finding_trainset.npz flow."""
    if not OUT_NPZ.exists():
        raise SystemExit(
            f"# cache {OUT_NPZ} not found — run a full build first "
            f"(python {Path(__file__).name}) to generate it.")
    d = np.load(OUT_NPZ)
    X, y = d["X"], d["y"]
    thr = float(d["prob_threshold"]) if "prob_threshold" in d.files else 0.5
    coef, intercept, classes, cv_auc = fit_logreg(X, y)
    prov = {
        "builder": "modules/logreg_signal_noise/build_rinalmo_finding.py --from-cache",
        "date": date.today().isoformat(),
        "cache": OUT_NPZ.name,
        "finding_pca": OUT_PCA.name,
        "window_size": int(d["window_size"]) if "window_size" in d.files else WINDOW,
        "train_cv_auc": round(cv_auc, 4),
        "note": ("logreg re-fit from cached PCA-64 features; threshold reused from "
                 "the full-build held-out calibration (no re-embedding)."),
    }
    write_model_json(coef, intercept, classes, X.shape[1], thr, prov)
    print(f"# Refit from cache {OUT_NPZ.name} -> {OUT_JSON.name} "
          f"(feature_dim={X.shape[1]}, prob_threshold={thr}, train_cv_auc={cv_auc:.3f})")


def main():
    global _PCA_MEAN, _PCA_COMPONENTS
    rng = random.Random(SEED)
    device = get_device()
    print(f"# Device: {device} | window={WINDOW} train_stride={TRAIN_STRIDE} "
          f"deploy_stride={DEPLOY_STRIDE} overlap_nt={OVERLAP_NT} pca_k={PCA_K}")

    accessor = TwoBitAccessor(str(TWOBIT_PATH))
    chrom_sizes = accessor.chrom_sizes()
    bed = read_bed12_file(str(BED_PATH))
    biotype_map = load_biotype_map()
    by_bt = group_single_exon_ncrna(bed, biotype_map, rng)
    print("# single-exon structured ncRNA per biotype:",
          {b: len(v) for b, v in by_bt.items()})

    # --- locus-level split BEFORE tiling (no window leakage) --------------
    train_tx, holdout_tx = [], []
    for b, ts in by_bt.items():
        holdout_tx += [(t, b) for t in ts[:N_HOLDOUT_PER_BT]]
        train_tx += [(t, b) for t in ts[N_HOLDOUT_PER_BT:N_HOLDOUT_PER_BT + N_TRAIN_PER_BT]]
    print(f"# train loci: {len(train_tx)} | holdout loci: {len(holdout_tx)}")

    # --- build training windows (overlap-labeled) -------------------------
    train_seqs, train_labels = [], []
    n_regions = 0
    region_lengths = []
    for t, _b in train_tx:
        region = build_region(t, accessor, chrom_sizes)
        if region is None:
            continue
        seq, span = region
        n_regions += 1
        region_lengths.append(len(seq))
        for p, w in tile(seq, WINDOW, TRAIN_STRIDE):
            train_seqs.append(w)
            train_labels.append(1 if overlap_nt(p, WINDOW, span) >= OVERLAP_NT else 0)
    n_pos = int(sum(train_labels))
    print(f"# train gene regions: {n_regions} -> {len(train_seqs)} windows "
          f"({n_pos} positive / {len(train_seqs) - n_pos} negative)")

    # intergenic negative windows (one region per gene region, tiled)
    rng_neg = random.Random(SEED + 1)
    neg_lengths = [rng_neg.choice(region_lengths) for _ in range(n_regions)]
    for reg in sample_intergenic_regions(neg_lengths, accessor, chrom_sizes, rng_neg):
        for _p, w in tile(reg, WINDOW, TRAIN_STRIDE):
            train_seqs.append(w)
            train_labels.append(0)
    train_labels = np.array(train_labels, dtype=int)
    print(f"# total train windows: {len(train_seqs)} "
          f"({int(train_labels.sum())} positive / {int((train_labels == 0).sum())} negative)")

    # --- embed + fit finding PCA-64 ---------------------------------------
    model, tokenize_fn, extract_fn = load_model(MODEL_NAME, device)
    print("# Embedding training windows (isolated, mean-pooled)...")
    X_raw = mean_embed(train_seqs, model, tokenize_fn, extract_fn, device)

    print(f"# Fitting finding PCA-{PCA_K} on {X_raw.shape[0]} window embeddings...")
    pca = PCA(n_components=PCA_K, random_state=SEED).fit(X_raw)
    _PCA_MEAN = pca.mean_.astype(np.float32)
    _PCA_COMPONENTS = pca.components_.astype(np.float32)
    print(f"#   explained variance (top {PCA_K}): {pca.explained_variance_ratio_.sum():.3f}")

    X = apply_finding_pca(X_raw)

    # --- StandardScaler + balanced logreg, folded to plain linear model ---
    coef, intercept, classes, cv_auc = fit_logreg(X, train_labels)
    print(f"# train 5-fold CV AUC (overlap-labeled vs intergenic): {cv_auc:.3f}")
    print("# scaler-fold equivalence check passed")

    # --- save finding PCA (same npz key format as fit_pca.py) -------------
    np.savez_compressed(
        OUT_PCA,
        mean=pca.mean_,
        components=pca.components_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        n_components=PCA_K,
        n_samples=X_raw.shape[0],
        input_dim=X_raw.shape[1],
    )
    print(f"# Saved {OUT_PCA.name} (components {pca.components_.shape})")

    # --- held-out calibration: real scan ROC ------------------------------
    print("# Building held-out gene + intergenic regions for calibration...")
    hold_regions, hold_spans = [], []
    for t, _b in holdout_tx:
        region = build_region(t, accessor, chrom_sizes)
        if region is not None:
            hold_regions.append(region[0])
            hold_spans.append(region[1])
    rng_h = random.Random(SEED + 2)
    bg_regions = sample_intergenic_regions(
        [len(s) for s in hold_regions], accessor, chrom_sizes, rng_h)
    print(f"# holdout: {len(hold_regions)} gene regions, {len(bg_regions)} intergenic")

    print("# Scanning held-out regions (deployment-faithful)...")
    pos_tracks = [scan_region(s, model, tokenize_fn, extract_fn, device, coef, intercept)
                  for s in hold_regions]
    bg_tracks = [scan_region(s, model, tokenize_fn, extract_fn, device, coef, intercept)
                 for s in bg_regions]

    thresholds = np.linspace(0.05, 0.95, 37)
    det_strict = np.zeros(len(thresholds))
    det_lax = np.zeros(len(thresholds))
    fp = np.zeros(len(thresholds))
    for ti, thr in enumerate(thresholds):
        strict, lax = [], []
        for (starts, probs), span in zip(pos_tracks, hold_spans):
            if len(probs) == 0:
                strict.append(False); lax.append(False); continue
            isl = _get_islands(probs >= thr, starts, WINDOW)
            strict.append(recovery(isl, span) >= 0.5)
            lax.append(max_overlap_nt(isl, span) >= 40)
        det_strict[ti] = np.mean(strict)
        det_lax[ti] = np.mean(lax)
        fired = []
        for starts, probs in bg_tracks:
            if len(probs) == 0:
                fired.append(False); continue
            fired.append(len(_get_islands(probs >= thr, starts, WINDOW)) > 0)
        fp[ti] = np.mean(fired)

    print("\n# Held-out island-scan ROC (RiNALMo finding, PCA-64):")
    print(f"#   {'thr':>5s} {'bg-FP':>7s} {'det>=50%':>9s} {'det>=40nt':>10s}")
    for thr, f, ds, dl in zip(thresholds, fp, det_strict, det_lax):
        print(f"#   {thr:5.2f} {f:7.3f} {ds:9.3f} {dl:10.3f}")

    calibrated = {}
    for budget in FP_BUDGETS:
        ok = fp <= budget
        if ok.any():
            i = np.where(ok)[0][np.argmax(det_lax[ok])]
            calibrated[budget] = {
                "prob_threshold": round(float(thresholds[i]), 3),
                "det_strict": round(float(det_strict[i]), 3),
                "det_lax": round(float(det_lax[i]), 3),
                "bg_fp": round(float(fp[i]), 3),
            }
        else:
            calibrated[budget] = None
    print("\n# Calibrated operating points (max recall within budget):")
    for budget, c in calibrated.items():
        print(f"#   background-FP<={budget:.0%}: {c}")

    # Deploy threshold: recall-first (<=20% budget), fall back to <=10%.
    chosen = calibrated.get(0.20) or calibrated.get(0.10)
    deploy_thr = chosen["prob_threshold"] if chosen else 0.5
    deploy_budget = 0.20 if calibrated.get(0.20) else 0.10

    # --- write deployed finding classifier --------------------------------
    provenance = {
        "builder": "modules/logreg_signal_noise/build_rinalmo_finding.py",
        "notebook": "notebooks/island_scan_param_sweep.ipynb",
        "date": date.today().isoformat(),
        "signal": "structured ncRNA (tRNA/miRNA/snoRNA/snRNA/misc_RNA/scaRNA), lncRNA EXCLUDED",
        "labeling": f"overlap: window positive if it covers >={OVERLAP_NT}nt of gene body",
        "noise": "intergenic",
        "finding_pca": OUT_PCA.name,
        "trainset_cache": OUT_NPZ.name,
        "window_size": WINDOW,
        "train_stride": TRAIN_STRIDE,
        "deploy_stride": DEPLOY_STRIDE,
        "smooth_window": SMOOTH_WINDOW,
        "embed": "isolated per-window mean-pool (RiNALMo raw 1280 -> PCA-64)",
        "train_cv_auc": round(cv_auc, 4),
        "calibration": {f"{int(b*100)}pct_fp": c for b, c in calibrated.items()},
        "threshold_basis": f"max recall (>=40nt overlap) at background-FP<={deploy_budget:.0%}",
    }
    write_model_json(coef, intercept, classes, X.shape[1], deploy_thr, provenance)
    # Cache trainset features + the calibrated threshold so `--from-cache` can
    # reproduce this exact model (logreg re-fit) without re-embedding.
    np.savez_compressed(
        OUT_NPZ, X=X.astype(np.float32), y=train_labels,
        prob_threshold=np.float64(deploy_thr),
        window_size=np.int64(WINDOW), train_stride=np.int64(TRAIN_STRIDE),
        overlap_nt=np.int64(OVERLAP_NT), pca_k=np.int64(PCA_K),
    )
    print(f"\n# Saved {OUT_JSON.name} (feature_dim={X.shape[1]}, prob_threshold={deploy_thr})")
    print(f"# Saved {OUT_NPZ.name} (trainset cache; re-fit via --from-cache)")
    print(f"\n# ==> set model_registry rinalmo island_scan.prob_threshold = {deploy_thr}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build the RiNALMo island-finding classifier (PCA-64 + logreg).")
    ap.add_argument(
        "--from-cache", action="store_true",
        help="Reproducible fast path: re-fit the logreg from the cached trainset "
             "features + calibrated threshold (no embedding / PCA / scan). Requires a "
             "prior full build to have written the cache.")
    args = ap.parse_args()
    if args.from_cache:
        refit_from_cache()
    else:
        main()
