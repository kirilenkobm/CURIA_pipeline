#!/usr/bin/env python3
"""
build_rinalmo_logreg.py: Build the RiNALMo signal-vs-noise logistic-regression model.

Reproduces the recipe from notebooks/rinalmo_signal_noise.ipynb, but projects
through the SAME global PCA the GPU executor uses (rinalmo_pca_k16.npz, fit on
position-level embeddings) so the trained classifier matches pipeline inputs.

Pipeline order (matches gpu_executor.py mean_pool path): per-token RiNALMo
embedding -> mean-pool over tokens (raw 1280-dim) -> global PCA -> logreg.

Outputs (paths from model_registry):
  - modules/logreg_signal_noise/train_rinalmo.npz
  - modules/logreg_signal_noise/logreg_noise_model_rinalmo.json

Usage:
    python modules/logreg_signal_noise/build_rinalmo_logreg.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.model_registry import load_model, get_logreg_path, get_model_config
from modules.global_PCA.apply_pca import apply_pca, load_pca
from modules.pipeline import short_ncrna as sn

from pyrion import TwoBitAccessor
from pyrion.io.bed import read_bed12_file

# --- config (mirrors rinalmo_signal_noise.ipynb) --------------------------
MODEL_NAME = "rinalmo"
WINDOW_SIZE = 72
N_SIGNAL = 500
N_NOISE = 500
SEED = 42
SIGNAL_BIOTYPES = {"tRNA", "snoRNA", "miRNA", "snRNA", "misc_RNA", "scaRNA"}

BED_PATH = REPO_ROOT / "input_data" / "reference_annotation" / "hg38.input.w.tRNA.bed"
TWOBIT_PATH = REPO_ROOT / "input_data" / "2bit" / "hg38.2bit"
META_PATH = REPO_ROOT / "input_data" / "reference_annotation" / "test_sample.metadata.tsv"

OUT_NPZ = Path(__file__).resolve().parent / "train_rinalmo.npz"
OUT_JSON = get_logreg_path(MODEL_NAME)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    rng = random.Random(SEED)
    device = get_device()
    print(f"# Device: {device}")

    # --- load reference data ----------------------------------------------
    accessor = TwoBitAccessor(str(TWOBIT_PATH))
    bed_data = read_bed12_file(str(BED_PATH))

    biotype_map = {}
    with open(META_PATH) as f:
        header = f.readline().rstrip("\n").split("\t")
        ti, bi = header.index("transcript_id"), header.index("transcript_biotype")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > max(ti, bi):
                biotype_map[parts[ti]] = parts[bi]

    # --- signal windows (one random 72-nt window per transcript) ----------
    signal_windows = []
    for t in bed_data:
        is_trna = t.id.startswith("tRNA-") and "Und-NNN" not in t.id
        bt = "tRNA" if is_trna else biotype_map.get(t.id.split(".")[0])
        if bt not in SIGNAL_BIOTYPES:
            continue
        seq = sn._get_spliced_sequence(t, accessor)
        if not seq or "N" in seq.upper() or len(seq) < WINDOW_SIZE:
            continue
        seq = seq.upper().replace("T", "U")
        start = rng.randint(0, len(seq) - WINDOW_SIZE)
        signal_windows.append(seq[start:start + WINDOW_SIZE])
    rng.shuffle(signal_windows)
    signal_windows = signal_windows[:N_SIGNAL]
    print(f"# Signal windows: {len(signal_windows)}")

    # --- noise windows (random intergenic) --------------------------------
    chrom_sizes = accessor.chrom_sizes()
    chroms = [c for c in chrom_sizes
              if c.startswith("chr") and "_" not in c and c != "chrM"]
    rng_n = random.Random(SEED + 1)
    noise_windows = []
    attempts = 0
    while len(noise_windows) < N_NOISE and attempts < N_NOISE * 20:
        attempts += 1
        chrom = rng_n.choice(chroms)
        start = rng_n.randint(0, chrom_sizes[chrom] - WINDOW_SIZE)
        try:
            seq = str(accessor.fetch(chrom, start, start + WINDOW_SIZE)).upper().replace("T", "U")
        except Exception:
            continue
        if "N" not in seq and len(seq) == WINDOW_SIZE:
            noise_windows.append(seq)
    print(f"# Noise windows: {len(noise_windows)}")

    # --- embed (per-token RiNALMo -> mean-pool raw 1280-dim) --------------
    model, tokenize_fn, extract_fn = load_model(MODEL_NAME, device)

    def mean_embed(seqs):
        out = []
        B = 64
        for i in range(0, len(seqs), B):
            batch = [s.upper().replace("T", "U") for s in seqs[i:i + B]]
            tokens = tokenize_fn(batch)
            with torch.no_grad():
                reps = extract_fn(model, tokens)
            for j, s in enumerate(batch):
                emb = reps[j, 1:1 + len(s), :].mean(dim=0)  # mean-pool raw dim
                out.append(emb.cpu().float().numpy())
        return np.array(out)

    print("# Embedding signal + noise windows...")
    X_signal = mean_embed(signal_windows)
    X_noise = mean_embed(noise_windows)

    # --- project through the SAME global PCA the executor uses ------------
    pca_model = load_pca(model_name=MODEL_NAME)
    k = pca_model["n_components"]
    X_signal = apply_pca(X_signal, pca_model=pca_model)
    X_noise = apply_pca(X_noise, pca_model=pca_model)
    print(f"# Projected to {k}-dim via {get_model_config(MODEL_NAME)['pca_file']}")

    X = np.vstack([X_signal, X_noise])
    y = np.concatenate([np.ones(len(X_signal)), np.zeros(len(X_noise))]).astype(int)

    # --- stratified 80/20 split + balanced logreg -------------------------
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print("\n# Classification report (test):")
    print(classification_report(y_test, clf.predict(X_test), target_names=["noise", "signal"]))
    print(f"# ROC-AUC: {auc:.4f}")

    # --- save artifacts ---------------------------------------------------
    np.savez_compressed(OUT_NPZ, X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)
    model_dict = {
        "coefficients": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "classes": clf.classes_.tolist(),
        "feature_dim": int(X.shape[1]),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(model_dict, f, indent=2)
    print(f"\n# Saved {OUT_NPZ.name} and {OUT_JSON.name} (AUC={auc:.4f})")


if __name__ == "__main__":
    main()
