#!/usr/bin/env python3
"""Compare the new GBM orthology model against the legacy logreg on the same data.

Read-only: mutates no files. Reports (1) on the legacy training table (ORTH/PARA):
ROC-AUC of each model vs the TOGA label + model-vs-model agreement + confusion; and
(2) on the real hg38-vs-mm39 table (which carries the 7 runtime features): ORTH/PARA
agreement between the two models.
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

from rna_toga import _load_model, _predict_proba, classify_table

SCRIPT_LOCATION = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_LOCATION))
LEGACY = os.path.join(SCRIPT_LOCATION, "model.json")
GBM = os.path.join(SCRIPT_LOCATION, "gbm_model.json")

LEGACY_TABLE = os.path.join(ROOT, "big_test/toga_mini_results/toga_classification_table.tsv")
HG38_MM39_TABLE = os.path.join(ROOT, "hg38_vs_mm39_sncRNA_only/toga_mini_results/original_toga_classification_table.tsv")


def _proba(df, model_path):
    """Score raw ORTH probabilities (SPAN/P_PGENES rules NOT applied — pure model score)."""
    return _predict_proba(df, _load_model(model_path))


def on_legacy_table(path):
    print("=" * 70)
    print(f"LEGACY TABLE (same rows legacy was trained on): {path}")
    df = pd.read_csv(path, sep=",", engine="python")
    df = df[df["label"].isin(["ORTH", "PARA"])].copy().fillna(0.0)
    if "single_exon" not in df.columns:
        df["single_exon"] = (df["ex_num"] == 1).astype(int)
    y = (df["label"] == "ORTH").astype(int).values
    print(f"rows: {len(df):,}  (ORTH={int(y.sum()):,} PARA={int((1-y).sum()):,})")

    p_leg = _proba(df, LEGACY)
    p_gbm = _proba(df, GBM)
    print(f"\nROC-AUC vs TOGA label:  legacy logreg={roc_auc_score(y, p_leg):.4f}   GBM={roc_auc_score(y, p_gbm):.4f}")
    agree = ((p_leg >= 0.5) == (p_gbm >= 0.5)).mean()
    print(f"model-vs-model agreement @0.5: {agree:.3f}")
    print("legacy confusion [PARA;ORTH]@0.5:", confusion_matrix(y, (p_leg >= 0.5)).tolist())
    print("GBM    confusion [PARA;ORTH]@0.5:", confusion_matrix(y, (p_gbm >= 0.5)).tolist())


def on_hg38_mm39_table(path):
    print("\n" + "=" * 70)
    print(f"REAL hg38-vs-mm39 TABLE: {path}")
    if not os.path.isfile(path):
        print("  (not found — skipping)")
        return
    df = pd.read_csv(path, sep=",", engine="python").fillna(0.0)
    print(f"rows: {len(df):,}")
    # full pipeline classification (incl. SPAN / P_PGENES rules) under each model
    leg = classify_table(df, LEGACY)["label"]
    gbm = classify_table(df, GBM)["label"]
    print("\nlabel distribution:")
    print(pd.DataFrame({"legacy": leg.value_counts(), "GBM": gbm.value_counts()}).fillna(0).astype(int))
    same = (leg == gbm).mean()
    print(f"\nfull-label agreement: {same:.3f}")
    orth_leg, orth_gbm = (leg == "ORTH"), (gbm == "ORTH")
    print(f"ORTH calls: legacy={int(orth_leg.sum()):,}  GBM={int(orth_gbm.sum()):,}  "
          f"(GBM/legacy = {orth_gbm.sum()/max(1, orth_leg.sum()):.2f}x)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legacy-table", default=LEGACY_TABLE)
    ap.add_argument("--hg38-mm39-table", default=HG38_MM39_TABLE)
    args = ap.parse_args()

    for p in (LEGACY, GBM):
        if not os.path.isfile(p):
            raise SystemExit(f"Model not found: {p} (train the GBM first)")
    with open(GBM) as f:
        meta = json.load(f).get("evaluation", {})
    print("GBM training eval:", {k: meta[k] for k in meta if k.startswith("cv_") or k == "heldout_auc_gbm7"})

    on_legacy_table(args.legacy_table)
    on_hg38_mm39_table(args.hg38_mm39_table)


if __name__ == "__main__":
    main()
