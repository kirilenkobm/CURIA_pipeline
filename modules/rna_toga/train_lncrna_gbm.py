#!/usr/bin/env python3
"""Train a gradient-boosted-tree orthology classifier for RNA TOGA.

Trained on the same data as the legacy logreg (``train_lncrna_logreg.py``) so the two
are directly comparable, but using the 7 features that ``rna_toga.extract_features``
actually produces at runtime and exporting to a plain-JSON tree dump that
``rna_toga._gbm_predict_proba`` can score with no sklearn dependency at inference.

Outputs ``gbm_model.json`` with: the tree ensemble, the standardiser (mean/scale),
the log-odds init and learning rate, the global threshold and per-biotype thresholds
(lncRNA is relaxed for high-recall candidate generation), and a comparison vs the
legacy logreg on the same rows.
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score, confusion_matrix

# Runtime feature set: exactly what rna_toga.extract_features emits (synteny is log1p'd,
# single_exon is derived from ex_num). loc_exo/intr_perc/chain_qlen/... are NOT produced
# at runtime, so they must not enter the deployed model.
FEATURES_7 = ["synteny_log1p", "gl_exo", "flank_cov", "exlen_to_qlen", "exon_perc", "ex_num", "single_exon"]
FEATURES_3 = ["synteny_log1p", "gl_exo", "flank_cov"]

SCRIPT_LOCATION = os.path.dirname(os.path.abspath(__file__))
LEGACY_MODEL_PATH = os.path.join(SCRIPT_LOCATION, "model.json")


def load_data(classification_table, biotypes_table=None, sample_size=20000, seed=42):
    """Load the TOGA classification table, keep ORTH/PARA, optionally filter lncRNA, sample per class."""
    df = pd.read_csv(classification_table, sep=",", engine="python")
    print(f"Loaded {len(df):,} rows from {classification_table}")

    df = df[df["label"].isin(["ORTH", "PARA"])].copy()
    print(f"ORTH/PARA rows: {len(df):,}  (ORTH={int((df.label=='ORTH').sum()):,} PARA={int((df.label=='PARA').sum()):,})")

    if biotypes_table:
        bt = pd.read_csv(biotypes_table, sep="\t")
        lnc = set(bt[bt["biotype"] == "lncRNA"]["transcript_id"])
        df = df[df["transcript_id"].isin(lnc)].copy()
        print(f"Filtered to lncRNA: {len(df):,} rows")

    if sample_size and sample_size > 0:
        parts = []
        for lab in ("ORTH", "PARA"):
            sub = df[df["label"] == lab]
            parts.append(sub.sample(min(sample_size, len(sub)), random_state=seed))
        df = pd.concat(parts)
        print(f"Sampled to {len(df):,} rows (<= {sample_size:,} per class)")

    # derived features
    df["synteny_log1p"] = np.log1p(df["synteny"].fillna(0.0))
    if "single_exon" not in df.columns:
        df["single_exon"] = (df["ex_num"] == 1).astype(int)
    df = df.fillna(0.0)
    return df


def legacy_logreg_proba(df):
    """Score rows with the shipped legacy model.json formula (for the comparison)."""
    with open(LEGACY_MODEL_PATH) as f:
        m = json.load(f)
    c = m["coefficients"]
    score = (
        c["synteny_log1p"] * np.log1p(df["synteny"].fillna(0.0))
        + c["gl_exo"] * df["gl_exo"].fillna(0.0)
        + c["flank_cov"] * df["flank_cov"].fillna(0.0)
        + c["intercept"]
    )
    return 1.0 / (1.0 + np.exp(-score))


def export_gbm(gbm, scaler, features):
    """Serialise a fitted GradientBoostingClassifier to a plain dict (pure-python scorable)."""
    trees = []
    for est in gbm.estimators_[:, 0]:
        t = est.tree_
        trees.append({
            "f": t.feature.tolist(),           # feature index per node (leaf = -2)
            "t": t.threshold.tolist(),          # split threshold (leaf = -2.0)
            "l": t.children_left.tolist(),      # left child (leaf = -1)
            "r": t.children_right.tolist(),     # right child (leaf = -1)
            "v": [float(v[0][0]) for v in t.value],
        })
    prior = gbm.init_.class_prior_             # weighted priors (sample_weight passed to init_)
    init = float(np.log(prior[1] / prior[0]))
    return {
        "features": list(features),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "init": init,
        "lr": float(gbm.learning_rate),
        "trees": trees,
    }


def gbm_proba_local(X, m):
    """Reference pure-python/numpy scorer used to round-trip-check the export."""
    mean = np.asarray(m["mean"]); scale = np.asarray(m["scale"])
    Z = (X - mean) / scale
    raw = np.full(len(X), float(m["init"]))
    for tr in m["trees"]:
        f = np.asarray(tr["f"]); thr = np.asarray(tr["t"])
        left = np.asarray(tr["l"]); right = np.asarray(tr["r"]); val = np.asarray(tr["v"])
        node = np.zeros(len(X), dtype=int)
        while True:
            is_leaf = left[node] == -1
            if is_leaf.all():
                break
            act = ~is_leaf
            cur = node[act]
            go_left = Z[act, f[cur]] <= thr[cur]
            node[act] = np.where(go_left, left[cur], right[cur])
        raw += float(m["lr"]) * val[node]
    return 1.0 / (1.0 + np.exp(-raw))


def cv_auc(df, features, clf, seed=42):
    X = StandardScaler().fit_transform(df[features].values)
    y = (df["label"] == "ORTH").astype(int).values
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("classification_table", help="TOGA classification table (CSV) — same input as train_lncrna_logreg.py")
    ap.add_argument("-b", "--biotypes", help="Optional biotypes TSV to filter lncRNA")
    ap.add_argument("-o", "--output", default=os.path.join(SCRIPT_LOCATION, "gbm_model.json"))
    ap.add_argument("-s", "--sample-size", type=int, default=20000, help="Rows per class (0 = all)")
    ap.add_argument("--n-estimators", type=int, default=150)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--lncrna-threshold", type=float, default=0.3,
                    help="Relaxed ORTH threshold for lncRNA candidate generation (default 0.3)")
    args = ap.parse_args()

    df = load_data(args.classification_table, args.biotypes, args.sample_size)
    if len(df) == 0:
        raise SystemExit("No data after filtering")
    y = (df["label"] == "ORTH").astype(int).values

    # ---- model comparison on the same rows (5-fold CV ROC-AUC) ----
    print("\n5-fold CV ROC-AUC (same rows):")
    auc_lr3 = cv_auc(df, FEATURES_3, LogisticRegression(max_iter=1000, class_weight="balanced"))
    auc_gb3 = cv_auc(df, FEATURES_3, GradientBoostingClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth))
    auc_gb7 = cv_auc(df, FEATURES_7, GradientBoostingClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth))
    print(f"  logreg-3 : {auc_lr3:.4f}")
    print(f"  GBM-3    : {auc_gb3:.4f}")
    print(f"  GBM-7    : {auc_gb7:.4f}")

    # ---- final GBM-7 on a train/test split, balanced sample weights ----
    Xtr, Xte, ytr, yte, dtr, dte = train_test_split(
        df[FEATURES_7].values, y, df, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    gbm = GradientBoostingClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
    gbm.fit(scaler.transform(Xtr), ytr, sample_weight=compute_sample_weight("balanced", ytr))

    pte = gbm.predict_proba(scaler.transform(Xte))[:, 1]
    cm = confusion_matrix(yte, (pte >= 0.5).astype(int))
    print(f"\nHeld-out GBM-7 AUC={roc_auc_score(yte, pte):.4f}  confusion(@0.5) [PARA;ORTH]={cm.tolist()}")
    print("feature importances:")
    for f_, imp in sorted(zip(FEATURES_7, gbm.feature_importances_), key=lambda x: -x[1]):
        print(f"  {f_:16s} {imp:.3f}")

    # ---- agreement with the legacy logreg on the full sampled set ----
    p_leg = legacy_logreg_proba(df).values
    p_gbm_all = gbm.predict_proba(scaler.transform(df[FEATURES_7].values))[:, 1]
    agree = ((p_leg >= 0.5) == (p_gbm_all >= 0.5)).mean()
    print(f"\nGBM vs legacy logreg agreement (@0.5, all sampled rows): {agree:.3f}")

    # ---- export + round-trip check ----
    model_json = export_gbm(gbm, scaler, FEATURES_7)
    check = gbm_proba_local(df[FEATURES_7].values, model_json)
    max_diff = float(np.abs(check - p_gbm_all).max())
    print(f"\nJSON round-trip max abs diff vs sklearn: {max_diff:.2e}")

    model_json.update({
        "model_type": "gbm",
        "threshold": 0.5,
        "biotype_thresholds": {"lncRNA": args.lncrna_threshold},
        "class_labels": {"0": "PARA", "1": "ORTH"},
        "evaluation": {
            "cv_auc_logreg3": float(auc_lr3),
            "cv_auc_gbm3": float(auc_gb3),
            "cv_auc_gbm7": float(auc_gb7),
            "heldout_auc_gbm7": float(roc_auc_score(yte, pte)),
            "heldout_confusion": cm.tolist(),
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "roundtrip_max_abs_diff": max_diff,
        },
        "comparison_with_legacy": {"agreement_at_0.5": float(agree)},
    })
    with open(args.output, "w") as f:
        json.dump(model_json, f, indent=2)
    print(f"\nWrote {args.output}  ({len(model_json['trees'])} trees)")


if __name__ == "__main__":
    main()
