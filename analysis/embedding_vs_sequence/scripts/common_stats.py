#!/usr/bin/env python3
"""Statistical helpers shared by analyze_short.py / analyze_island.py:
grouped cross-validation, ROC/PR-AUC with bootstrap CIs, paired AUC-delta CIs,
effect sizes, and nearest-neighbour matching. All preprocessing (standardisation)
is fit strictly inside each training fold via an sklearn Pipeline.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logreg():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                                   class_weight="balanced")),
    ])


def grouped_cv_oof(X, y, groups, seeds=(0, 1, 2, 3, 4), n_splits=5):
    """Return dict with per-(seed,fold) ROC/PR AUC and pooled out-of-fold scores.

    StratifiedGroupKFold ensures no group (e.g. gene) spans train and test.
    Standardisation is fit inside each fold. Averages over `seeds` shuffles.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    fold_roc, fold_pr = [], []
    # pooled OOF from the first seed (for a single ROC/PR curve + bootstrap)
    oof_scores = np.full(len(y), np.nan)
    for si, seed in enumerate(seeds):
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y, groups):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            model = make_logreg()
            model.fit(X[tr], y[tr])
            p = model.predict_proba(X[te])[:, 1]
            fold_roc.append(roc_auc_score(y[te], p))
            fold_pr.append(average_precision_score(y[te], p))
            if si == 0:
                oof_scores[te] = p
    return dict(
        roc=np.array(fold_roc), pr=np.array(fold_pr),
        roc_mean=float(np.mean(fold_roc)), roc_sd=float(np.std(fold_roc)),
        pr_mean=float(np.mean(fold_pr)), pr_sd=float(np.std(fold_pr)),
        oof=oof_scores,
    )


def paired_fold_delta(res_a, res_b):
    """Paired per-fold delta (b - a) with Wilcoxon over folds (same fold order)."""
    a, b = res_a["roc"], res_b["roc"]
    n = min(len(a), len(b))
    d_roc = b[:n] - a[:n]
    da, db = res_a["pr"], res_b["pr"]
    d_pr = db[:n] - da[:n]
    out = dict(d_roc_mean=float(np.mean(d_roc)), d_pr_mean=float(np.mean(d_pr)))
    try:
        out["wilcoxon_roc_p"] = float(wilcoxon(b[:n], a[:n]).pvalue)
    except Exception:
        out["wilcoxon_roc_p"] = float("nan")
    return out


def bootstrap_auc_ci(y, score, n_boot=2000, seed=0, kind="roc"):
    """Stratified bootstrap CI for a single-score AUC (score higher => positive)."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y, score = y[ok], score[ok]
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    f = roc_auc_score if kind == "roc" else average_precision_score
    point = float(f(y, score))
    boots = []
    for _ in range(n_boot):
        bi = np.concatenate([rng.choice(pos, len(pos), replace=True),
                             rng.choice(neg, len(neg), replace=True)])
        if len(np.unique(y[bi])) < 2:
            continue
        boots.append(f(y[bi], score[bi]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_bootstrap_delta_auc(y, score_base, score_ext, n_boot=2000, seed=0, kind="roc"):
    """Paired bootstrap CI for AUC(ext) - AUC(base) on the SAME rows."""
    y = np.asarray(y, dtype=int)
    a = np.asarray(score_base, dtype=float)
    b = np.asarray(score_ext, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    y, a, b = y[ok], a[ok], b[ok]
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    f = roc_auc_score if kind == "roc" else average_precision_score
    point = float(f(y, b) - f(y, a))
    boots = []
    for _ in range(n_boot):
        bi = np.concatenate([rng.choice(pos, len(pos), replace=True),
                             rng.choice(neg, len(neg), replace=True)])
        if len(np.unique(y[bi])) < 2:
            continue
        boots.append(f(y[bi], b[bi]) - f(y[bi], a[bi]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def _cluster_index(clusters):
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    return uniq, {c: np.where(clusters == c)[0] for c in uniq}


def cluster_bootstrap_auc(y, score, clusters, n_boot=2000, seed=0, kind="roc"):
    """AUC point estimate + 95% CI via CLUSTER bootstrap: resample whole reference
    loci (clusters) with replacement so correlated rows sharing a reference move
    together. score higher => positive."""
    y = np.asarray(y, int); score = np.asarray(score, float)
    ok = np.isfinite(score)
    y, score, clusters = y[ok], score[ok], np.asarray(clusters)[ok]
    uniq, idx = _cluster_index(clusters)
    f = roc_auc_score if kind == "roc" else average_precision_score
    point = float(f(y, score))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        rows = np.concatenate([idx[c] for c in pick])
        if len(np.unique(y[rows])) < 2:
            continue
        boots.append(f(y[rows], score[rows]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def cluster_bootstrap_delta_auc(y, score_a, score_b, clusters, n_boot=2000, seed=0, kind="roc"):
    """Paired AUC(b)-AUC(a) with cluster bootstrap over reference loci."""
    y = np.asarray(y, int); a = np.asarray(score_a, float); b = np.asarray(score_b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    y, a, b, clusters = y[ok], a[ok], b[ok], np.asarray(clusters)[ok]
    uniq, idx = _cluster_index(clusters)
    f = roc_auc_score if kind == "roc" else average_precision_score
    point = float(f(y, b) - f(y, a))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        rows = np.concatenate([idx[c] for c in pick])
        if len(np.unique(y[rows])) < 2:
            continue
        boots.append(f(y[rows], b[rows]) - f(y[rows], a[rows]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def cliffs_delta(x, y):
    """Cliff's delta effect size for x vs y (P(x>y) - P(x<y)); robust, in [-1,1]."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    # rank-based O(n log n)
    allv = np.concatenate([x, y])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    rx = ranks[:len(x)].sum()
    u = rx - len(x) * (len(x) + 1) / 2.0
    return float(2.0 * u / (len(x) * len(y)) - 1.0)


def cohens_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / sp) if sp > 0 else float("nan")


def mannwhitney_p(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 3 or len(y) < 3:
        return float("nan")
    return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)


def bootstrap_diff_ci(x, y, n_boot=2000, seed=0, stat=np.median):
    """CI for stat(x) - stat(y) via independent bootstrap."""
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    y = np.asarray(y, dtype=float); y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = float(stat(x) - stat(y))
    boots = [stat(rng.choice(x, len(x), True)) - stat(rng.choice(y, len(y), True))
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def greedy_nn_match(pos_df, neg_df, cols, caliper_sd=0.5, exact=None, seed=0):
    """Greedy nearest-neighbour matching of positives to negatives on standardised
    `cols` (euclidean), optionally requiring exact agreement on `exact` columns.
    Returns (matched_pos_idx, matched_neg_idx)."""
    rng = np.random.default_rng(seed)
    both = np.vstack([pos_df[cols].to_numpy(float), neg_df[cols].to_numpy(float)])
    mu, sd = both.mean(0), both.std(0)
    sd[sd == 0] = 1.0
    P = (pos_df[cols].to_numpy(float) - mu) / sd
    N = (neg_df[cols].to_numpy(float) - mu) / sd
    caliper = caliper_sd * np.sqrt(len(cols))
    used = np.zeros(len(neg_df), dtype=bool)
    mp, mn = [], []
    p_order = rng.permutation(len(pos_df))
    neg_ex = neg_df[exact].to_numpy() if exact else None
    pos_ex = pos_df[exact].to_numpy() if exact else None
    for pi in p_order:
        d = np.sqrt(((N - P[pi]) ** 2).sum(1))
        d[used] = np.inf
        if exact:
            mask = (neg_ex == pos_ex[pi]).all(1)
            d[~mask] = np.inf
        j = int(np.argmin(d))
        if np.isfinite(d[j]) and d[j] <= caliper:
            used[j] = True
            mp.append(int(pos_df.index[pi]))
            mn.append(int(neg_df.index[j]))
    return mp, mn
