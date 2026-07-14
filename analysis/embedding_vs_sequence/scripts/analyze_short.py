#!/usr/bin/env python3
"""Analysis A (short ncRNAs): does the RiNALMo MMD score add information about
annotation support beyond conventional nucleotide sequence-similarity metrics?

A1 descriptive correlations; A2 sequence-only vs sequence+MMD grouped-CV models;
A3 identity-matched analysis; A4 residual-MMD analysis. Writes short_model_results.tsv,
short_analysis_summary.json and figures/.  GPU-free (reads short_ncrna_metrics.tsv).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_stats as cs  # noqa: E402

BASE = HERE.parent
FIG = BASE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SEQ = ["ident_levenshtein", "sw_norm", "kmer3_cos", "kmer4_cos",
       "len_ratio", "gc_diff", "dinuc_dist"]
# sign so that higher score => more likely SUPPORTED (for single-feature AUC)
SIGN = dict(ident_levenshtein=1, sw_norm=1, kmer3_cos=1, kmer4_cos=1,
            gc_diff=-1, dinuc_dist=-1, len_ratio=1, mmd=-1)
LABEL = "overlap_any"


def main():
    df = pd.read_csv(BASE / "short_ncrna_metrics.tsv", sep="\t")
    y = df[LABEL].astype(int).to_numpy()
    groups = df["gene_id"].to_numpy()
    results = []      # rows for short_model_results.tsv
    summ = {}         # numbers for README

    summ["n"] = int(len(df))
    summ["n_supported"] = int(y.sum())
    summ["support_frac"] = float(y.mean())

    # ---------------- A1: descriptive correlations ----------------
    a1 = {}
    for f in ["ident_levenshtein", "sw_norm", "kmer3_cos", "kmer4_cos", "dinuc_dist"]:
        pr = pearsonr(df["mmd"], df[f])
        sp = spearmanr(df["mmd"], df[f])
        a1[f] = dict(pearson_r=round(float(pr[0]), 3), pearson_p=float(pr[1]),
                     spearman_rho=round(float(sp[0]), 3), spearman_p=float(sp[1]))
    summ["A1_mmd_vs"] = a1

    _plot_a1(df, y)

    # ---------------- A2: sequence-only vs sequence+MMD ----------------
    seeds = (0, 1, 2, 3, 4)

    def run(features, tag, subset=None):
        d = df if subset is None else df.loc[subset]
        yy = d[LABEL].astype(int).to_numpy()
        gg = d["gene_id"].to_numpy()
        X = d[features].to_numpy(float)
        r = cs.grouped_cv_oof(X, yy, gg, seeds=seeds)
        results.append(dict(analysis="A2", model=tag, n=len(d),
                            n_features=len(features),
                            roc_auc=round(r["roc_mean"], 4), roc_sd=round(r["roc_sd"], 4),
                            pr_auc=round(r["pr_mean"], 4), pr_sd=round(r["pr_sd"], 4)))
        return r

    base = run(SEQ, "baseline_seq")
    ext = run(SEQ + ["mmd"], "seq+mmd")
    base_bt = run(SEQ + _biotype_cols(df), "baseline_seq+biotype")
    ext_bt = run(SEQ + ["mmd"] + _biotype_cols(df), "seq+biotype+mmd")

    # paired delta (ext - base), same rows via seed-0 OOF
    for a, b, name in [(base, ext, "seq -> seq+mmd"),
                       (base_bt, ext_bt, "seq+biotype -> +mmd")]:
        ok = np.isfinite(a["oof"]) & np.isfinite(b["oof"])
        d_roc, lo_r, hi_r = cs.paired_bootstrap_delta_auc(y[ok], a["oof"][ok], b["oof"][ok], kind="roc")
        d_pr, lo_p, hi_p = cs.paired_bootstrap_delta_auc(y[ok], a["oof"][ok], b["oof"][ok], kind="pr")
        fold = cs.paired_fold_delta(a, b)
        results.append(dict(analysis="A2_delta", model=name, n=int(ok.sum()),
                            d_roc_auc=round(d_roc, 4), d_roc_ci=f"[{lo_r:.4f},{hi_r:.4f}]",
                            d_pr_auc=round(d_pr, 4), d_pr_ci=f"[{lo_p:.4f},{hi_p:.4f}]",
                            fold_d_roc=round(fold["d_roc_mean"], 4),
                            wilcoxon_p=round(fold["wilcoxon_roc_p"], 4)))
        summ.setdefault("A2_delta", {})[name] = dict(
            d_roc=round(d_roc, 4), d_roc_ci=[round(lo_r, 4), round(hi_r, 4)],
            d_pr=round(d_pr, 4), d_pr_ci=[round(lo_p, 4), round(hi_p, 4)])

    # single-feature AUCs (threshold-free, oriented)
    single = {}
    for f in ["ident_levenshtein", "sw_norm", "mmd"]:
        s = SIGN[f] * df[f].to_numpy(float)
        pt, lo, hi = cs.bootstrap_auc_ci(y, s, kind="roc")
        single[f] = dict(roc_auc=round(pt, 4), ci=[round(lo, 4), round(hi, 4)])
        results.append(dict(analysis="A2_single", model=f"{f}_alone", n=len(df),
                            roc_auc=round(pt, 4), roc_ci=f"[{lo:.4f},{hi:.4f}]"))
    summ["A2_single_feature"] = single
    summ["A2_models"] = dict(
        baseline=dict(roc=round(base["roc_mean"], 4), pr=round(base["pr_mean"], 4)),
        extended=dict(roc=round(ext["roc_mean"], 4), pr=round(ext["pr_mean"], 4)),
        baseline_bt=dict(roc=round(base_bt["roc_mean"], 4), pr=round(base_bt["pr_mean"], 4)),
        extended_bt=dict(roc=round(ext_bt["roc_mean"], 4), pr=round(ext_bt["pr_mean"], 4)))

    # biotype robustness: drop dominant biotype (miRNA)
    keep = df["biotype"] != "miRNA"
    base_nm = run(SEQ, "baseline_seq_no_miRNA", subset=keep)
    ext_nm = run(SEQ + ["mmd"], "seq+mmd_no_miRNA", subset=keep)
    summ["A2_no_miRNA"] = dict(baseline_roc=round(base_nm["roc_mean"], 4),
                               extended_roc=round(ext_nm["roc_mean"], 4),
                               delta=round(ext_nm["roc_mean"] - base_nm["roc_mean"], 4))
    # per-biotype MMD-alone AUC (where both classes present, n>=40)
    perbt = {}
    for bt, g in df.groupby("biotype"):
        yb = g[LABEL].astype(int).to_numpy()
        if len(g) >= 40 and 0 < yb.mean() < 1:
            pt, lo, hi = cs.bootstrap_auc_ci(yb, -g["mmd"].to_numpy(float), kind="roc")
            perbt[bt] = dict(n=len(g), support=round(float(yb.mean()), 3),
                             mmd_auc=round(pt, 3), ci=[round(lo, 3), round(hi, 3)])
    summ["A2_mmd_alone_per_biotype"] = perbt

    # ---------------- A3: identity-matched ----------------
    summ["A3_bins"] = _a3_bins(df, y, results)
    summ["A3_matched"] = _a3_matched(df, results)

    # ---------------- A4: residual MMD ----------------
    summ["A4_residual"] = _a4_residual(df, y, results)

    # ---------------- robustness: dedup by gene ----------------
    ddf = df.sort_values("mmd").drop_duplicates("gene_id", keep="first")
    yd = ddf[LABEL].astype(int).to_numpy()
    gd = ddf["gene_id"].to_numpy()
    bd = cs.grouped_cv_oof(ddf[SEQ].to_numpy(float), yd, gd, seeds=seeds)
    ed = cs.grouped_cv_oof(ddf[SEQ + ["mmd"]].to_numpy(float), yd, gd, seeds=seeds)
    summ["robustness_dedup_gene"] = dict(n=len(ddf), baseline_roc=round(bd["roc_mean"], 4),
                                         extended_roc=round(ed["roc_mean"], 4),
                                         delta=round(ed["roc_mean"] - bd["roc_mean"], 4))
    # exclude near-identical (ident>0.95) and very short (len_query<30)
    keep2 = (df["ident_levenshtein"] <= 0.95) & (df["len_query"] >= 30)
    b2 = run(SEQ, "baseline_seq_no_nearident", subset=keep2)
    e2 = run(SEQ + ["mmd"], "seq+mmd_no_nearident", subset=keep2)
    summ["robustness_no_nearident"] = dict(n=int(keep2.sum()),
                                           baseline_roc=round(b2["roc_mean"], 4),
                                           extended_roc=round(e2["roc_mean"], 4),
                                           delta=round(e2["roc_mean"] - b2["roc_mean"], 4))

    pd.DataFrame(results).to_csv(BASE / "short_model_results.tsv", sep="\t", index=False)
    (BASE / "short_analysis_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    print(f"\n# wrote {BASE/'short_model_results.tsv'} and short_analysis_summary.json")


def _biotype_cols(df):
    # one-hot top biotypes; ensure columns exist on df (added in place, cached)
    if not hasattr(_biotype_cols, "cols"):
        oh = pd.get_dummies(df["biotype"], prefix="bt").astype(float)
        for c in oh.columns:
            df[c] = oh[c].to_numpy()
        _biotype_cols.cols = list(oh.columns)
    return _biotype_cols.cols


def _a3_bins(df, y, results):
    edges = [(30, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 70)]
    ident = df["ident_levenshtein"].to_numpy() * 100
    out = {}
    for lo, hi in edges:
        m = (ident >= lo) & (ident < hi)
        sup = df.loc[m & (df[LABEL] == 1), "mmd"].to_numpy()
        uns = df.loc[m & (df[LABEL] == 0), "mmd"].to_numpy()
        rec = dict(n=int(m.sum()), n_sup=len(sup), n_uns=len(uns))
        if len(sup) >= 10 and len(uns) >= 10:
            diff, lo_c, hi_c = cs.bootstrap_diff_ci(sup, uns, stat=np.median)
            rec.update(median_sup=round(float(np.median(sup)), 4),
                       median_uns=round(float(np.median(uns)), 4),
                       median_diff=round(diff, 4), diff_ci=[round(lo_c, 4), round(hi_c, 4)],
                       cliffs_delta=round(cs.cliffs_delta(sup, uns), 3),
                       cohens_d=round(cs.cohens_d(sup, uns), 3),
                       mannwhitney_p=cs.mannwhitney_p(sup, uns))
        out[f"{lo}-{hi}"] = rec
        results.append(dict(analysis="A3_bin", model=f"ident_{lo}-{hi}", **rec))
    return out


def _a3_matched(df, results):
    cols = ["ident_levenshtein", "sw_norm", "len_ratio"]
    pos = df[df[LABEL] == 1].copy()
    neg = df[df[LABEL] == 0].copy()
    mp, mn = cs.greedy_nn_match(pos, neg, cols, caliper_sd=0.5, exact=["biotype"])
    if len(mp) < 10:
        # relax: no exact biotype
        mp, mn = cs.greedy_nn_match(pos, neg, cols, caliper_sd=0.5, exact=None)
    mmd_pos = df.loc[mp, "mmd"].to_numpy()
    mmd_neg = df.loc[mn, "mmd"].to_numpy()
    diff = mmd_pos - mmd_neg
    rec = dict(n_pairs=len(mp),
               median_mmd_supported=round(float(np.median(mmd_pos)), 4),
               median_mmd_unsupported=round(float(np.median(mmd_neg)), 4),
               median_paired_diff=round(float(np.median(diff)), 4),
               mean_paired_diff=round(float(np.mean(diff)), 4),
               cliffs_delta=round(cs.cliffs_delta(mmd_pos, mmd_neg), 3))
    try:
        rec["wilcoxon_p"] = float(wilcoxon(mmd_pos, mmd_neg).pvalue)
    except Exception:
        rec["wilcoxon_p"] = float("nan")
    # bootstrap CI on the paired median difference
    rng = np.random.default_rng(0)
    boots = [np.median(rng.choice(diff, len(diff), True)) for _ in range(2000)]
    rec["paired_diff_ci"] = [round(float(np.percentile(boots, 2.5)), 4),
                             round(float(np.percentile(boots, 97.5)), 4)]
    # check match quality
    rec["match_ident_gap_median"] = round(float(np.median(
        np.abs(df.loc[mp, "ident_levenshtein"].to_numpy() - df.loc[mn, "ident_levenshtein"].to_numpy()))), 4)
    results.append(dict(analysis="A3_matched", model="NN_matched", **{k: v for k, v in rec.items()
                                                                       if not isinstance(v, list)}))
    return rec


def _a4_residual(df, y, results):
    X = df[SEQ].to_numpy(float)
    lr = LinearRegression().fit(X, df["mmd"].to_numpy(float))
    resid = df["mmd"].to_numpy(float) - lr.predict(X)
    r2 = lr.score(X, df["mmd"].to_numpy(float))
    rs = resid[y == 1]
    ru = resid[y == 0]
    rec = dict(seq_explains_mmd_r2=round(float(r2), 4),
               resid_median_supported=round(float(np.median(rs)), 4),
               resid_median_unsupported=round(float(np.median(ru)), 4),
               cliffs_delta=round(cs.cliffs_delta(rs, ru), 3),
               cohens_d=round(cs.cohens_d(rs, ru), 3),
               mannwhitney_p=cs.mannwhitney_p(rs, ru))
    # descriptive residual-only AUC for support (note: full-data fit, descriptive)
    pt, lo, hi = cs.bootstrap_auc_ci(y, -resid, kind="roc")
    rec["resid_only_auc_descriptive"] = dict(roc_auc=round(pt, 4), ci=[round(lo, 4), round(hi, 4)])
    results.append(dict(analysis="A4", model="residual_mmd", **{k: v for k, v in rec.items()
                                                                if not isinstance(v, dict)}))
    _plot_a4(resid, y)
    return rec


# ---------------- figures ----------------
def _plot_a1(df, y):
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    c = np.where(y == 1, "#2166ac", "#b2182b")
    for a, f, xlab in [(ax[0, 0], "ident_levenshtein", "Levenshtein identity"),
                       (ax[0, 1], "sw_norm", "normalized nucleotide-SW")]:
        a.scatter(df[f], df["mmd"], s=6, c=c, alpha=0.35, linewidths=0)
        a.set_xlabel(xlab); a.set_ylabel("RiNALMo MMD (distance)")
        a.set_title(f"MMD vs {xlab}")
    # 40-60% identity zoom
    m = (df["ident_levenshtein"] >= 0.40) & (df["ident_levenshtein"] <= 0.60)
    az = ax[1, 0]
    az.scatter(df.loc[m, "ident_levenshtein"], df.loc[m, "mmd"],
               s=10, c=np.where(y[m.to_numpy()] == 1, "#2166ac", "#b2182b"), alpha=0.5, linewidths=0)
    az.set_xlabel("Levenshtein identity (40-60% zoom)"); az.set_ylabel("MMD")
    az.set_title(f"40-60% identity (n={int(m.sum())})")
    # MMD by support, stratified by identity bin
    ab = ax[1, 1]
    edges = [(30, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 70)]
    ident = df["ident_levenshtein"].to_numpy() * 100
    xs = np.arange(len(edges))
    for cls, col, off in [(1, "#2166ac", -0.15), (0, "#b2182b", 0.15)]:
        meds = []
        for lo, hi in edges:
            sel = (ident >= lo) & (ident < hi) & (y == cls)
            meds.append(np.median(df.loc[sel, "mmd"]) if sel.sum() else np.nan)
        ab.bar(xs + off, meds, width=0.3, color=col,
               label=("supported" if cls == 1 else "unsupported"))
    ab.set_xticks(xs); ab.set_xticklabels([f"{lo}-{hi}" for lo, hi in edges], rotation=45)
    ab.set_xlabel("identity bin (%)"); ab.set_ylabel("median MMD"); ab.legend()
    ab.set_title("median MMD by support within identity bins")
    fig.tight_layout(); fig.savefig(FIG / "A1_mmd_vs_sequence.png", dpi=140); plt.close(fig)


def _plot_a4(resid, y):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bins = np.linspace(resid.min(), resid.max(), 50)
    ax.hist(resid[y == 1], bins=bins, alpha=0.55, density=True, color="#2166ac", label="supported")
    ax.hist(resid[y == 0], bins=bins, alpha=0.55, density=True, color="#b2182b", label="unsupported")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("residual MMD (MMD - seq-metric prediction)")
    ax.set_ylabel("density"); ax.legend(); ax.set_title("A4: residual MMD by annotation support")
    fig.tight_layout(); fig.savefig(FIG / "A4_residual_mmd.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
