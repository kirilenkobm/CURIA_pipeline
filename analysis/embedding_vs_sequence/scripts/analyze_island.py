#!/usr/bin/env python3
"""Analysis B (islands): does the RiNALMo embedding-SW score discriminate the correct
syntenic correspondence beyond conventional nucleotide similarity?

Reads the MPS dump (data/island_embed_<tag>.npz) which holds the per-pair embedding-SW
distances for assigned pairs (diagonal), cross-locus negatives (off-diagonal), dinuc
shuffles, and tiled within-locus windows -- plus the sequences, so nucleotide metrics
are recomputed on IDENTICAL pairs here.

IMPORTANT: every AUC in Analysis B uses the RECOMPUTED embedding scores consistently
(diagonal, off-diagonal, shuffle and windows all come from the same fresh scorer). The
stored pipeline `diag_mmd` is used ONLY as a reproduction sanity check (see the
`reproduction` block) -- never inside an AUC. All AUC CIs are CLUSTER bootstraps that
resample whole reference islands (loci), so correlated rows sharing a reference move
together.

B1 assigned vs cross-locus; B2 within-locus ranking; B3 low-sequence-similarity subset.
Writes island_model_results.tsv, island_analysis_summary.json, figures/, and caches the
expensive nucleotide NxN matrices to data/island_ntmat_<tag>.npz.
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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_stats as cs  # noqa: E402
import seqmetrics as sm  # noqa: E402

BASE = HERE.parent
FIG = BASE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DATA = BASE / "data"

# stored embedding-only cross-vs-true AUC from analysis/scratch/island_null_*.json
STORED_AUC = {"mouse": 0.898, "cow": 0.907}
STORED_SANITY_R = {"mouse": 0.852, "cow": None}   # island_null_test.py on-GPU sanity_r
SEQ = ["sw_norm", "sw_aligned_ident", "ident_levenshtein", "kmer3_cos", "kmer4_cos",
       "len_ratio", "gc_diff", "dinuc_dist"]


def _pair_metrics(ra, qa, qra):
    best = None
    for qc in (qa, qra):
        raw, swn, swi = sm.sw_metrics(ra, qc)
        if best is None or raw > best[0]:
            best = (raw, swn, swi, float(sm._edit_ident(ra, qc)),
                    sm.kmer_cosine(ra, qc, 3), sm.kmer_cosine(ra, qc, 4),
                    sm.dinuc_dist(ra, qc), qc)
    raw, swn, swi, lev, k3, k4, dn, qc = best
    return dict(sw_raw=raw, sw_norm=swn, sw_aligned_ident=swi, ident_levenshtein=lev,
                kmer3_cos=k3, kmer4_cos=k4, dinuc_dist=dn,
                len_ratio=min(len(ra), len(qc)) / max(len(ra), len(qc)),
                gc_diff=abs(sm.gc_content(ra) - sm.gc_content(qc)))


def _nt_matrices(tag, meta):
    """NxN nucleotide sw_raw/sw_norm/sw_aligned_ident (best orientation) + shuffle;
    cached to disk so re-runs skip the ~7 min/species SW."""
    cache = DATA / f"island_ntmat_{tag}.npz"
    n = len(meta)
    if cache.exists():
        z = np.load(cache)
        if z["S_norm"].shape[0] == n:
            return z["S_raw"], z["S_norm"], z["S_ident"], z["sh_norm"]
    ref_enc = [sm.enc(m["ref_seq"]) for m in meta]
    qf_enc = [sm.enc(m["query_seq"]) for m in meta]
    qr_enc = [sm.enc(sm.revcomp(m["query_seq"])) for m in meta]
    S_raw = np.zeros((n, n)); S_norm = np.zeros((n, n)); S_ident = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r1 = sm.sw_metrics(ref_enc[i], qf_enc[j])
            r2 = sm.sw_metrics(ref_enc[i], qr_enc[j])
            S_raw[i, j], S_norm[i, j], S_ident[i, j] = r1 if r1[0] >= r2[0] else r2
    sh_norm = np.array([max(sm.sw_metrics(ref_enc[i], sm.enc(meta[i]["shuf_seq"]))[1],
                            sm.sw_metrics(ref_enc[i], sm.enc(sm.revcomp(meta[i]["shuf_seq"])))[1])
                        for i in range(n)])
    np.savez(cache, S_raw=S_raw, S_norm=S_norm, S_ident=S_ident, sh_norm=sh_norm)
    return S_raw, S_norm, S_ident, sh_norm


def analyze(tag, results, summ):
    npz = np.load(DATA / f"island_embed_{tag}.npz", allow_pickle=True)
    D = npz["D"]; diag = npz["diag"]; d_shuf = npz["d_shuf"]
    stored = npz["stored_diag_mmd"]
    meta = list(npz["meta"]); n = len(meta)
    genes = np.array([m["gene_id"] for m in meta])
    S_raw, S_norm, S_ident, sh_norm = _nt_matrices(tag, meta)
    off = ~np.eye(n, dtype=bool)

    # ---------------- reproduction diagnostics (stored is sanity ONLY) ----------------
    dd = np.abs(diag - stored)
    n_above_cap = int((diag > 0.1 + 1e-9).sum())      # fresh recompute above the 0.1 cap
    repro = dict(
        pearson_r=round(float(pearsonr(diag, stored)[0]), 3),
        spearman_rho=round(float(spearmanr(diag, stored)[0]), 3),
        median_abs_delta=round(float(np.median(dd)), 5),
        mean_abs_delta=round(float(dd.mean()), 5),
        frac_within_0p02=round(float((dd < 0.02).mean()), 3),
        n_recomputed_above_0p1_cap=n_above_cap,
        stored_range=[round(float(stored.min()), 3), round(float(stored.max()), 3)],
        note=("stored diag_mmd is right-censored at max_match_dist=0.1; a few islands "
              "re-score above the cap and dominate Pearson while Spearman/AUC are preserved"),
    )

    # ---------------- B1: assigned (diag) vs cross-locus (off-diag) ----------------
    y = np.concatenate([np.ones(n), np.zeros(off.sum())]).astype(int)
    clu = np.concatenate([np.arange(n), np.repeat(np.arange(n), n - 1)])  # by reference island
    emb_alone = np.concatenate([-diag, -D[off]])
    ntsw_alone = np.concatenate([np.diag(S_norm), S_norm[off]])
    ntid_alone = np.concatenate([np.diag(S_ident), S_ident[off]])

    b1 = {"reproduction": repro}
    for name, score in [("emb_sw_alone", emb_alone), ("nt_sw_alone", ntsw_alone),
                        ("nt_ident_alone", ntid_alone)]:
        pt, lo, hi = cs.cluster_bootstrap_auc(y, score, clu, kind="roc")
        b1[name] = dict(roc=round(pt, 4), roc_ci=[round(lo, 4), round(hi, 4)])
        results.append(dict(analysis="B1_alone", species=tag, model=name, n_pos=n,
                            n_neg=int(off.sum()), roc_auc=round(pt, 4),
                            roc_ci_cluster=f"[{lo:.4f},{hi:.4f}]"))
    # paired Δ(emb - nt_sw), cluster CI
    dpt, dlo, dhi = cs.cluster_bootstrap_delta_auc(y, ntsw_alone, emb_alone, clu, kind="roc")
    b1["delta_emb_minus_ntsw"] = dict(d_roc=round(dpt, 4), ci=[round(dlo, 4), round(dhi, 4)])
    b1["stored_emb_auc_nulltest"] = STORED_AUC[tag]
    results.append(dict(analysis="B1_alone_delta", species=tag, model="emb - nt_sw",
                        d_roc=round(dpt, 4), roc_ci_cluster=f"[{dlo:.4f},{dhi:.4f}]"))

    # fitted seq vs seq+emb (GroupKFold by ref island); cluster-bootstrap Δ over OOF
    rng = np.random.default_rng(0)
    K = 15
    rows, grp = [], []
    for i in range(n):
        m = _pair_metrics(sm.enc(meta[i]["ref_seq"]), sm.enc(meta[i]["query_seq"]),
                          sm.enc(sm.revcomp(meta[i]["query_seq"])))
        m.update(emb_sw=(1.0 / diag[i] - 1.0) if diag[i] > 0 else 1e6, label=1); rows.append(m); grp.append(i)
        js = rng.choice([j for j in range(n) if j != i], min(K, n - 1), replace=False)
        for j in js:
            mm = _pair_metrics(sm.enc(meta[i]["ref_seq"]), sm.enc(meta[j]["query_seq"]),
                               sm.enc(sm.revcomp(meta[j]["query_seq"])))
            dij = D[i, j]
            mm.update(emb_sw=(1.0 / dij - 1.0) if dij > 0 else 1e6, label=0); rows.append(mm); grp.append(i)
    fd = pd.DataFrame(rows); grp = np.array(grp); yy = fd["label"].to_numpy()
    seq_oof = _group_oof(fd[SEQ].to_numpy(float), yy, grp)
    ext_oof = _group_oof(fd[SEQ + ["emb_sw"]].to_numpy(float), yy, grp)
    d_roc, lo_r, hi_r = cs.cluster_bootstrap_delta_auc(yy, seq_oof, ext_oof, grp, kind="roc")
    d_pr, lo_p, hi_p = cs.cluster_bootstrap_delta_auc(yy, seq_oof, ext_oof, grp, kind="pr")
    b1["fitted"] = dict(seq_roc=round(roc_auc_score(yy, seq_oof), 4),
                        seqemb_roc=round(roc_auc_score(yy, ext_oof), 4),
                        d_roc=round(d_roc, 4), d_roc_ci_cluster=[round(lo_r, 4), round(hi_r, 4)],
                        d_pr=round(d_pr, 4), d_pr_ci_cluster=[round(lo_p, 4), round(hi_p, 4)])
    results.append(dict(analysis="B1_fitted", species=tag, model="seq_vs_seq+emb", n=len(fd),
                        seq_roc=round(roc_auc_score(yy, seq_oof), 4),
                        seqemb_roc=round(roc_auc_score(yy, ext_oof), 4), d_roc=round(d_roc, 4),
                        d_roc_ci_cluster=f"[{lo_r:.4f},{hi_r:.4f}]"))

    # robustness: restrict positives to well-reproduced diagonal (|diag-stored|<0.02)
    good = dd < 0.02
    yg = np.concatenate([np.ones(int(good.sum())), np.zeros(off.sum())]).astype(int)
    clug = np.concatenate([np.arange(n)[good], np.repeat(np.arange(n), n - 1)])
    embg = np.concatenate([-diag[good], -D[off]])
    ntg = np.concatenate([np.diag(S_norm)[good], S_norm[off]])
    pe, le, he = cs.cluster_bootstrap_auc(yg, embg, clug, kind="roc")
    dg, dgl, dgh = cs.cluster_bootstrap_delta_auc(yg, ntg, embg, clug, kind="roc")
    b1["robustness_wellreproduced_diag"] = dict(
        n_pos=int(good.sum()), emb_auc=round(pe, 4), emb_ci=[round(le, 4), round(he, 4)],
        delta_emb_minus_ntsw=round(dg, 4), delta_ci=[round(dgl, 4), round(dgh, 4)])

    # ---------------- B3: low-sequence-similarity subset (pre-registered) ----------------
    diag_norm = np.diag(S_norm); diag_ident = np.diag(S_ident)
    q_thresh = float(np.quantile(diag_norm, 0.25))
    b3 = {"sw_norm_q25_threshold": round(q_thresh, 4)}
    for strat_name, mask in [("bottom_quartile_sw_norm", diag_norm <= q_thresh),
                             ("aligned_ident_below_0.5", diag_ident < 0.5)]:
        idx = np.where(mask)[0]
        rec = dict(n_assigned=len(idx))
        if len(idx) >= 8:
            sub_off = off[idx]
            ys = np.concatenate([np.ones(len(idx)), np.zeros(sub_off.sum())]).astype(int)
            clus = np.concatenate([idx, np.repeat(idx, n - 1)])
            emb_s = np.concatenate([-diag[idx], -D[idx][sub_off]])
            nt_s = np.concatenate([diag_norm[idx], S_norm[idx][sub_off]])
            pe, le, he = cs.cluster_bootstrap_auc(ys, emb_s, clus, kind="roc")
            pn, ln, hn = cs.cluster_bootstrap_auc(ys, nt_s, clus, kind="roc")
            dd3, dl3, dh3 = cs.cluster_bootstrap_delta_auc(ys, nt_s, emb_s, clus, kind="roc")
            rec["emb_auc"] = dict(roc=round(pe, 4), ci=[round(le, 4), round(he, 4)])
            rec["nt_sw_auc"] = dict(roc=round(pn, 4), ci=[round(ln, 4), round(hn, 4)])
            rec["delta_emb_minus_ntsw"] = dict(d_roc=round(dd3, 4), ci=[round(dl3, 4), round(dh3, 4)])
            # assigned vs shuffle within stratum, bootstrap over islands
            fr = (diag[idx] < d_shuf[idx])
            frac, flo, fhi = _boot_mean_ci(fr)
            rec["assigned_vs_shuffle"] = dict(
                d_real_median=round(float(np.median(diag[idx])), 4),
                d_shuf_median=round(float(np.median(d_shuf[idx])), 4),
                frac_real_lt_shuf=round(frac, 3), frac_ci=[round(flo, 3), round(fhi, 3)],
                cliffs_delta=round(cs.cliffs_delta(d_shuf[idx], diag[idx]), 3))
        b3[strat_name] = rec
        results.append(dict(analysis="B3", species=tag, model=strat_name, n=rec["n_assigned"],
                            emb_auc=rec.get("emb_auc", {}).get("roc"),
                            emb_ci=str(rec.get("emb_auc", {}).get("ci")),
                            nt_auc=rec.get("nt_sw_auc", {}).get("roc"),
                            nt_ci=str(rec.get("nt_sw_auc", {}).get("ci")),
                            delta=rec.get("delta_emb_minus_ntsw", {}).get("d_roc"),
                            delta_ci=str(rec.get("delta_emb_minus_ntsw", {}).get("ci"))))

    fr_all = (diag < d_shuf); frac, flo, fhi = _boot_mean_ci(fr_all)
    b1["assigned_vs_shuffle_all"] = dict(
        d_real_median=round(float(np.median(diag)), 4), d_shuf_median=round(float(np.median(d_shuf)), 4),
        frac_real_lt_shuf=round(frac, 3), frac_ci=[round(flo, 3), round(fhi, 3)],
        nt_real_median=round(float(np.median(diag_norm)), 4), nt_shuf_median=round(float(np.median(sh_norm)), 4),
        nt_frac_real_gt_shuf=round(float((diag_norm > sh_norm).mean()), 3))

    b2 = _b2_windows(npz, tag, results)
    summ[tag] = dict(n=n, B1=b1, B2=b2, B3=b3)
    _plot_b1(tag, y, emb_alone, ntsw_alone, diag, d_shuf, diag_norm, sh_norm)
    return summ


def _boot_mean_ci(mask, n_boot=2000, seed=0):
    x = np.asarray(mask, float)
    rng = np.random.default_rng(seed)
    b = [rng.choice(x, len(x), True).mean() for _ in range(n_boot)]
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _group_oof(X, y, groups, n_splits=5):
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        model = cs.make_logreg(); model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    return oof


def _b2_windows(npz, tag, results):
    wr = list(npz["win_records"])
    if not wr:
        return {"note": "no window records"}
    recs = []
    for r in wr:
        ra = sm.enc(r["ref_seq"]); nt = []
        for w in r["win_seqs"]:
            if isinstance(w, str) and len(w) >= 40:
                a = sm.enc(w)
                nt.append(max(sm.sw_metrics(ra, a)[1], sm.sw_metrics(ra, sm.enc(sm.revcomp(w)))[1]))
            else:
                nt.append(-1.0)
        recs.append(dict(assigned_idx=int(r["assigned_idx"]),
                         emb=np.asarray(r["emb_dists"], float), nt=np.asarray(nt, float)))

    def rank_stats(records, key, higher_better, min_cand=1):
        t1, rr, pct = [], [], []
        for r in records:
            sc = r[key]; k = len(sc)
            if k < min_cand:
                continue
            order = np.argsort(-sc if higher_better else sc)
            rank = int(np.where(order == r["assigned_idx"])[0][0])
            t1.append(rank == 0); rr.append(1.0 / (rank + 1)); pct.append(rank / (k - 1) if k > 1 else 0.0)
        return dict(n=len(t1), top1=round(float(np.mean(t1)), 3), mrr=round(float(np.mean(rr)), 3),
                    median_pct=round(float(np.median(pct)), 3))

    b2 = dict(all_loci=dict(emb=rank_stats(recs, "emb", False), nt_sw=rank_stats(recs, "nt", True)),
              multicand_only=dict(emb=rank_stats(recs, "emb", False, 2),
                                  nt_sw=rank_stats(recs, "nt", True, 2)))
    m = b2["multicand_only"]
    results.append(dict(analysis="B2", species=tag, model="within_locus_ranking_multicand",
                        n=m["emb"]["n"], emb_top1=m["emb"]["top1"], nt_top1=m["nt_sw"]["top1"],
                        emb_mrr=m["emb"]["mrr"], nt_mrr=m["nt_sw"]["mrr"]))
    return b2


def _plot_b1(tag, y, emb, nt, diag, d_shuf, diag_norm, sh_norm):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].hist(emb[y == 1], bins=40, density=True, alpha=0.6, color="#2166ac", label="assigned")
    ax[0].hist(emb[y == 0], bins=40, density=True, alpha=0.6, color="#b2182b", label="cross-locus")
    ax[0].set_title(f"{tag}: embedding-SW"); ax[0].set_xlabel("-embed dist"); ax[0].legend()
    ax[1].hist(nt[y == 1], bins=40, density=True, alpha=0.6, color="#2166ac", label="assigned")
    ax[1].hist(nt[y == 0], bins=40, density=True, alpha=0.6, color="#b2182b", label="cross-locus")
    ax[1].set_title(f"{tag}: nucleotide-SW (sw_norm)"); ax[1].set_xlabel("sw_norm"); ax[1].legend()
    ax[2].scatter(diag_norm, diag, s=12, alpha=0.5, label="assigned")
    ax[2].scatter(sh_norm, d_shuf, s=12, alpha=0.5, color="#b2182b", label="dinuc-shuffle")
    ax[2].set_xlabel("nucleotide sw_norm"); ax[2].set_ylabel("embedding dist")
    ax[2].set_title(f"{tag}: real vs shuffle"); ax[2].legend()
    fig.tight_layout(); fig.savefig(FIG / f"B1_{tag}.png", dpi=140); plt.close(fig)


def main():
    results, summ = [], {}
    for tag in ["mouse", "cow"]:
        if not (DATA / f"island_embed_{tag}.npz").exists():
            print(f"# missing dump for {tag}, skipping"); continue
        analyze(tag, results, summ)
    pd.DataFrame(results).to_csv(BASE / "island_model_results.tsv", sep="\t", index=False)
    (BASE / "island_analysis_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    print(f"\n# wrote {BASE/'island_model_results.tsv'} and island_analysis_summary.json")


if __name__ == "__main__":
    main()
