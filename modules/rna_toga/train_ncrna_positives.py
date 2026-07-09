#!/usr/bin/env python3
"""Retrain the RNA-TOGA orthology GBM as a DROP-IN of the deployed model that differs
ONLY by injected curated ncRNA positives.

Motivation: the deployed GBM-7 is well-calibrated and does NOT over-promote (unlike a
lean/rebalanced synteny-first model, which floods with syntenic single-exon pseudogenes
that are feature-identical to genuine single-exon lncRNAs like NEAT1). The single real
failure is a handful of deeply-conserved lncRNAs whose protein-coding-derived training
labels are simply wrong (NEAT1, MALAT1 -> PARA). We fix exactly those by relabelling the
named genes ORTH and upweighting them, keeping everything else identical:
  * same feature set (FEATURES_7), same 20k/class balanced sample + balanced weights,
    same threshold / per-biotype threshold as train_lncrna_gbm.py.
  * curated positives are injected AFTER sampling (otherwise the random subsample drops
    their single rows) and upweighted.

Scope is honest: this recovers the NAMED genes (and near neighbours), NOT unknown
NEAT1-like lncRNAs — those are not separable from retrocopies by these per-chain
features (see the archetype sweep), so more curation / the oracle is the general fix.

Writes gbm_model_ncrna.json. Does NOT overwrite the committed gbm_model.json and does
NOT wire itself into model_registry.
"""
import os
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score

from train_lncrna_gbm import FEATURES_7, load_data, export_gbm, gbm_proba_local

HERE = os.path.dirname(os.path.abspath(__file__))

# deeply conserved lncRNAs to force-positive (ENSG base -> name). Extend this list as
# more curated ncRNA orthologs become available.
CURATED_POSITIVES = {
    "ENSG00000245532": "NEAT1", "ENSG00000251562": "MALAT1", "ENSG00000229807": "XIST",
    "ENSG00000214548": "MEG3", "ENSG00000269821": "KCNQ1OT1", "ENSG00000130600": "H19",
    "ENSG00000234741": "GAS5",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("classification_table", nargs="?",
                    default=os.path.join(HERE, "original_toga_classification_table.tsv"))
    ap.add_argument("-o", "--output", default=os.path.join(HERE, "gbm_model_ncrna.json"))
    ap.add_argument("--pos-weight", type=float, default=500.0)
    ap.add_argument("--sample-size", type=int, default=20000)
    ap.add_argument("--lncrna-threshold", type=float, default=0.3)
    ap.add_argument("--n-estimators", type=int, default=150)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--run-table",
                    default=os.path.join(HERE, "..", "..", "rinalmo_version_outputs",
                                         "hg38_vs_mm39", "toga_results",
                                         "original_toga_classification_table.tsv"),
                    help="optional run table for an A/B gene-count delta vs the deployed model")
    args = ap.parse_args()

    df = load_data(args.classification_table, None, args.sample_size)   # same regime as deployed

    # inject curated positives from the FULL table (they are dropped by the subsample)
    full = pd.read_csv(args.classification_table, sep=",", engine="python")
    full = full[full["label"].isin(["ORTH", "PARA"])].copy()
    tid = full["transcript_id"].astype(str)
    pos_rows = []
    for ensg, name in CURATED_POSITIVES.items():
        hit = full[tid.str.contains(ensg)]
        if len(hit):
            r = hit.copy(); was = r["label"].value_counts().to_dict(); r["label"] = "ORTH"
            pos_rows.append(r)
            print(f"  positive {name:9s} ({ensg}): {len(r)} row(s) was {was} -> ORTH")
    pos = pd.concat(pos_rows) if pos_rows else full.iloc[:0]
    pos["synteny_log1p"] = np.log1p(pos["synteny"].fillna(0.0))
    if "single_exon" not in pos.columns:
        pos["single_exon"] = (pos["ex_num"] == 1).astype(int)
    pos = pos.fillna(0.0)

    # relabel any of these that were already in the sample, then append the rest
    df = df[~df["transcript_id"].isin(set(pos["transcript_id"]))]
    train = pd.concat([df, pos[df.columns]], ignore_index=True)
    is_pos = train["transcript_id"].isin(set(pos["transcript_id"])).to_numpy()

    y = (train["label"] == "ORTH").astype(int).to_numpy()
    X = train[FEATURES_7].to_numpy()
    print(f"\ntrain rows: {len(train):,}  ORTH={int(y.sum()):,}  PARA={int((1-y).sum()):,}  positives={int(is_pos.sum())}")

    w = compute_sample_weight("balanced", y)
    w[is_pos] *= args.pos_weight

    scaler = StandardScaler().fit(X)
    gbm = GradientBoostingClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
    gbm.fit(scaler.transform(X), y, sample_weight=w)

    model = export_gbm(gbm, scaler, FEATURES_7)
    p_all = gbm.predict_proba(scaler.transform(X))[:, 1]
    rt = float(np.abs(gbm_proba_local(X, model) - p_all).max())
    model.update({"model_type": "gbm", "threshold": 0.5,
                  "biotype_thresholds": {"lncRNA": args.lncrna_threshold},
                  "class_labels": {"0": "PARA", "1": "ORTH"},
                  "notes": f"deployed GBM-7 regime + {int(is_pos.sum())} curated ncRNA positives "
                           f"(weight x{args.pos_weight:g}). Drop-in for gbm_model.json."})
    with open(args.output, "w") as f:
        json.dump(model, f, indent=2)
    print(f"round-trip max abs diff={rt:.2e}  |  wrote {args.output}")

    # ---- A/B vs deployed on the run table (gene-level ORTH), if available ----
    if os.path.isfile(args.run_table):
        from rna_toga import classify_table
        r = pd.read_csv(args.run_table)
        feat = ["chain_id", "transcript_id", "gl_exo", "exlen_to_qlen", "synteny",
                "flank_cov", "exon_perc", "ex_num", "biotype"]
        r["oldL"] = classify_table(r[feat], os.path.join(HERE, "gbm_model.json")).label.to_numpy()
        r["newL"] = classify_table(r[feat], args.output).label.to_numpy()
        r["base"] = r.transcript_id.astype(str).str.replace("U_", "", regex=False).str.split(".").str[0]
        g = r.groupby("base").agg(old=("oldL", lambda x: (x == "ORTH").any()),
                                  new=("newL", lambda x: (x == "ORTH").any()))
        print(f"\nA/B on {os.path.basename(args.run_table)}:")
        print(f"  genes ORTH: deployed={int(g.old.sum()):,}  new={int(g.new.sum()):,}  "
              f"(delta {int(g.new.sum()-g.old.sum()):+,}; gained {int((~g.old&g.new).sum())}, lost {int((g.old&~g.new).sum())})")
        for name, e in [("NEAT1", "ENSG00000245532"), ("MALAT1", "ENSG00000251562")]:
            sub = r[r.base == e]
            print(f"  {name}: deployed={'ORTH' if (sub.oldL=='ORTH').any() else 'PARA'} -> new={'ORTH' if (sub.newL=='ORTH').any() else 'PARA'}")


if __name__ == "__main__":
    main()
