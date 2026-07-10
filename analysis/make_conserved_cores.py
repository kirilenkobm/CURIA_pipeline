#!/usr/bin/env python3
"""Export the list of lncRNAs carrying conserved cores across many mammals.

This is the supplementary table referenced by the paper (§ "Candidate conserved
regions absent from current annotations"): lncRNA loci whose embedding-similarity
"cores" recur across nearly all species in the panel. The table is large
(~10^3 rows), so the preprint only links to the repo copy produced here.

Definition (per the deprecated island_phylo_conservation.ipynb, cells 53-94),
generalized to however many hg38_vs_* pairs are present:
  * cluster reference islands into cross-species cores (genomic-overlap merge);
  * keep lncRNA cores detected in >= (N_species - 1) species, with mean reference
    length >= --min-len and mean match distance <= --max-dist;
  * aggregate per gene; classify each gene by proximity to protein-coding loci
    (intergenic / near-coding +/-5 kb / antisense / sense-overlapping).

The headline "~700 candidate intergenic lncRNAs" is the intergenic subset.

Reuses _load_islands / _cluster_cores / _best_per_core_species / SPECIES_META
from make_figures.py so the core definition matches Figure 6 exactly.

Usage:
    python analysis/make_conserved_cores.py \
        --results-dir rinalmo_version_outputs \
        --out preprint_results/lncRNAs_with_conserved_cores.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import make_figures as mf  # _load_islands, _cluster_cores, _best_per_core_species, PHYLO/META

REPO = Path(__file__).resolve().parents[1]
ANNOT = REPO / "input_data" / "reference_annotation"


# --------------------------------------------------------------------------
# Coding-proximity classification (hg38 gene loci)
# --------------------------------------------------------------------------
def _gene_loci() -> pd.DataFrame:
    """Build hg38 gene loci (bare ENSG -> chrom/start/end/strand/biotype) from the
    reference annotation, so lncRNA loci can be classified by proximity to
    protein-coding genes."""
    bed = pd.read_csv(ANNOT / "hg38.primary_only.bed", sep="\t", header=None,
                      usecols=[0, 1, 2, 3, 5],
                      names=["chrom", "start", "end", "tx_id", "strand"],
                      dtype={"start": int, "end": int})
    meta = pd.read_csv(ANNOT / "hg38.primary_only.transcript_metadata.tsv", sep="\t")
    bed = bed.merge(meta, left_on="tx_id", right_on="transcript_id", how="left")
    gtypes = (pd.read_csv(ANNOT / "hg38_gene_names.txt", sep="\t",
                          usecols=["Gene stable ID", "Gene type"])
              .drop_duplicates("Gene stable ID")
              .set_index("Gene stable ID")["Gene type"])
    bed["gene_bare"] = bed["gene_id"].str.split(".").str[0]
    bt = bed["gene_bare"].map(gtypes)
    if "transcript_biotype" in bed.columns:
        bt = bt.fillna(bed["transcript_biotype"])
    bed["gene_biotype"] = bt
    return (bed.groupby("gene_bare")
            .agg(chrom=("chrom", "first"), start=("start", "min"),
                 end=("end", "max"), strand=("strand", "first"),
                 biotype=("gene_biotype", "first"))
            .reset_index())


def _classify_proximity(lnc: pd.DataFrame, coding: pd.DataFrame,
                        proximity_bp: int = 5000) -> pd.DataFrame:
    """Per lncRNA gene: intergenic / near_coding_5kb / antisense /
    sense_overlapping / antisense+sense, plus overlapping coding partners."""
    rows = []
    for chrom, lnc_ch in lnc.groupby("chrom"):
        cod_ch = coding[coding["chrom"] == chrom]
        if cod_ch.empty:
            rows += [(g, "intergenic", "") for g in lnc_ch["gene_bare"]]
            continue
        ls = lnc_ch["start"].to_numpy()[:, None]; le = lnc_ch["end"].to_numpy()[:, None]
        cs = cod_ch["start"].to_numpy()[None, :]; ce = cod_ch["end"].to_numpy()[None, :]
        ov = (ls < ce) & (cs < le)                                   # direct overlap
        near = (ls < ce + proximity_bp) & (cs - proximity_bp < le)   # within +/-5 kb
        lstr = lnc_ch["strand"].to_numpy(); cstr = cod_ch["strand"].to_numpy()
        cgene = cod_ch["gene_bare"].to_numpy()
        for i, (g, s) in enumerate(zip(lnc_ch["gene_bare"].to_numpy(), lstr)):
            hits = np.where(ov[i])[0]
            if len(hits) == 0:
                cat = "near_coding_5kb" if near[i].any() else "intergenic"
                rows.append((g, cat, ""))
            else:
                hs = cstr[hits]
                anti, sense = (hs != s).any(), (hs == s).any()
                cat = "antisense+sense" if (anti and sense) else ("antisense" if anti else "sense_overlapping")
                rows.append((g, cat, ",".join(cgene[hits][:5])))
    return pd.DataFrame(rows, columns=["gene_bare", "overlap_category", "coding_partners"])


# --------------------------------------------------------------------------
def build(results_dir: Path, min_species: int | None, min_len: int,
          max_dist: float) -> tuple[pd.DataFrame, dict]:
    df, present = mf._load_islands(results_dir)
    if df is None:
        raise SystemExit(f"no island results under {results_dir}")
    n_sp = len(present)
    if min_species is None:
        min_species = max(1, n_sp - 1)

    df = mf._cluster_cores(df)
    best = mf._best_per_core_species(df)  # best (min diag_mmd) per (core, species)

    # biotype per union transcript (island gene_id == metadata transcript_id)
    meta_tsv = results_dir / f"hg38_vs_{present[0]}" / "reference_union_transcripts_metadata.tsv"
    bmeta = pd.read_csv(meta_tsv, sep="\t")
    biotype_map = dict(zip(bmeta["transcript_id"], bmeta["biotype"]))

    # per-core conservation stats
    core = (best.groupby("core_id")
            .agg(gene_id=("gene_id", "first"),
                 n_species=("species", "nunique"),
                 mean_dist=("diag_mmd", "mean"), median_dist=("diag_mmd", "median"),
                 mean_ref_len=("ref_len", "mean"))
            .reset_index())
    core["biotype"] = core["gene_id"].map(biotype_map)
    core["gene_bare"] = core["gene_id"].str.extract(r"(ENSG\d+)", expand=False)

    # qualifying lncRNA cores
    q = core[(core["biotype"] == "lncRNA") &
             (core["n_species"] >= min_species) &
             (core["mean_ref_len"] >= min_len) &
             (core["mean_dist"] <= max_dist)].copy()

    genes = (q.groupby("gene_bare")
             .agg(n_cores=("core_id", "count"), max_species=("n_species", "max"),
                  mean_species=("n_species", "mean"),
                  best_mean_dist=("mean_dist", "min"),
                  best_median_dist=("median_dist", "min"),
                  max_core_len=("mean_ref_len", "max"))
             .reset_index())

    # gene names + coding-proximity
    loci = _gene_loci()
    gtypes = (pd.read_csv(ANNOT / "hg38_gene_names.txt", sep="\t",
                          usecols=["Gene stable ID", "Gene name"])
              .drop_duplicates("Gene stable ID")
              .set_index("Gene stable ID")["Gene name"])
    genes["gene_name"] = genes["gene_bare"].map(gtypes).fillna("")
    prox = _classify_proximity(loci[loci["biotype"] == "lncRNA"],
                               loci[loci["biotype"] == "protein_coding"])
    genes = genes.merge(prox, on="gene_bare", how="left")
    genes["overlap_category"] = genes["overlap_category"].fillna("unknown")
    genes = genes.sort_values(["overlap_category", "best_mean_dist"])

    cols = ["gene_bare", "gene_name", "overlap_category", "n_cores", "max_species",
            "mean_species", "best_mean_dist", "best_median_dist", "max_core_len",
            "coding_partners"]
    genes = genes[cols].round({"mean_species": 2, "best_mean_dist": 4,
                               "best_median_dist": 4, "max_core_len": 0})
    meta = {"n_species": n_sp, "species": present, "min_species": min_species,
            "min_len": min_len, "max_dist": max_dist,
            "n_total": len(genes),
            "n_intergenic": int((genes["overlap_category"] == "intergenic").sum())}
    return genes, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path,
                    default=REPO / "rinalmo_version_outputs")
    ap.add_argument("--out", type=Path,
                    default=REPO / "preprint_results" / "lncRNAs_with_conserved_cores.tsv")
    ap.add_argument("--min-species", type=int, default=None,
                    help="min species with the core (default: N_species - 1)")
    ap.add_argument("--min-len", type=int, default=120, help="min mean core length (bp)")
    ap.add_argument("--max-dist", type=float, default=0.10,
                    help="max mean cosine-SW distance (RiNALMo diag_mmd is capped at 0.1)")
    args = ap.parse_args()

    genes, meta = build(args.results_dir, args.min_species, args.min_len, args.max_dist)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# lncRNAs with conserved cores across mammals\n"
              f"# panel: {meta['n_species']} species ({', '.join(meta['species'])})\n"
              f"# criteria: lncRNA cores in >= {meta['min_species']} species, "
              f"mean length >= {meta['min_len']} bp, mean cosine-SW distance <= {meta['max_dist']}\n"
              f"# total genes: {meta['n_total']}  |  intergenic (>5 kb from coding): {meta['n_intergenic']}\n")
    with open(args.out, "w") as fh:
        fh.write(header)
        genes.to_csv(fh, sep="\t", index=False)
    print(f"wrote {args.out}")
    print(f"  {meta['n_total']} conserved-core lncRNA genes "
          f"({meta['n_intergenic']} intergenic) across {meta['n_species']} species "
          f"(>= {meta['min_species']} required)")


if __name__ == "__main__":
    main()
