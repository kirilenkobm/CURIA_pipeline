#!/usr/bin/env python3
"""Export reproducible recurrent-core tables from completed CURIA results.

The export starts from accepted rows in every
``preprint_results/hg38_vs_*/island_alignment_results.tsv`` file, reconstructs
human-reference cores using the same overlap clustering as Figure 6, and writes:

* recurrent_gene_summary.tsv  -- one row per broad-set lncRNA gene locus;
* recurrent_core_summary.tsv  -- one row per broad-set reference core;
* recurrent_core_matches.tsv  -- one best accepted row per core and query assembly;
* distance_sensitivity.tsv    -- fixed species quorum, varying distance cutoff;
* quorum_sensitivity_d_le_0.02.tsv -- fixed distance cutoff, varying quorum;
* README.md                   -- definitions and provenance.

The broad set requires an accepted match (the matcher ceiling is d <= 0.10) in
at least 17 of 19 query assemblies and a mean reference-island length >=120 bp.
Strict subsets additionally require the stated per-species distance cutoff in
the stated number of assemblies; they do not use a mean-distance gate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import make_figures as mf

REPO = Path(__file__).resolve().parents[1]
ANNOT = REPO / "input_data" / "reference_annotation"
DEFAULT_OUT = REPO / "paper" / "recurrent_core_tables"
DISTANCE_CUTOFFS = (0.10, 0.05, 0.04, 0.03, 0.02, 0.01)
QUORUMS_AT_002 = (17, 15, 12, 9)


def _gene_loci() -> pd.DataFrame:
    """Build hg38 gene loci (versionless ENSG -> locus and biotype)."""
    bed = pd.read_csv(
        ANNOT / "hg38.primary_only.bed",
        sep="\t",
        header=None,
        usecols=[0, 1, 2, 3, 5],
        names=["chrom", "start", "end", "tx_id", "strand"],
        dtype={"start": int, "end": int},
    )
    meta = pd.read_csv(ANNOT / "hg38.primary_only.transcript_metadata.tsv", sep="\t")
    bed = bed.merge(meta, left_on="tx_id", right_on="transcript_id", how="left")
    gtypes = (
        pd.read_csv(
            ANNOT / "hg38_gene_names.txt",
            sep="\t",
            usecols=["Gene stable ID", "Gene type"],
        )
        .drop_duplicates("Gene stable ID")
        .set_index("Gene stable ID")["Gene type"]
    )
    bed["gene_bare"] = bed["gene_id"].str.split(".").str[0]
    biotype = bed["gene_bare"].map(gtypes)
    if "transcript_biotype" in bed.columns:
        biotype = biotype.fillna(bed["transcript_biotype"])
    bed["gene_biotype"] = biotype
    return (
        bed.groupby("gene_bare")
        .agg(
            chrom=("chrom", "first"),
            start=("start", "min"),
            end=("end", "max"),
            strand=("strand", "first"),
            biotype=("gene_biotype", "first"),
        )
        .reset_index()
    )


def _classify_proximity(
    lnc: pd.DataFrame, coding: pd.DataFrame, proximity_bp: int = 5000
) -> pd.DataFrame:
    """Classify each lncRNA gene locus relative to protein-coding loci."""
    rows = []
    for chrom, lnc_chrom in lnc.groupby("chrom"):
        coding_chrom = coding[coding["chrom"] == chrom]
        if coding_chrom.empty:
            rows.extend((gene, "intergenic", "") for gene in lnc_chrom["gene_bare"])
            continue
        ls = lnc_chrom["start"].to_numpy()[:, None]
        le = lnc_chrom["end"].to_numpy()[:, None]
        cs = coding_chrom["start"].to_numpy()[None, :]
        ce = coding_chrom["end"].to_numpy()[None, :]
        overlap = (ls < ce) & (cs < le)
        near = (ls < ce + proximity_bp) & (cs - proximity_bp < le)
        lstrand = lnc_chrom["strand"].to_numpy()
        cstrand = coding_chrom["strand"].to_numpy()
        cgenes = coding_chrom["gene_bare"].to_numpy()
        for i, (gene, strand) in enumerate(zip(lnc_chrom["gene_bare"], lstrand)):
            hits = np.where(overlap[i])[0]
            if len(hits) == 0:
                category = "near_coding_5kb" if near[i].any() else "intergenic"
                rows.append((gene, category, ""))
                continue
            hit_strands = cstrand[hits]
            antisense = (hit_strands != strand).any()
            sense = (hit_strands == strand).any()
            category = (
                "antisense+sense"
                if antisense and sense
                else ("antisense" if antisense else "sense_overlapping")
            )
            rows.append((gene, category, ",".join(cgenes[hits][:5])))
    return pd.DataFrame(
        rows, columns=["gene_bare", "overlap_category", "coding_partners"]
    )


def _reference_strands(results_dir: Path, assembly: str) -> dict[str, str]:
    bed = pd.read_csv(
        results_dir / f"hg38_vs_{assembly}" / "reference_union_transcripts.bed",
        sep="\t",
        header=None,
        usecols=[3, 5],
        names=["gene_id", "ref_strand"],
    )
    return dict(zip(bed["gene_id"], bed["ref_strand"]))


def _effective_aligned_length(row: pd.Series) -> int:
    """Recover matcher eff_nt from the exported per-chain alignment spans."""
    total = 0
    for chain in range(1, int(row["n_chains"]) + 1):
        values = [
            row.get(f"chain{chain}_ref_from"),
            row.get(f"chain{chain}_ref_to"),
            row.get(f"chain{chain}_q_from"),
            row.get(f"chain{chain}_q_to"),
        ]
        if any(pd.isna(value) for value in values):
            continue
        ref_from, ref_to, query_from, query_to = map(int, values)
        total += ((ref_to - ref_from) + (query_to - query_from)) // 2
    return total


def _annotations() -> tuple[pd.Series, pd.DataFrame]:
    names = (
        pd.read_csv(
            ANNOT / "hg38_gene_names.txt",
            sep="\t",
            usecols=["Gene stable ID", "Gene name"],
        )
        .drop_duplicates("Gene stable ID")
        .set_index("Gene stable ID")["Gene name"]
    )
    loci = _gene_loci()
    proximity = _classify_proximity(
        loci[loci["biotype"] == "lncRNA"], loci[loci["biotype"] == "protein_coding"]
    )
    return names, proximity


def _prepare_matches(results_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    rows, assemblies = mf._load_islands(results_dir)
    if rows is None:
        raise SystemExit(f"no island results under {results_dir}")
    rows = mf._cluster_cores(rows)
    rows["source_core_id"] = rows["core_id"]
    rows["gene_bare"] = rows["gene_id"].str.extract(r"(ENSG\d+)", expand=False)

    metadata = pd.read_csv(
        results_dir
        / f"hg38_vs_{assemblies[0]}"
        / "reference_union_transcripts_metadata.tsv",
        sep="\t",
    )
    biotype = dict(zip(metadata["transcript_id"], metadata["biotype"]))
    rows["biotype"] = rows["gene_id"].map(biotype)
    rows["ref_strand"] = rows["gene_id"].map(
        _reference_strands(results_dir, assemblies[0])
    )
    rows["species"] = rows["species"].map(lambda assembly: mf._meta(assembly)["name"])
    rows = rows.rename(columns={"species": "assembly", "diag_mmd": "embedding_sw_distance"})
    # _load_islands' species column held assembly IDs; restore both labels.
    rows["species"] = rows["assembly"]
    reverse_names = {mf._meta(a)["name"]: a for a in assemblies}
    rows["assembly"] = rows["species"].map(reverse_names)
    rows["species"] = rows["assembly"].map(lambda a: mf._meta(a)["name"])
    rows["effective_aligned_length"] = rows.apply(_effective_aligned_length, axis=1)

    # Public IDs are stable with respect to genomic ordering and do not expose
    # run-specific R0/R1 island numbering.
    core_order = (
        rows[["source_core_id", "gene_bare", "core_start"]]
        .drop_duplicates()
        .sort_values(["gene_bare", "core_start", "source_core_id"])
    )
    core_order["core_number"] = core_order.groupby("gene_bare").cumcount() + 1
    core_order["core_id"] = (
        core_order["gene_bare"] + "_core_" + core_order["core_number"].astype(str)
    )
    rows = rows.drop(columns="core_id").merge(
        core_order[["source_core_id", "core_id"]], on="source_core_id", how="left"
    )
    if rows["core_id"].isna().any() or core_order["core_id"].duplicated().any():
        raise RuntimeError("failed to construct unique public core IDs")
    return rows, assemblies


def build_exports(
    results_dir: Path, broad_quorum: int = 17, min_len: int = 120
) -> tuple[dict[str, pd.DataFrame], dict]:
    rows, assemblies = _prepare_matches(results_dir)
    best = (
        rows.sort_values("embedding_sw_distance")
        .drop_duplicates(["core_id", "assembly"], keep="first")
        .copy()
    )

    core = (
        best.groupby("core_id")
        .agg(
            gene_bare=("gene_bare", "first"),
            source_gene_id=("gene_id", "first"),
            biotype=("biotype", "first"),
            ref_chrom=("ref_chrom", "first"),
            ref_start=("ref_start", "min"),
            ref_end=("ref_end", "max"),
            ref_strand=("ref_strand", "first"),
            mean_reference_island_length=("ref_len", "mean"),
            n_species_accepted=("assembly", "nunique"),
            mean_dist=("embedding_sw_distance", "mean"),
            median_dist=("embedding_sw_distance", "median"),
            max_dist=("embedding_sw_distance", "max"),
            mean_aligned_length=("effective_aligned_length", "mean"),
            median_aligned_length=("effective_aligned_length", "median"),
            species_present=("assembly", lambda x: ",".join(sorted(set(x)))),
        )
        .reset_index()
    )
    core["reference_core_length"] = core["ref_end"] - core["ref_start"]
    for cutoff in (0.05, 0.04, 0.03, 0.02, 0.01):
        counts = (
            best[best["embedding_sw_distance"] <= cutoff]
            .groupby("core_id")["assembly"]
            .nunique()
        )
        core[f"n_species_d_le_{cutoff:.2f}"] = (
            core["core_id"].map(counts).fillna(0).astype(int)
        )

    names, proximity = _annotations()
    core["gene_name"] = core["gene_bare"].map(names).fillna("")
    core = core.merge(proximity, on="gene_bare", how="left")
    core["overlap_category"] = core["overlap_category"].fillna("unknown")
    core["coding_partners"] = core["coding_partners"].fillna("").replace("", ".")

    eligible = core[
        (core["biotype"] == "lncRNA")
        & (core["mean_reference_island_length"] >= min_len)
    ].copy()
    broad = eligible[eligible["n_species_accepted"] >= broad_quorum].copy()
    broad_ids = set(broad["core_id"])

    matches = best[best["core_id"].isin(broad_ids)].copy()
    matches["accepted"] = True
    for cutoff in (0.05, 0.04, 0.03, 0.02, 0.01):
        matches[f"d_le_{cutoff:.2f}"] = matches["embedding_sw_distance"] <= cutoff
    match_columns = [
        "core_id", "gene_bare", "species", "assembly", "ref_chrom", "ref_start",
        "ref_end", "ref_strand", "query_chrom", "query_start", "query_end",
        "embedding_sw_distance", "effective_aligned_length", "n_chains",
        "ref_island", "query_island", "accepted", "d_le_0.05", "d_le_0.04", "d_le_0.03",
        "d_le_0.02", "d_le_0.01",
    ]
    matches = matches[match_columns].sort_values(["gene_bare", "core_id", "assembly"])

    gene = (
        broad.groupby("gene_bare")
        .agg(
            gene_name=("gene_name", "first"),
            overlap_category=("overlap_category", "first"),
            coding_partners=("coding_partners", "first"),
            n_cores=("core_id", "count"),
            max_species_accepted=("n_species_accepted", "max"),
            mean_species_per_core=("n_species_accepted", "mean"),
            best_mean_dist=("mean_dist", "min"),
            best_median_dist=("median_dist", "min"),
            max_reference_core_length=("reference_core_length", "max"),
            max_species_d_le_0_05=("n_species_d_le_0.05", "max"),
            max_species_d_le_0_04=("n_species_d_le_0.04", "max"),
            max_species_d_le_0_03=("n_species_d_le_0.03", "max"),
            max_species_d_le_0_02=("n_species_d_le_0.02", "max"),
            max_species_d_le_0_01=("n_species_d_le_0.01", "max"),
        )
        .reset_index()
        .sort_values(["overlap_category", "best_mean_dist", "gene_bare"])
    )

    intergenic = set(proximity.loc[proximity["overlap_category"] == "intergenic", "gene_bare"])

    def sensitivity_row(cutoff: float, quorum: int, interpretation: str = "") -> dict:
        count_column = "n_species_accepted" if cutoff == 0.10 else f"n_species_d_le_{cutoff:.2f}"
        selected = eligible[eligible[count_column] >= quorum]
        loci = selected["gene_bare"].dropna().unique()
        return {
            "per_species_cutoff": cutoff,
            "required_species": quorum,
            "cores": len(selected),
            "lncRNA_loci": len(loci),
            "intergenic_loci": sum(gene_id in intergenic for gene_id in loci),
            "interpretation": interpretation,
        }

    labels = {
        0.10: "broad accepted set",
        0.05: "intermediate set",
        0.04: "intermediate-stringency set",
        0.03: "stringent set",
        0.02: "highly stringent set",
        0.01: "ultra-stringent set",
    }
    distance = pd.DataFrame(
        [sensitivity_row(cutoff, broad_quorum, labels[cutoff]) for cutoff in DISTANCE_CUTOFFS]
    )
    broad_loci = int(distance.loc[distance["per_species_cutoff"] == 0.10, "lncRNA_loci"].iloc[0])
    distance["percent_of_broad_loci"] = (100 * distance["lncRNA_loci"] / broad_loci).round(1)
    distance = distance[
        ["per_species_cutoff", "required_species", "cores", "lncRNA_loci",
         "percent_of_broad_loci", "intergenic_loci", "interpretation"]
    ]

    quorum = pd.DataFrame([sensitivity_row(0.02, q) for q in QUORUMS_AT_002]).drop(
        columns="interpretation"
    )

    core_columns = [
        "core_id", "gene_bare", "gene_name", "source_gene_id", "ref_chrom",
        "ref_start", "ref_end", "ref_strand", "reference_core_length",
        "mean_reference_island_length", "overlap_category", "coding_partners",
        "n_species_accepted", "n_species_d_le_0.05", "n_species_d_le_0.04", "n_species_d_le_0.03",
        "n_species_d_le_0.02", "n_species_d_le_0.01", "mean_dist", "median_dist",
        "max_dist", "mean_aligned_length", "median_aligned_length", "species_present",
    ]
    broad = broad[core_columns].sort_values(["gene_bare", "ref_start", "core_id"])

    float_rounding = {
        "mean_reference_island_length": 1,
        "mean_dist": 4,
        "median_dist": 4,
        "max_dist": 4,
        "mean_aligned_length": 1,
        "median_aligned_length": 1,
    }
    broad = broad.round(float_rounding)
    gene = gene.round(
        {"mean_species_per_core": 2, "best_mean_dist": 4, "best_median_dist": 4}
    )

    tables = {
        "recurrent_gene_summary.tsv": gene,
        "recurrent_core_summary.tsv": broad,
        "recurrent_core_matches.tsv": matches,
        "distance_sensitivity.tsv": distance,
        "quorum_sensitivity_d_le_0.02.tsv": quorum,
    }
    metadata = {
        "assemblies": assemblies,
        "broad_quorum": broad_quorum,
        "min_len": min_len,
        "n_broad_cores": len(broad),
        "n_broad_loci": gene["gene_bare"].nunique(),
        "n_broad_intergenic": int((gene["overlap_category"] == "intergenic").sum()),
    }
    return tables, metadata


def _readme(metadata: dict) -> str:
    assemblies = ", ".join(metadata["assemblies"])
    return f"""# Recurrent lncRNA core tables

These tables are generated from the accepted island-match rows in the completed
human-to-query CURIA runs. Run `make -C paper cores` from the repository root to
recreate them.

## Broad and strict sets

The broad set contains lncRNA cores with an accepted match in at least
{metadata['broad_quorum']} of {len(metadata['assemblies'])} query assemblies and a
mean reference-island length of at least {metadata['min_len']} bp. The pipeline
accepts island matches at embedding-SW distance `d <= 0.10`; consequently, the
broad criterion primarily measures recurrence at the pipeline acceptance ceiling.
Strict quorum columns count assemblies in which the best accepted match for that
core also satisfies the indicated per-species cutoff. Lower distance is better.

The current broad export contains {metadata['n_broad_cores']:,} cores in
{metadata['n_broad_loci']:,} annotated human lncRNA loci, including
{metadata['n_broad_intergenic']:,} loci more than 5 kb from and not overlapping a
protein-coding gene.

## Files

* `recurrent_gene_summary.tsv`: one row per broad-set versionless ENSG locus.
  `best_mean_dist` and `best_median_dist` are minima across that gene's cores;
  `mean_species_per_core` is the mean broad-acceptance count across its cores.
* `recurrent_core_summary.tsv`: one row per overlap-aggregated human-reference
  core, including coordinates, accepted and strict species counts, distance and
  aligned-length summaries, and the assemblies with an accepted match.
* `recurrent_core_matches.tsv`: one row per core x query assembly, retaining the
  lowest-distance accepted match when multiple accepted rows exist.
* `distance_sensitivity.tsv`: recurrent-core counts at a fixed >=17-species quorum
  while varying the per-species distance cutoff.
* `quorum_sensitivity_d_le_0.02.tsv`: counts at fixed `d <= 0.02` while varying
  the required number of query assemblies.

## Definitions

`core_id`
: Stable public identifier assigned by genomic order within a versionless ENSG
  locus. A gene can contain multiple non-overlapping cores.

Coordinates
: All reference and query coordinates are BED-style, 0-based, half-open. Reference
  coordinates use hg38. Query assembly is given in the `assembly` column.

Core reconstruction
: Accepted reference-island intervals from all species runs are pooled within each
  source gene and merged by overlap in hg38 coordinates, matching Figure 6.

`embedding_sw_distance`
: CURIA cosine-dotplot Smith-Waterman distance `d = 1/(1+s)`, where lower values
  indicate a stronger retained match. The saved match tables contain accepted rows
  only (`d <= 0.10`) and therefore have a restricted score range.

`effective_aligned_length`
: Reconstructed from the saved per-chain reference and query spans using the same
  integer formula as the matcher.

`overlap_category`
: Relationship of the complete human lncRNA gene locus to protein-coding gene
  loci: `intergenic`, `near_coding_5kb`, `antisense`, `sense_overlapping`, or
  `antisense+sense`.

Not stored in the completed island-match tables
: Query strand, nucleotide Smith-Waterman score, nucleotide aligned identity,
  source chain ID, and rank among rejected candidates were not persisted by the
  pipeline and are therefore not included or inferred here.

## Query assemblies

{assemblies}
"""


def write_exports(tables: dict[str, pd.DataFrame], metadata: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        table.to_csv(out_dir / filename, sep="\t", index=False)
        print(f"wrote {out_dir / filename} ({len(table):,} rows)")
    (out_dir / "README.md").write_text(_readme(metadata))
    print(f"wrote {out_dir / 'README.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO / "preprint_results")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--broad-quorum", type=int, default=17)
    parser.add_argument("--min-len", type=int, default=120)
    parser.add_argument(
        "--legacy-out",
        type=Path,
        default=REPO / "paper" / "lncRNAs_with_conserved_cores.tsv",
        help="also write the gene summary to the historical repository path",
    )
    args = parser.parse_args()
    tables, metadata = build_exports(args.results_dir, args.broad_quorum, args.min_len)
    write_exports(tables, metadata, args.out_dir)
    if args.legacy_out:
        tables["recurrent_gene_summary.tsv"].to_csv(args.legacy_out, sep="\t", index=False)
        print(f"wrote {args.legacy_out} ({len(tables['recurrent_gene_summary.tsv']):,} rows)")


if __name__ == "__main__":
    main()
