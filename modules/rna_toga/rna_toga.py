#!/usr/bin/env python3
"""RNA TOGA pipeline (simplified logreg classifier)."""
import sys
from collections import defaultdict
from datetime import datetime as dt
import argparse
import os
import json

import pandas as pd
import numpy as np

from pyrion.ops import find_intersections
from pyrion.ops import chains_to_arrays
from pyrion.ops import transcripts_to_arrays
from pyrion.ops import intervals_to_array

from pyrion.ops import split_genome_alignment
from pyrion.ops import merge_transcript_intervals
from pyrion.ops.chains import project_intervals_through_chain_strict

from pyrion.core.intervals import AnnotatedIntervalSet
from pyrion.core.intervals import RegionType

from pyrion import read_chain_file, read_gene_data, read_bed12_file
from typing import Tuple

ORTH = "ORTH"
PARA = "PARA"
SPAN = "SPAN"
PROCESSED_PSEUDOGENES = "P_PGENES"

ORTHOLOGY_THRESHOLD = 0.5
SPANNING_SCORE = -1.0
PROCESSED_PSEUDOGENE_SCORE = -2.0

SCRIPT_LOCATION = os.path.dirname(__file__)
LOGREG_MODEL_PATH = os.path.join(SCRIPT_LOCATION, "model.json")

FEATURE_COLUMNS = (
    "chain_id",
    "transcript_id",
    "gl_exo",
    "exlen_to_qlen",
    "synteny",
    "flank_cov",
    "exon_perc",
    "ex_num",
)


def ensure_parent_directory(file_path: str) -> None:
    """Ensure the parent directory for a file path exists."""
    directory_path = os.path.dirname(file_path) or "."
    os.makedirs(directory_path, exist_ok=True)


def compute_region_lengths(region_set: AnnotatedIntervalSet) -> Tuple[int, int]:
    rt = region_set.region_types
    iv = region_set.intervals

    def total(rt_code):
        return int(np.sum(iv[rt == rt_code, 1] - iv[rt == rt_code, 0]))

    cds_len = total(RegionType.CDS)
    flank_len = total(RegionType.FLANK_LEFT) + total(RegionType.FLANK_RIGHT)
    exon_len = cds_len

    return flank_len, exon_len


def write_orthologous_regions(pairs_to_q_intervals, chains, transcripts, output_file):
    f = open(output_file, "w")
    f.write("transcript_id\tchain_id\tregion\ttranscript_strand\tchain_strand\n")
    for (t_id, c_id), (start, end) in pairs_to_q_intervals.items():
        transcript_obj = transcripts.get_by_id(t_id)
        chain_obj = chains.get_by_chain_id(c_id)
        q_chrom = chain_obj.q_chrom

        transcript_strand = transcript_obj.strand
        chain_q_strand = chain_obj.q_strand

        # Normalize coordinates: ensure start < end
        # Pyrion's chain projection can return reversed coords for reverse-strand alignments
        # The strand information is preserved separately in chain_q_strand
        if start > end:
            start, end = end, start

        region = f"{q_chrom}:{start}-{end}"
        f.write(f"{t_id}\t{c_id}\t{region}\t{transcript_strand}\t{chain_q_strand}\n")

    f.close()


def map_orthologs(ortholog_map, chains, transcripts):
    pairs_to_q_intervals = {}

    for num, (chain_id, transcript_ids) in enumerate(ortholog_map.items()):
        ts = [transcripts.get_by_id(t_id) for t_id in transcript_ids]
        transcript_arr, transcript_id_arr = transcripts_to_arrays(ts)
        chain = chains.get_by_chain_id(chain_id)
        projected_list = project_intervals_through_chain_strict(
            transcript_arr, chain.blocks, chain.q_strand
        )
        for t_id, elem in zip(transcript_id_arr, projected_list):
            # Filter out failed liftovers (intervals that are [0, 0])
            if elem[0][0] == 0 and elem[0][1] == 0:
                # Skip this transcript - liftover failed
                continue
            pairs_to_q_intervals[(t_id, chain_id)] = elem[0]
    return pairs_to_q_intervals


def assign_label(pred, threshold=ORTHOLOGY_THRESHOLD):
    if pred == SPANNING_SCORE:
        return SPAN
    elif pred == PROCESSED_PSEUDOGENE_SCORE:
        return PROCESSED_PSEUDOGENES
    elif pred < threshold:
        return PARA
    else:
        return ORTH


def _load_model(model_path: str) -> dict:
    """Load either a logistic-regression or a GBM model. Dispatches on `model_type`."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, "r") as f:
        model = json.load(f)
    model_type = model.get("model_type", "logistic_regression")
    common = {
        "type": model_type,
        "threshold": float(model.get("threshold", ORTHOLOGY_THRESHOLD)),
        "biotype_thresholds": model.get("biotype_thresholds", {}) or {},
    }
    if model_type == "gbm":
        required = {"features", "mean", "scale", "init", "lr", "trees"}
        missing = required - set(model.keys())
        if missing:
            raise ValueError(f"Missing keys in GBM model: {sorted(missing)}")
        common["gbm"] = model
        return common
    coeffs = model.get("coefficients", {})
    required = {"synteny_log1p", "gl_exo", "flank_cov", "intercept"}
    missing = required - set(coeffs.keys())
    if missing:
        raise ValueError(f"Missing coefficients in model: {sorted(missing)}")
    common["coeffs"] = coeffs
    return common


def _build_feature_matrix(df: pd.DataFrame, features) -> np.ndarray:
    """Assemble the feature matrix in the model's declared order (synteny is log1p'd)."""
    cols = []
    for name in features:
        if name == "synteny_log1p":
            cols.append(np.log1p(df["synteny"].fillna(0.0).to_numpy(dtype=float)))
        else:
            cols.append(df[name].fillna(0.0).to_numpy(dtype=float))
    return np.column_stack(cols)


def _logreg_predict_proba(df: pd.DataFrame, coeffs: dict) -> np.ndarray:
    synteny_log1p = np.log1p(df["synteny"].fillna(0.0))
    gl_exo = df["gl_exo"].fillna(0.0)
    flank_cov = df["flank_cov"].fillna(0.0)
    score = (
        coeffs["synteny_log1p"] * synteny_log1p
        + coeffs["gl_exo"] * gl_exo
        + coeffs["flank_cov"] * flank_cov
        + coeffs["intercept"]
    )
    return 1.0 / (1.0 + np.exp(-score))


def _gbm_predict_proba(df: pd.DataFrame, gbm: dict) -> np.ndarray:
    """Score a GBM exported by train_lncrna_gbm.py — numpy-vectorised, no sklearn needed."""
    X = _build_feature_matrix(df, gbm["features"])
    Z = (X - np.asarray(gbm["mean"])) / np.asarray(gbm["scale"])
    raw = np.full(len(df), float(gbm["init"]), dtype=float)
    lr = float(gbm["lr"])
    for tr in gbm["trees"]:
        f = np.asarray(tr["f"]); thr = np.asarray(tr["t"])
        left = np.asarray(tr["l"]); right = np.asarray(tr["r"]); val = np.asarray(tr["v"])
        node = np.zeros(len(df), dtype=int)
        while True:
            is_leaf = left[node] == -1
            if is_leaf.all():
                break
            act = ~is_leaf
            cur = node[act]
            go_left = Z[act, f[cur]] <= thr[cur]
            node[act] = np.where(go_left, left[cur], right[cur])
        raw += lr * val[node]
    return 1.0 / (1.0 + np.exp(-raw))


def _predict_proba(df: pd.DataFrame, model: dict) -> np.ndarray:
    if model["type"] == "gbm":
        return _gbm_predict_proba(df, model["gbm"])
    return _logreg_predict_proba(df, model["coeffs"])


def classify_table(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    df = df.copy()
    df["single_exon"] = (df["ex_num"] == 1).astype(int)
    for col in ("gl_exo", "exlen_to_qlen", "synteny", "flank_cov", "exon_perc", "ex_num"):
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    model = _load_model(model_path)

    df["pred"] = np.nan
    df.loc[(df["exon_perc"] == 0) & (df["synteny"] > 1), "pred"] = SPANNING_SCORE

    mask = df["pred"].isna()
    if mask.any():
        df.loc[mask, "pred"] = _predict_proba(df.loc[mask], model)

    processed_pseudogenes_mask = (
        (df["single_exon"] == 0)
        & (df["synteny"] == 1)
        & (df["exlen_to_qlen"] > 0.95)
        & (df["pred"] < ORTHOLOGY_THRESHOLD)
        & (df["exon_perc"] > 0.65)
    )

    df.loc[processed_pseudogenes_mask, "pred"] = PROCESSED_PSEUDOGENE_SCORE

    # per-biotype threshold: lncRNA is relaxed (high-recall candidate generation), others keep global
    global_thr = model["threshold"]
    bt_thr = model["biotype_thresholds"]
    if bt_thr and "biotype" in df.columns:
        thresholds = df["biotype"].map(lambda b: float(bt_thr.get(b, global_thr))).to_numpy()
    else:
        thresholds = np.full(len(df), global_thr, dtype=float)
    df["label"] = [assign_label(p, t) for p, t in zip(df["pred"].to_numpy(), thresholds)]
    return df


def extract_features(all_chain_ids, chain_to_ts_intersection, chains, transcripts, reference_chrom_sizes,
                     giant_chains_transcripts_mapping, split_giant_chains_by_id):
    features = []
    total_units = len(all_chain_ids)
    progress_step = max(1, total_units // 20) if total_units > 0 else 1

    for num, chain_ids_tup in enumerate(all_chain_ids):
        if num % progress_step == 0:
            print(f"extract_features: processing unit {num}/{total_units}...")
        if chain_ids_tup[1] == -1:
            transcript_ov_data = chain_to_ts_intersection[chain_ids_tup[0]]
            chain = chains.get_by_chain_id(chain_ids_tup[0])
            intersected_transcripts = [transcripts.get_by_id(str(t)) for t, _ in transcript_ov_data]
        else:
            intersected_transcript_ids = giant_chains_transcripts_mapping[chain_ids_tup]
            intersected_transcripts = [transcripts.get_by_id(str(t)) for t in intersected_transcript_ids]
            chain = split_giant_chains_by_id[chain_ids_tup]

        synteny = len(set(transcripts.get_gene_by_transcript_id(t.id) for t in intersected_transcripts))
        chain_blocks_len = chain.aligned_length()
        chain_q_len = chain.q_length()

        if chain_q_len < 100:
            continue

        exon_intervals = merge_transcript_intervals(intersected_transcripts)
        exon_intervals_np_array = intervals_to_array(exon_intervals)
        chain_blocks_np_array = chain.blocks_in_target()

        intersect_all_exons_all_chain_blocks = find_intersections(exon_intervals_np_array, chain_blocks_np_array)
        overlapped_exon_bases = sum(
            np.fromiter((overlap for pairs in intersect_all_exons_all_chain_blocks.values() for _, overlap in pairs),
                        dtype=int)
        )
        global_exo = overlapped_exon_bases / chain_blocks_len if chain_blocks_len > 0 else 0

        if global_exo > 0.95:
            continue

        exon_length_to_query_length = overlapped_exon_bases / chain_q_len if chain_q_len > 0 else 0

        for t in intersected_transcripts:
            region_data = t.get_annotated_regions(chrom_sizes=reference_chrom_sizes, flank_size=10000)
            region_types = region_data.region_types
            intervals = region_data.intervals

            flank_len, exon_len = compute_region_lengths(region_data)
            local_overlaps = find_intersections(intervals, chain_blocks_np_array, region_types)

            overlap_sums = defaultdict(int)

            for rtype, overlaps in local_overlaps.items():
                if rtype in (RegionType.FLANK_LEFT, RegionType.FLANK_RIGHT):
                    key = RegionType.FLANK_LEFT  # collapsed into one
                elif rtype == RegionType.CDS:
                    key = rtype
                else:
                    continue
                overlap_sums[key] += sum(overlap_len for _, overlap_len in overlaps)

            total_cds_overlap = overlap_sums[RegionType.CDS]
            total_flank_overlap = overlap_sums[RegionType.FLANK_LEFT]

            flank_feature = total_flank_overlap / flank_len if flank_len > 0 else 0
            exon_perc = total_cds_overlap / exon_len if exon_len > 0 else 0

            features.append((
                chain_ids_tup[0],
                t.id,
                global_exo,
                exon_length_to_query_length,
                synteny,
                flank_feature,
                exon_perc,
                len(t.blocks),
            ))

    return features


def split_giant_chains(chain_to_ts_intersection, chains, transcripts):
    giant_chain_ids = set([c.chain_id for c in chains if c.t_length() > 2_000_000])
    giant_chain_mapping, split_giant_chains_by_id = {}, {}

    for chain_id in giant_chain_ids:
        gigachain = chains.get_by_chain_id(chain_id)
        its = [transcripts.get_by_id(str(t)) for t, _ in chain_to_ts_intersection[chain_id]]

        subchains_and_transcripts = split_genome_alignment(gigachain, its)
        subchains = subchains_and_transcripts[0]
        trans_attached = subchains_and_transcripts[1]

        for s in subchains:
            split_giant_chains_by_id[(chain_id, s.child_id)] = s

        for k, v in trans_attached.items():
            giant_chain_mapping[(chain_id, k)] = v

    chain_ids_normal = [(k, -1) for k in chain_to_ts_intersection.keys() if k not in giant_chain_ids]
    chain_ids_split = list(giant_chain_mapping.keys())
    all_chain_ids = chain_ids_split + chain_ids_normal

    return all_chain_ids, giant_chain_mapping, split_giant_chains_by_id


def intersect_chains_and_transcripts(chains, transcripts, reference_chromosomes, chain_t_chromosomes):
    chain_to_ts_intersection = defaultdict(list)
    for chrom in set(reference_chromosomes) & set(chain_t_chromosomes):
        chrom_chains = chains.get_by_target_chrom(chrom)
        chrom_transcripts = transcripts.get_by_chrom(chrom)

        if not chrom_chains or not chrom_transcripts:
            continue

        chain_arr, chain_id_arr = chains_to_arrays(chrom_chains)
        transcript_arr, transcript_id_arr = transcripts_to_arrays(chrom_transcripts)

        intersections = find_intersections(chain_arr, transcript_arr, chain_id_arr, transcript_id_arr)
        chain_to_ts_intersection.update(intersections)

    return chain_to_ts_intersection


def read_chrom_sizes(file_path: str) -> dict:
    f = open(file_path, "r")
    lines_parts = [x.split("\t") for x in f.readlines() if x and not x.startswith("#")]
    f.close()
    chrom_sizes = {x[0]: int(x[1]) for x in lines_parts}
    return chrom_sizes


def parse_args():
    app = argparse.ArgumentParser()
    app.add_argument("chain_file", help="Path to genome alignment file in chain format")
    app.add_argument("transcript_file", help="Path to transcripts bed12 file")
    app.add_argument(
        "isoforms_file",
        help="Isoforms mapping TSV with columns: gene_id, transcript_id",
    )
    app.add_argument("reference_chrom_sizes", help="Path to reference chromosome sizes file (tab-separated: chrom_name\tsize)")
    app.add_argument("out_orthologous_regions_mapping", help="Output with orthologous regions")
    app.add_argument("out_classification_table", help="Classification table with predictions")
    app.add_argument(
        "--model-path",
        default=LOGREG_MODEL_PATH,
        help="Path to the orthology model JSON (logreg model.json or a GBM gbm_model.json). "
             f"Default: {LOGREG_MODEL_PATH}",
    )
    app.add_argument(
        "--projection-mode",
        choices=["orthologous", "best-chain"],
        default="orthologous",
        help="How to pick the query search space. 'orthologous' (default): only "
             "chains the ncRNA-orthology classifier labels ORTH. 'best-chain': keep "
             "all ORTH calls as-is AND add the top-K score-ranked chains for "
             "non-ORTH (PARA/SPAN) transcripts -- rescues deep-divergence false "
             "negatives (e.g. marsupial/opossum) without changing ORTH loci.",
    )
    app.add_argument(
        "--best-chain-topk",
        type=int,
        default=1,
        help="best-chain mode: number of top-scoring chains kept per transcript "
             "(lowest chain id = highest score). Default 1.",
    )

    if len(sys.argv) == 1:
        app.print_help()
        sys.exit(0)

    return app.parse_args()


def _time_delta(t0: dt):
    return dt.now() - t0


def _read_biotype_map(metadata_file: str) -> dict:
    """Read transcript_id -> biotype from the union metadata TSV, if it carries a biotype column."""
    try:
        meta = pd.read_csv(metadata_file, sep="\t", dtype=str)
    except Exception:
        return {}
    bcol = next((c for c in meta.columns if c.lower() in ("biotype", "transcript_biotype")), None)
    if bcol is None or "transcript_id" not in meta.columns:
        return {}
    return dict(zip(meta["transcript_id"], meta[bcol]))


def run_toga_mini(
    chain_file: str,
    transcript_file: str,
    isoforms_file: str,
    reference_chrom_sizes: str,
    out_orthologous_regions_mapping: str,
    out_classification_table: str,
    model_path: str = LOGREG_MODEL_PATH,
    projection_mode: str = "orthologous",
    best_chain_topk: int = 1,
):
    ensure_parent_directory(out_orthologous_regions_mapping)
    ensure_parent_directory(out_classification_table)

    if not os.path.isfile(model_path):
        print(f"Error: orthology model not found at {model_path}")
        sys.exit(1)
    print(f"Using orthology model: {model_path}")

    t0 = dt.now()
    print(f"{t0}: Reading input files...")
    chains = read_chain_file(chain_file, 25_000)
    print(f"Parsed: {len(chains)} from {chain_file} in {_time_delta(t0)}")
    transcripts = read_bed12_file(transcript_file)
    gene_data = read_gene_data(
        isoforms_file,
        gene_column="gene_id",
        transcript_id_column="transcript_id",
    )
    transcripts.bind_gene_data(gene_data)
    print(f"Parsed {len(transcripts)} transcripts from {transcript_file} in {_time_delta(t0)}")

    reference_chromosomes = transcripts.get_all_chromosomes()
    reference_chrom_sizes = read_chrom_sizes(reference_chrom_sizes)
    print(f" Found lengths for {len(reference_chromosomes)} reference chromosomes")

    chain_t_chromosomes = chains.get_reference_chromosomes()
    print(f"{_time_delta(t0)}: Input files read.")

    shared_chromosomes = set(reference_chromosomes) & set(chain_t_chromosomes)
    num_chains_shared = sum(len(chains.get_by_target_chrom(chrom)) for chrom in shared_chromosomes if chains.get_by_target_chrom(chrom))
    num_transcripts_shared = sum(len(transcripts.get_by_chrom(chrom)) for chrom in shared_chromosomes if transcripts.get_by_chrom(chrom))
    print(f"{_time_delta(t0)}: Intersecting {num_transcripts_shared} transcripts vs {num_chains_shared} chains across {len(shared_chromosomes)} chromosomes...")

    chain_to_ts_intersection = intersect_chains_and_transcripts(chains, transcripts, reference_chromosomes, chain_t_chromosomes)
    total_intersections = sum(len(v) for v in chain_to_ts_intersection.values())
    print(f"{_time_delta(t0)}: Intersected {total_intersections} chain–transcript pairs.")

    print(f"{_time_delta(t0)}: Splitting giant chains...")
    all_chain_ids, giant_chains_transcripts_mapping, split_giant_chains_by_id = split_giant_chains(chain_to_ts_intersection, chains, transcripts)
    num_units_total = len(all_chain_ids)
    num_units_split = sum(1 for cid in all_chain_ids if cid[1] != -1)
    num_units_normal = num_units_total - num_units_split
    print(f"{_time_delta(t0)}: Giant chains split. Units: {num_units_total} (split: {num_units_split}, normal: {num_units_normal}).")

    print(f"{_time_delta(t0)}: Extracting features...")
    features = extract_features(all_chain_ids, chain_to_ts_intersection, chains, transcripts, reference_chrom_sizes, giant_chains_transcripts_mapping, split_giant_chains_by_id)
    df = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    print(f"{_time_delta(t0)}: Features extracted. {len(features)} entries in total")

    # attach transcript biotype (enables per-biotype thresholds, e.g. relaxed lncRNA);
    # the metadata file is the same union metadata passed as isoforms_file.
    biotype_map = _read_biotype_map(isoforms_file)
    if biotype_map:
        df["biotype"] = df["transcript_id"].map(biotype_map)

    print(f"{_time_delta(t0)}: Classifying table...")
    classified_df = classify_table(df, model_path)
    print(f"{_time_delta(t0)}: Table classified.")

    print(f"{_time_delta(t0)}: Writing classification table...")
    classified_df.to_csv(out_classification_table, index=False)
    print(f"{_time_delta(t0)}: Classification table written.")

    if projection_mode == "best-chain":
        # Hybrid relaxed search space: keep every ORTH call exactly as the
        # classifier assigns it (identical to the default mode), and ADD the top-K
        # score-ranked chains ONLY for transcripts the model did not call ORTH
        # (PARA / SPAN / ...). This rescues deep-divergence false negatives
        # (e.g. MALAT1 labeled PARA despite strong chain coverage) without
        # perturbing confidently-orthologous loci. Chain ids are score-ordered
        # (lowest = best), so head(K) after ascending sort takes the best chains.
        # Genuinely unaligned loci still fail projection later -> nothing fabricated.
        orth = classified_df[classified_df["label"] == ORTH]
        orth_tids = set(orth["transcript_id"])
        rest = classified_df[~classified_df["transcript_id"].isin(orth_tids)].copy()
        rest["_cid_int"] = pd.to_numeric(rest["chain_id"], errors="coerce")
        rest = (
            rest.sort_values("_cid_int", kind="stable")
            .groupby("transcript_id", sort=False)
            .head(max(1, best_chain_topk))
        )
        print(f"{_time_delta(t0)}: best-chain mode: {len(orth_tids)} ORTH kept as-is"
              f" + top-{best_chain_topk} chain(s) for {rest['transcript_id'].nunique()}"
              f" non-ORTH transcripts...")
        combined = pd.concat([orth, rest], ignore_index=True)
        ortholog_map = (
            combined.groupby("chain_id")["transcript_id"].apply(list).to_dict()
        )
    else:
        print(f"{_time_delta(t0)}: Making orthologous regions mapping...")
        ortholog_map = (
            classified_df[classified_df["label"] == ORTH]
            .groupby("chain_id")["transcript_id"]
            .apply(list)
            .to_dict()
        )
    pairs_to_q_intervals = map_orthologs(ortholog_map, chains, transcripts)
    write_orthologous_regions(pairs_to_q_intervals, chains, transcripts, out_orthologous_regions_mapping)
    print(f"{_time_delta(t0)}: Orthologous regions mapping written.")


def main():
    args = parse_args()
    run_toga_mini(
        args.chain_file,
        args.transcript_file,
        args.isoforms_file,
        args.reference_chrom_sizes,
        args.out_orthologous_regions_mapping,
        args.out_classification_table,
        args.model_path,
        projection_mode=args.projection_mode,
        best_chain_topk=args.best_chain_topk,
    )


if __name__ == "__main__":
    main()
