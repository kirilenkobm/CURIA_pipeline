#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import signal
import sys
from pathlib import Path

# Make GPU executor importable when running from repo root
MODULES_DIR = Path(__file__).resolve().parent / "modules"
GPU_EXECUTOR_DIR = MODULES_DIR / "GPU_executor"
TOGA_MINI_DIR = MODULES_DIR / "TOGA_mini"
UTILS_DIR = MODULES_DIR / "utils"

from modules.GPU_executor.gpu_executor import ExecutorConfig, run_gpu_executor
from modules.TOGA_mini.toga_mini import run_toga_mini
from modules.utils.chrom_sizes import write_chrom_sizes_from_2bit
from modules.utils.ultimate_isoforms import collapse_to_ultimate_isoforms
from modules.utils.toga_postprocess import write_rna_orthologous_regions
from modules.utils.short_ncrna import write_short_ncrna_joblist, run_short_ncrna_scheduler
from modules.utils.short_ncrna_bed import write_short_ncrna_bed
from modules.utils.merge_query_regions import merge_query_regions
from modules.utils.query_islands_scanner import write_query_islands_joblist, run_query_islands_scanner
from modules.utils.reference_islands_scanner import write_reference_islands_joblist, run_reference_islands_scanner
from modules.utils.output_paths import OutputPaths


def parse_args():
    parser = argparse.ArgumentParser(description="CURIA pipeline")
    parser.add_argument("--ref-bed12", required=True, help="Reference annotation in BED12 format")
    parser.add_argument("--biomart-tsv", required=True, help="Biomart TSV (gene/biotype mapping)")
    parser.add_argument("--chain", required=True, help="Chain file (can be .gz)")
    parser.add_argument("--ref-2bit", required=True, help="Reference genome in .2bit")
    parser.add_argument("--query-2bit", required=True, help="Query genome in .2bit")
    parser.add_argument(
        "--ref-preprocessed",
        help="Path to preprocessed reference lncRNA data (optional)",
    )
    parser.add_argument("--gpu-max-batch", type=int, default=160, help="Max GPU batch size")
    parser.add_argument("--gpu-min-batch", type=int, default=32, help="Min GPU batch size before timeout")
    parser.add_argument("--short-max-workers", type=int, default=128, help="Max concurrent short ncRNA jobs")
    parser.add_argument("--gpu-logger", action="store_true", help="Enable GPU utilization logging every 3s")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip pipeline steps if their output files already exist (useful for debugging/resuming)",
    )
    parser.add_argument(
        "--test-cap-jobs",
        type=int,
        help="Process no more than N jobs per step (for quick testing)",
    )

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)
    return parser.parse_args()


def start_gpu_executor(args):
    ctx = mp.get_context("spawn")
    input_q = ctx.Queue()
    output_q = ctx.Queue()
    cfg = ExecutorConfig(
        max_batch=args.gpu_max_batch,
        min_batch=args.gpu_min_batch,
        enable_logging=args.gpu_logger,
    )
    proc = ctx.Process(target=run_gpu_executor, args=(input_q, output_q, cfg), name="gpu_executor")
    proc.daemon = True
    proc.start()
    return proc, input_q, output_q


def shutdown_gpu_executor(proc, input_q):
    if input_q is not None:
        input_q.put(None)
    if proc is not None:
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()


def run_toga_step(
    args,
    paths: OutputPaths,
    skip_completed: bool,
) -> None:
    if skip_completed and paths.toga_regions.exists() and paths.toga_classification.exists():
        print("# [SKIP] TOGA outputs exist, skipping TOGA mini.")
        return

    print(f"# Saving ultimate isoforms to {paths.ultimate_bed}")
    write_chrom_sizes_from_2bit(args.ref_2bit, paths.chrom_sizes)  # noqa
    collapse_to_ultimate_isoforms(
        args.ref_bed12,
        args.biomart_tsv,
        paths.ultimate_bed,  # noqa
        paths.ultimate_meta,  # noqa
        ultimate_to_isoforms_path=str(paths.ultimate_to_isoforms),
    )

    print("# Running TOGA mini...")
    se_model_path = TOGA_MINI_DIR / "chain_classification_models" / "se_model.dat"
    me_model_path = TOGA_MINI_DIR / "chain_classification_models" / "me_model.dat"

    print(f"Using models:\nSE: {se_model_path}\nME: {me_model_path}")

    run_toga_mini(
        args.chain,
        str(paths.ultimate_bed),
        str(paths.ultimate_meta),
        str(paths.chrom_sizes),
        str(paths.toga_regions),
        str(paths.toga_classification),
        str(se_model_path),
        str(me_model_path),
    )


def run_reference_islands_step(
    args,
    paths: OutputPaths,
    input_q,
    output_q,
    skip_completed: bool,
) -> Path:
    """
    Run Step 2: Reference transcript island scanning.

    This preprocessing step only needs to run once per reference genome.
    Results are saved to --ref-preprocessed path and can be reused across query species.
    """
    ref_islands_json = paths.preprocessed_reference

    if skip_completed and ref_islands_json.exists():
        print(f"# [SKIP] Reference islands JSON exists at {ref_islands_json}, skipping Step 2.")
        return ref_islands_json

    print("# === Step 2: Scanning reference transcripts for functional islands ===")

    # Create joblist from non-short ultimate isoforms
    ref_islands_joblist = paths.output_dir / "joblists" / "reference_islands_joblist.txt"
    if skip_completed and ref_islands_joblist.exists():
        print(f"# [SKIP] Reference islands joblist exists.")
    else:
        print("# Preparing reference islands joblist...")
        write_reference_islands_joblist(
            str(paths.rna_toga_regions),
            str(paths.ultimate_bed),
            str(paths.short_joblist),
            str(ref_islands_joblist),
        )

    # Scan transcripts for islands
    ref_islands_sqlite = paths.intermediate_sqlite_dir / "reference_islands.db"
    if skip_completed and ref_islands_json.exists():
        print(f"# [SKIP] Reference islands results exist.")
    else:
        print("# Running reference islands scanner...")
        # Ensure directories exist before async operations
        paths.intermediate_sqlite_dir.mkdir(parents=True, exist_ok=True)
        ref_islands_json.parent.mkdir(parents=True, exist_ok=True)

        logreg_model_path = MODULES_DIR / "logreg_signal_noise" / "logreg_noise_model.pkl"
        run_reference_islands_scanner(
            str(ref_islands_joblist),
            str(args.ref_2bit),
            input_q,
            output_q,
            str(ref_islands_sqlite),
            str(logreg_model_path),
            str(ref_islands_json),
            max_concurrent=args.short_max_workers,
            test_cap_jobs=args.test_cap_jobs,
        )

    return ref_islands_json


def main():
    args = parse_args()
    proc = None
    input_q = None
    output_q = None

    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\n# Interrupted by user (Ctrl+C). Shutting down gracefully...")
        shutdown_gpu_executor(proc, input_q)
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        print("# Starting GPU executor...")
        proc, input_q, output_q = start_gpu_executor(args)
        # sanity check all modules
        # read and validate input
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ref_preprocessed_override = None
        if args.ref_preprocessed:
            ref_preprocessed_override = Path(args.ref_preprocessed)
        paths = OutputPaths(output_dir, ref_preprocessed_override=ref_preprocessed_override)

        # Ensure mappings, intermediate sqlite, toga results, and joblists directories exist
        paths.mappings_dir.mkdir(parents=True, exist_ok=True)
        paths.intermediate_sqlite_dir.mkdir(parents=True, exist_ok=True)
        paths.toga_results_dir.mkdir(parents=True, exist_ok=True)
        paths.joblists_dir.mkdir(parents=True, exist_ok=True)

        run_toga_step(
            args,
            paths,
            args.skip_completed,
        )

        # Post-process TOGA results
        if args.skip_completed and paths.rna_toga_regions.exists():
            print("# [SKIP] RNA TOGA regions exist, skipping post-processing.")
        else:
            print("# Post-processing TOGA results for RNA biotypes...")
            write_rna_orthologous_regions(
                str(paths.toga_regions),
                str(paths.ultimate_meta),
                str(paths.ultimate_bed),
                str(paths.rna_toga_regions),
            )

        # Prepare a short ncRNA joblist
        if args.skip_completed and paths.short_joblist.exists():
            print("# [SKIP] Short ncRNA joblist exists, skipping preparation.")
        else:
            print("# Preparing short ncRNA joblist...")
            write_short_ncrna_joblist(
                str(paths.rna_toga_regions),
                str(paths.ultimate_bed),
                str(paths.short_joblist),
                max_length=160,
            )

        # Run a short ncRNA scheduler
        if args.skip_completed and paths.short_sqlite.exists():
            print("# [SKIP] Short ncRNA sqlite exists, skipping scheduler.")
        else:
            print("# Running short ncRNA scheduler...")
            run_short_ncrna_scheduler(
                str(paths.short_joblist),
                str(paths.ultimate_bed),
                str(args.ref_2bit),
                str(args.query_2bit),
                input_q,
                output_q,
                str(paths.short_sqlite),
                max_concurrent=args.short_max_workers,
                dump_tsv_path=str(output_dir / "temp_shortrna_results.tsv"),
            )

        # Step 2: Reference transcript island scanning (reusable across query species)
        ref_islands_json = run_reference_islands_step(
            args,
            paths,
            input_q,
            output_q,
            args.skip_completed,
        )
        print(f"# Reference islands data: {ref_islands_json}")

        # Merge query regions
        if args.skip_completed and paths.query_regions_clusters.exists():
            print("# [SKIP] Query regions clusters exist, skipping merge.")
        else:
            print("# Preparing merged long ncRNA query regions...")
            merge_query_regions(
                paths.rna_toga_regions,
                paths.short_joblist,
                paths.merged_query_mapping,
                paths.long_jobs,
                paths.query_regions_clusters,
                ultimate_to_query_path=paths.ultimate_to_query,
            )

        # Write short ncRNA BED
        if args.skip_completed and paths.short_bed.exists():
            print("# [SKIP] Short ncRNA BED exists, skipping write.")
        else:
            print("# Writing short ncRNA BED9 annotation...")
            write_short_ncrna_bed(
                str(paths.short_sqlite),
                str(paths.short_bed),
            )

        # Step 3: Prepare query islands joblist (filtered by Step 2 results)
        if args.skip_completed and paths.query_islands_joblist.exists():
            print("# [SKIP] Query islands joblist exists, skipping preparation.")
        else:
            print("# === Step 3: Preparing query islands scanner joblist ===")
            write_query_islands_joblist(
                str(paths.query_regions_clusters),
                str(ref_islands_json),
                str(paths.query_islands_joblist),
            )

        # Run query islands scanner
        if args.skip_completed and paths.query_islands_sqlite.exists():
            print("# [SKIP] Query islands sqlite exists, skipping scanner.")
        else:
            print("# Running query islands scanner...")
            run_query_islands_scanner(
                str(paths.query_islands_joblist),
                str(args.query_2bit),
                input_q,
                output_q,
                str(paths.query_islands_sqlite),
                str(MODULES_DIR / "logreg_signal_noise" / "logreg_noise_model.pkl"),
                max_concurrent=args.short_max_workers,
                output_json_path=str(paths.query_islands_json),
            )

        # fill the jobs queue with long preprocessing jobs - execute
        # fill the jobs queue with long jobs - execute
        # process the results
    finally:
        print("Shutting down GPU executor...")
        shutdown_gpu_executor(proc, input_q)


if __name__ == "__main__":
    main()
