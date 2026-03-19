#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import sys
import time
import json
from pathlib import Path

# Make GPU executor importable when running from repo root
MODULES_DIR = Path(__file__).resolve().parent / "modules"
GPU_EXECUTOR_DIR = MODULES_DIR / "GPU_executor"
RNA_TOGA_DIR = MODULES_DIR / "rna_toga"
UTILS_DIR = MODULES_DIR / "utils"

from pyrion import read_chain_file

from modules.GPU_executor.gpu_executor import ExecutorConfig, run_gpu_executor
from modules.utils.chrom_sizes import write_chrom_sizes_from_2bit
from modules.utils.output_paths import OutputPaths
from modules.converters.union_transcript import collapse_to_union_transcripts
from modules.converters.short_ncrna_bed import write_short_ncrna_bed
from modules.converters.island_alignment_bed import write_island_alignment_beds
from modules.converters.islands_bed import write_reference_islands_bed, write_query_islands_bed
from modules.pipeline.toga_postprocess import write_rna_orthologous_regions
from modules.pipeline.short_ncrna import write_short_ncrna_joblist, run_short_ncrna_scheduler
from modules.pipeline.reference_islands_liftover import liftover_reference_islands
from modules.pipeline.query_islands_scanner import write_query_islands_joblist, run_query_islands_scanner
from modules.pipeline.reference_islands_scanner import write_reference_islands_joblist, run_reference_islands_scanner
from modules.pipeline.island_alignment import (
    write_island_alignment_joblist,
    run_island_alignment_scheduler,
)
from modules.utils.cleanup_outputs import cleanup_and_reorganize
from modules.utils.input_validation import validate_all_inputs, ValidationError


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
    parser.add_argument("--cpu-max-workers", type=int, default=128, help="Max concurrent CPU workers for all pipeline steps")
    parser.add_argument("--gpu-logger", action="store_true", help="Enable GPU utilization logging every 3s")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip pipeline steps if their output files already exist (useful for debugging/resuming)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip automatic cleanup and reorganization of outputs (keep all intermediate files)",
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
        print("# [SKIP] TOGA outputs exist, skipping RNA TOGA.")
        return

    print(f"# Saving union transcripts to {paths.union_bed}")
    write_chrom_sizes_from_2bit(args.ref_2bit, paths.chrom_sizes)  # noqa
    collapse_to_union_transcripts(
        args.ref_bed12,
        args.biomart_tsv,
        paths.union_bed,  # noqa
        paths.union_meta,  # noqa
        union_to_isoforms_path=str(paths.union_to_isoforms),
    )

    print("# Running RNA TOGA...")
    toga_script = RNA_TOGA_DIR / "rna_toga.py"

    toga_cmd = [
        sys.executable,
        str(toga_script),
        args.chain,
        str(paths.union_bed),
        str(paths.union_meta),
        str(paths.chrom_sizes),
        str(paths.toga_regions),
        str(paths.toga_classification),
    ]

    print("RNA TOGA called with:")
    print(f"  chain: {args.chain}")
    print(f"  bed: {paths.union_bed}")
    print(f"  metadata: {paths.union_meta}")
    print(f"  chrom_sizes: {paths.chrom_sizes}")
    print(f"  output_regions: {paths.toga_regions}")
    print(f"  output_classification: {paths.toga_classification}")
    print(f"\nRunning command: {' '.join(toga_cmd)}\n")

    # Run TOGA in a separate subprocess to isolate native runtimes and avoid segmentation faults.
    # Set PYTHONUNBUFFERED to force immediate output flushing
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        toga_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    # Stream output line by line with immediate flushing
    for line in process.stdout:
        print(f"[TOGA] {line}", end="", flush=True)

    # Wait for process to complete
    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"RNA TOGA failed with exit code {return_code}")

    print("# RNA TOGA completed successfully.")


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

    # Create joblist from non-short union transcripts
    ref_islands_joblist = paths.output_dir / "joblists" / "reference_islands_joblist.txt"
    if skip_completed and ref_islands_joblist.exists():
        print(f"# [SKIP] Reference islands joblist exists.")
    else:
        print("# Preparing reference islands joblist...")
        write_reference_islands_joblist(
            str(paths.rna_toga_regions),
            str(paths.union_bed),
            str(paths.short_joblist),
            str(ref_islands_joblist),
        )

    # Scan transcripts for islands
    if skip_completed and ref_islands_json.exists():
        print(f"# [SKIP] Reference islands results exist.")
    else:
        print("# Running reference islands scanner...")
        # Ensure directories exist before async operations
        paths.intermediate_sqlite_dir.mkdir(parents=True, exist_ok=True)
        ref_islands_json.parent.mkdir(parents=True, exist_ok=True)

        logreg_model_path = MODULES_DIR / "logreg_signal_noise" / "logreg_noise_model.json"
        run_reference_islands_scanner(
            str(ref_islands_joblist),
            str(args.ref_2bit),
            input_q,
            output_q,
            str(paths.reference_islands_sqlite),
            str(logreg_model_path),
            str(ref_islands_json),
            max_concurrent=args.cpu_max_workers,
        )

    return ref_islands_json


def main():
    t0 = time.time()
    args = parse_args()
    proc = None
    input_q = None

    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\n# Interrupted by user (Ctrl+C). Shutting down gracefully...")
        shutdown_gpu_executor(proc, input_q)
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Validate inputs before starting GPU executor (fail fast)
        try:
            validate_all_inputs(
                args.ref_bed12,
                args.biomart_tsv,
                args.chain,
                args.ref_2bit,
                args.query_2bit,
                args.ref_preprocessed,
            )
        except ValidationError as e:
            print(f"\n# INPUT VALIDATION FAILED:\n{e}\n", file=sys.stderr)
            sys.exit(1)

        print("# Starting GPU executor...")
        proc, input_q, output_q = start_gpu_executor(args)
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
                str(paths.union_meta),
                str(paths.union_bed),
                str(paths.rna_toga_regions),
            )

        # Prepare a short ncRNA joblist
        if args.skip_completed and paths.short_joblist.exists():
            print("# [SKIP] Short ncRNA joblist exists, skipping preparation.")
        else:
            print("# Preparing short ncRNA joblist...")
            write_short_ncrna_joblist(
                str(paths.rna_toga_regions),
                str(paths.union_bed),
                str(paths.short_joblist),
                max_length=160,
            )

        # Run a short ncRNA scheduler
        # Check for final output (short_bed), not temporary sqlite
        if args.skip_completed and paths.short_bed.exists():
            print("# [SKIP] Short ncRNA BED exists (final output), skipping scheduler.")
        else:
            print("# Running short ncRNA scheduler...")
            run_short_ncrna_scheduler(
                str(paths.short_joblist),
                str(paths.union_bed),
                str(args.ref_2bit),
                str(args.query_2bit),
                input_q,
                output_q,
                str(paths.short_sqlite),
                max_concurrent=args.cpu_max_workers,
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

        # Generate BED12 for reference islands
        if args.skip_completed and paths.reference_islands_bed.exists():
            print("# [SKIP] Reference islands BED exists, skipping generation.")
        else:
            print("# Writing reference islands BED12...")
            paths.intermediate_bed_dir.mkdir(parents=True, exist_ok=True)
            write_reference_islands_bed(
                str(ref_islands_json),
                str(paths.reference_islands_bed),
            )

        # Load genome alignment chains — reused across pipeline steps.
        # Currently TOGA runs as a subprocess and loads its own copy;
        # TODO: refactor TOGA to accept pre-loaded chains and avoid double-loading.
        print("# Loading genome alignment chains...")
        genome_chains = read_chain_file(args.chain, 25_000)
        print(f"# Loaded {len(genome_chains)} chains")

        # Step 2.5: Liftover reference islands → targeted query regions
        # Replaces the legacy merge_query_regions (full transcript → full query locus)
        # with flanked island liftover for dramatically reduced scanning workload
        if args.skip_completed and paths.query_regions_clusters.exists():
            print("# [SKIP] Query regions clusters exist, skipping island liftover.")
        else:
            print("# === Step 2.5: Liftover reference islands to query regions ===")
            liftover_reference_islands(
                chain_path=args.chain,
                ref_islands_json_path=str(ref_islands_json),
                rna_regions_path=paths.rna_toga_regions,
                short_joblist_path=paths.short_joblist,
                clusters_json_path=paths.query_regions_clusters,
                union_to_query_path=paths.union_to_query,
                chains=genome_chains,
            )

        # Write short ncRNA BED (only if we ran the scheduler above)
        if not (args.skip_completed and paths.short_bed.exists()):
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
        # Check if output JSON exists (final output, not just sqlite)
        skip_query_islands = False
        if args.skip_completed and paths.query_islands_json.exists():
            try:
                with open(paths.query_islands_json) as f:
                    data = json.load(f)
                    if len(data) > 0:
                        print(f"# [SKIP] Query islands JSON exists with {len(data)} entries, skipping scanner.")
                        skip_query_islands = True
            except (json.JSONDecodeError, Exception):
                pass

        if not skip_query_islands:
            print("# Running query islands scanner...")
            run_query_islands_scanner(
                str(paths.query_islands_joblist),
                str(args.query_2bit),
                input_q,
                output_q,
                str(paths.query_islands_sqlite),
                str(MODULES_DIR / "logreg_signal_noise" / "logreg_noise_model.json"),
                max_concurrent=args.cpu_max_workers,
                output_json_path=str(paths.query_islands_json),
            )

        # Generate BED12 for query islands
        if args.skip_completed and paths.query_islands_bed.exists():
            print("# [SKIP] Query islands BED exists, skipping generation.")
        else:
            print("# Writing query islands BED12...")
            paths.intermediate_bed_dir.mkdir(parents=True, exist_ok=True)
            write_query_islands_bed(
                str(paths.query_islands_json),
                str(paths.union_to_query),
                str(paths.query_islands_bed),
            )

        # Step 4: Island alignment via windowed MMD
        if args.skip_completed and paths.island_alignment_joblist.exists():
            print("# [SKIP] Island alignment joblist exists, skipping preparation.")
        else:
            print("# === Step 4: Preparing island alignment joblist ===")
            write_island_alignment_joblist(
                str(ref_islands_json),
                str(paths.union_to_query),
                str(paths.query_islands_json),
                str(paths.island_alignment_joblist),
            )

        # Run island alignment scheduler
        # Check if file exists AND has more than just header (indicating completion)
        skip_island_alignment = False
        if args.skip_completed and paths.island_alignment_results.exists():
            with open(paths.island_alignment_results) as f:
                lines = sum(1 for _ in f)
                if lines > 1:  # More than just header
                    print(f"# [SKIP] Island alignment results exist ({lines} lines), skipping alignment.")
                    skip_island_alignment = True

        if not skip_island_alignment:
            print("# Running island alignment...")
            run_island_alignment_scheduler(
                str(paths.island_alignment_joblist),
                str(args.ref_2bit),
                str(args.query_2bit),
                str(ref_islands_json),
                str(paths.union_to_query),
                str(paths.query_islands_json),
                input_q,
                output_q,
                str(paths.island_alignment_sqlite),
                str(paths.island_alignment_results),
                max_concurrent=args.cpu_max_workers,
            )

        print(f"# Pipeline completed! Island alignment results: {paths.island_alignment_results}")

        # Generate BED12 files from island alignments
        if args.skip_completed and paths.aligned_islands_ref_bed.exists() and paths.aligned_islands_query_bed.exists():
            print("# [SKIP] Aligned islands BED files exist, skipping BED generation.")
        else:
            print("# Writing aligned islands BED12 files...")
            paths.intermediate_bed_dir.mkdir(parents=True, exist_ok=True)
            write_island_alignment_beds(
                str(paths.island_alignment_results),
                str(ref_islands_json),
                str(paths.query_islands_json),
                str(paths.aligned_islands_ref_bed),
                str(paths.aligned_islands_query_bed),
            )

        # Cleanup and reorganize outputs (unless --no-cleanup)
        if not args.no_cleanup:
            cleanup_and_reorganize(output_dir, verbose=True)
        else:
            print("# [SKIP] Cleanup disabled via --no-cleanup flag")

    finally:
        elapsed = time.time() - t0
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"# Total pipeline wall time: {hours:02d}:{minutes:02d}:{seconds:02d} ({elapsed:.1f}s)")
        print("Shutting down GPU executor...")
        shutdown_gpu_executor(proc, input_q)


if __name__ == "__main__":
    main()
