#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

# Make GPU executor importable when running from repo root
MODULES_DIR = Path(__file__).resolve().parent / "modules"
GPU_EXECUTOR_DIR = MODULES_DIR / "GPU_executor"
TOGA_MINI_DIR = MODULES_DIR / "TOGA_mini"
UTILS_DIR = MODULES_DIR / "utils"

from modules.GPU_executor.gpu_executor import ExecutorConfig, run_gpu_executor
from modules.TOGA_mini.toga_mini import run_toga_mini
from utils.chrom_sizes import write_chrom_sizes_from_2bit
from utils.ultimate_isoforms import collapse_to_ultimate_isoforms
from utils.toga_postprocess import write_rna_orthologous_regions
from utils.short_ncrna import write_short_ncrna_joblist, run_short_ncrna_scheduler
from utils.short_ncrna_bed import write_short_ncrna_bed


def parse_args():
    parser = argparse.ArgumentParser(description="CURIA pipeline")
    parser.add_argument("--ref-bed12", required=True, help="Reference annotation in BED12 format")
    parser.add_argument("--biomart-tsv", required=True, help="Biomart TSV (gene/biotype mapping)")
    parser.add_argument("--chain", required=True, help="Chain file (can be .gz)")
    parser.add_argument("--ref-2bit", required=True, help="Reference genome in .2bit")
    parser.add_argument("--query-2bit", required=True, help="Query genome in .2bit")
    parser.add_argument("--gpu-max-batch", type=int, default=128, help="Max GPU batch size")
    parser.add_argument("--short-max-workers", type=int, default=64, help="Max concurrent short ncRNA jobs")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Skip TOGA mini if output files already exist",
    )

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)
    return parser.parse_args()


def start_gpu_executor(args):
    ctx = mp.get_context("spawn")
    input_q = ctx.Queue()
    output_q = ctx.Queue()
    cfg = ExecutorConfig(max_batch=args.gpu_max_batch)
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
    ultimate_bed_path: Path,
    ultimate_meta_path: Path,
    chrom_sizes_path: Path,
    toga_regions_path: Path,
    toga_classification_path: Path,
    debug: bool,
) -> None:
    if debug and toga_regions_path.exists() and toga_classification_path.exists():
        print("# TOGA outputs exist; skipping TOGA mini.")
        return

    print("# Saving ultimate isoforms to")
    write_chrom_sizes_from_2bit(args.ref_2bit, chrom_sizes_path)  # noqa
    collapse_to_ultimate_isoforms(
        args.ref_bed12,
        args.biomart_tsv,
        ultimate_bed_path,  # noqa
        ultimate_meta_path,  # noqa
    )

    print("# Running TOGA mini...")
    run_toga_mini(
        args.chain,
        str(ultimate_bed_path),
        str(ultimate_meta_path),
        str(chrom_sizes_path),
        str(toga_regions_path),
        str(toga_classification_path),
        str(TOGA_MINI_DIR / "chain_classification_models" / "se_model.dat"),
        str(TOGA_MINI_DIR / "chain_classification_models" / "me_model.dat"),
    )


def main():
    args = parse_args()
    proc = None
    input_q = None
    output_q = None
    try:
        print("# Starting GPU executor...")
        proc, input_q, output_q = start_gpu_executor(args)
        # sanity check all modules
        # read and validate input
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ultimate_bed_path = output_dir / "ultimate_isoforms.bed"
        ultimate_meta_path = output_dir / "ultimate_isoforms.tsv"
        toga_regions_path = output_dir / "toga_orthologous_regions.tsv"
        toga_classification_path = output_dir / "toga_classification_table.tsv"
        rna_toga_regions_path = output_dir / "rna_orthologous_regions.tsv"
        short_joblist_path = output_dir / "short_ncRNA_joblist.txt"
        short_sqlite_path = output_dir / "short_ncRNA_results.sqlite"
        short_bed_path = output_dir / "intermediate_bed_fliles" / "short_rna_annotation_intermediate.bed"
        chrom_sizes_path = output_dir / "reference.chrom.sizes.tsv"

        run_toga_step(
            args,
            ultimate_bed_path,
            ultimate_meta_path,
            chrom_sizes_path,
            toga_regions_path,
            toga_classification_path,
            args.debug,
        )

        print("# Post-processing TOGA results for RNA biotypes...")
        write_rna_orthologous_regions(
            str(toga_regions_path),
            str(ultimate_meta_path),
            str(ultimate_bed_path),
            str(rna_toga_regions_path),
        )

        # process toga results, split into short and long jobs
        print("# Preparing short ncRNA joblist...")
        write_short_ncrna_joblist(
            str(rna_toga_regions_path),
            str(ultimate_bed_path),
            str(short_joblist_path),
            max_length=160,
        )

        print("# Running short ncRNA scheduler...")
        run_short_ncrna_scheduler(
            str(short_joblist_path),
            str(ultimate_bed_path),
            str(args.ref_2bit),
            str(args.query_2bit),
            input_q,
            output_q,
            str(short_sqlite_path),
            max_concurrent=args.short_max_workers,
            dump_tsv_path=str(output_dir / "temp_shortrna_results.tsv"),
        )

        print("# Writing short ncRNA BED9 annotation...")
        write_short_ncrna_bed(
            str(short_sqlite_path),
            str(short_bed_path),
        )

        # fill the jobs queue with long preprocessing jobs - execute
        # fill the jobs queue with long jobs - execute
        # process the results
        pass
    finally:
        print("Shutting down GPU executor...")
        shutdown_gpu_executor(proc, input_q)


if __name__ == "__main__":
    main()
