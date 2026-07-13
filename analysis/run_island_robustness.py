#!/usr/bin/env python3
"""Run the complete island-match robustness analysis with one command.

This is the canonical entry point for the two complementary null tests and the
summary plot. The older scripts remain importable and independently runnable so
that existing commands and cached outputs are not broken.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
import time
from argparse import Namespace
from pathlib import Path

import island_null_test
import island_searchspace_null
import plot_searchspace_null


class _Tee:
    """Write progress to the terminal and an immediately flushed report file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _banner(number: int, total: int, label: str) -> None:
    print(f"\n{'=' * 72}\n[{number}/{total}] {label}\n{'=' * 72}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run all island robustness tests and rebuild the summary plot."
    )
    ap.add_argument("--pair", required=True,
                    help="pair output directory, e.g. preprint_results/hg38_vs_mm39")
    ap.add_argument("--ref-2bit", required=True)
    ap.add_argument("--query-2bit", required=True)
    ap.add_argument("--n", type=int, default=None,
                    help="use the same sample size for both tests")
    ap.add_argument("--specificity-n", type=int, default=250,
                    help="sample size for cross-locus/composition tests (default: 250)")
    ap.add_argument("--searchspace-n", type=int, default=400,
                    help="sample size for within-locus test (default: 400)")
    ap.add_argument("--max-windows", type=int, default=100,
                    help="maximum same-locus windows per island (default: 100)")
    ap.add_argument("--model", default="rinalmo")
    ap.add_argument("--gpu-max-batch", type=int, default=64)
    ap.add_argument("--gpu-max-tokens", type=int, default=8192)
    ap.add_argument("--output",
                    help="text report path (default: analysis/scratch/<pair>.txt)")
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()

    pair = Path(args.pair)
    required = [pair / "island_alignment_results.tsv",
                pair / "mappings" / "union_to_query.json",
                pair / "mappings" / "query_regions_clusters.json",
                Path(args.ref_2bit), Path(args.query_2bit)]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        ap.error("required input(s) not found:\n  " + "\n  ".join(missing))

    specificity_n = args.n if args.n is not None else args.specificity_n
    searchspace_n = args.n if args.n is not None else args.searchspace_n
    common = dict(pair=str(pair), ref_2bit=args.ref_2bit,
                  query_2bit=args.query_2bit, model=args.model,
                  gpu_max_batch=args.gpu_max_batch,
                  gpu_max_tokens=args.gpu_max_tokens)
    total_steps = 2 if args.skip_plot else 3
    all_started = time.monotonic()
    output = (Path(args.output) if args.output else
              Path(__file__).resolve().parent / "scratch" / f"{pair.name}.txt")
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", buffering=1) as report:
        with contextlib.redirect_stdout(_Tee(sys.stdout, report)):
            print(f"# live report: {output}", flush=True)
            _banner(1, total_steps,
                    f"Cross-locus specificity + composition null (n={specificity_n})")
            asyncio.run(island_null_test._amain(Namespace(**common, n=specificity_n)))

            _banner(2, total_steps,
                    f"Within-locus positional null (n={searchspace_n}, "
                    f"max_windows={args.max_windows})")
            asyncio.run(island_searchspace_null._amain(
                Namespace(**common, n=searchspace_n, max_windows=args.max_windows)
            ))

            if not args.skip_plot:
                _banner(3, total_steps, "Rebuild cached-results summary plot")
                plot_searchspace_null.main()

            elapsed = time.monotonic() - all_started
            print(f"\n# complete: {pair.name} in {elapsed/60:.1f} minutes", flush=True)
            print(f"# report: {output}", flush=True)


if __name__ == "__main__":
    main()
