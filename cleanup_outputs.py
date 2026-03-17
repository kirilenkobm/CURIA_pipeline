#!/usr/bin/env python3
"""
Standalone script to cleanup and reorganize existing CURIA pipeline outputs.

Usage:
    ./cleanup_outputs.py <output_dir>
    ./cleanup_outputs.py smoke_test_output/
"""

import argparse
import sys
from pathlib import Path

# Make modules importable
MODULES_DIR = Path(__file__).resolve().parent / "modules"
sys.path.insert(0, str(MODULES_DIR.parent))

from modules.utils.cleanup_outputs import cleanup_and_reorganize


def main():
    parser = argparse.ArgumentParser(
        description="Cleanup and reorganize CURIA pipeline outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    ./cleanup_outputs.py smoke_test_output/
    python cleanup_outputs.py my_analysis_results/

This will:
  - Move annotations to query_annotation/
  - Move TOGA results to toga_results/
  - Rename union transcripts for clarity
  - Remove temporary files (SQLite DBs, joblists, technical files)
        """,
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="CURIA output directory to cleanup",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"ERROR: Directory does not exist: {output_dir}", file=sys.stderr)
        sys.exit(1)

    if not output_dir.is_dir():
        print(f"ERROR: Not a directory: {output_dir}", file=sys.stderr)
        sys.exit(1)

    # Check if this looks like a CURIA output directory
    expected_markers = [
        output_dir / "island_alignment_results.tsv",
        output_dir / "preprocessed_reference.json",
        output_dir / "mappings",
    ]

    if not any(marker.exists() for marker in expected_markers):
        print(f"WARNING: {output_dir} doesn't look like a CURIA output directory", file=sys.stderr)
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    try:
        cleanup_and_reorganize(output_dir, verbose=not args.quiet)
        if not args.quiet:
            print(f"\n✓ Successfully cleaned up: {output_dir}")
    except Exception as e:
        print(f"ERROR: Cleanup failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
