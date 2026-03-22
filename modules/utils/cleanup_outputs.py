#!/usr/bin/env python3
"""
Cleanup temporary CURIA pipeline files.

Removes SQLite databases, joblists, and other intermediate artefacts
that are no longer needed once the final outputs have been written.
"""

import shutil
from pathlib import Path
from typing import List


def cleanup_temp_files(output_dir: Path, verbose: bool = True) -> None:
    """Remove temporary/intermediate files from a finished pipeline run."""
    if verbose:
        print("\n# === Cleaning up temporary files ===")

    removals: List[tuple[Path, str]] = [
        (output_dir / "intermediate_sqlite_dbs", "SQLite databases"),
        (output_dir / "joblists", "joblists"),
        (output_dir / "temp_shortrna_results.tsv", "temporary short RNA results"),
        (output_dir / "toga_results" / "reference_chrom_sizes.tsv",
         "reference chromosome sizes (technical)"),
    ]

    for legacy in ("intermediate_bed_files", "toga_mini_results"):
        p = output_dir / legacy
        if p.exists():
            removals.append((p, f"legacy {legacy}"))

    query_islands_json = output_dir / "mappings" / "query_islands.json"
    if query_islands_json.exists():
        removals.append((query_islands_json, "redundant query_islands.json"))

    for path, desc in removals:
        if path.exists():
            if verbose:
                rel = path.relative_to(output_dir) if output_dir in path.parents else path.name
                print(f"  Removing {desc}: {rel}")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        elif verbose and path.parent.exists():
            print(f"  [SKIP] {desc} not found")

    if verbose:
        print("# Cleanup complete!")
