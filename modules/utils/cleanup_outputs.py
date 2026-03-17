#!/usr/bin/env python3
"""
Cleanup and reorganize CURIA pipeline outputs into a user-friendly structure.

Removes temporary files (SQLite DBs, joblists, technical files) and reorganizes
essential outputs into a cleaner directory structure.
"""

import shutil
from pathlib import Path
from typing import List


def cleanup_and_reorganize(output_dir: Path, verbose: bool = True) -> None:
    """
    Cleanup temporary files and reorganize outputs into final structure.

    Final structure:
        output_dir/
        ├── query_annotation/
        │   ├── short_ncRNA.bed              # Short ncRNA annotations (≤160bp)
        │   ├── lncRNA_islands.bed           # Aligned lncRNA islands
        │   └── all_islands.bed              # Reference + query islands (for QC)
        ├── island_alignments.tsv            # Island alignment results with scores
        ├── preprocessed_reference.json      # Reusable reference islands
        ├── mappings/
        │   ├── union_to_isoforms.json       # Union transcript → original isoforms
        │   └── union_to_query.json          # Union transcript → query regions
        └── toga_results/
            ├── rna_orthologous_regions.tsv  # RNA TOGA orthology calls
            └── toga_orthologous_regions.tsv # Original TOGA output

    Args:
        output_dir: Pipeline output directory
        verbose: Print cleanup progress
    """

    if verbose:
        print("\n# === Cleaning up and reorganizing outputs ===")

    # Define paths
    annotation_dir = output_dir / "query_annotation"
    toga_dir = output_dir / "toga_results"

    intermediate_bed = output_dir / "intermediate_bed_files"
    intermediate_sqlite = output_dir / "intermediate_sqlite_dbs"
    joblists = output_dir / "joblists"
    toga_mini = output_dir / "toga_mini_results"
    mappings = output_dir / "mappings"

    # Create final directories
    annotation_dir.mkdir(exist_ok=True)
    toga_dir.mkdir(exist_ok=True)

    # --- Move essential outputs ---
    moves: List[tuple[Path, Path, str]] = [
        # Annotations
        (intermediate_bed / "short_rna_annotation_intermediate.bed",
         annotation_dir / "short_ncRNA.bed",
         "Short ncRNA annotations"),
        (intermediate_bed / "aligned_islands_query.bed",
         annotation_dir / "lncRNA_islands.bed",
         "Aligned lncRNA islands"),
        (intermediate_bed / "reference_islands.bed",
         annotation_dir / "reference_islands.bed",
         "Reference islands (QC)"),
        (intermediate_bed / "query_islands.bed",
         annotation_dir / "query_islands.bed",
         "Query islands (QC)"),

        # TOGA results
        (toga_mini / "rna_orthologous_regions.tsv",
         toga_dir / "rna_orthologous_regions.tsv",
         "RNA orthology regions"),
        (toga_mini / "toga_orthologous_regions.tsv",
         toga_dir / "toga_orthologous_regions.tsv",
         "Original TOGA output"),
    ]

    for src, dst, desc in moves:
        if src.exists():
            if verbose:
                print(f"  Moving {desc}: {src.name} → {dst.relative_to(output_dir)}")
            shutil.move(str(src), str(dst))
        elif verbose:
            print(f"  [SKIP] {desc} not found: {src.name}")

    # Keep mappings in place but rename union_transcripts files for clarity
    union_bed = output_dir / "union_transcripts.bed"
    union_meta = output_dir / "union_transcripts_metadata.tsv"
    if union_bed.exists():
        new_union_bed = output_dir / "reference_union_transcripts.bed"
        if verbose:
            print(f"  Renaming: union_transcripts.bed → reference_union_transcripts.bed")
        shutil.move(str(union_bed), str(new_union_bed))
    if union_meta.exists():
        new_union_meta = output_dir / "reference_union_transcripts_metadata.tsv"
        if verbose:
            print(f"  Renaming: union_transcripts_metadata.tsv → reference_union_transcripts_metadata.tsv")
        shutil.move(str(union_meta), str(new_union_meta))

    # --- Remove temporary files/directories ---
    removals: List[tuple[Path, str]] = [
        (intermediate_bed, "intermediate BED directory"),
        (intermediate_sqlite, "SQLite databases"),
        (joblists, "joblists"),
        (toga_mini, "TOGA intermediate directory"),
        (output_dir / "temp_shortrna_results.tsv", "temporary short RNA results"),
        (toga_dir / "reference_chrom_sizes.tsv", "reference chromosome sizes (technical)"),
    ]

    # Also remove chrom sizes from toga_mini if it still exists
    chrom_sizes_old = toga_mini / "reference_chrom_sizes.tsv"
    if chrom_sizes_old.exists():
        removals.append((chrom_sizes_old, "reference chromosome sizes (technical)"))

    for path, desc in removals:
        if path.exists():
            if verbose:
                print(f"  Removing {desc}: {path.relative_to(output_dir) if output_dir in path.parents else path.name}")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        elif verbose and path.parent.exists():
            # Only mention if parent exists (otherwise already cleaned)
            print(f"  [SKIP] {desc} not found")

    # Remove query_islands.json from mappings (redundant with query_islands.bed)
    query_islands_json = mappings / "query_islands.json"
    if query_islands_json.exists():
        if verbose:
            print(f"  Removing redundant: {query_islands_json.relative_to(output_dir)}")
        query_islands_json.unlink()

    if verbose:
        print("# Cleanup complete!\n")
        print(f"# Final outputs in: {output_dir}")
        print(f"#   - Query annotations: {annotation_dir.relative_to(output_dir)}/")
        print(f"#   - Island alignments: island_alignment_results.tsv")
        print(f"#   - Preprocessed reference: preprocessed_reference.json")
        print(f"#   - Transcript mappings: mappings/")
        print(f"#   - TOGA results: {toga_dir.relative_to(output_dir)}/")
