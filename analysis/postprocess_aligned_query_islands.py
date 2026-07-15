#!/usr/bin/env python3
"""Add gene names, match scores, and UCSC colors to aligned query islands.

The script operates on one standard CURIA output directory. It reads
``query_annotation/aligned_query_islands.bed`` without modifying it and writes
``query_annotation/aligned_query_islands_human_readable.bed`` as BED9 with
itemRgb.
The original item identity from the aligned BED (for example
``U_ENSG00000251562.13:Q5``) is preserved exactly, and an available gene symbol
is appended.
The BED score is a display confidence scaled from 1000 at d=0 to 0 at the
acceptance threshold (d=0.10); distance is not added to the item name.

Example:
    .venv/bin/python analysis/postprocess_aligned_query_islands.py \
        preprint_results/hg38_vs_mm39
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from pathlib import Path


DARK_BLUE = (8, 48, 107)
PALE_BLUE = (198, 219, 239)


def bare_gene_id(value: str) -> str:
    value = value.removeprefix("U_")
    return re.sub(r"\.\d+$", "", value)


def find_gene_names_file(output_dir: Path, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Gene-name table not found: {path}")
        return path

    repo_root = Path(__file__).resolve().parents[1]
    candidates = (
        output_dir / "reference_gene_names.tsv",
        output_dir / "reference_gene_names.txt",
        repo_root / "input_data/reference_annotation/hg38_gene_names.txt",
    )
    return next((path for path in candidates if path.exists()), None)


def load_gene_names(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}

    names: dict[str, str] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return names
        id_column = next(
            (c for c in ("Gene stable ID", "gene_id", "Gene ID") if c in reader.fieldnames),
            None,
        )
        name_column = next(
            (c for c in ("Gene name", "gene_name", "symbol") if c in reader.fieldnames),
            None,
        )
        if id_column is None or name_column is None:
            raise ValueError(
                f"{path} must contain a gene-ID column and a gene-name column"
            )
        for row in reader:
            gene_id = bare_gene_id(row.get(id_column, "").strip())
            gene_name = row.get(name_column, "").strip()
            if gene_id and gene_name and gene_id not in names:
                names[gene_id] = gene_name
    return names


def load_matches(path: Path) -> dict[tuple[str, str, int, int], tuple[float, str]]:
    matches: dict[tuple[str, str, int, int], tuple[float, str]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "gene_id",
            "query_island",
            "type",
            "query_chrom",
            "query_start",
            "query_end",
            "diag_mmd",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            if row["type"] != "match":
                continue
            key = (
                row["gene_id"],
                row["query_chrom"],
                int(row["query_start"]),
                int(row["query_end"]),
            )
            distance = float(row["diag_mmd"])
            previous = matches.get(key)
            if previous is None or distance < previous[0]:
                matches[key] = (distance, row["ref_island"])
    return matches


def parse_gene_id(name: str) -> str:
    match = re.search(r"(U_ENSG\d+(?:\.\d+)?|ENSG\d+(?:\.\d+)?)", name)
    if not match:
        raise ValueError(f"Cannot recover gene ID from BED name: {name}")
    gene_id = match.group(1)
    if not gene_id.startswith("U_"):
        gene_id = f"U_{gene_id}"
    return gene_id


def display_values(distance: float, max_distance: float) -> tuple[int, str]:
    confidence = max(0.0, min(1.0, 1.0 - distance / max_distance))
    score = round(1000 * confidence)
    rgb = tuple(
        round(pale + confidence * (dark - pale))
        for pale, dark in zip(PALE_BLUE, DARK_BLUE)
    )
    return score, ",".join(map(str, rgb))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="CURIA output directory")
    parser.add_argument(
        "--gene-names",
        help="optional TSV gene-name table; auto-detected for the bundled hg38 input",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.10,
        help="distance represented by score 0 and the palest color (default: 0.10)",
    )
    args = parser.parse_args()
    if args.max_distance <= 0:
        parser.error("--max-distance must be positive")

    output_dir = Path(args.output_dir)
    bed_path = output_dir / "query_annotation/aligned_query_islands.bed"
    output_path = output_dir / "query_annotation/aligned_query_islands_human_readable.bed"
    matches_path = output_dir / "island_alignment_results.tsv"
    if not bed_path.exists():
        raise FileNotFoundError(f"Aligned-query BED not found: {bed_path}")
    if not matches_path.exists():
        raise FileNotFoundError(f"Island match table not found: {matches_path}")

    names_path = find_gene_names_file(output_dir, args.gene_names)
    gene_names = load_gene_names(names_path)
    matches = load_matches(matches_path)

    written = named = 0
    parent = output_path.parent
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=parent)
    try:
        with os.fdopen(fd, "w") as out, bed_path.open() as source:
            out.write(
                'track name="CURIA_query_islands" '
                'description="CURIA query islands; darker blue indicates lower embedding-SW distance" '
                'visibility=2 itemRgb="On"\n'
            )
            for line_number, line in enumerate(source, 1):
                if not line.strip() or line.startswith(("track ", "browser ", "#")):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 6:
                    raise ValueError(f"{bed_path}:{line_number}: expected at least 6 BED fields")
                chrom, start_text, end_text, old_name, _, strand = fields[:6]
                start, end = int(start_text), int(end_text)
                gene_id = parse_gene_id(old_name)
                key = (gene_id, chrom, start, end)
                if key not in matches:
                    raise KeyError(
                        f"{bed_path}:{line_number}: no matching score row for "
                        f"{gene_id} at {chrom}:{start}-{end}"
                    )
                distance, ref_island = matches[key]
                symbol = gene_names.get(bare_gene_id(gene_id))
                item_name = f"{old_name}.{ref_island}"
                if symbol:
                    item_name = f"{item_name}|{symbol}"
                    named += 1
                score, rgb = display_values(distance, args.max_distance)
                out.write(
                    f"{chrom}\t{start}\t{end}\t{item_name}\t{score}\t{strand}"
                    f"\t{start}\t{end}\t{rgb}\n"
                )
                written += 1
        os.replace(temporary_name, output_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    source_note = str(names_path) if names_path else "none available"
    print(f"# Read {written} islands without modifying: {bed_path}")
    print(f"# Wrote human-readable UCSC track: {output_path}")
    print(f"# Added gene symbols to {named}/{written} records (source: {source_note})")
    print(f"# Color scale: d=0 dark blue; d={args.max_distance:g} pale blue")


if __name__ == "__main__":
    main()
