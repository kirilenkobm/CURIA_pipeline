#!/usr/bin/env python3
"""Replace U_ENSG* identifiers with human-readable gene names.

Reads a gene-name lookup table (BioMart TSV with columns:
Gene stable ID, Gene name, ...) and substitutes every occurrence
of U_ENSG<digits> in the input file with the corresponding gene name.

Suffixes after the ENSG ID are preserved:
    U_ENSG00000157873       -> TNFRSF14
    U_ENSG00000157873.1     -> TNFRSF14.1      (version)
    U_ENSG00000157873.1.42  -> TNFRSF14.1.42   (version + chain)

Non-ENSG identifiers (e.g. U_tRNA-...) are left untouched.
Writes to stdout by default, or to a file with -o.

Usage:
    python helpers/resolve_gene_names.py GENE_NAMES_TSV INPUT_FILE
    python helpers/resolve_gene_names.py GENE_NAMES_TSV INPUT_FILE -o output.tsv
"""

import argparse
import re
import sys
from pathlib import Path

ENSG_PATTERN = re.compile(r"U_(ENSG\d+)")


def load_gene_names(path: str) -> dict[str, str]:
    """Build ENSG ID -> gene name map (first gene name per ID wins)."""
    mapping: dict[str, str] = {}
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            ensg_id, gene_name = parts[0], parts[1]
            if ensg_id not in mapping and gene_name:
                mapping[ensg_id] = gene_name
    return mapping


def resolve_line(line: str, gene_map: dict[str, str]) -> str:
    return ENSG_PATTERN.sub(
        lambda m: gene_map.get(m.group(1), m.group(0)),
        line,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Replace U_ENSG* IDs with gene names in a BED/TSV file",
    )
    parser.add_argument("gene_names", help="Gene names TSV (BioMart export)")
    parser.add_argument("input_file", help="BED or TSV file to process")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument(
        "--stats", action="store_true",
        help="Print replacement stats to stderr",
    )
    args = parser.parse_args()

    gene_map = load_gene_names(args.gene_names)

    resolved, unresolved = 0, set()
    out = open(args.output, "w") if args.output else sys.stdout
    try:
        with open(args.input_file) as f:
            for line in f:
                if args.stats:
                    for m in ENSG_PATTERN.finditer(line):
                        if m.group(1) in gene_map:
                            resolved += 1
                        else:
                            unresolved.add(m.group(1))
                out.write(resolve_line(line, gene_map))
    finally:
        if args.output:
            out.close()

    if args.stats:
        print(f"Resolved: {resolved} occurrences", file=sys.stderr)
        if unresolved:
            print(f"Unresolved ({len(unresolved)} unique): "
                  f"{', '.join(sorted(unresolved)[:10])}"
                  f"{'...' if len(unresolved) > 10 else ''}",
                  file=sys.stderr)


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    main()
