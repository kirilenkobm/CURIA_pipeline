#!/usr/bin/env python3
"""
Input file validation for CURIA pipeline.

Checks for file existence, non-zero size, basic format validity,
and chain-genome compatibility.
"""

import gzip
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from pyrion import TwoBitAccessor


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def validate_file_exists_and_nonempty(file_path: str, file_type: str) -> None:
    """Check if file exists and is not empty (0 bytes)."""
    path = Path(file_path)

    if not path.exists():
        raise ValidationError(f"{file_type} does not exist: {file_path}")

    if not path.is_file():
        raise ValidationError(f"{file_type} is not a file: {file_path}")

    # Check file size
    size = path.stat().st_size
    if size == 0:
        raise ValidationError(f"{file_type} is empty (0 bytes): {file_path}")

    # Warn if suspiciously small
    if size < 100:  # Less than 100 bytes is suspicious for any input
        print(f"  WARNING: {file_type} is very small ({size} bytes): {file_path}")


def validate_bed12(bed_path: str) -> Tuple[int, Set[str]]:
    """
    Validate BED12 file format and extract chromosome names.

    Returns:
        (num_records, chromosome_names)
    """
    chroms = set()
    num_records = 0

    try:
        with open(bed_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split("\t")
                if len(parts) < 12:
                    raise ValidationError(
                        f"BED12 file has < 12 columns at line {line_num}: {bed_path}\n"
                        f"  Got {len(parts)} columns, expected 12"
                    )

                chrom = parts[0]
                chroms.add(chrom)
                num_records += 1

                # Validate start/end are integers
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                    if start >= end:
                        raise ValidationError(
                            f"BED12 file has start >= end at line {line_num}: {bed_path}"
                        )
                except ValueError:
                    raise ValidationError(
                        f"BED12 file has non-integer coordinates at line {line_num}: {bed_path}"
                    )
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Failed to parse BED12 file {bed_path}: {e}")

    if num_records == 0:
        raise ValidationError(f"BED12 file contains no records: {bed_path}")

    return num_records, chroms


def validate_tsv_has_header(tsv_path: str, file_type: str) -> int:
    """Validate TSV file has header and at least one data row."""
    try:
        with open(tsv_path) as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) == 0:
            raise ValidationError(f"{file_type} is empty: {tsv_path}")

        if len(lines) == 1:
            raise ValidationError(f"{file_type} has header but no data rows: {tsv_path}")

        # Check that header looks reasonable (has tabs)
        if "\t" not in lines[0]:
            print(f"  WARNING: {file_type} header may not be tab-separated: {tsv_path}")

        return len(lines) - 1  # Number of data rows

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Failed to parse {file_type} {tsv_path}: {e}")


def validate_chain_file(chain_path: str) -> Tuple[Set[str], Set[str], int]:
    """
    Validate chain file and extract reference/query chromosome names.

    Returns:
        (ref_chroms, query_chroms, num_chains)
    """
    ref_chroms = set()
    query_chroms = set()
    num_chains = 0

    # Handle gzipped files
    opener = gzip.open if chain_path.endswith('.gz') else open

    try:
        with opener(chain_path, 'rt') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                if line.startswith("chain"):
                    parts = line.split()
                    if len(parts) < 13:
                        raise ValidationError(
                            f"Chain file has malformed chain header at line {line_num}: {chain_path}\n"
                            f"  Expected at least 13 fields, got {len(parts)}"
                        )

                    # chain score tName tSize tStrand tStart tEnd qName qSize qStrand qStart qEnd id
                    ref_chrom = parts[2]
                    query_chrom = parts[7]

                    ref_chroms.add(ref_chrom)
                    query_chroms.add(query_chrom)
                    num_chains += 1

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Failed to parse chain file {chain_path}: {e}")

    if num_chains == 0:
        raise ValidationError(f"Chain file contains no chain records: {chain_path}")

    return ref_chroms, query_chroms, num_chains


def validate_2bit_file(twobit_path: str, file_type: str) -> Set[str]:
    """
    Validate 2bit file and extract chromosome names.

    Returns:
        Set of chromosome names
    """
    try:
        accessor = TwoBitAccessor(twobit_path)
        chroms = set(accessor.chrom_sizes())

        if len(chroms) == 0:
            raise ValidationError(f"{file_type} contains no sequences: {twobit_path}")

        return chroms

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Failed to open {file_type} {twobit_path}: {e}")


def check_chain_genome_compatibility(
    ref_2bit_chroms: Set[str],
    query_2bit_chroms: Set[str],
    chain_ref_chroms: Set[str],
    chain_query_chroms: Set[str],
) -> List[str]:
    """
    Check if chain file chromosomes match the genome files.

    Returns list of warning messages (empty if all OK).
    """
    warnings = []

    # Check reference genome overlap
    chain_ref_missing = chain_ref_chroms - ref_2bit_chroms
    if chain_ref_missing:
        pct_missing = 100 * len(chain_ref_missing) / len(chain_ref_chroms)
        warnings.append(
            f"  Chain file references {len(chain_ref_missing)} chromosomes "
            f"not found in reference 2bit ({pct_missing:.1f}% of chain chroms):\n"
            f"    {', '.join(sorted(list(chain_ref_missing)[:10]))}"
            + ("..." if len(chain_ref_missing) > 10 else "")
        )

    # Check query genome overlap
    chain_query_missing = chain_query_chroms - query_2bit_chroms
    if chain_query_missing:
        pct_missing = 100 * len(chain_query_missing) / len(chain_query_chroms)
        warnings.append(
            f"  Chain file references {len(chain_query_missing)} chromosomes "
            f"not found in query 2bit ({pct_missing:.1f}% of chain chroms):\n"
            f"    {', '.join(sorted(list(chain_query_missing)[:10]))}"
            + ("..." if len(chain_query_missing) > 10 else "")
        )

    # Check if any chains are usable
    ref_usable = len(chain_ref_chroms - chain_ref_missing)
    query_usable = len(chain_query_chroms - chain_query_missing)

    if ref_usable == 0:
        warnings.append("  ERROR: No chain chromosomes match reference genome!")
    if query_usable == 0:
        warnings.append("  ERROR: No chain chromosomes match query genome!")

    return warnings


def validate_bed_genome_compatibility(
    bed_chroms: Set[str],
    genome_chroms: Set[str],
    bed_file: str,
) -> List[str]:
    """
    Check if BED file chromosomes exist in reference genome.

    Returns list of warning messages (empty if all OK).
    """
    warnings = []

    bed_missing = bed_chroms - genome_chroms
    if bed_missing:
        pct_missing = 100 * len(bed_missing) / len(bed_chroms)
        warnings.append(
            f"  BED file references {len(bed_missing)} chromosomes "
            f"not found in reference genome ({pct_missing:.1f}% of BED chroms):\n"
            f"    {', '.join(sorted(list(bed_missing)[:10]))}"
            + ("..." if len(bed_missing) > 10 else "")
        )

        if len(bed_missing) == len(bed_chroms):
            warnings.append(f"  ERROR: No BED chromosomes match reference genome!")

    return warnings


def validate_all_inputs(
    ref_bed12: str,
    biomart_tsv: str,
    chain: str,
    ref_2bit: str,
    query_2bit: str,
    ref_preprocessed: Optional[str] = None,
) -> None:
    """
    Comprehensive input validation.

    Raises ValidationError if critical issues found.
    Prints warnings for non-critical issues.
    """
    print("# Validating input files...")

    # 1. Check existence and non-zero size
    input_files = {
        "Reference BED12": ref_bed12,
        "Biomart TSV": biomart_tsv,
        "Chain file": chain,
        "Reference 2bit": ref_2bit,
        "Query 2bit": query_2bit,
    }

    errors = []
    for name, path in input_files.items():
        try:
            validate_file_exists_and_nonempty(path, name)
        except ValidationError as e:
            errors.append(str(e))

    if ref_preprocessed:
        try:
            validate_file_exists_and_nonempty(ref_preprocessed, "Preprocessed reference")
        except ValidationError as e:
            errors.append(str(e))

    if errors:
        raise ValidationError("\n".join(errors))

    print("  ✓ All files exist and are non-empty")

    # 2. Validate BED12 format
    print("# Validating BED12 format...")
    try:
        num_bed_records, bed_chroms = validate_bed12(ref_bed12)
        print(f"  ✓ BED12 valid: {num_bed_records} records, {len(bed_chroms)} chromosomes")
    except ValidationError as e:
        raise ValidationError(f"BED12 validation failed:\n{e}")

    # 3. Validate Biomart TSV
    print("# Validating Biomart TSV...")
    try:
        num_biomart_rows = validate_tsv_has_header(biomart_tsv, "Biomart TSV")
        print(f"  ✓ Biomart TSV valid: {num_biomart_rows} data rows")
    except ValidationError as e:
        raise ValidationError(f"Biomart TSV validation failed:\n{e}")

    # 4. Validate 2bit files
    print("# Validating genome files...")
    try:
        ref_2bit_chroms = validate_2bit_file(ref_2bit, "Reference 2bit")
        print(f"  ✓ Reference 2bit valid: {len(ref_2bit_chroms)} sequences")
    except ValidationError as e:
        raise ValidationError(f"Reference 2bit validation failed:\n{e}")

    try:
        query_2bit_chroms = validate_2bit_file(query_2bit, "Query 2bit")
        print(f"  ✓ Query 2bit valid: {len(query_2bit_chroms)} sequences")
    except ValidationError as e:
        raise ValidationError(f"Query 2bit validation failed:\n{e}")

    # 5. Validate chain file
    print("# Validating chain file...")
    try:
        chain_ref_chroms, chain_query_chroms, num_chains = validate_chain_file(chain)
        print(f"  ✓ Chain file valid: {num_chains} chains")
        print(f"    Reference chroms: {len(chain_ref_chroms)}")
        print(f"    Query chroms: {len(chain_query_chroms)}")
    except ValidationError as e:
        raise ValidationError(f"Chain file validation failed:\n{e}")

    # 6. Check chain-genome compatibility
    print("# Checking chain-genome compatibility...")
    warnings = check_chain_genome_compatibility(
        ref_2bit_chroms,
        query_2bit_chroms,
        chain_ref_chroms,
        chain_query_chroms,
    )

    if warnings:
        has_error = any("ERROR:" in w for w in warnings)
        if has_error:
            print("# FATAL ERRORS detected:")
            for w in warnings:
                print(w)
            raise ValidationError(
                "Chain file chromosomes do not match genome files. "
                "Please verify you are using the correct chain and genome files."
            )
        else:
            print("# Warnings (may be OK if using chromosome subsets):")
            for w in warnings:
                print(w)
    else:
        print("  ✓ Chain file compatible with both genomes")

    # 7. Check BED-genome compatibility
    print("# Checking BED-genome compatibility...")
    bed_warnings = validate_bed_genome_compatibility(
        bed_chroms,
        ref_2bit_chroms,
        ref_bed12,
    )

    if bed_warnings:
        has_error = any("ERROR:" in w for w in bed_warnings)
        if has_error:
            print("# FATAL ERRORS detected:")
            for w in bed_warnings:
                print(w)
            raise ValidationError(
                "BED file chromosomes do not match reference genome. "
                "Please verify chromosome naming (e.g., 'chr1' vs '1')."
            )
        else:
            print("# Warnings (may be OK if using chromosome subsets):")
            for w in bed_warnings:
                print(w)
    else:
        print("  ✓ BED file compatible with reference genome")

    print("# ✓ All input validation checks passed\n")
