#!/usr/bin/env python3
"""
Input file validation for CURIA pipeline.

Checks for file existence, non-zero size, basic format validity,
and chain-genome compatibility.
"""

import gzip
from pathlib import Path
from typing import Dict, List, Set, Tuple

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


def validate_chain_file(chain_path: str) -> None:
    """
    Validate chain file exists and is non-empty.

    TODO: Check what are cheap ways to validate chain format without full parsing.
    TOGA will crash if something is fundamentally wrong with the chain file.
    """
    # Chain file existence/size already checked by validate_file_exists_and_nonempty
    pass


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


def _diagnose_naming_mismatch(
    chain_chroms: Set[str],
    twobit_chroms: Set[str],
) -> str:
    """Try to identify the naming convention mismatch between chain and 2bit."""
    if not chain_chroms or not twobit_chroms:
        return ""

    missing = chain_chroms - twobit_chroms

    if not missing:
        return ""

    # Check version suffix mismatch: chain has "NW_123" but 2bit has "NW_123.1"
    strip_version = {c.rsplit(".", 1)[0]: c for c in twobit_chroms if "." in c}
    fixable_by_adding = sum(1 for c in missing if c in strip_version)
    if fixable_by_adding > len(missing) * 0.5:
        sample_chain = sorted(missing)[0]
        sample_2bit = strip_version.get(sample_chain, "?")
        return (
            f"    Likely cause: accession version suffix mismatch.\n"
            f"    Chain uses '{sample_chain}' but 2bit has '{sample_2bit}'.\n"
            f"    The chain and 2bit files may be from different assembly versions."
        )

    # Check chr-prefix mismatch: chain has "chr1" but 2bit has "1" or vice versa
    add_chr = {f"chr{c}": c for c in twobit_chroms if not c.startswith("chr")}
    strip_chr = {c[3:]: c for c in twobit_chroms if c.startswith("chr")}
    fixable_add = sum(1 for c in missing if c in add_chr)
    fixable_strip = sum(1 for c in missing if c in strip_chr)
    if fixable_add > len(missing) * 0.5:
        return "    Likely cause: 2bit lacks 'chr' prefix that chains expect."
    if fixable_strip > len(missing) * 0.5:
        return "    Likely cause: 2bit uses 'chr' prefix but chains do not."

    return ""


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
        msg = (
            f"  Chain file references {len(chain_ref_missing)} reference chromosomes "
            f"not found in reference 2bit ({pct_missing:.1f}% of chain chroms):\n"
            f"    {', '.join(sorted(list(chain_ref_missing)[:10]))}"
            + ("..." if len(chain_ref_missing) > 10 else "")
        )
        diagnosis = _diagnose_naming_mismatch(chain_ref_chroms, ref_2bit_chroms)
        if diagnosis:
            msg += f"\n{diagnosis}"
        warnings.append(msg)

    # Check query genome overlap
    chain_query_missing = chain_query_chroms - query_2bit_chroms
    if chain_query_missing:
        pct_missing = 100 * len(chain_query_missing) / len(chain_query_chroms)
        msg = (
            f"  Chain file references {len(chain_query_missing)} query chromosomes "
            f"not found in query 2bit ({pct_missing:.1f}% of chain chroms):\n"
            f"    {', '.join(sorted(list(chain_query_missing)[:10]))}"
            + ("..." if len(chain_query_missing) > 10 else "")
        )
        diagnosis = _diagnose_naming_mismatch(chain_query_chroms, query_2bit_chroms)
        if diagnosis:
            msg += f"\n{diagnosis}"
        warnings.append(msg)

    # Check if any chains are usable
    ref_usable = len(chain_ref_chroms - chain_ref_missing)
    query_usable = len(chain_query_chroms - chain_query_missing)

    if ref_usable == 0:
        warnings.append("  FATAL: No chain chromosomes match reference genome!")
    if query_usable == 0:
        warnings.append("  FATAL: No chain chromosomes match query genome!")

    return warnings


def _open_maybe_gzip(path: str):
    """Open a text file transparently whether it is gzip-compressed or plain."""
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt")
    return open(path, "rt")


def sample_chain_chromosomes(
    chain_path: str,
    max_headers: int = 200_000,
) -> Tuple[Set[str], Set[str], bool]:
    """
    Cheaply extract reference/query chromosome names from chain HEADER lines
    only (never parses alignment blocks), stopping after ``max_headers``.

    A uniform naming-convention mismatch (accession version suffix, ``chr``
    prefix) shows up in the first handful of headers, so a bounded sample is
    enough for a fail-fast startup gate; the authoritative full check still runs
    later once chains are parsed into memory.

    Returns (ref_chroms, query_chroms, truncated).
    Chain header: ``chain score tName tSize tStrand tStart tEnd qName ...``.
    """
    ref_chroms: Set[str] = set()
    query_chroms: Set[str] = set()
    n = 0
    with _open_maybe_gzip(chain_path) as fh:
        for line in fh:
            if not line.startswith("chain"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            ref_chroms.add(parts[2])
            query_chroms.add(parts[7])
            n += 1
            if n >= max_headers:
                return ref_chroms, query_chroms, True
    return ref_chroms, query_chroms, False


def build_chrom_name_remap(
    chain_chroms: Set[str],
    twobit_chroms: Set[str],
) -> Dict[str, str]:
    """
    Map chain chromosome names -> 2bit names for the two fixable naming-
    convention mismatches: accession version suffix (``X`` vs ``X.1``) and
    ``chr`` prefix (``1`` vs ``chr1``). Only unambiguous 1:1 fixes are emitted;
    names already in the 2bit or with no clear counterpart are left unmapped
    (the rewrite passes them through untouched).
    """
    remap: Dict[str, str] = {}
    missing = chain_chroms - twobit_chroms
    if not missing:
        return remap

    # 2bit 'X.N' indexed by its version-stripped base (only unambiguous bases)
    strip_ver: Dict[str, List[str]] = {}
    for name in twobit_chroms:
        base = name.rsplit(".", 1)[0]
        if base != name:
            strip_ver.setdefault(base, []).append(name)

    for c in missing:
        # chain 'X' -> 2bit 'X.N' (add version, unambiguous)
        if c in strip_ver and len(strip_ver[c]) == 1:
            remap[c] = strip_ver[c][0]
            continue
        # chain 'X.N' -> 2bit 'X' (strip version)
        base = c.rsplit(".", 1)[0]
        if base != c and base in twobit_chroms:
            remap[c] = base
            continue
        # chain 'X' -> 2bit 'chrX' (add chr prefix)
        if f"chr{c}" in twobit_chroms:
            remap[c] = f"chr{c}"
            continue
        # chain 'chrX' -> 2bit 'X' (strip chr prefix)
        if c.startswith("chr") and c[3:] in twobit_chroms:
            remap[c] = c[3:]
            continue
    return remap


def _mismatch_is_aliasable(
    chain_ref: Set[str],
    chain_query: Set[str],
    ref_2bit_chroms: Set[str],
    query_2bit_chroms: Set[str],
) -> bool:
    """
    True if a fatal chain<->2bit name mismatch is fully resolved by version-
    suffix / ``chr``-prefix remapping — i.e. exactly what AliasedTwoBitAccessor
    does at query-fetch time — rather than a genuine different-assembly
    mismatch. Applies the remap to the chain names and rechecks for a fatal.
    """
    remap = build_chrom_name_remap(chain_query, query_2bit_chroms)
    remap.update(build_chrom_name_remap(chain_ref, ref_2bit_chroms))
    if not remap:
        return False
    remapped_q = {remap.get(c, c) for c in chain_query}
    remapped_r = {remap.get(c, c) for c in chain_ref}
    recheck = check_chain_genome_compatibility(
        ref_2bit_chroms, query_2bit_chroms, remapped_r, remapped_q,
    )
    return not any("FATAL:" in w for w in recheck)


def precheck_chain_2bit_compatibility(
    chain_path: str,
    ref_2bit_chroms: Set[str],
    query_2bit_chroms: Set[str],
    auto_fix: bool = False,
) -> str:
    """
    Fail-fast startup gate: sample chain headers and compare chromosome names
    to the (already-read) 2bit genome headers BEFORE any heavy work runs.

    Returns the (unchanged) chain path. On a fatal name mismatch: aborts unless
    ``auto_fix`` and the mismatch is a version-suffix / ``chr``-prefix issue that
    AliasedTwoBitAccessor will resolve at query-fetch time, in which case it logs
    and proceeds. Raises ValidationError on an unfixable fatal mismatch.
    """
    print("# Checking chain-genome compatibility (fail-fast)...")
    chain_ref, chain_query, truncated = sample_chain_chromosomes(chain_path)
    note = " (sampled)" if truncated else ""
    print(
        f"#   Chain{note}: {len(chain_ref)} ref chroms, {len(chain_query)} query chroms; "
        f"2bit: {len(ref_2bit_chroms)} ref, {len(query_2bit_chroms)} query"
    )

    warnings = check_chain_genome_compatibility(
        ref_2bit_chroms, query_2bit_chroms, chain_ref, chain_query,
    )
    if not warnings:
        print("#   ✓ chain chromosomes present in both genomes")
        return chain_path

    has_fatal = any("FATAL:" in w for w in warnings)
    for w in warnings:
        print(w)

    if not has_fatal:
        print("#   (non-fatal — proceeding; full check runs after chains load)")
        return chain_path

    if not auto_fix:
        raise ValidationError(
            "Chain chromosomes do not match the genome 2bit files (see above).\n"
            "If this is a naming-convention mismatch (accession version suffix or "
            "'chr' prefix), re-run with --auto-fix-chrom-names to resolve the names "
            "on the fly and continue."
        )

    # --- auto-fix path: resolve names on the fly, no file rewrite ----------
    remap = build_chrom_name_remap(chain_query, query_2bit_chroms)
    remap.update(build_chrom_name_remap(chain_ref, ref_2bit_chroms))
    if not remap or not _mismatch_is_aliasable(
        chain_ref, chain_query, ref_2bit_chroms, query_2bit_chroms
    ):
        raise ValidationError(
            "Chain-genome mismatch is not an auto-fixable naming convention "
            "(no version-suffix or chr-prefix mapping resolves it). Verify that "
            "the chain and 2bit files are from the same assemblies."
        )
    example = next(iter(remap.items()))
    print(
        f"#   --auto-fix-chrom-names: version/'chr'-prefix mismatch "
        f"(e.g. {example[0]} -> {example[1]}); query-2bit names will be resolved "
        f"on the fly at fetch time (no file written)."
    )
    return chain_path


def validate_chain_2bit_compatibility(
    chains,
    ref_2bit_path: str,
    query_2bit_path: str,
    auto_fix: bool = False,
) -> None:
    """
    Fast chain-vs-2bit chromosome validation using already-loaded chains.

    Extracts reference/query chromosome sets from the in-memory chain
    collection and compares against 2bit headers (instant reads).
    Raises ValidationError on fatal mismatches; prints warnings otherwise.

    When ``auto_fix`` is set and a fatal mismatch is a version-suffix /
    ``chr``-prefix naming issue that AliasedTwoBitAccessor resolves at
    query-fetch time, it is downgraded to a warning instead of aborting.
    """
    chain_ref_chroms = set(chains.get_reference_chromosomes())
    chain_query_chroms = set(chains.get_query_chromosomes())

    ref_2bit_chroms = set(TwoBitAccessor(ref_2bit_path).chrom_sizes())
    query_2bit_chroms = set(TwoBitAccessor(query_2bit_path).chrom_sizes())

    print(f"# Chain-genome compatibility check:")
    print(f"#   Chain:     {len(chain_ref_chroms)} ref chroms, {len(chain_query_chroms)} query chroms")
    print(f"#   Ref 2bit:  {len(ref_2bit_chroms)} sequences")
    print(f"#   Query 2bit: {len(query_2bit_chroms)} sequences")

    warnings = check_chain_genome_compatibility(
        ref_2bit_chroms, query_2bit_chroms,
        chain_ref_chroms, chain_query_chroms,
    )

    if not warnings:
        print(f"#   ✓ All chain chromosomes found in both genomes")
        return

    has_fatal = any("FATAL:" in w for w in warnings)

    if has_fatal:
        if auto_fix and _mismatch_is_aliasable(
            chain_ref_chroms, chain_query_chroms, ref_2bit_chroms, query_2bit_chroms
        ):
            print("# Chain-genome name mismatch — resolved on the fly by "
                  "query-2bit aliasing (--auto-fix-chrom-names):")
            for w in warnings:
                print(w)
            return
        print("# FATAL chain-genome incompatibility detected:")
        for w in warnings:
            print(w)
        raise ValidationError(
            "Chain file chromosomes do not match genome 2bit files. "
            "Ensure chain, reference 2bit, and query 2bit are from the same assemblies."
        )

    print("# Chain-genome compatibility warnings:")
    for w in warnings:
        print(w)


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
    reference_metadata: str,
    chain: str,
    ref_2bit: str,
    query_2bit: str,
    auto_fix_chrom_names: bool = False,
) -> str:
    """
    Comprehensive input validation.

    Raises ValidationError if critical issues found.
    Prints warnings for non-critical issues.

    Returns the chain path to use (unchanged). With ``auto_fix_chrom_names``, a
    version-suffix / ``chr``-prefix chain<->2bit mismatch is tolerated and
    resolved on the fly at query-fetch time (see AliasedTwoBitAccessor) rather
    than aborting.
    """
    print("# Validating input files...")

    # 1. Check existence and non-zero size
    input_files = {
        "Reference BED12": ref_bed12,
        "Reference metadata TSV": reference_metadata,
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

    # 3. Validate reference metadata TSV
    print("# Validating reference metadata TSV...")
    try:
        num_metadata_rows = validate_tsv_has_header(reference_metadata, "Reference metadata TSV")
        print(f"  ✓ Reference metadata TSV valid: {num_metadata_rows} data rows")
    except ValidationError as e:
        raise ValidationError(f"Reference metadata TSV validation failed:\n{e}")

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
        validate_chain_file(chain)
        print(f"  ✓ Chain file exists and is non-empty")
    except ValidationError as e:
        raise ValidationError(f"Chain file validation failed:\n{e}")

    # 6. Fail-fast chain <-> genome compatibility (cheap header sample), so a
    # name mismatch aborts in seconds instead of ~16 min into the run. May
    # return a corrected chain path when --auto-fix-chrom-names is set.
    effective_chain = precheck_chain_2bit_compatibility(
        chain,
        ref_2bit_chroms,
        query_2bit_chroms,
        auto_fix=auto_fix_chrom_names,
    )

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
    return effective_chain
