from __future__ import annotations

from typing import Dict, Iterable, Tuple


def _is_rna_biotype(biotype: str) -> bool:
    if not biotype:
        return False
    b_norm = biotype.strip().lower().replace("-", "_").replace(" ", "_")
    if b_norm == "protein_coding":
        return False
    if "rna" in b_norm:
        return True
    if b_norm in {"ribozyme"}:
        return True
    return False


def _load_biotypes(metadata_tsv_path: str) -> Dict[str, str]:
    with open(metadata_tsv_path, "r") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            tid_idx = header.index("transcript_id")
            biotype_idx = header.index("biotype")
        except ValueError:
            raise ValueError("Expected columns transcript_id and biotype in metadata TSV")

        biotypes: Dict[str, str] = {}
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(tid_idx, biotype_idx):
                continue
            transcript_id = parts[tid_idx]
            biotype = parts[biotype_idx]
            biotypes[transcript_id] = biotype
    return biotypes


def _parse_bed12_lengths(bed12_path: str) -> Dict[str, int]:
    lengths: Dict[str, int] = {}
    with open(bed12_path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 12:
                continue
            transcript_id = parts[3]
            block_sizes = parts[10].split(",")
            size_sum = 0
            for size in block_sizes:
                if not size:
                    continue
                try:
                    size_sum += int(size)
                except ValueError:
                    continue
            if size_sum == 0:
                try:
                    size_sum = int(parts[2]) - int(parts[1])
                except ValueError:
                    size_sum = 0
            lengths[transcript_id] = size_sum
    return lengths


def write_rna_orthologous_regions(
    toga_regions_path: str,
    ultimate_meta_path: str,
    ultimate_bed_path: str,
    out_path: str,
) -> Tuple[int, int]:
    biotypes = _load_biotypes(ultimate_meta_path)
    lengths = _parse_bed12_lengths(ultimate_bed_path)

    kept = 0
    total = 0

    with open(toga_regions_path, "r") as src, open(out_path, "w") as dst:
        header = src.readline().rstrip("\n")
        if header:
            dst.write(f"{header}\tbiotype\ttranscript_length\n")
        else:
            dst.write("transcript_id\tchain_id\tregion\ttranscript_strand\tchain_strand\tbiotype\ttranscript_length\n")

        for line in src:
            if not line.strip():
                continue
            total += 1
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            transcript_id = parts[0]
            if not transcript_id.startswith("U_"):
                continue
            biotype = biotypes.get(transcript_id)
            if not _is_rna_biotype(biotype):
                continue
            transcript_length = lengths.get(transcript_id, 0)
            dst.write(f"{line.rstrip()}\t{biotype}\t{transcript_length}\n")
            kept += 1

    return kept, total
