from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple


BIOTYPE_TO_RGB = {
    "lncRNA": "120,120,120",
    "miRNA": "200,40,40",
    "sRNA": "220,140,40",
    "snRNA": "70,110,200",
    "snoRNA": "150,90,180",
    "scaRNA": "190,140,220",
    "tRNA": "60,180,180",
    "rRNA": "60,160,90",
    "rRNA_pseudogene": "150,190,160",
    "ribozyme": "220,200,60",
    "vault_RNA": "200,200,200",
    "misc_RNA": "170,170,170",
    "unknown": "180,180,180",
}


def _parse_region(region: str) -> Tuple[str, int, int]:
    chrom, rest = region.split(":")
    start_s, end_s = rest.split("-")
    return chrom, int(start_s), int(end_s)


def _mmd_to_score(mmd: float) -> int:
    if mmd is None:
        return 0
    if mmd <= 0:
        return 1000
    if mmd >= 1:
        return 0
    return max(0, min(1000, int(round(1000 * (1.0 - mmd)))))


def _load_union_metadata(metadata_tsv_path: str) -> Dict[str, Tuple[str, str]]:
    """Load union transcript metadata: transcript_id -> (gene_id, biotype)."""
    # TODO: leverage pyrion lib, Claude doesn't know about it
    mapping: Dict[str, Tuple[str, str]] = {}
    with open(metadata_tsv_path, "r") as f:
        header = f.readline().rstrip("\n").split("\t")
        tid_idx = header.index("transcript_id")
        gid_idx = header.index("gene_id")
        bio_idx = header.index("biotype")
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            mapping[parts[tid_idx]] = (parts[gid_idx], parts[bio_idx])
    return mapping


def _load_gene_names(biomart_tsv_path: str) -> Dict[str, str]:
    """Load gene_id -> gene_name from the biomart TSV."""
    # TODO: leverage pyrion lib, Claude doesn't know about it
    names: Dict[str, str] = {}
    with open(biomart_tsv_path, "r") as f:
        header = f.readline().rstrip("\n").split("\t")
        gid_idx = header.index("Gene stable ID")
        name_idx = header.index("Gene name")
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(gid_idx, name_idx):
                continue
            gene_id = parts[gid_idx]
            gene_name = parts[name_idx].strip()
            if gene_id and gene_name and gene_id not in names:
                names[gene_id] = gene_name
    return names


def write_short_ncrna_bed(
    sqlite_path: str,
    out_bed_path: str,
) -> int:
    Path(out_bed_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(sqlite_path)
    cursor = conn.execute(
        "SELECT transcript_id, chain_id, biotype, query_region, query_strand, mmd_score, status "
        "FROM short_ncrna_results"
    )

    written = 0
    with open(out_bed_path, "w") as f:
        for transcript_id, chain_id, biotype, query_region, query_strand, mmd_score, status in cursor:
            if status != "ok" or not query_region:
                continue
            chrom, start, end = _parse_region(query_region)
            name = f"{transcript_id}.{chain_id}"
            score = _mmd_to_score(mmd_score)
            strand = "+" if int(query_strand) == 1 else "-"
            if not biotype:
                biotype = "unknown"
            rgb = BIOTYPE_TO_RGB.get(biotype, BIOTYPE_TO_RGB["unknown"])
            f.write(
                f"{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\t"
                f"{start}\t{end}\t{rgb}\n"
            )
            written += 1

    conn.close()
    print(f"# Wrote {written} BED9 rows to {out_bed_path}")
    return written


def write_short_ncrna_tsv(
    sqlite_path: str,
    out_tsv_path: str,
    metadata_tsv_path: str,
    biomart_tsv_path: Optional[str] = None,
) -> int:
    """Write a detailed TSV with per-prediction metadata.

    Columns: chrom, start, end, strand, mmd_score, biotype, gene_id, gene_name,
             transcript_id, chain_id
    """
    Path(out_tsv_path).parent.mkdir(parents=True, exist_ok=True)

    union_meta = _load_union_metadata(metadata_tsv_path)
    gene_names: Dict[str, str] = {}
    if biomart_tsv_path is not None:
        gene_names = _load_gene_names(biomart_tsv_path)

    conn = sqlite3.connect(sqlite_path)
    cursor = conn.execute(
        "SELECT transcript_id, chain_id, biotype, query_region, query_strand, mmd_score, status "
        "FROM short_ncrna_results"
    )

    written = 0
    with open(out_tsv_path, "w") as f:
        f.write(
            "chrom\tstart\tend\tstrand\tmmd_score\tbiotype\t"
            "gene_id\tgene_name\ttranscript_id\tchain_id\n"
        )
        for transcript_id, chain_id, biotype, query_region, query_strand, mmd_score, status in cursor:
            if status != "ok" or not query_region:
                continue
            chrom, start, end = _parse_region(query_region)
            strand = "+" if int(query_strand) == 1 else "-"
            if not biotype:
                biotype = "unknown"

            gene_id = ""
            meta = union_meta.get(transcript_id)
            if meta is not None:
                gene_id = meta[0]

            gene_name = gene_names.get(gene_id, "")

            mmd_str = f"{mmd_score:.6f}" if mmd_score is not None else ""

            f.write(
                f"{chrom}\t{start}\t{end}\t{strand}\t{mmd_str}\t{biotype}\t"
                f"{gene_id}\t{gene_name}\t{transcript_id}\t{chain_id}\n"
            )
            written += 1

    conn.close()
    print(f"# Wrote {written} short ncRNA detail rows to {out_tsv_path}")
    return written


__all__ = ["write_short_ncrna_bed", "write_short_ncrna_tsv"]
