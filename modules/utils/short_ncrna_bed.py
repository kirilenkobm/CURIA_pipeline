from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Tuple


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


__all__ = ["write_short_ncrna_bed"]
