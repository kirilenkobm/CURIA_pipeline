"""I/O + interval helpers for the RNA-FM vs RiNALMo output comparison notebook.

Kept in a module (not notebook cells) only because the file-parsing details are
boring boilerplate; all analysis / interpretation lives in the notebook itself.
Interval math is delegated to `pyrion` (GenomicInterval / GenomicIntervalsCollection).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyrion import GenomicInterval as GI
from pyrion import GenomicIntervalsCollection as GIC
from pyrion import Strand

BED12_COLS = [
    "chrom", "start", "end", "name", "score", "strand",
    "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts",
]


# ---------------------------------------------------------------- file loading

def resolve(base: Path, name: str) -> Path:
    """Old preprint__deprecated layout has files at top level; new layout nests them under
    query_annotation/. Find the file in whichever place it lives."""
    for cand in (base / name, base / "query_annotation" / name):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{name!r} not found under {base}")


def read_details(base: Path) -> pd.DataFrame:
    """short_ncRNA_details.tsv (has header)."""
    return pd.read_csv(resolve(base, "short_ncRNA_details.tsv"), sep="\t",
                       dtype={"chrom": str})


def read_alignment(base: Path) -> pd.DataFrame:
    """island_alignment_results.tsv (has header)."""
    return pd.read_csv(resolve(base, "island_alignment_results.tsv"), sep="\t",
                       dtype={"ref_chrom": str, "query_chrom": str})


def read_bed12(base: Path, name: str) -> pd.DataFrame:
    """A headerless BED12 island file."""
    return pd.read_csv(resolve(base, name), sep="\t", header=None,
                       names=BED12_COLS, dtype={"chrom": str})


def blocks_of(row) -> list[tuple[int, int]]:
    """Expand a BED12 row into absolute (start, end) block intervals."""
    sizes = [int(x) for x in str(row["blockSizes"]).rstrip(",").split(",") if x != ""]
    starts = [int(x) for x in str(row["blockStarts"]).rstrip(",").split(",") if x != ""]
    cs = int(row["start"])
    return [(cs + s, cs + s + sz) for s, sz in zip(starts, sizes)]


def gene_of_aligned(name: str) -> str:
    """'U_ENSG00000099869.8_aligned' -> 'U_ENSG00000099869.8'."""
    return name[:-len("_aligned")] if name.endswith("_aligned") else name


def gene_of_raw(name: str) -> str:
    """'U_ENSG00000309243.1.1_island_0' -> 'U_ENSG00000309243.1.1' (union id)."""
    return name.split("_island_")[0]


# ------------------------------------------------------- interval math (pyrion)

def _col(chrom: str, ivs: list[tuple[int, int]]) -> GIC:
    return GIC.from_intervals([GI(chrom, s, e, Strand.PLUS) for s, e in ivs]).merge_close(0)


def covered_bp(chrom: str, ivs: list[tuple[int, int]]) -> int:
    if not ivs:
        return 0
    return sum(i.length() for i in _col(chrom, ivs).to_intervals_list())


def intersection_bp(chrom: str, a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    if not a or not b:
        return 0
    r = _col(chrom, a).intersect(_col(chrom, b))
    return sum(i.length() for i in r.to_intervals_list())


def jaccard(chrom_a: str, ivs_a, chrom_b: str, ivs_b) -> float:
    """bp Jaccard of two interval sets. 0 if on different chromosomes."""
    if chrom_a != chrom_b:
        return 0.0
    inter = intersection_bp(chrom_a, ivs_a, ivs_b)
    union = covered_bp(chrom_a, ivs_a) + covered_bp(chrom_b, ivs_b) - inter
    return inter / union if union > 0 else 0.0


def reciprocal_overlap(chrom_a, s_a, e_a, chrom_b, s_b, e_b) -> float:
    """Reciprocal-overlap fraction of two single intervals (min of the two
    coverage fractions), 0 on different chromosomes."""
    if chrom_a != chrom_b:
        return 0.0
    inter = GI(chrom_a, s_a, e_a, Strand.PLUS).overlap(GI(chrom_b, s_b, e_b, Strand.PLUS))
    if inter <= 0:
        return 0.0
    return min(inter / (e_a - s_a), inter / (e_b - s_b))
