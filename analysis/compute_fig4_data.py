#!/usr/bin/env python3
"""Precompute the data-heavy panels of Figure 4 (short-ncRNA MMD behaviour).

Panels A and B need the reference/query 2bit files (sequence identity) and the
mm39 GENCODE annotation (overlap) plus numba/pyrion --- too heavy to run inside
make_figures.py. Following the compute_fig1_embeddings.py / compute_fig3b_dotplot.py
pattern, this writes a light cache (analysis/data/fig4_mmd.npz + .json) that the
`fig4_mmd` builder loads without torch / 2bit / numba on the path.

Panel C (MMD by biotype) is cheap and stays in make_figures.py (reads the TSV).

Usage:
    .venv/bin/python analysis/compute_fig4_data.py \
        --results rinalmo_version_outputs/hg38_vs_mm39 \
        --ref-2bit input_data/2bit/hg38.2bit --query-2bit input_data/2bit/mm39.2bit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for sub in ("pipeline", "RiNALMo"):
    sys.path.insert(0, str(REPO / "modules" / sub))
from short_ncrna import _get_spliced_sequence, _extract_sequence  # noqa: E402
from pyrion import TwoBitAccessor                                 # noqa: E402
from pyrion.core.intervals import GenomicInterval                 # noqa: E402
from pyrion.core.strand import Strand                             # noqa: E402
from pyrion.io.bed import read_bed12_file                         # noqa: E402
from pyrion.ops.interval_ops import intersect_intervals           # noqa: E402

from numba import njit  # noqa: E402

DATA = REPO / "analysis" / "data"


@njit(cache=True)
def _edit_ident(a, b):
    """Levenshtein-based identity of two int-encoded seqs: 1 - dist/max(len)."""
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    prev = np.arange(lb + 1)
    cur = np.zeros(lb + 1, dtype=np.int64)
    for i in range(1, la + 1):
        cur[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d = prev[j] + 1
            l = cur[j - 1] + 1
            diag = prev[j - 1] + cost
            m = d if d < l else l
            m = m if m < diag else diag
            cur[j] = m
        prev, cur = cur, prev
    return 1.0 - prev[lb] / max(la, lb)


_MAP = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4}


def _enc(s):
    return np.array([_MAP.get(c, 4) for c in s.upper()], dtype=np.int64)


def _seqid_vs_mmd(sh, args):
    """Panel A: pairwise sequence identity (%) vs MMD for matched short ncRNAs.

    Mirrors analysis/pair_numbers.py: sample <=max-seqid loci, splice the
    reference transcript and extract the matched query span from the 2bit files,
    and Levenshtein-identity the two."""
    ref_acc = TwoBitAccessor(str(args.ref_2bit))
    qry_acc = TwoBitAccessor(str(args.query_2bit))
    ref_bed = {t.id: t for t in read_bed12_file(str(args.results / "reference_union_transcripts.bed"))}
    _edit_ident(_enc("ACGU"), _enc("ACGU"))  # warm up numba

    ids, mmds = [], []
    sub = sh.sample(min(len(sh), args.max_seqid), random_state=0)
    for _, r in sub.iterrows():
        t = ref_bed.get(r["transcript_id"])
        if t is None:
            continue
        try:
            rseq = _get_spliced_sequence(t, ref_acc)
            qseq = _extract_sequence(qry_acc, r["chrom"], int(r["start"]), int(r["end"]),
                                     -1 if r["strand"] == "-" else 1)
        except Exception:
            continue
        if not rseq or not qseq or "N" in (rseq + qseq).upper():
            continue
        ids.append(_edit_ident(_enc(rseq), _enc(qseq)) * 100)
        mmds.append(float(r["mmd_score"]))
    return np.asarray(ids), np.asarray(mmds)


def _annotation_overlap(sh, args):
    """Panel B: exonic overlap of each short-ncRNA locus with mm39 annotation.

    Same method as preprint__deprecated/island_annotation_overlap.ipynb: for each
    locus, find annotation transcripts spanning it and intersect the locus with
    their exon blocks; overlap fraction = best exonic bp / locus length. tRNAs are
    added because GENCODE under-annotates them."""
    anno_dir = args.annotation
    beds = [anno_dir / "mm39_gencode_all_transcripts.bed"]
    trna = anno_dir / "mm39-tRNAs.bed"
    if trna.exists():
        beds.append(trna)
    annos = [read_bed12_file(str(b)) for b in beds]

    mmd, any_ov, ov50 = [], [], []
    for _, r in sh.iterrows():
        start, end = int(r["start"]), int(r["end"])
        llen = end - start
        if llen <= 0:
            continue
        strand = Strand.MINUS if r["strand"] == "-" else Strand.PLUS
        gi = GenomicInterval(r["chrom"], start, end, strand)
        loc = np.array([[start, end]], dtype=np.int32)
        best_bp = 0
        for anno in annos:
            for t in anno.get_transcripts_in_interval(gi):
                isect = intersect_intervals(loc, t.blocks)
                if len(isect) > 0:
                    bp = int(np.sum(isect[:, 1] - isect[:, 0]))
                    if bp > best_bp:
                        best_bp = bp
        mmd.append(float(r["mmd_score"]))
        any_ov.append(best_bp > 0)
        ov50.append(best_bp / llen >= 0.5)
    return np.asarray(mmd), np.asarray(any_ov, dtype=bool), np.asarray(ov50, dtype=bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path,
                    default=REPO / "rinalmo_version_outputs/hg38_vs_mm39")
    ap.add_argument("--ref-2bit", type=Path, default=REPO / "input_data/2bit/hg38.2bit")
    ap.add_argument("--query-2bit", type=Path, default=REPO / "input_data/2bit/mm39.2bit")
    ap.add_argument("--annotation", type=Path,
                    default=REPO / "input_data/mm39_annotation_validation")
    ap.add_argument("--max-seqid", type=int, default=1500, help="cap loci for seq-id (speed)")
    args = ap.parse_args()

    sh = pd.read_csv(args.results / "query_annotation/short_ncRNA_details.tsv", sep="\t")

    print(f"# seq-identity vs MMD (sampling <= {args.max_seqid} of {len(sh)} loci) ...")
    seqid, mmd_a = _seqid_vs_mmd(sh, args)
    from scipy.stats import pearsonr, spearmanr
    pr = float(pearsonr(seqid, mmd_a)[0]) if len(seqid) > 2 else float("nan")
    sp = float(spearmanr(seqid, mmd_a)[0]) if len(seqid) > 2 else float("nan")
    print(f"#   n={len(seqid)}  Pearson r={pr:.3f}  Spearman rho={sp:.3f}")

    print(f"# annotation overlap for all {len(sh)} loci ...")
    mmd_b, any_ov, ov50 = _annotation_overlap(sh, args)
    print(f"#   any-overlap {100*any_ov.mean():.1f}%   >=50% {100*ov50.mean():.1f}%")
    lo = mmd_b < 0.1
    if lo.any():
        print(f"#   any-overlap at MMD<0.1: {100*any_ov[lo].mean():.1f}%")

    # Panel C (cheap): MMD by biotype, carried in the cache so the builder needs
    # no results dir on disk.
    C_mmd = sh["mmd_score"].to_numpy(dtype=float)
    C_biotype = sh["biotype"].astype(str).to_numpy(dtype="<U32")

    DATA.mkdir(parents=True, exist_ok=True)
    np.savez(DATA / "fig4_mmd.npz",
             A_seqid=seqid, A_mmd=mmd_a,
             B_mmd=mmd_b, B_any=any_ov, B_over50=ov50,
             C_mmd=C_mmd, C_biotype=C_biotype)
    (DATA / "fig4_mmd.json").write_text(json.dumps({
        "pair": args.results.name,
        "seqid_pearson_r": round(pr, 3),
        "seqid_spearman_rho": round(sp, 3),
        "seqid_n": int(len(seqid)),
        "overlap_any_frac": round(float(any_ov.mean()), 3),
        "overlap_50_frac": round(float(ov50.mean()), 3),
    }, indent=2))
    print(f"wrote {DATA/'fig4_mmd.npz'}")


if __name__ == "__main__":
    main()
