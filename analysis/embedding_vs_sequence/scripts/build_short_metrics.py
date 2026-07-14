#!/usr/bin/env python3
"""Analysis A table builder (GPU-free).

For every human->mouse short-ncRNA prediction, assemble the stored embedding MMD
score, the annotation-support label (any/50/99% exonic overlap, recomputed exactly
as compute_fig4_data.py does), the biotype, and every conventional nucleotide
sequence metric (seqmetrics.py). Reference sequence = spliced human transcript;
query sequence = extracted mouse locus. No RiNALMo/GPU.

Output: analysis/embedding_vs_sequence/short_ncrna_metrics.tsv

Run:
  .venv/bin/python analysis/embedding_vs_sequence/scripts/build_short_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for sub in ("pipeline", "RiNALMo"):
    sys.path.insert(0, str(REPO / "modules" / sub))

import seqmetrics as sm  # noqa: E402
from short_ncrna import _get_spliced_sequence, _extract_sequence  # noqa: E402
from pyrion import TwoBitAccessor  # noqa: E402
from pyrion.core.intervals import GenomicInterval  # noqa: E402
from pyrion.core.strand import Strand  # noqa: E402
from pyrion.io.bed import read_bed12_file  # noqa: E402
from pyrion.ops.interval_ops import intersect_intervals  # noqa: E402

RESULTS = REPO / "preprint_results/hg38_vs_mm39"
REF_2BIT = REPO / "input_data/2bit/hg38.2bit"
QRY_2BIT = REPO / "input_data/2bit/mm39.2bit"
ANNO = REPO / "input_data/mm39_annotation_validation"
OUT = REPO / "analysis/embedding_vs_sequence/short_ncrna_metrics.tsv"


def load_annotation():
    beds = [ANNO / "mm39_gencode_all_transcripts.bed"]
    trna = ANNO / "mm39-tRNAs.bed"
    if trna.exists():
        beds.append(trna)
    return [read_bed12_file(str(b)) for b in beds]


def overlap_bp(annos, chrom, start, end, strand):
    strand_e = Strand.MINUS if strand == "-" else Strand.PLUS
    gi = GenomicInterval(chrom, start, end, strand_e)
    loc = np.array([[start, end]], dtype=np.int32)
    best = 0
    for anno in annos:
        for t in anno.get_transcripts_in_interval(gi):
            isect = intersect_intervals(loc, t.blocks)
            if len(isect) > 0:
                bp = int(np.sum(isect[:, 1] - isect[:, 0]))
                if bp > best:
                    best = bp
    return best


def main():
    sh = pd.read_csv(RESULTS / "query_annotation/short_ncRNA_details.tsv", sep="\t")
    ref_acc = TwoBitAccessor(str(REF_2BIT))
    qry_acc = TwoBitAccessor(str(QRY_2BIT))
    ref_bed = {t.id: t for t in read_bed12_file(str(RESULTS / "reference_union_transcripts.bed"))}
    annos = load_annotation()

    rows = []
    drops = {"no_ref_transcript": 0, "empty_seq": 0}
    for _, r in sh.iterrows():
        t = ref_bed.get(r["transcript_id"])
        if t is None:
            drops["no_ref_transcript"] += 1
            continue
        try:
            rseq = _get_spliced_sequence(t, ref_acc)
            qseq = _extract_sequence(qry_acc, r["chrom"], int(r["start"]), int(r["end"]),
                                     -1 if r["strand"] == "-" else 1)
        except Exception:
            drops["empty_seq"] += 1
            continue
        if not rseq or not qseq:
            drops["empty_seq"] += 1
            continue

        m = sm.all_metrics(rseq, qseq, best_orientation=False)
        start, end = int(r["start"]), int(r["end"])
        llen = end - start
        bp = overlap_bp(annos, r["chrom"], start, end,
                        "-" if r["strand"] == "-" else "+") if llen > 0 else 0
        rows.append(dict(
            transcript_id=r["transcript_id"], gene_id=r["gene_id"],
            biotype=r["biotype"], chrom=r["chrom"], start=start, end=end,
            strand=r["strand"], chain_id=r["chain_id"],
            mmd=float(r["mmd_score"]),
            overlap_bp=bp, locus_len=llen,
            overlap_any=bool(bp > 0),
            overlap_50=bool(llen > 0 and bp / llen >= 0.5),
            overlap_99=bool(llen > 0 and bp / llen >= 0.99),
            **m,
        ))

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, sep="\t", index=False)

    # --- sanity: recomputed Levenshtein identity vs the cached fig4 A_seqid sample
    print(f"# rows written: {len(df)}  drops: {drops}")
    print(f"# biotypes:\n{df.biotype.value_counts()}")
    print(f"# any-overlap frac = {df.overlap_any.mean():.3f} (expect ~0.524)")
    print(f"# MMD vs ident_levenshtein Pearson = "
          f"{np.corrcoef(df.mmd, df.ident_levenshtein)[0,1]:.3f} (expect ~ -0.5..-0.8)")
    print(f"# has_N rows: {int(df.has_N.sum())}")
    print(f"# wrote {OUT}")


if __name__ == "__main__":
    main()
