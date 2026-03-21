from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from pyrion import read_bed12_file, read_gene_data
from pyrion.core.genes import Transcript, TranscriptsCollection
from pyrion.core.strand import Strand
from pyrion.ops.transcript_serialization import save_transcripts_collection_to_bed12

LNCRNA_BIOTYPE = "lncRNA"


# ------------------------------------------------------------------
# Union-find for clustering overlapping lncRNA union transcripts
# ------------------------------------------------------------------

class _UnionFind:
    def __init__(self):
        self._parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        while self._parent.get(x, x) != x:
            self._parent[x] = self._parent.get(self._parent[x], self._parent[x])
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def groups(self, keys: List[str]) -> Dict[str, List[str]]:
        clusters: Dict[str, List[str]] = defaultdict(list)
        for k in keys:
            clusters[self.find(k)].append(k)
        return dict(clusters)


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    sorted_ivs = sorted(intervals)
    merged = [sorted_ivs[0]]
    for start, end in sorted_ivs[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _merge_overlapping_lncrna_unions(
    union_transcripts: List[Transcript],
    metadata_rows: List[Tuple[str, str, str]],
    union_to_isoforms: dict,
) -> Tuple[List[Transcript], List[Tuple[str, str, str]], dict]:
    """Cluster lncRNA union transcripts by exon overlap, merge each cluster."""
    uid_to_transcript = {}
    uid_to_meta = {}
    lncrna_uids: List[str] = []
    non_lncrna_transcripts: List[Transcript] = []
    non_lncrna_meta: List[Tuple[str, str, str]] = []

    for t, meta_row in zip(union_transcripts, metadata_rows):
        uid_to_transcript[t.id] = t
        uid_to_meta[t.id] = meta_row
        if meta_row[2] == LNCRNA_BIOTYPE:
            lncrna_uids.append(t.id)
        else:
            non_lncrna_transcripts.append(t)
            non_lncrna_meta.append(meta_row)

    if len(lncrna_uids) < 2:
        return union_transcripts, metadata_rows, union_to_isoforms

    # Group by (chrom, strand)
    chrom_strand_groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for uid in lncrna_uids:
        t = uid_to_transcript[uid]
        chrom_strand_groups[(t.chrom, str(t.strand))].append(uid)

    uf = _UnionFind()
    for group_uids in chrom_strand_groups.values():
        if len(group_uids) < 2:
            continue
        block_index: List[Tuple[int, int, str]] = []
        for uid in group_uids:
            for block in uid_to_transcript[uid].blocks:
                block_index.append((int(block[0]), int(block[1]), uid))
        block_index.sort()

        active_end = -1
        active_uids: List[str] = []
        for start, end, uid in block_index:
            if start < active_end:
                for prev in active_uids:
                    uf.union(uid, prev)
                active_end = max(active_end, end)
                if uid not in active_uids:
                    active_uids.append(uid)
            else:
                active_end = end
                active_uids = [uid]

    clusters = uf.groups(lncrna_uids)

    merged_transcripts: List[Transcript] = list(non_lncrna_transcripts)
    merged_meta: List[Tuple[str, str, str]] = list(non_lncrna_meta)
    merged_isoforms = {
        uid: union_to_isoforms[uid]
        for uid in union_to_isoforms
        if uid_to_meta.get(uid, ("", "", ""))[2] != LNCRNA_BIOTYPE
    }

    for members in clusters.values():
        if len(members) == 1:
            uid = members[0]
            merged_transcripts.append(uid_to_transcript[uid])
            merged_meta.append(uid_to_meta[uid])
            merged_isoforms[uid] = union_to_isoforms[uid]
            continue

        # Pick the first ID (by original insertion order)
        representative = members[0]
        rep_t = uid_to_transcript[representative]
        all_blocks: List[Tuple[int, int]] = []
        all_isoforms: List[str] = []
        all_gene_ids: List[str] = []
        for uid in members:
            t = uid_to_transcript[uid]
            for block in t.blocks:
                all_blocks.append((int(block[0]), int(block[1])))
            all_isoforms.extend(union_to_isoforms[uid]["isoforms"])
            all_gene_ids.append(uid_to_meta[uid][1])

        merged_blocks = _merge_intervals(all_blocks)
        blocks_arr = np.array(merged_blocks, dtype=np.int64)
        merged_t = Transcript(
            blocks=blocks_arr,
            strand=rep_t.strand,
            chrom=rep_t.chrom,
            id=representative,
        )
        merged_transcripts.append(merged_t)
        merged_meta.append((representative, uid_to_meta[representative][1], LNCRNA_BIOTYPE))
        merged_isoforms[representative] = {
            "isoforms": all_isoforms,
            "biotype": LNCRNA_BIOTYPE,
            "merged_gene_ids": all_gene_ids,
        }

    return merged_transcripts, merged_meta, merged_isoforms


def collapse_to_union_transcripts(
    bed12_path: str,
    metadata_tsv_path: str,
    out_bed12_path: str,
    out_metadata_tsv_path: str,
    union_to_isoforms_path: str | None = None,
    id_prefix: str = "U_",
) -> Tuple[str, str]:
    transcripts = read_bed12_file(bed12_path)
    gene_data = read_gene_data(
        metadata_tsv_path,
        gene_column="gene_id",
        transcript_id_column="transcript_id",
        transcript_type_column="transcript_biotype",
    )
    transcripts.bind_gene_data(gene_data)

    union_transcripts: List[Transcript] = []
    metadata_rows: List[Tuple[str, str, str]] = []
    union_to_isoforms = {}

    for gene_id in gene_data.gene_ids:
        gene = transcripts.get_gene_by_id(gene_id)
        if gene is None:
            continue

        union_transcript = gene.to_union_transcript(id_prefix=id_prefix)

        original_biotypes = [t.biotype for t in gene.transcripts if t.biotype]
        biotype = "unknown"
        if original_biotypes:
            if any(b and "protein_coding" in b.lower().replace("-", "_").replace(" ", "_") for b in original_biotypes):
                biotype = "protein_coding"
            else:
                biotype = original_biotypes[0]

        union_transcripts.append(union_transcript)
        metadata_rows.append((union_transcript.id, gene_id, biotype))
        union_to_isoforms[union_transcript.id] = {
            "isoforms": [t.id for t in gene.transcripts],
            "biotype": biotype,
        }

    # Merge overlapping lncRNA union transcripts into single locus unions
    union_transcripts, metadata_rows, union_to_isoforms = _merge_overlapping_lncrna_unions(
        union_transcripts, metadata_rows, union_to_isoforms,
    )

    out_bed12_path = str(out_bed12_path)
    out_metadata_tsv_path = str(out_metadata_tsv_path)
    Path(out_bed12_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_metadata_tsv_path).parent.mkdir(parents=True, exist_ok=True)

    collection = TranscriptsCollection(union_transcripts)
    save_transcripts_collection_to_bed12(collection, out_bed12_path)

    with open(out_metadata_tsv_path, "w") as f:
        f.write("transcript_id\tgene_id\tbiotype\n")
        for transcript_id, gene_id, biotype in metadata_rows:
            f.write(f"{transcript_id}\t{gene_id}\t{biotype}\n")

    if union_to_isoforms_path:
        with open(union_to_isoforms_path, "w") as f:
            json.dump(union_to_isoforms, f, indent=2)

    return out_bed12_path, out_metadata_tsv_path
