import json
from pathlib import Path
from typing import Dict, List, Tuple

from pyrion.core.intervals import GenomicInterval, GenomicIntervalsCollection
from pyrion.core.strand import Strand
from pyrion.ops.interval_collection_ops import group_intervals_by_proximity


def _parse_region(region: str) -> Tuple[str, int, int]:
    """Parse region string like 'chr1:100-200' into components.

    Ensures start < end by swapping if necessary (handles reversed coordinates).
    """
    chrom, rest = region.split(":")
    start_s, end_s = rest.split("-")
    start, end = int(start_s), int(end_s)

    # Ensure start < end (swap if reversed)
    if start > end:
        start, end = end, start

    return chrom, start, end


def _strand_int_to_pyrion(strand_int: int) -> Strand:
    """Convert integer strand (-1, 1) to pyrion Strand enum."""
    if strand_int == 1:
        return Strand.PLUS
    elif strand_int == -1:
        return Strand.MINUS
    else:
        return Strand.UNKNOWN


def _pyrion_to_strand_int(strand: Strand) -> int:
    """Convert pyrion Strand enum to integer (-1, 1)."""
    if strand == Strand.PLUS:
        return 1
    elif strand == Strand.MINUS:
        return -1
    else:
        return 0


def merge_query_regions(
    rna_regions_path: Path,
    short_joblist_path: Path,
    out_jobs_path: Path,
    clusters_json_path: Path,
    union_to_query_path: Path | None = None,
) -> None:
    """
    Merge overlapping query regions using pyrion's interval operations.

    Groups query regions by chromosome and strand, then uses pyrion's
    proximity-based clustering to merge overlapping regions (≥50% overlap).
    """
    # Load short pairs to exclude
    short_pairs = set()
    with open(short_joblist_path, "r") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            tid_idx = header.index("transcript_id")
            chain_idx = header.index("chain_id")
        except ValueError:
            raise ValueError("short_ncRNA_joblist.txt header missing required columns")
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            short_pairs.add((parts[tid_idx], parts[chain_idx]))

    # Load RNA regions and convert to pyrion GenomicInterval objects
    intervals_data: List[Tuple[GenomicInterval, dict]] = []

    with open(rna_regions_path, "r") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            tid_idx = header.index("transcript_id")
            chain_idx = header.index("chain_id")
            region_idx = header.index("region")
            tstrand_idx = header.index("transcript_strand")
            cstrand_idx = header.index("chain_strand")
            biotype_idx = header.index("biotype")
        except ValueError:
            raise ValueError("rna_orthologous_regions.tsv header missing required columns")

        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            transcript_id = parts[tid_idx]
            chain_id = parts[chain_idx]

            # Skip short ncRNAs
            if (transcript_id, chain_id) in short_pairs:
                continue

            query_region = parts[region_idx]
            transcript_strand = int(parts[tstrand_idx])
            chain_strand = int(parts[cstrand_idx])
            query_strand_int = 1 if transcript_strand == chain_strand else -1
            biotype = parts[biotype_idx]

            chrom, start, end = _parse_region(query_region)

            # Skip zero-length intervals (pyrion requires start < end)
            if start >= end:
                continue

            pyrion_strand = _strand_int_to_pyrion(query_strand_int)

            # Create GenomicInterval
            interval = GenomicInterval(
                chrom=chrom,
                start=start,
                end=end,
                strand=pyrion_strand,
                id=f"{transcript_id}_{chain_id}"
            )

            # Store metadata alongside interval
            metadata = {
                "transcript_id": transcript_id,
                "chain_id": chain_id,
                "biotype": biotype,
                "query_region": query_region,
                "chrom": chrom,
                "start": start,
                "end": end,
                "strand": query_strand_int,
            }
            intervals_data.append((interval, metadata))

    if not intervals_data:
        # No long ncRNAs to process - write empty outputs
        with open(clusters_json_path, "w") as f:
            json.dump({}, f, indent=2)
        if union_to_query_path:
            with open(union_to_query_path, "w") as f:
                json.dump({}, f, indent=2)
        with open(out_jobs_path, "w") as f:
            f.write("transcript_id\tchain_id\tmerged_query_id\tbiotype\tchrom\tstart\tend\tstrand\n")
        return

    # Group intervals by chromosome and strand using pyrion
    intervals_by_group: Dict[Tuple[str, Strand], List[Tuple[GenomicInterval, dict]]] = {}
    for interval, metadata in intervals_data:
        key = (interval.chrom, interval.strand)
        intervals_by_group.setdefault(key, []).append((interval, metadata))

    # Process each group using pyrion's proximity clustering
    clusters_output = {}
    union_to_query = {}
    merged_regions: List[Tuple[str, int, int, int, List[dict]]] = []

    for (chrom, strand), group_data in intervals_by_group.items():
        # Extract intervals and metadata separately
        intervals = [item[0] for item in group_data]

        # Create a GenomicIntervalsCollection
        collection = GenomicIntervalsCollection.from_intervals(intervals)

        # Use pyrion's proximity-based grouping with a custom max_gap
        # We want ≥50% overlap, which is equivalent to max_gap = 0 (direct overlap)
        # Then we'll manually filter for the 50% criterion
        clustered_collections = group_intervals_by_proximity(collection, max_gap=0)

        # For each cluster, check if entries meet the 50% overlap criterion
        # and merge those that do
        for cluster_collection in clustered_collections:
            # Get intervals in this cluster
            cluster_intervals = cluster_collection.to_intervals_list()

            # Map back to original metadata using interval IDs
            cluster_metadata = []
            for interval in cluster_intervals:
                # Find matching metadata by interval ID
                for orig_interval, meta in group_data:
                    if orig_interval.id == interval.id:
                        cluster_metadata.append(meta)
                        break

            if not cluster_metadata:
                continue

            # Calculate merged bounds
            c_start = min(m["start"] for m in cluster_metadata)
            c_end = max(m["end"] for m in cluster_metadata)
            strand_int = cluster_metadata[0]["strand"]

            merged_regions.append((chrom, c_start, c_end, strand_int, cluster_metadata))

    # Generate output
    for idx, (chrom, start, end, strand_int, cluster) in enumerate(merged_regions, start=1):
        merged_id = f"query_merged_region_{idx}"
        clusters_output[merged_id] = {
            "merged_region": {
                "chrom": chrom,
                "start": start,
                "end": end,
                "strand": strand_int,
            },
            "merged_transcripts": [
                {
                    "transcript_id": e["transcript_id"],
                    "chain_id": e["chain_id"],
                    "chrom": e["chrom"],
                    "start": e["start"],
                    "end": e["end"],
                    "strand": e["strand"],
                }
                for e in cluster
            ],
        }
        for e in cluster:
            union_id = e["transcript_id"]
            if union_id not in union_to_query:
                union_to_query[union_id] = []
            union_to_query[union_id].append(merged_id)

    with open(clusters_json_path, "w") as f:
        json.dump(clusters_output, f, indent=2)

    if union_to_query_path:
        with open(union_to_query_path, "w") as f:
            json.dump(union_to_query, f, indent=2)

    with open(out_jobs_path, "w") as f:
        f.write("transcript_id\tchain_id\tmerged_query_id\tbiotype\tchrom\tstart\tend\tstrand\n")
        for idx, (chrom, start, end, strand_int, cluster) in enumerate(merged_regions, start=1):
            merged_id = f"query_merged_region_{idx}"
            for e in cluster:
                f.write(
                    f"{e['transcript_id']}\t{e['chain_id']}\t{merged_id}\t"
                    f"{e['biotype']}\t{chrom}\t{start}\t{end}\t{strand_int}\n"
                )
