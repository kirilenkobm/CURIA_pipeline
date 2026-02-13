import json
from pathlib import Path
from typing import Dict, List, Tuple
# TODO: check if pyrion's vectorized ops are used

def _parse_region(region: str) -> Tuple[str, int, int]:
    chrom, rest = region.split(":")
    start_s, end_s = rest.split("-")
    return chrom, int(start_s), int(end_s)


def _overlap_len(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def merge_query_regions(
    rna_regions_path: Path,
    short_joblist_path: Path,
    out_json_path: Path,
    out_jobs_path: Path,
    clusters_json_path: Path,
    ultimate_to_query_path: Path | None = None,
) -> None:
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

    entries = []
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
            if (transcript_id, chain_id) in short_pairs:
                continue
            query_region = parts[region_idx]
            transcript_strand = int(parts[tstrand_idx])
            chain_strand = int(parts[cstrand_idx])
            query_strand = 1 if transcript_strand == chain_strand else -1
            biotype = parts[biotype_idx]
            chrom, start, end = _parse_region(query_region)
            entries.append(
                {
                    "transcript_id": transcript_id,
                    "chain_id": chain_id,
                    "biotype": biotype,
                    "query_region": query_region,
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "strand": query_strand,
                }
            )

    groups: Dict[Tuple[str, int], List[dict]] = {}
    for e in entries:
        groups.setdefault((e["chrom"], e["strand"]), []).append(e)

    merged_mapping: Dict[str, List[dict]] = {}
    clusters_output = {}
    ultimate_to_query = {}
    merged_regions: List[Tuple[str, int, int, int, List[dict]]] = []
    for (chrom, strand), group in groups.items():
        group_sorted = sorted(group, key=lambda x: x["start"])
        clusters: List[List[dict]] = []
        for entry in group_sorted:
            placed = False
            for cluster in clusters:
                if any(
                    _overlap_len((entry["start"], entry["end"]), (c["start"], c["end"]))
                    >= 0.5 * min(entry["end"] - entry["start"], c["end"] - c["start"])
                    for c in cluster
                ):
                    cluster.append(entry)
                    placed = True
                    break
            if not placed:
                clusters.append([entry])

        for cluster in clusters:
            c_start = min(e["start"] for e in cluster)
            c_end = max(e["end"] for e in cluster)
            merged_regions.append((chrom, c_start, c_end, strand, cluster))

    for idx, (chrom, start, end, strand, cluster) in enumerate(merged_regions, start=1):
        merged_id = f"query_merged_region_{idx}"
        merged_mapping[merged_id] = [
            {
                "transcript_id": e["transcript_id"],
                "chain_id": e["chain_id"],
                "query_region": e["query_region"],
                "strand": e["strand"],
                "biotype": e["biotype"],
            }
            for e in cluster
        ]
        clusters_output[merged_id] = {
            "merged_region": {
                "chrom": chrom,
                "start": start,
                "end": end,
                "strand": strand,
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
            ultimate_id = e["transcript_id"]
            if ultimate_id not in ultimate_to_query:
                ultimate_to_query[ultimate_id] = []
            ultimate_to_query[ultimate_id].append(merged_id)

    with open(out_json_path, "w") as f:
        json.dump(merged_mapping, f, indent=2)

    with open(clusters_json_path, "w") as f:
        json.dump(clusters_output, f, indent=2)

    if ultimate_to_query_path:
        with open(ultimate_to_query_path, "w") as f:
            json.dump(ultimate_to_query, f, indent=2)

    with open(out_jobs_path, "w") as f:
        f.write("transcript_id\tchain_id\tmerged_query_id\tbiotype\tchrom\tstart\tend\tstrand\n")
        for idx, (chrom, start, end, strand, cluster) in enumerate(merged_regions, start=1):
            merged_id = f"query_merged_region_{idx}"
            for e in cluster:
                f.write(
                    f"{e['transcript_id']}\t{e['chain_id']}\t{merged_id}\t"
                    f"{e['biotype']}\t{chrom}\t{start}\t{end}\t{strand}\n"
                )

