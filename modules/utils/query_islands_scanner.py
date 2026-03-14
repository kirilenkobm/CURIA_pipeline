from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import label

import pyrion
from pyrion import TwoBitAccessor
from pyrion.core.nucleotide_sequences import NucleotideSequence


@dataclass(frozen=True)
class QueryIslandScanJob:
    merged_query_id: str
    chrom: str
    start: int
    end: int
    strand: int


def _parse_region(region: str) -> Tuple[str, int, int]:
    chrom, rest = region.split(":")
    start_s, end_s = rest.split("-")
    return chrom, int(start_s), int(end_s)


def write_query_islands_joblist(
    query_regions_clusters_path: str,
    ref_islands_json_path: str,
    out_path: str,
    min_query_length: int = 128,
    max_query_length: int = 1_500_000,
    max_query_ref_ratio: float = 4.0,
) -> int:
    """
    Create joblist from merged query regions clusters for island scanning.

    FILTERS:
    - Only transcripts with islands in the reference
    - Query region length >= min_query_length (default 128 bp)
    - Query region length <= max_query_length (default 1.5M bp)
    - Query region length <= max_query_ref_ratio * reference transcript length (default 4x)

    Format: merged_query_id, chrom, start, end, strand
    """
    # Load reference islands data to find transcripts with islands
    with open(ref_islands_json_path, "r") as f:
        ref_islands_data = json.load(f)

    transcripts_with_islands = {
        tid for tid, data in ref_islands_data.items()
        if data.get("islands")
    }

    print(f"# Reference: {len(transcripts_with_islands)} transcripts have functional islands")

    with open(query_regions_clusters_path, "r") as f:
        clusters = json.load(f)

    # Filtering statistics
    filter_stats = {
        "total": len(clusters),
        "no_islands": 0,
        "too_short": 0,
        "too_long_abs": 0,
        "too_long_ratio": 0,
        "kept": 0,
    }

    # Track unique transcripts and query mappings in kept clusters
    kept_transcripts = set()
    total_query_mappings = 0

    with open(out_path, "w") as dst:
        dst.write("merged_query_id\tchrom\tstart\tend\tstrand\n")

        for merged_id, cluster_data in clusters.items():
            # Get ultimate transcript IDs from merged_transcripts
            merged_transcripts = cluster_data.get("merged_transcripts", [])
            ultimate_ids = [t["transcript_id"] for t in merged_transcripts]

            # Filter 1: Check if ANY of the ultimate transcripts have islands
            if not any(uid in transcripts_with_islands for uid in ultimate_ids):
                filter_stats["no_islands"] += 1
                continue

            merged_region = cluster_data["merged_region"]
            chrom = merged_region["chrom"]
            start = merged_region["start"]
            end = merged_region["end"]
            strand = merged_region["strand"]
            query_length = end - start

            # Filter 2: Too short
            if query_length < min_query_length:
                filter_stats["too_short"] += 1
                continue

            # Filter 3: Too long (absolute)
            if query_length > max_query_length:
                filter_stats["too_long_abs"] += 1
                continue

            # Filter 4: Too long relative to reference
            # Check against ALL ultimate transcripts in this cluster
            too_long_for_all = True
            for uid in ultimate_ids:
                if uid in ref_islands_data:
                    ref_length = ref_islands_data[uid].get("total_length", 0)
                    if ref_length > 0 and query_length <= max_query_ref_ratio * ref_length:
                        too_long_for_all = False
                        break

            if too_long_for_all:
                filter_stats["too_long_ratio"] += 1
                continue

            filter_stats["kept"] += 1
            kept_transcripts.update(ultimate_ids)
            total_query_mappings += len(merged_transcripts)
            dst.write(f"{merged_id}\t{chrom}\t{start}\t{end}\t{strand}\n")

    # Print filtering summary
    print(f"# Query island scan job filtering summary:")
    print(f"#   Total query region clusters: {filter_stats['total']}")
    print(f"#   Filtered - no islands in reference: {filter_stats['no_islands']}")
    print(f"#   Filtered - too short (<{min_query_length} bp): {filter_stats['too_short']}")
    print(f"#   Filtered - too long (>{max_query_length:,} bp): {filter_stats['too_long_abs']}")
    print(f"#   Filtered - too long (>{max_query_ref_ratio}x reference): {filter_stats['too_long_ratio']}")
    print(f"#   Kept for island scanning: {filter_stats['kept']} query regions")
    print(f"#     -> {len(kept_transcripts)} unique reference transcripts, {total_query_mappings} total query mappings")

    return filter_stats["kept"]


def _extract_sequence(
    accessor: TwoBitAccessor,
    chrom: str,
    start: int,
    end: int,
    strand: int,
) -> str:
    seq_obj = accessor.fetch(chrom, start, end)
    if strand == -1:
        seq_obj = seq_obj.reverse_complement()
    if not seq_obj.is_rna:
        seq_obj = seq_obj.toggle_type()
    return seq_obj.to_string()


def _smooth_signal(s: np.ndarray, window_len: int = 5) -> np.ndarray:
    """Box filter smoothing."""
    if window_len <= 1:
        return s
    kernel = np.ones(window_len) / window_len
    return np.convolve(s, kernel, mode='same')


def _get_islands(mask: np.ndarray, positions: np.ndarray, window_size: int) -> List[Dict]:
    """Extract continuous islands from binary mask."""
    labels, num_features = label(mask)
    islands = []
    for i in range(1, num_features + 1):
        idx = np.where(labels == i)[0]
        start_pos = int(positions[idx[0]])
        end_pos = int(positions[idx[-1]] + window_size)
        islands.append({
            'start': start_pos,
            'end': end_pos,
            'indices': idx,
            'max_prob': 0.0,
        })
    return islands


class GPUClient:
    def __init__(self, input_queue, output_queue, loop: asyncio.AbstractEventLoop):
        self._input = input_queue
        self._output = output_queue
        self._loop = loop
        self._pending: Dict[Tuple[str, str], asyncio.Future] = {}
        self._lock = threading.Lock()
        self._stopping = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    async def embed(self, worker_id: str, sequence_id: str, sequence: str, mean_pool: bool = False) -> np.ndarray:
        future = self._loop.create_future()
        key = (worker_id, sequence_id)
        with self._lock:
            self._pending[key] = future
        self._input.put((worker_id, sequence_id, sequence, {"mean_pool": mean_pool}))
        return await future

    def stop(self) -> None:
        """Signal reader thread to stop."""
        self._stopping = True

    def _reader(self) -> None:
        while not self._stopping:
            try:
                payload = self._output.get(timeout=0.5)
                # Batched results: payload is a list of (worker_id, sequence_id, emb) tuples
                for worker_id, sequence_id, emb in payload:
                    key = (worker_id, sequence_id)
                    if self._loop.is_closed():
                        return
                    self._loop.call_soon_threadsafe(self._resolve_future, key, emb)
            except Exception:
                # Timeout or event loop closed - check stopping flag
                if self._stopping or self._loop.is_closed():
                    return
                continue

    def _resolve_future(self, key: Tuple[str, str], emb: np.ndarray) -> None:
        with self._lock:
            future = self._pending.pop(key, None)
        if future is not None and not future.done():
            future.set_result(emb)


def _load_joblist(joblist_path: str) -> List[QueryIslandScanJob]:
    jobs: List[QueryIslandScanJob] = []
    with open(joblist_path, "r") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            id_idx = header.index("merged_query_id")
            chrom_idx = header.index("chrom")
            start_idx = header.index("start")
            end_idx = header.index("end")
            strand_idx = header.index("strand")
        except ValueError:
            raise ValueError("query_islands_joblist.txt header missing required columns")

        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            jobs.append(
                QueryIslandScanJob(
                    merged_query_id=parts[id_idx],
                    chrom=parts[chrom_idx],
                    start=int(parts[start_idx]),
                    end=int(parts[end_idx]),
                    strand=int(parts[strand_idx]),
                )
            )
    return jobs


def _ensure_table(conn: sqlite3.Connection, table: str, columns: Dict[str, str]) -> None:
    cols_sql = ", ".join(f"{name} {ctype}" for name, ctype in columns.items())
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({cols_sql})")

    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, ctype in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ctype}")


async def _sqlite_writer(queue: asyncio.Queue, sqlite_path: str) -> None:
    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    columns = {
        "merged_query_id": "TEXT NOT NULL",
        "island_number": "INTEGER NOT NULL",
        "chrom": "TEXT NOT NULL",
        "start": "INTEGER NOT NULL",
        "end": "INTEGER NOT NULL",
        "strand": "INTEGER NOT NULL",
        "max_prob": "REAL",
        "avg_prob": "REAL",
    }
    _ensure_table(conn, "query_islands", columns)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_query_islands ON query_islands(merged_query_id, island_number)"
    )

    while True:
        item = await queue.get()
        if item is None:
            break

        placeholders = ", ".join(["?"] * len(item))
        columns_sql = ", ".join(item.keys())
        updates = ", ".join(f"{k}=excluded.{k}" for k in item.keys())
        sql = (
            f"INSERT INTO query_islands ({columns_sql}) VALUES ({placeholders}) "
            f"ON CONFLICT(merged_query_id, island_number) DO UPDATE SET {updates}"
        )
        conn.execute(sql, tuple(item.values()))
        conn.commit()

    conn.close()


async def _process_query_island_scan(
    job: QueryIslandScanJob,
    query_accessor: TwoBitAccessor,
    gpu: GPUClient,
    logreg_model,
    pca_model,
    window_size: int = 72,
    stride: int = 16,
    smooth_window: int = 5,
    prob_threshold: float = 0.25,
    batch_size: int = 128,
) -> List[Dict]:
    """
    Process a single query region:
    - Extract sequence
    - Create 72nt sliding windows
    - Get embeddings (mean pooled)
    - Apply PCA + LogReg
    - Smooth probabilities
    - Find islands
    """
    from modules.logreg_signal_noise.apply_logreg import score_embeddings

    # Extract full sequence
    sequence = _extract_sequence(query_accessor, job.chrom, job.start, job.end, job.strand)
    seq_len = len(sequence)

    if seq_len < window_size:
        return []

    # Create sliding windows
    windows = []
    positions = []
    for i in range(0, seq_len - window_size + 1, stride):
        windows.append(sequence[i:i + window_size])
        positions.append(i)

    if not windows:
        return []

    positions = np.array(positions)

    # Get embeddings - send all windows at once, then await results
    # This avoids IPC overhead from awaiting each window individually
    tasks = []
    for i, seq in enumerate(windows):
        task = gpu.embed(job.merged_query_id, f"win:{positions[i]}", seq, mean_pool=True)
        tasks.append(task)

    # Await all embeddings in parallel
    embeddings = await asyncio.gather(*tasks)
    embeddings = np.stack(embeddings, axis=0)

    # Apply LogReg model (via PCA)
    probs, _ = score_embeddings(embeddings, model=logreg_model, pca_model=pca_model)

    # Smooth probabilities
    probs_smooth = _smooth_signal(probs, smooth_window)

    # Threshold to find islands
    mask = probs_smooth >= prob_threshold
    islands = _get_islands(mask, positions, window_size)

    # Compute max_prob for each island
    for island in islands:
        idx = island['indices']
        island['max_prob'] = float(np.max(probs_smooth[idx]))
        island['avg_prob'] = float(np.mean(probs_smooth[idx]))

    # Convert to genomic coordinates
    results = []
    for island_num, island in enumerate(islands, start=1):
        if job.strand == 1:
            genomic_start = job.start + island['start']
            genomic_end = job.start + island['end']
        else:
            genomic_end = job.end - island['start']
            genomic_start = job.end - island['end']

        results.append({
            "merged_query_id": job.merged_query_id,
            "island_number": island_num,
            "chrom": job.chrom,
            "start": genomic_start,
            "end": genomic_end,
            "strand": job.strand,
            "max_prob": island['max_prob'],
            "avg_prob": island['avg_prob'],
        })

    return results


async def _worker(
    job_queue: asyncio.Queue,
    result_queue: asyncio.Queue,
    query_accessor: TwoBitAccessor,
    gpu: GPUClient,
    logreg_model,
    pca_model,
    window_size: int,
    stride: int,
    smooth_window: int,
    prob_threshold: float,
    batch_size: int,
    t0: float,
    total_jobs: int,
    max_concurrent: int,
) -> None:
    while True:
        job = await job_queue.get()
        if job is None:
            break

        if job_queue.qsize() % 100 == 0:
            elapsed = time.monotonic() - t0
            remaining = job_queue.qsize() - max_concurrent
            started = total_jobs - remaining
            pct_started = (started / total_jobs * 100) if total_jobs > 0 else 0
            print(f"# Query island scan jobs: {started}/{total_jobs} started ({pct_started:.1f}%), {remaining} in queue (elapsed {elapsed:.1f}s)")

        try:
            islands = await _process_query_island_scan(
                job,
                query_accessor,
                gpu,
                logreg_model,
                pca_model,
                window_size,
                stride,
                smooth_window,
                prob_threshold,
                batch_size,
            )
            for island in islands:
                await result_queue.put(island)
        except Exception as exc:
            print(f"# Error processing {job.merged_query_id}: {exc}")


def run_query_islands_scanner(
    joblist_path: str,
    query_2bit_path: str,
    gpu_input_queue,
    gpu_output_queue,
    sqlite_path: str,
    logreg_model_path: str,
    max_concurrent: int = 10,
    window_size: int = 72,
    stride: int = 16,
    smooth_window: int = 5,
    prob_threshold: float = 0.25,
    batch_size: int = 128,
    output_json_path: Optional[str] = None,
) -> None:
    """
    Scan orthologous query regions for functional islands.

    This identifies potential functional regions (islands) within query genome regions
    that are orthologous to reference lncRNAs. These islands will later be used
    for lncRNA annotation in subsequent pipeline steps.
    """
    from modules.logreg_signal_noise.apply_logreg import load_logreg_model

    async def _run() -> None:
        t0 = time.monotonic()
        jobs = _load_joblist(joblist_path)
        print(f"# Loaded {len(jobs)} query region island scanning jobs.")
        print(f"# Query islands scanner workers: {max_concurrent}")

        # Load model (PCA not needed - GPU executor already applies it)
        logreg_model = load_logreg_model(logreg_model_path)
        pca_model = None  # Not used - embeddings are already PCA-reduced by GPU executor

        query_accessor = TwoBitAccessor(query_2bit_path)

        loop = asyncio.get_running_loop()
        gpu = GPUClient(gpu_input_queue, gpu_output_queue, loop)

        result_queue: asyncio.Queue = asyncio.Queue()
        writer_task = asyncio.create_task(_sqlite_writer(result_queue, sqlite_path))

        job_queue: asyncio.Queue = asyncio.Queue()
        total_jobs = len(jobs)
        for job in jobs:
            await job_queue.put(job)
        for _ in range(max_concurrent):
            await job_queue.put(None)

        workers = [
            asyncio.create_task(
                _worker(
                    job_queue,
                    result_queue,
                    query_accessor,
                    gpu,
                    logreg_model,
                    pca_model,
                    window_size,
                    stride,
                    smooth_window,
                    prob_threshold,
                    batch_size,
                    t0,
                    total_jobs,
                    max_concurrent,
                )
            )
            for _ in range(max_concurrent)
        ]
        await asyncio.gather(*workers)

        await result_queue.put(None)
        await writer_task

        # Stop GPU client reader thread
        gpu.stop()

        elapsed_total = time.monotonic() - t0
        print(f"# Query islands scanner finished in {elapsed_total:.1f}s.")

        # Export to JSON if requested
        if output_json_path:
            conn = sqlite3.connect(sqlite_path)
            cursor = conn.execute(
                "SELECT merged_query_id, island_number, chrom, start, end, strand, max_prob, avg_prob "
                "FROM query_islands ORDER BY merged_query_id, island_number"
            )

            islands_by_query = {}
            for merged_query_id, island_num, chrom, start, end, strand, max_prob, avg_prob in cursor:
                if merged_query_id not in islands_by_query:
                    islands_by_query[merged_query_id] = []
                islands_by_query[merged_query_id].append({
                    "island_number": island_num,
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "max_prob": max_prob,
                    "avg_prob": avg_prob,
                })

            with open(output_json_path, "w") as f:
                json.dump(islands_by_query, f, indent=2)

            conn.close()
            print(f"# Exported islands to {output_json_path}")

    asyncio.run(_run())


__all__ = [
    "write_query_islands_joblist",
    "run_query_islands_scanner",
]
