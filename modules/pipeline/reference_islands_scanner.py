#!/usr/bin/env python3
"""
Reference transcript island scanner (Step 2).

Scans reference lncRNA transcripts (non-short union transcripts) for functional islands
using RNA-FM embeddings + LogReg model. Results are saved to SQLite and JSON for reuse
across multiple query species.

This is a preprocessing step that only needs to run once per reference genome.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import label

import pyrion
from pyrion import TwoBitAccessor

from modules.utils.signal_processing import smooth_signal


@dataclass(frozen=True)
class ReferenceIslandScanJob:
    transcript_id: str
    chrom: str
    start: int
    end: int
    strand: int
    exon_blocks: List[Tuple[int, int]]  # List of (start, end) tuples


def write_reference_islands_joblist(
    rna_toga_regions_path: str,
    union_bed_path: str,
    short_joblist_path: str,
    out_joblist_path: str,
) -> int:
    """
    Create joblist for reference transcript island scanning.

    Selects union transcripts that:
    - Appear in RNA TOGA regions (orthologous)
    - Are NOT in the short ncRNA joblist (length > 160)

    Returns number of jobs created.
    """
    # Load short transcript IDs to exclude
    short_ids = set()
    with open(short_joblist_path, "r") as f:
        f.readline()  # Skip header
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                short_ids.add(parts[0])  # transcript_id column

    # Load union transcripts
    transcripts = pyrion.read_bed12_file(union_bed_path)
    print(f"  Loaded {len(transcripts)} union transcripts")

    # Load RNA TOGA regions to filter for orthologous transcripts
    orthologous_ids = set()
    with open(rna_toga_regions_path, "r") as f:
        f.readline()  # Skip header
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                orthologous_ids.add(parts[0])  # transcript_id column

    print(f"  Found {len(orthologous_ids)} orthologous IDs")
    print(f"  Excluding {len(short_ids)} short ncRNA IDs")
    candidates = orthologous_ids - short_ids
    print(f"  Candidates to process: {len(candidates)}")

    # Filter: orthologous AND not short
    kept = 0
    not_found = 0
    with open(out_joblist_path, "w") as dst:
        dst.write("transcript_id\tchrom\tstart\tend\tstrand\texon_blocks\n")

        for tid in orthologous_ids:
            if tid in short_ids:
                continue

            t = transcripts.get_by_id(tid)
            if t is None:
                not_found += 1
                if not_found <= 5:  # Show first 5 missing
                    print(f"  WARNING: Transcript {tid} not found in BED file")
                continue

            blocks_str = ";".join(f"{int(b[0])},{int(b[1])}" for b in t.blocks)
            iv = t.transcript_interval
            dst.write(f"{tid}\t{iv.chrom}\t{int(iv.start)}\t{int(iv.end)}\t{iv.strand}\t{blocks_str}\n")
            kept += 1

    if not_found > 5:
        print(f"  WARNING: {not_found} total transcripts not found in BED file")

    print(f"# Prepared {kept} reference transcript island scanning jobs (excluded {len(short_ids)} short ncRNAs).")
    return kept


def _load_joblist(joblist_path: str) -> List[ReferenceIslandScanJob]:
    """Load reference island scan joblist."""
    jobs: List[ReferenceIslandScanJob] = []
    with open(joblist_path, "r") as f:
        header = f.readline().strip().split("\t")
        tid_idx = header.index("transcript_id")
        chrom_idx = header.index("chrom")
        start_idx = header.index("start")
        end_idx = header.index("end")
        strand_idx = header.index("strand")
        blocks_idx = header.index("exon_blocks")

        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")

            # Parse exon blocks
            blocks = []
            for block_str in parts[blocks_idx].split(";"):
                s, e = block_str.split(",")
                blocks.append((int(s), int(e)))

            # Parse strand: '+' -> 1, '-' -> -1
            strand_str = parts[strand_idx]
            strand = 1 if strand_str == '+' else -1

            jobs.append(ReferenceIslandScanJob(
                transcript_id=parts[tid_idx],
                chrom=parts[chrom_idx],
                start=int(parts[start_idx]),
                end=int(parts[end_idx]),
                strand=strand,
                exon_blocks=blocks,
            ))
    return jobs


def _extract_exonic_sequence(
    accessor: TwoBitAccessor,
    chrom: str,
    strand: int,
    exon_blocks: List[Tuple[int, int]],
) -> str:
    """Extract concatenated exonic sequence."""
    exon_seqs = []
    for start, end in exon_blocks:
        seq_obj = accessor.fetch(chrom, start, end)
        if strand == -1:
            seq_obj = seq_obj.reverse_complement()
        if not seq_obj.is_rna:
            seq_obj = seq_obj.toggle_type()
        exon_seqs.append(seq_obj.to_string())
    return ''.join(exon_seqs)


# Note: _smooth_signal moved to modules.utils.signal_processing.smooth_signal


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


def _map_spliced_to_genomic(
    exon_blocks: List[Tuple[int, int]],
    strand: int,
    spliced_start: int,
    spliced_end: int,
) -> List[Tuple[int, int]]:
    """
    Map spliced coordinates to genomic segments.
    Returns list of (genomic_start, genomic_end) tuples.
    """
    segments = []
    offset = 0

    for g_start, g_end in exon_blocks:
        exon_len = g_end - g_start
        a = max(spliced_start, offset)
        b = min(spliced_end, offset + exon_len)

        if a < b:
            exon_off_a = a - offset
            exon_off_b = b - offset

            if strand == -1:
                seg_end = g_end - exon_off_a
                seg_start = g_end - exon_off_b
            else:
                seg_start = g_start + exon_off_a
                seg_end = g_start + exon_off_b

            segments.append((seg_start, seg_end))

        offset += exon_len
        if offset >= spliced_end:
            break

    return segments


class GPUClient:
    """Async GPU client for embeddings."""
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


def _ensure_table(conn: sqlite3.Connection, table: str, columns: Dict[str, str]) -> None:
    """Ensure SQLite table exists with required columns."""
    cols_sql = ", ".join(f"{name} {ctype}" for name, ctype in columns.items())
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({cols_sql})")

    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, ctype in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ctype}")


async def _sqlite_writer(queue: asyncio.Queue, sqlite_path: str) -> None:
    """Write islands to SQLite database."""
    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    columns = {
        "transcript_id": "TEXT NOT NULL",
        "island_number": "INTEGER NOT NULL",
        "chrom": "TEXT NOT NULL",
        "start": "INTEGER NOT NULL",
        "end": "INTEGER NOT NULL",
        "strand": "INTEGER NOT NULL",
        "score": "REAL",
    }
    _ensure_table(conn, "reference_islands", columns)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_ref_islands ON reference_islands(transcript_id, island_number)"
    )

    while True:
        item = await queue.get()
        if item is None:
            break

        placeholders = ", ".join(["?"] * len(item))
        columns_sql = ", ".join(item.keys())
        updates = ", ".join(f"{k}=excluded.{k}" for k in item.keys())
        sql = (
            f"INSERT INTO reference_islands ({columns_sql}) VALUES ({placeholders}) "
            f"ON CONFLICT(transcript_id, island_number) DO UPDATE SET {updates}"
        )
        conn.execute(sql, tuple(item.values()))
        conn.commit()

    conn.close()


async def _process_transcript(
    job: ReferenceIslandScanJob,
    ref_accessor: TwoBitAccessor,
    gpu: GPUClient,
    logreg_model,
    window_size: int = 72,
    stride: int = 16,
    smooth_window: int = 5,
    prob_threshold: float = 0.25,
    batch_size: int = 128,
) -> Tuple[List[Dict], int, int]:
    """
    Process a single reference transcript.

    Returns:
        islands: List of island dicts
        total_length: Genomic span (start to end)
        sum_exons_length: Sum of exon lengths
    """
    from modules.logreg_signal_noise.apply_logreg import score_embeddings

    # Extract exonic sequence
    sequence = _extract_exonic_sequence(
        ref_accessor, job.chrom, job.strand, job.exon_blocks
    )
    seq_len = len(sequence)

    # Calculate lengths
    total_length = job.end - job.start
    sum_exons_length = sum(end - start for start, end in job.exon_blocks)

    if seq_len < window_size:
        return [], total_length, sum_exons_length

    # Create sliding windows
    windows = []
    positions = []
    for i in range(0, seq_len - window_size + 1, stride):
        windows.append(sequence[i:i + window_size])
        positions.append(i)

    if not windows:
        return [], total_length, sum_exons_length

    positions = np.array(positions)

    # Get embeddings - send all windows at once, then await results
    # This avoids IPC overhead from awaiting each window individually
    tasks = []
    for i, seq in enumerate(windows):
        task = gpu.embed(job.transcript_id, f"win:{positions[i]}", seq, mean_pool=True)
        tasks.append(task)

    # Await ALL tasks AFTER the loop
    embeddings = await asyncio.gather(*tasks)
    embeddings = np.stack(embeddings, axis=0)

    # Apply LogReg model (expects 16-dim PCA-reduced embeddings)
    probs, _ = score_embeddings(embeddings, model=logreg_model, pca_model=None)

    # Smooth probabilities
    probs_smooth = smooth_signal(probs, smooth_window)

    # Threshold to find islands
    mask = probs_smooth >= prob_threshold
    islands = _get_islands(mask, positions, window_size)

    # Compute scores for each island
    for island in islands:
        idx = island['indices']
        island['max_prob'] = float(np.max(probs_smooth[idx]))
        island['avg_prob'] = float(np.mean(probs_smooth[idx]))

    # Map to genomic coordinates
    results = []
    for island_num, island in enumerate(islands, start=1):
        # Get genomic segments for this island
        segments = _map_spliced_to_genomic(
            job.exon_blocks, job.strand, island['start'], island['end']
        )

        # Store each segment
        for seg_start, seg_end in segments:
            results.append({
                "transcript_id": job.transcript_id,
                "island_number": island_num,
                "chrom": job.chrom,
                "start": seg_start,
                "end": seg_end,
                "strand": job.strand,
                "score": island['max_prob'],
            })

    return results, total_length, sum_exons_length


async def _worker(
    job_queue: asyncio.Queue,
    result_queue: asyncio.Queue,
    transcript_metadata: Dict,
    metadata_lock: asyncio.Lock,
    completed_counter: Dict,  # Shared counter for completed jobs
    counter_lock: asyncio.Lock,
    ref_accessor: TwoBitAccessor,
    gpu: GPUClient,
    logreg_model,
    window_size: int,
    stride: int,
    smooth_window: int,
    prob_threshold: float,
    batch_size: int,
    t0: float,
    total_jobs: int,
    max_concurrent: int,
) -> None:
    """Worker for processing reference transcripts."""
    while True:
        job = await job_queue.get()
        if job is None:
            break

        try:
            transcript_start = time.monotonic()
            islands, total_length, sum_exons_length = await _process_transcript(
                job,
                ref_accessor,
                gpu,
                logreg_model,
                window_size,
                stride,
                smooth_window,
                prob_threshold,
                batch_size,
            )
            transcript_time = time.monotonic() - transcript_start

            # Only warn about extremely slow transcripts (>120s) - these might indicate issues
            # Note: Long transcripts (>10kb exonic) can legitimately take 60-90s
            if transcript_time > 120.0:
                print(f"# Note: Transcript {job.transcript_id} took {transcript_time:.1f}s (exons: {sum_exons_length:,} bp)")

            # Store metadata
            async with metadata_lock:
                transcript_metadata[job.transcript_id] = {
                    "total_length": total_length,
                    "sum_exons_length": sum_exons_length,
                    "num_islands": len(set(isl['island_number'] for isl in islands)),
                }

            # Send islands to writer
            for island in islands:
                await result_queue.put(island)

            # Update completion counter and maybe log progress
            async with counter_lock:
                completed_counter['count'] += 1
                completed = completed_counter['count']

                # Log every 100 completions OR every 30s OR when <100 remaining
                current_time = time.monotonic()
                remaining = total_jobs - completed
                last_log_time = completed_counter['last_log_time']
                should_log = (
                    completed % 100 == 0 or
                    (current_time - last_log_time) >= 30.0 or
                    remaining < 100
                )

                if should_log:
                    elapsed = current_time - t0
                    pct_completed = (completed / total_jobs * 100) if total_jobs > 0 else 0
                    print(f"# Reference island scan: {completed}/{total_jobs} completed ({pct_completed:.1f}%), {remaining} in queue (elapsed {elapsed:.1f}s)")
                    completed_counter['last_log_time'] = current_time

        except Exception as exc:
            print(f"# Error processing {job.transcript_id}: {exc}")


def run_reference_islands_scanner(
    joblist_path: str,
    ref_2bit_path: str,
    gpu_input_queue,
    gpu_output_queue,
    sqlite_path: str,
    logreg_model_path: str,
    output_json_path: str,
    max_concurrent: int = 10,
    window_size: int = 72,
    stride: int = 16,
    smooth_window: int = 5,
    prob_threshold: float = 0.25,
    batch_size: int = 128,
    test_cap_jobs: int = None,
) -> None:
    """
    Scan reference transcripts for functional islands.

    This preprocessing step only needs to run once per reference genome.
    Results can be reused across multiple query species.
    """
    from modules.logreg_signal_noise.apply_logreg import load_logreg_model

    async def _run() -> None:
        t0 = time.monotonic()
        jobs = _load_joblist(joblist_path)
        if test_cap_jobs is not None:
            jobs = jobs[:test_cap_jobs]
            print(f"# [TEST MODE] Capped to {len(jobs)} jobs (--test-cap-jobs={test_cap_jobs})")
        print(f"# Loaded {len(jobs)} reference transcript island scanning jobs.")
        print(f"# Reference islands scanner workers: {max_concurrent}")

        # Load model
        logreg_model = load_logreg_model(logreg_model_path)

        ref_accessor = TwoBitAccessor(ref_2bit_path)

        loop = asyncio.get_running_loop()
        gpu = GPUClient(gpu_input_queue, gpu_output_queue, loop)

        # Shared metadata storage
        transcript_metadata: Dict = {}
        metadata_lock = asyncio.Lock()

        # Shared completion counter and last log time
        completed_counter: Dict = {'count': 0, 'last_log_time': t0}
        counter_lock = asyncio.Lock()

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
                    transcript_metadata,
                    metadata_lock,
                    completed_counter,
                    counter_lock,
                    ref_accessor,
                    gpu,
                    logreg_model,
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

        # Signal SQLite writer to finish
        await result_queue.put(None)
        await writer_task

        # Stop GPU client reader thread
        gpu.stop()

        elapsed_total = time.monotonic() - t0
        print(f"# Reference islands scanner finished in {elapsed_total:.1f}s.")

        # Export to JSON
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.execute(
            "SELECT transcript_id, island_number, chrom, start, end, strand, score "
            "FROM reference_islands ORDER BY transcript_id, island_number"
        )

        islands_by_transcript = {}
        for tid, island_num, chrom, start, end, strand, score in cursor:
            if tid not in islands_by_transcript:
                islands_by_transcript[tid] = []
            islands_by_transcript[tid].append({
                "island_number": island_num,
                "chrom": chrom,
                "start": start,
                "end": end,
                "strand": strand,
                "score": score,
            })

        conn.close()

        # Build final JSON with metadata
        output_data = {}
        for tid, metadata in transcript_metadata.items():
            output_data[tid] = {
                "total_length": metadata["total_length"],
                "sum_exons_length": metadata["sum_exons_length"],
                "islands": islands_by_transcript.get(tid, []),
            }

        with open(output_json_path, "w") as f:
            json.dump(output_data, f, indent=2)

        transcripts_with_islands = sum(1 for v in output_data.values() if v["islands"])
        total_islands = sum(len(v["islands"]) for v in output_data.values())
        transcripts_without_islands = len(output_data) - transcripts_with_islands
        print(
            f"# Reference islands summary: {len(output_data)} transcripts analyzed, "
            f"{transcripts_with_islands} with islands ({total_islands} total islands), "
            f"{transcripts_without_islands} without islands"
        )
        print(f"# Exported results to {output_json_path}")

    asyncio.run(_run())


__all__ = [
    "write_reference_islands_joblist",
    "run_reference_islands_scanner",
]
