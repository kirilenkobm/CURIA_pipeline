from __future__ import annotations

import asyncio
import time
import math
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import pyrion
from pyrion import TwoBitAccessor
from pyrion.core.nucleotide_sequences import NucleotideSequence


@dataclass(frozen=True)
class ShortRNAJob:
    transcript_id: str
    chain_id: str
    biotype: str
    transcript_region: str
    transcript_strand: int
    query_region: str
    query_strand: int


def _parse_region(region: str) -> Tuple[str, int, int]:
    chrom, rest = region.split(":")
    start_s, end_s = rest.split("-")
    return chrom, int(start_s), int(end_s)


def _strand_to_int(strand: str) -> int:
    if strand in ("+", "1", 1, True):
        return 1
    if strand in ("-", "-1", -1, False):
        return -1
    raise ValueError(f"Unsupported strand: {strand}")


def _load_transcript_regions(bed12_path: str) -> Dict[str, Tuple[str, int, int]]:
    regions: Dict[str, Tuple[str, int, int]] = {}
    with open(bed12_path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            transcript_id = parts[3]
            strand = _strand_to_int(parts[5])
            exon_count = 1
            if len(parts) >= 10:
                try:
                    exon_count = int(parts[9])
                except ValueError:
                    exon_count = 1
                    if len(parts) >= 12:
                        block_sizes = [s for s in parts[10].split(",") if s]
                        exon_count = len(block_sizes) or 1
            regions[transcript_id] = (f"{chrom}:{start}-{end}", strand, exon_count)
    return regions


def write_short_ncrna_joblist(
    rna_orthologous_regions_path: str,
    ultimate_bed_path: str,
    out_path: str,
    max_length: int = 160,
) -> int:
    transcript_regions = _load_transcript_regions(ultimate_bed_path)

    kept = 0
    total = 0
    with open(rna_orthologous_regions_path, "r") as src, open(out_path, "w") as dst:
        header = src.readline().rstrip("\n").split("\t")
        try:
            tid_idx = header.index("transcript_id")
            chain_idx = header.index("chain_id")
            region_idx = header.index("region")
            tstrand_idx = header.index("transcript_strand")
            cstrand_idx = header.index("chain_strand")
            biotype_idx = header.index("biotype")
            tlen_idx = header.index("transcript_length")
        except ValueError:
            raise ValueError("rna_orthologous_regions.tsv header missing required columns")

        dst.write(
            "transcript_id\tchain_id\tbiotype\ttranscript_region\t"
            "transcript_strand\tquery_region\tquery_strand\n"
        )

        for line in src:
            if not line.strip():
                continue
            total += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(tlen_idx, cstrand_idx):
                continue

            try:
                tlen = int(parts[tlen_idx])
            except ValueError:
                continue
            if tlen > max_length:
                continue

            transcript_id = parts[tid_idx]
            chain_id = parts[chain_idx]
            query_region = parts[region_idx]
            transcript_strand = int(parts[tstrand_idx])
            chain_strand = int(parts[cstrand_idx])
            biotype = parts[biotype_idx]

            ref_region_data = transcript_regions.get(transcript_id)
            if ref_region_data is None:
                continue
            transcript_region, ref_strand_from_bed, exon_count = ref_region_data
            if exon_count > 1:
                continue
            if transcript_strand not in (1, -1):
                transcript_strand = ref_strand_from_bed

            query_strand = 1 if transcript_strand == chain_strand else -1

            dst.write(
                f"{transcript_id}\t{chain_id}\t{biotype}\t{transcript_region}\t"
                f"{transcript_strand}\t{query_region}\t{query_strand}\n"
            )
            kept += 1

    print(f"# Prepared {kept} short ncRNA jobs (from {total} RNA regions).")
    return kept


def _get_spliced_sequence(transcript, accessor: TwoBitAccessor) -> str:
    seq_obj: Optional[NucleotideSequence] = None
    for block in transcript.blocks:
        block_seq = accessor.fetch(transcript.chrom, int(block[0]), int(block[1]))
        if seq_obj is None:
            seq_obj = block_seq
        else:
            seq_obj = seq_obj.merge(block_seq)

    if seq_obj is None:
        return ""

    if transcript.strand == -1:
        seq_obj = seq_obj.reverse_complement()

    if not seq_obj.is_rna:
        seq_obj = seq_obj.toggle_type()

    return seq_obj.to_string()


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


def _add_flanks(
    query_seq: str,
    chrom: str,
    start: int,
    end: int,
    strand: int,
    ref_length: int,
    accessor: TwoBitAccessor,
    flank_ratio: float,
) -> Tuple[str, int, int]:
    target_length = int(ref_length * (1 + flank_ratio))
    if len(query_seq) >= target_length:
        return query_seq, start, end

    to_add = target_length - len(query_seq)
    flank_each_side = to_add // 2

    new_start = max(0, start - flank_each_side)
    new_end = end + flank_each_side + (to_add % 2)

    extended = _extract_sequence(accessor, chrom, new_start, new_end, strand)
    return extended, new_start, new_end


def _pairwise_sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        return np.full((X.shape[0], Y.shape[0]), np.inf, dtype=np.float64)
    x_norm = (X ** 2).sum(axis=1)[:, None]
    y_norm = (Y ** 2).sum(axis=1)[None, :]
    dists = x_norm + y_norm - 2.0 * (X @ Y.T)
    return np.maximum(dists, 0.0)


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
    dists_sq = _pairwise_sq_dists(X, Y)
    if gamma is None:
        dists = np.sqrt(dists_sq)
        median = np.median(dists)
        if median <= 0:
            median = 1.0
        gamma = 1.0 / (2.0 * median * median + 1e-10)
    return np.exp(-gamma * dists_sq)


def _compute_mmd(X: np.ndarray, Y: np.ndarray) -> float:
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        return float("inf")

    K_XX = _rbf_kernel(X, X)
    K_YY = _rbf_kernel(Y, Y)
    K_XY = _rbf_kernel(X, Y)

    mmd_sq = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1))
    mmd_sq += (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1))
    mmd_sq -= 2.0 * K_XY.mean()

    return math.sqrt(max(mmd_sq, 0.0))


def _prepare_ref_mmd(ref_emb: np.ndarray) -> Tuple[float, float, float, int]:
    n = ref_emb.shape[0]
    if n < 2:
        return 1.0, 0.0, 0.0, n

    ref_dists = _pairwise_sq_dists(ref_emb, ref_emb)
    ref_dists = np.sqrt(ref_dists)
    median = np.median(ref_dists)
    if median <= 0:
        median = 1.0
    gamma = 1.0 / (2.0 * median * median + 1e-10)

    K_XX = np.exp(-gamma * (_pairwise_sq_dists(ref_emb, ref_emb)))
    K_XX_sum = float(K_XX.sum())
    K_XX_trace = float(np.trace(K_XX))
    return gamma, K_XX_sum, K_XX_trace, n


def _compute_mmd_with_ref(
    ref_ctx: Tuple[float, float, float, int],
    ref_emb: np.ndarray,
    Y: np.ndarray,
    context: Optional[Dict[str, object]] = None,
) -> float:
    gamma, K_XX_sum, K_XX_trace, n = ref_ctx
    m = Y.shape[0]
    if n < 2 or m < 2:
        return float("inf")
    if not np.isfinite(ref_emb).all() or not np.isfinite(Y).all():
        if context is not None:
            print("# Warning: non-finite embeddings before MMD", context)
        return float("inf")

    K_YY = np.exp(-gamma * (_pairwise_sq_dists(Y, Y)))
    K_XY = np.exp(-gamma * (_pairwise_sq_dists(ref_emb, Y)))

    mmd_sq = (K_XX_sum - K_XX_trace) / (n * (n - 1))
    mmd_sq += (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1))
    mmd_sq -= 2.0 * K_XY.mean()
    return math.sqrt(max(float(mmd_sq), 0.0))


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


def _load_joblist(joblist_path: str) -> List[ShortRNAJob]:
    jobs: List[ShortRNAJob] = []
    with open(joblist_path, "r") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            tid_idx = header.index("transcript_id")
            chain_idx = header.index("chain_id")
            biotype_idx = header.index("biotype")
            tregion_idx = header.index("transcript_region")
            tstrand_idx = header.index("transcript_strand")
            qregion_idx = header.index("query_region")
            qstrand_idx = header.index("query_strand")
        except ValueError:
            raise ValueError("short_ncRNA_joblist.txt header missing required columns")

        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            jobs.append(
                ShortRNAJob(
                    transcript_id=parts[tid_idx],
                    chain_id=parts[chain_idx],
                    biotype=parts[biotype_idx],
                    transcript_region=parts[tregion_idx],
                    transcript_strand=int(parts[tstrand_idx]),
                    query_region=parts[qregion_idx],
                    query_strand=int(parts[qstrand_idx]),
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
        "transcript_id": "TEXT NOT NULL",
        "chain_id": "TEXT NOT NULL",
        "biotype": "TEXT",
        "query_region": "TEXT",
        "query_strand": "INTEGER",
        "mmd_score": "REAL",
        "aligned_length": "INTEGER",
        "status": "TEXT",
    }
    _ensure_table(conn, "short_ncrna_results", columns)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_short_ncrna_key ON short_ncrna_results(transcript_id, chain_id)"
    )

    while True:
        item = await queue.get()
        if item is None:
            break

        data = item.copy()
        data.setdefault("status", "ok")

        placeholders = ", ".join(["?"] * len(data))
        columns_sql = ", ".join(data.keys())
        updates = ", ".join(f"{k}=excluded.{k}" for k in data.keys())
        sql = (
            f"INSERT INTO short_ncrna_results ({columns_sql}) VALUES ({placeholders}) "
            f"ON CONFLICT(transcript_id, chain_id) DO UPDATE SET {updates}"
        )
        conn.execute(sql, tuple(data.values()))
        conn.commit()

    conn.close()


async def _process_short_job(
    job: ShortRNAJob,
    transcripts_by_id,
    ref_accessor: TwoBitAccessor,
    query_accessor: TwoBitAccessor,
    gpu: GPUClient,
    flank_ratio: float,
    min_length_ratio: float,
    max_length_ratio: float,
    window_step: int,
    perturbation_range: int,
) -> Dict[str, object]:
    t = transcripts_by_id.get(job.transcript_id)
    if t is None:
        return {
            "transcript_id": job.transcript_id,
            "chain_id": job.chain_id,
            "biotype": job.biotype,
            "status": "missing_transcript",
        }

    ref_seq = _get_spliced_sequence(t, ref_accessor)
    ref_length = len(ref_seq)
    if ref_length == 0:
        return {
            "transcript_id": job.transcript_id,
            "chain_id": job.chain_id,
            "biotype": job.biotype,
            "status": "empty_ref",
        }

    q_chrom, q_start, q_end = _parse_region(job.query_region)
    query_seq = _extract_sequence(query_accessor, q_chrom, q_start, q_end, job.query_strand)
    if len(query_seq) == 0:
        return {
            "transcript_id": job.transcript_id,
            "chain_id": job.chain_id,
            "biotype": job.biotype,
            "status": "empty_query",
        }

    if (q_end - q_start) > ref_length * 10:
        return {
            "transcript_id": job.transcript_id,
            "chain_id": job.chain_id,
            "biotype": job.biotype,
            "status": "query_too_long",
        }

    extended_query, ext_start, ext_end = _add_flanks(
        query_seq, q_chrom, q_start, q_end, job.query_strand, ref_length, query_accessor, flank_ratio
    )

    ref_emb = await gpu.embed(job.transcript_id + "|" + job.chain_id, "ref", ref_seq, mean_pool=False)
    if not np.isfinite(ref_emb).all():
        print(
            "# Warning: non-finite ref embedding",
            {
                "transcript_id": job.transcript_id,
                "chain_id": job.chain_id,
                "biotype": job.biotype,
                "transcript_region": job.transcript_region,
                "transcript_strand": job.transcript_strand,
                "query_region": job.query_region,
                "query_strand": job.query_strand,
                "ref_length": len(ref_seq),
                "ref_seq": ref_seq,
            },
        )
        return {
            "transcript_id": job.transcript_id,
            "chain_id": job.chain_id,
            "biotype": job.biotype,
            "status": "nan_ref_emb",
        }
    ref_ctx = _prepare_ref_mmd(ref_emb)

    best_mmd = float("inf")
    best_start = 0
    best_end = ref_length
    any_valid_window = False

    query_len = len(extended_query)
    if query_len < ref_length:
        return {
            "transcript_id": job.transcript_id,
            "chain_id": job.chain_id,
            "biotype": job.biotype,
            "status": "query_shorter_than_ref",
        }

    # Send all window embedding requests at once for better GPU utilization
    start_positions = list(range(0, query_len - ref_length + 1, window_step))
    window_tasks = []
    for start in start_positions:
        window_seq = extended_query[start:start + ref_length]
        window_id = f"win:{start}:{start + ref_length}"
        task = gpu.embed(job.transcript_id + "|" + job.chain_id, window_id, window_seq, mean_pool=False)
        window_tasks.append((start, window_seq, task))

    # Await all embeddings
    for start, window_seq, task in window_tasks:
        window_emb = await task
        if not np.isfinite(window_emb).all():
            print(
                "# Warning: non-finite window embedding",
                {
                    "transcript_id": job.transcript_id,
                    "chain_id": job.chain_id,
                    "biotype": job.biotype,
                    "window_start": start,
                    "window_end": start + ref_length,
                    "window_length": len(window_seq),
                },
            )
            continue
        mmd = _compute_mmd_with_ref(
            ref_ctx,
            ref_emb,
            window_emb,
            context={
                "transcript_id": job.transcript_id,
                "chain_id": job.chain_id,
                "biotype": job.biotype,
                "window_start": start,
                "window_end": start + ref_length,
                "window_length": len(window_seq),
            },
        )
        any_valid_window = True
        if mmd < best_mmd:
            best_mmd = mmd
            best_start = start
            best_end = start + ref_length

    if not any_valid_window or not np.isfinite(best_mmd):
        return {
            "transcript_id": job.transcript_id,
            "chain_id": job.chain_id,
            "biotype": job.biotype,
            "status": "nan_windows",
        }

    min_len = int(ref_length * min_length_ratio)
    max_len = int(ref_length * max_length_ratio)

    # Send all perturbation embedding requests at once
    perturb_best_mmd = best_mmd
    perturb_best_end = best_end
    any_valid_perturb = False

    cand_tasks = []
    for p in range(-perturbation_range, perturbation_range + 1):
        end = best_end + p
        if end <= best_start or end > query_len:
            continue
        cand_len = end - best_start
        if cand_len < min_len or cand_len > max_len:
            continue
        cand_seq = extended_query[best_start:end]
        cand_id = f"end:{best_start}:{end}"
        task = gpu.embed(job.transcript_id + "|" + job.chain_id, cand_id, cand_seq, mean_pool=False)
        cand_tasks.append((end, cand_seq, task))

    # Await all perturbation embeddings
    for end, cand_seq, task in cand_tasks:
        cand_emb = await task
        if not np.isfinite(cand_emb).all():
            print(
                "# Warning: non-finite perturbation embedding",
                {
                    "transcript_id": job.transcript_id,
                    "chain_id": job.chain_id,
                    "biotype": job.biotype,
                    "perturbation_end": end,
                    "perturbation_length": len(cand_seq),
                },
            )
            continue
        mmd = _compute_mmd_with_ref(
            ref_ctx,
            ref_emb,
            cand_emb,
            context={
                "transcript_id": job.transcript_id,
                "chain_id": job.chain_id,
                "biotype": job.biotype,
                "perturbation_end": end,
                "perturbation_length": len(cand_seq),
            },
        )
        any_valid_perturb = True
        if mmd < perturb_best_mmd:
            perturb_best_mmd = mmd
            perturb_best_end = end

    if not any_valid_perturb or not np.isfinite(perturb_best_mmd):
        return {
            "transcript_id": job.transcript_id,
            "chain_id": job.chain_id,
            "biotype": job.biotype,
            "status": "nan_perturbations",
        }

    if job.query_strand == 1:
        final_start = ext_start + best_start
        final_end = ext_start + perturb_best_end
    else:
        final_end = ext_end - best_start
        final_start = ext_end - perturb_best_end

    final_region = f"{q_chrom}:{final_start}-{final_end}"
    aligned_length = abs(final_end - final_start)

    return {
        "transcript_id": job.transcript_id,
        "chain_id": job.chain_id,
        "biotype": job.biotype,
        "query_region": final_region,
        "query_strand": job.query_strand,
        "mmd_score": float(perturb_best_mmd),
        "aligned_length": int(aligned_length),
        "status": "ok",
    }


def run_short_ncrna_scheduler(
    joblist_path: str,
    ultimate_bed_path: str,
    ref_2bit_path: str,
    query_2bit_path: str,
    gpu_input_queue,
    gpu_output_queue,
    sqlite_path: str,
    max_concurrent: int = 10,
    flank_ratio: float = 0.1,
    min_length_ratio: float = 0.8,
    max_length_ratio: float = 1.2,
    window_step: int = 1,
    perturbation_range: int = 5,
    dump_tsv_path: Optional[str] = None,
) -> None:
    async def _run() -> None:
        t0 = time.monotonic()
        jobs = _load_joblist(joblist_path)
        print(f"# Loaded {len(jobs)} short ncRNA jobs.")
        print(f"# Short ncRNA workers: {max_concurrent}")
        transcripts = pyrion.io.read_bed12_file(ultimate_bed_path)
        transcripts_by_id = {t.id: t for t in transcripts}

        ref_accessor = TwoBitAccessor(ref_2bit_path)
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

        async def _worker() -> None:
            while True:
                job = await job_queue.get()
                if job is None:
                    break
                if job_queue.qsize() % 200 == 0:
                    elapsed = time.monotonic() - t0
                    remaining = job_queue.qsize() - max_concurrent
                    started = total_jobs - remaining
                    pct_started = (started / total_jobs * 100) if total_jobs > 0 else 0
                    print(f"# Short ncRNA jobs: {started}/{total_jobs} started ({pct_started:.1f}%), {remaining} in queue (elapsed {elapsed:.1f}s)")
                try:
                    result = await _process_short_job(
                        job,
                        transcripts_by_id,
                        ref_accessor,
                        query_accessor,
                        gpu,
                        flank_ratio,
                        min_length_ratio,
                        max_length_ratio,
                        window_step,
                        perturbation_range,
                    )
                except Exception as exc:
                    result = {
                        "transcript_id": job.transcript_id,
                        "chain_id": job.chain_id,
                        "biotype": job.biotype,
                        "status": f"error:{exc}",
                    }
                await result_queue.put(result)

        workers = [asyncio.create_task(_worker()) for _ in range(max_concurrent)]
        await asyncio.gather(*workers)

        await result_queue.put(None)
        await writer_task

        # Stop GPU client reader thread
        gpu.stop()

        elapsed_total = time.monotonic() - t0
        print(f"# Short ncRNA scheduler finished in {elapsed_total:.1f}s.")

        # Print summary statistics
        conn = sqlite3.connect(sqlite_path)
        total_jobs = conn.execute("SELECT COUNT(*) FROM short_ncrna_results").fetchone()[0]
        ok_count = conn.execute(
            "SELECT COUNT(*) FROM short_ncrna_results WHERE status = 'ok'"
        ).fetchone()[0]
        error_count = conn.execute(
            "SELECT COUNT(*) FROM short_ncrna_results WHERE status LIKE 'error:%'"
        ).fetchone()[0]
        failed_count = total_jobs - ok_count - error_count

        # MMD-based quality breakdown for successful alignments
        perfect_count = conn.execute(
            "SELECT COUNT(*) FROM short_ncrna_results WHERE status = 'ok' AND mmd_score = 0.0"
        ).fetchone()[0]
        close_count = conn.execute(
            "SELECT COUNT(*) FROM short_ncrna_results WHERE status = 'ok' AND mmd_score > 0.0 AND mmd_score < 0.2"
        ).fetchone()[0]
        questionable_count = conn.execute(
            "SELECT COUNT(*) FROM short_ncrna_results WHERE status = 'ok' AND mmd_score >= 0.2 AND mmd_score < 0.5"
        ).fetchone()[0]
        mismatch_count = conn.execute(
            "SELECT COUNT(*) FROM short_ncrna_results WHERE status = 'ok' AND mmd_score >= 0.5"
        ).fetchone()[0]

        print(
            f"# Short ncRNA summary: {total_jobs} total, {ok_count} aligned successfully, "
            f"{failed_count} failed (various reasons), {error_count} errors"
        )
        print(
            f"#   Quality breakdown: {perfect_count} perfect (MMD=0.0), {close_count} close (MMD<0.2), "
            f"{questionable_count} questionable (0.2≤MMD<0.5), {mismatch_count} mismatches (MMD≥0.5)"
        )
        conn.close()

        if dump_tsv_path:
            conn = sqlite3.connect(sqlite_path)
            cursor = conn.execute(
                "SELECT transcript_id, chain_id, biotype, query_region, query_strand, "
                "mmd_score, aligned_length, status "
                "FROM short_ncrna_results"
            )
            columns = [desc[0] for desc in cursor.description]
            with open(dump_tsv_path, "w") as f:
                f.write("\t".join(columns) + "\n")
                for row in cursor:
                    f.write("\t".join("" if v is None else str(v) for v in row) + "\n")
            conn.close()
 
    asyncio.run(_run())


__all__ = [
    "write_short_ncrna_joblist",
    "run_short_ncrna_scheduler",
]
