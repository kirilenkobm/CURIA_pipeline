from __future__ import annotations

import asyncio
import math
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import pyrion
from pyrion import TwoBitAccessor


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


def _load_transcript_regions(bed12_path: str) -> Dict[str, Tuple[str, int]]:
    regions: Dict[str, Tuple[str, int]] = {}
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
            regions[transcript_id] = (f"{chrom}:{start}-{end}", strand)
    return regions


def write_short_ncrna_joblist(
    rna_orthologous_regions_path: str,
    ultimate_bed_path: str,
    out_path: str,
    max_length: int = 160,
) -> int:
    transcript_regions = _load_transcript_regions(ultimate_bed_path)

    kept = 0
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
            transcript_region, ref_strand_from_bed = ref_region_data
            if transcript_strand not in (1, -1):
                transcript_strand = ref_strand_from_bed

            query_strand = 1 if transcript_strand == chain_strand else -1

            dst.write(
                f"{transcript_id}\t{chain_id}\t{biotype}\t{transcript_region}\t"
                f"{transcript_strand}\t{query_region}\t{query_strand}\n"
            )
            kept += 1

    return kept


def _get_spliced_sequence(transcript, accessor: TwoBitAccessor) -> str:
    seq_parts = [
        str(accessor.fetch(transcript.chrom, int(b[0]), int(b[1]))).upper()
        for b in transcript.blocks
    ]
    seq = "".join(seq_parts)

    if transcript.strand == -1:
        comp = {"A": "U", "T": "A", "G": "C", "C": "G", "N": "N"}
        seq = "".join(comp.get(b, "N") for b in reversed(seq))
    else:
        seq = seq.replace("T", "U")
    return seq


def _extract_sequence(
    accessor: TwoBitAccessor,
    chrom: str,
    start: int,
    end: int,
    strand: int,
) -> str:
    seq = str(accessor.fetch(chrom, start, end)).upper()
    if strand == -1:
        comp = {"A": "U", "T": "A", "G": "C", "C": "G", "N": "N"}
        seq = "".join(comp.get(b, "N") for b in reversed(seq))
    else:
        seq = seq.replace("T", "U")
    return seq


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


class GPUClient:
    def __init__(self, input_queue, output_queue, loop: asyncio.AbstractEventLoop):
        self._input = input_queue
        self._output = output_queue
        self._loop = loop
        self._pending: Dict[Tuple[str, str], asyncio.Future] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    async def embed(self, worker_id: str, sequence_id: str, sequence: str, mean_pool: bool = False) -> np.ndarray:
        future = self._loop.create_future()
        key = (worker_id, sequence_id)
        with self._lock:
            self._pending[key] = future
        self._input.put((worker_id, sequence_id, sequence, {"mean_pool": mean_pool}))
        return await future

    def _reader(self) -> None:
        while True:
            worker_id, sequence_id, emb = self._output.get()
            key = (worker_id, sequence_id)
            self._loop.call_soon_threadsafe(self._resolve_future, key, emb)

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
            "status": "missing_transcript",
        }

    ref_seq = _get_spliced_sequence(t, ref_accessor)
    ref_length = len(ref_seq)
    if ref_length == 0:
        return {
            "transcript_id": job.transcript_id,
            "chain_id": job.chain_id,
            "status": "empty_ref",
        }

    q_chrom, q_start, q_end = _parse_region(job.query_region)
    query_seq = _extract_sequence(query_accessor, q_chrom, q_start, q_end, job.query_strand)
    if len(query_seq) == 0:
        return {
            "transcript_id": job.transcript_id,
            "chain_id": job.chain_id,
            "status": "empty_query",
        }

    if (q_end - q_start) > ref_length * 10:
        return {
            "transcript_id": job.transcript_id,
            "chain_id": job.chain_id,
            "status": "query_too_long",
        }

    extended_query, ext_start, ext_end = _add_flanks(
        query_seq, q_chrom, q_start, q_end, job.query_strand, ref_length, query_accessor, flank_ratio
    )

    ref_emb = await gpu.embed(job.transcript_id + "|" + job.chain_id, "ref", ref_seq, mean_pool=False)

    best_mmd = float("inf")
    best_start = 0
    best_end = ref_length

    query_len = len(extended_query)
    if query_len < ref_length:
        return {
            "transcript_id": job.transcript_id,
            "chain_id": job.chain_id,
            "status": "query_shorter_than_ref",
        }

    start_positions = range(0, query_len - ref_length + 1, window_step)
    for start in start_positions:
        window_seq = extended_query[start:start + ref_length]
        window_id = f"win:{start}:{start + ref_length}"
        window_emb = await gpu.embed(job.transcript_id + "|" + job.chain_id, window_id, window_seq, mean_pool=False)
        mmd = _compute_mmd(ref_emb, window_emb)
        if mmd < best_mmd:
            best_mmd = mmd
            best_start = start
            best_end = start + ref_length

    min_len = int(ref_length * min_length_ratio)
    max_len = int(ref_length * max_length_ratio)

    perturb_best_mmd = best_mmd
    perturb_best_end = best_end
    for p in range(-perturbation_range, perturbation_range + 1):
        end = best_end + p
        if end <= best_start or end > query_len:
            continue
        cand_len = end - best_start
        if cand_len < min_len or cand_len > max_len:
            continue
        cand_seq = extended_query[best_start:end]
        cand_id = f"end:{best_start}:{end}"
        cand_emb = await gpu.embed(job.transcript_id + "|" + job.chain_id, cand_id, cand_seq, mean_pool=False)
        mmd = _compute_mmd(ref_emb, cand_emb)
        if mmd < perturb_best_mmd:
            perturb_best_mmd = mmd
            perturb_best_end = end

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
) -> None:
    async def _run() -> None:
        jobs = _load_joblist(joblist_path)
        transcripts = pyrion.io.read_bed12_file(ultimate_bed_path)
        transcripts_by_id = {t.id: t for t in transcripts}

        ref_accessor = TwoBitAccessor(ref_2bit_path)
        query_accessor = TwoBitAccessor(query_2bit_path)

        loop = asyncio.get_running_loop()
        gpu = GPUClient(gpu_input_queue, gpu_output_queue, loop)

        result_queue: asyncio.Queue = asyncio.Queue()
        writer_task = asyncio.create_task(_sqlite_writer(result_queue, sqlite_path))

        job_queue: asyncio.Queue = asyncio.Queue()
        for job in jobs:
            await job_queue.put(job)
        for _ in range(max_concurrent):
            await job_queue.put(None)

        async def _worker() -> None:
            while True:
                job = await job_queue.get()
                if job is None:
                    break
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
                        "status": f"error:{exc}",
                    }
                await result_queue.put(result)

        workers = [asyncio.create_task(_worker()) for _ in range(max_concurrent)]
        await asyncio.gather(*workers)

        await result_queue.put(None)
        await writer_task

    asyncio.run(_run())


__all__ = [
    "write_short_ncrna_joblist",
    "run_short_ncrna_scheduler",
]
