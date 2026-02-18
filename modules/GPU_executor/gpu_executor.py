#!/usr/bin/env python3
"""
GPU executor for batched RNA-FM inference with PCA projection.

Expected job format on input queue:
    (worker_id, sequence_id, sequence, flags)

flags can be:
    - bool: mean_pool flag
    - dict: {"mean_pool": bool}

Output queue items:
    (worker_id, sequence_id, embedding)
"""

import os
# Fix macOS OpenMP conflict
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
import queue
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Add RNA-FM module to path
MODULES_DIR = Path(__file__).resolve().parents[1]
RNAFM_DIR = MODULES_DIR / "RNA-FM"
sys.path.insert(0, str(RNAFM_DIR))

import fm  # noqa: E402


@dataclass
class ExecutorConfig:
    max_batch: int = 256
    min_batch: int = 32
    collect_timeout: float = 0.01
    max_wait: float = 0.2
    device: str = "auto"
    enable_logging: bool = False


class PCAProjector:
    def __init__(self, pca_path: Path, device: torch.device):
        data = np.load(pca_path)
        self.mean = torch.from_numpy(data["mean"]).float().to(device)
        self.components = torch.from_numpy(data["components"]).float().to(device)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 640)
        return (x - self.mean) @ self.components.T


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize_sequence(seq: str) -> str:
    return seq.upper().replace("T", "U")


def _parse_mean_pool(flags) -> bool:
    if flags is None:
        return False
    if isinstance(flags, bool):
        return flags
    if isinstance(flags, dict):
        return bool(flags.get("mean_pool", False))
    return False


def _collect_batch(input_queue, cfg: ExecutorConfig):
    try:
        job = input_queue.get(timeout=0.1)
    except queue.Empty:
        return [], False

    if job is None:
        return [], True

    jobs = [job]
    start = time.time()
    stop_after = False

    while len(jobs) < cfg.max_batch:
        remaining = cfg.max_wait - (time.time() - start)
        if remaining <= 0:
            break
        timeout = min(cfg.collect_timeout, remaining)
        try:
            job = input_queue.get(timeout=timeout)
        except queue.Empty:
            if len(jobs) >= cfg.min_batch or (time.time() - start) >= cfg.max_wait:
                break
            continue

        if job is None:
            stop_after = True
            break

        jobs.append(job)

    return jobs, stop_after


def run_gpu_executor(input_queue, output_queue, cfg: ExecutorConfig | None = None):
    cfg = cfg or ExecutorConfig()
    device = get_device(cfg.device)
    if device.type == "cpu":
        print("# Warning: GPU executor running on CPU (CUDA/MPS unavailable).")

    model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.to(device)

    pca_path = MODULES_DIR / "global_PCA" / "rnafm_pca_k16.npz"
    pca = PCAProjector(pca_path, device)

    # Monitoring stats
    batch_sizes = []
    gpu_compute_times = []
    ipc_times = []
    last_log_time = time.time()
    log_interval = 3.0  # Log every 3 seconds

    while True:
        ipc_start = time.time()  # Start timing IPC (batch collection)
        jobs, stop_after = _collect_batch(input_queue, cfg)
        if not jobs and stop_after:
            break
        if not jobs:
            continue

        batch_collection_time = time.time() - ipc_start

        worker_ids = []
        sequence_ids = []
        sequences = []
        mean_flags = []

        for job in jobs:
            worker_id, sequence_id, sequence, flags = job
            worker_ids.append(worker_id)
            sequence_ids.append(sequence_id)
            sequences.append(_normalize_sequence(sequence))
            mean_flags.append(_parse_mean_pool(flags))

        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        # GPU compute timing
        gpu_start = time.time()
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12])
        gpu_compute_time = time.time() - gpu_start

        reps = results["representations"][12]

        token_embeds = []
        for i, seq in enumerate(sequences):
            length = len(seq)
            token_embeds.append(reps[i, 1:1 + length, :])

        no_mean_indices = [i for i, flag in enumerate(mean_flags) if not flag]
        mean_indices = [i for i, flag in enumerate(mean_flags) if flag]

        pca_token_slices = {}
        if no_mean_indices:
            concat_tokens = torch.cat([token_embeds[i] for i in no_mean_indices], dim=0)
            pca_concat = pca.project(concat_tokens)
            # One-shot GPU->CPU transfer
            pca_concat_cpu = pca_concat.detach().cpu()
            sizes = [token_embeds[i].shape[0] for i in no_mean_indices]
            splits = torch.split(pca_concat_cpu, sizes, dim=0)
            for idx, split in zip(no_mean_indices, splits):
                pca_token_slices[idx] = split

        pca_mean_vecs = {}
        if mean_indices:
            mean_vecs = torch.stack([token_embeds[i].mean(dim=0) for i in mean_indices], dim=0)
            pca_means = pca.project(mean_vecs)
            # One-shot GPU->CPU transfer
            pca_means_cpu = pca_means.detach().cpu()
            for idx, vec in zip(mean_indices, pca_means_cpu):
                pca_mean_vecs[idx] = vec

        # Batched output distribution - single queue put with all results
        output_start = time.time()
        payload = []
        for i in range(len(jobs)):
            if mean_flags[i]:
                emb = pca_mean_vecs[i].numpy()
            else:
                emb = pca_token_slices[i].numpy()
            payload.append((worker_ids[i], sequence_ids[i], emb))
        output_queue.put(payload)
        output_time = time.time() - output_start

        # Record stats
        batch_sizes.append(len(jobs))
        gpu_compute_times.append(gpu_compute_time)
        total_ipc_time = batch_collection_time + output_time
        ipc_times.append(total_ipc_time)

        # Log statistics every N seconds
        current_time = time.time()
        if cfg.enable_logging and current_time - last_log_time >= log_interval:
            try:
                in_queue_size = input_queue.qsize()
                out_queue_size = output_queue.qsize()
            except NotImplementedError:
                # Some queue implementations don't support qsize()
                in_queue_size = -1
                out_queue_size = -1

            if batch_sizes:
                avg_batch_size = sum(batch_sizes) / len(batch_sizes)
                avg_gpu_time = sum(gpu_compute_times) / len(gpu_compute_times)
                avg_ipc_time = sum(ipc_times) / len(ipc_times)
                total_time = avg_gpu_time + avg_ipc_time
                gpu_utilization = (avg_gpu_time / total_time * 100) if total_time > 0 else 0

                # Calculate throughput
                total_samples = sum(batch_sizes)
                elapsed = current_time - (last_log_time if last_log_time > 0 else current_time)
                throughput = total_samples / elapsed if elapsed > 0 else 0

                print(
                    f"[GPU] in_q={in_queue_size} out_q={out_queue_size} "
                    f"batch={avg_batch_size:.1f} "
                    f"gpu={avg_gpu_time*1000:.1f}ms ipc={avg_ipc_time*1000:.1f}ms "
                    f"util={gpu_utilization:.1f}% "
                    f"throughput={throughput:.0f}/s"
                )

                # Reset stats for next interval
                batch_sizes = []
                gpu_compute_times = []
                ipc_times = []

            last_log_time = current_time

        if stop_after:
            break


__all__ = ["ExecutorConfig", "run_gpu_executor"]
