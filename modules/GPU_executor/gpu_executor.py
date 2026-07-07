#!/usr/bin/env python3
"""
GPU executor for batched RNA foundation model inference with PCA projection.

Supports any model registered in model_registry (RNA-FM, RiNALMo, etc.).

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

MODULES_DIR = Path(__file__).resolve().parents[1]
if str(MODULES_DIR.parent) not in sys.path:
    sys.path.insert(0, str(MODULES_DIR.parent))

from modules.model_registry import load_model, get_pca_path, get_finding_pca_path


@dataclass
class ExecutorConfig:
    max_batch: int = 256
    min_batch: int = 32
    collect_timeout: float = 0.01
    max_wait: float = 0.2
    device: str = "auto"
    enable_logging: bool = False
    model_name: str = "rinalmo"


class PCAProjector:
    def __init__(self, pca_path: Path, device: torch.device):
        data = np.load(pca_path)
        self.mean = torch.from_numpy(data["mean"]).float().to(device)
        self.components = torch.from_numpy(data["components"]).float().to(device)

    def project(self, x: torch.Tensor) -> torch.Tensor:
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


def _parse_pca_role(flags, mean_pool: bool) -> str:
    """Which PCA space to project into: "find" (scanner) or "match" (alignment).

    An explicit ``pca_role`` in the flags dict wins. Otherwise we honour the
    invariant that has always held in this pipeline: the mean-pooled path is the
    signal/noise *finding* scanner, and the per-token path is island *matching*.
    Deriving the role from ``mean_pool`` keeps every existing caller working with
    no change, while the explicit flag decouples pooling from projection space for
    any future consumer that needs the other combination.
    """
    if isinstance(flags, dict):
        role = flags.get("pca_role")
        if role in ("find", "match"):
            return role
    return "find" if mean_pool else "match"


def _collect_batch(input_queue, cfg: ExecutorConfig):
    try:
        job = input_queue.get(timeout=0.01)
    except queue.Empty:
        return [], False

    if job is None:
        return [], True

    jobs = [job]
    stop_after = False

    while len(jobs) < cfg.max_batch:
        try:
            job = input_queue.get_nowait()
        except queue.Empty:
            break

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

    # Load model via registry
    model_name = cfg.model_name
    print(f"# GPU executor: loading {model_name}...")
    model, tokenize_fn, extract_fn = load_model(model_name, device)

    use_amp = device.type == "cuda"
    if use_amp:
        print("# GPU executor: AMP (float16) enabled")

    if device.type == "cuda":
        try:
            model = torch.compile(model)
            print("# GPU executor: torch.compile enabled (first batch will be slow)")
        except Exception as e:
            print(f"# GPU executor: torch.compile unavailable ({e})")

    # Two projectors: the matching PCA (per-token / island alignment) and the
    # finding PCA (mean-pooled / signal-noise scanner). They are the same file
    # for models without a dedicated finding PCA (e.g. RNA-FM), in which case we
    # avoid loading it twice.
    pca_path = get_pca_path(model_name)
    pca = PCAProjector(pca_path, device)
    find_pca_path = get_finding_pca_path(model_name)
    if find_pca_path == pca_path:
        pca_find = pca
        print(f"# GPU executor: {model_name} ready (PCA: {pca_path.name})")
    else:
        pca_find = PCAProjector(find_pca_path, device)
        print(f"# GPU executor: {model_name} ready "
              f"(match PCA: {pca_path.name}, find PCA: {find_pca_path.name})")

    # Monitoring stats
    batch_sizes = []
    gpu_compute_times = []
    ipc_times = []
    last_log_time = time.time()
    log_interval = 3.0

    while True:
        ipc_start = time.time()
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
        pca_roles = []

        for job in jobs:
            worker_id, sequence_id, sequence, flags = job
            worker_ids.append(worker_id)
            sequence_ids.append(sequence_id)
            sequences.append(_normalize_sequence(sequence))
            mean_pool = _parse_mean_pool(flags)
            mean_flags.append(mean_pool)
            pca_roles.append(_parse_pca_role(flags, mean_pool))

        tokens = tokenize_fn(sequences)

        gpu_start = time.time()
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_amp
        ):
            reps = extract_fn(model, tokens)
        gpu_compute_time = time.time() - gpu_start

        token_embeds = []
        for i, seq in enumerate(sequences):
            length = len(seq)
            token_embeds.append(reps[i, 1:1 + length, :])

        # Project each item into its PCA space, batched by (pooling, projector).
        # Pooling follows mean_pool; projector follows pca_role ("find" -> finding
        # PCA, "match" -> matching PCA). The two combinations used in practice are
        # (mean-pool, find) for the scanner and (per-token, match) for alignment;
        # the loop also covers the other two combinations for free.
        projectors = {"find": pca_find, "match": pca}
        pca_results = {}
        for do_mean in (False, True):
            for role, projector in projectors.items():
                group = [i for i in range(len(jobs))
                         if mean_flags[i] == do_mean and pca_roles[i] == role]
                if not group:
                    continue
                if do_mean:
                    vecs = torch.stack([token_embeds[i].mean(dim=0) for i in group], dim=0)
                    proj = projector.project(vecs).detach().cpu()
                    for idx, vec in zip(group, proj):
                        pca_results[idx] = vec
                else:
                    concat_tokens = torch.cat([token_embeds[i] for i in group], dim=0)
                    proj = projector.project(concat_tokens).detach().cpu()
                    sizes = [token_embeds[i].shape[0] for i in group]
                    for idx, split in zip(group, torch.split(proj, sizes, dim=0)):
                        pca_results[idx] = split

        output_start = time.time()
        payload = []
        for i in range(len(jobs)):
            payload.append((worker_ids[i], sequence_ids[i], pca_results[i].numpy()))
        output_queue.put(payload)
        output_time = time.time() - output_start

        batch_sizes.append(len(jobs))
        gpu_compute_times.append(gpu_compute_time)
        total_ipc_time = batch_collection_time + output_time
        ipc_times.append(total_ipc_time)

        current_time = time.time()
        if cfg.enable_logging and current_time - last_log_time >= log_interval:
            try:
                in_queue_size = input_queue.qsize()
                out_queue_size = output_queue.qsize()
            except NotImplementedError:
                in_queue_size = -1
                out_queue_size = -1

            if batch_sizes:
                avg_batch_size = sum(batch_sizes) / len(batch_sizes)
                avg_gpu_time = sum(gpu_compute_times) / len(gpu_compute_times)
                avg_ipc_time = sum(ipc_times) / len(ipc_times)
                total_time = avg_gpu_time + avg_ipc_time
                gpu_utilization = (avg_gpu_time / total_time * 100) if total_time > 0 else 0

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

                batch_sizes = []
                gpu_compute_times = []
                ipc_times = []

            last_log_time = current_time

        if stop_after:
            break


__all__ = ["ExecutorConfig", "run_gpu_executor"]
