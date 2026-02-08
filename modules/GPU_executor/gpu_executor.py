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
import sys
import time
import queue
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Fix macOS OpenMP conflict
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Add RNA-FM module to path
MODULES_DIR = Path(__file__).resolve().parents[1]
RNAFM_DIR = MODULES_DIR / "RNA-FM"
sys.path.insert(0, str(RNAFM_DIR))

import fm  # noqa: E402


@dataclass
class ExecutorConfig:
    max_batch: int = 256
    min_batch: int = 80
    collect_timeout: float = 0.01
    max_wait: float = 0.2
    device: str = "auto"


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

    model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.to(device)

    pca_path = MODULES_DIR / "global_PCA" / "rnafm_pca_k16.npz"
    pca = PCAProjector(pca_path, device)

    while True:
        jobs, stop_after = _collect_batch(input_queue, cfg)
        if not jobs and stop_after:
            break
        if not jobs:
            continue

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

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12])

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
            sizes = [token_embeds[i].shape[0] for i in no_mean_indices]
            splits = torch.split(pca_concat, sizes, dim=0)
            for idx, split in zip(no_mean_indices, splits):
                pca_token_slices[idx] = split

        pca_mean_vecs = {}
        if mean_indices:
            mean_vecs = torch.stack([token_embeds[i].mean(dim=0) for i in mean_indices], dim=0)
            pca_means = pca.project(mean_vecs)
            for idx, vec in zip(mean_indices, pca_means):
                pca_mean_vecs[idx] = vec

        for i in range(len(jobs)):
            if mean_flags[i]:
                emb = pca_mean_vecs[i].detach().cpu().numpy()
            else:
                emb = pca_token_slices[i].detach().cpu().numpy()
            output_queue.put((worker_ids[i], sequence_ids[i], emb))

        if stop_after:
            break


__all__ = ["ExecutorConfig", "run_gpu_executor"]
