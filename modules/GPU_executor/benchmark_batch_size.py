#!/usr/bin/env python3
"""Benchmark a reasonable batch size for the RNA foundation model at a fixed
sequence length. Defaults to RiNALMo; pass --model rnafm for the deprecated
RNA-FM comparison path."""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import torch

# Fix macOS OpenMP conflict
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.model_registry import load_model  # noqa: E402


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        preferred = "cuda" if torch.cuda.is_available() else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"

    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def random_seq(length: int) -> str:
    return "".join(random.choice("ACGU") for _ in range(length))


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def _mps_memory() -> tuple[int, int]:
    if not hasattr(torch, "mps"):
        return 0, 0
    current = 0
    driver = 0
    if hasattr(torch.mps, "current_allocated_memory"):
        current = torch.mps.current_allocated_memory()
    if hasattr(torch.mps, "driver_allocated_memory"):
        driver = torch.mps.driver_allocated_memory()
    return current, driver


def run_once(model, tokenize_fn, extract_fn, device: torch.device, batch_size: int, seq_len: int) -> float:
    seqs = [random_seq(seq_len) for _ in range(batch_size)]
    tokens = tokenize_fn(seqs)  # returns tokens already on device

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    with torch.no_grad():
        rep = extract_fn(model, tokens)
        _ = rep.sum().item()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    del tokens
    return elapsed


def try_batch(
    model,
    tokenize_fn,
    extract_fn,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    warmup: int,
) -> tuple[bool, float, int, int, int]:
    try:
        for _ in range(warmup):
            run_once(model, tokenize_fn, extract_fn, device, batch_size, seq_len)
        elapsed = run_once(model, tokenize_fn, extract_fn, device, batch_size, seq_len)
        peak = 0
        mps_current = 0
        mps_driver = 0
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated()
        elif device.type == "mps":
            mps_current, mps_driver = _mps_memory()
        return True, elapsed, peak, mps_current, mps_driver
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "alloc" in msg:
            return False, 0.0, 0, 0, 0
        raise


def format_bytes(num: int) -> str:
    if num <= 0:
        return "0"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}PB"


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark batch size for the RNA foundation model at a fixed sequence length.")
    parser.add_argument("--model", default="rinalmo", choices=["rinalmo", "rnafm"],
                        help="RNA foundation model (default: rinalmo; 'rnafm' is deprecated).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--seq-len", type=int, default=160)
    parser.add_argument("--min-batch", type=int, default=64)
    parser.add_argument("--max-batch", type=int, default=1024)
    parser.add_argument("--step", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--margin", type=float, default=0.8)
    parser.add_argument("--clear-between", action="store_true", help="Clear device cache between batch sizes")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device} | model: {args.model}")

    model, tokenize_fn, extract_fn = load_model(args.model, device)

    batches = list(range(args.min_batch, args.max_batch + 1, args.step))
    results = []

    for b in batches:
        print(f"\nTesting batch size {b}...")
        ok, elapsed, peak, mps_current, mps_driver = try_batch(
            model, tokenize_fn, extract_fn, device, b, args.seq_len, args.warmup
        )
        if not ok:
            print(f"Batch {b}: OOM or device error")
            break
        per_seq_ms = (elapsed / b) * 1000
        if device.type == "cuda":
            print(f"Batch {b}: {elapsed:.3f}s total | {per_seq_ms:.3f} ms/seq | peak {format_bytes(peak)}")
        elif device.type == "mps":
            parts = [f"Batch {b}: {elapsed:.3f}s total | {per_seq_ms:.3f} ms/seq"]
            if mps_current:
                parts.append(f"mps_current {format_bytes(mps_current)}")
            if mps_driver:
                parts.append(f"mps_driver {format_bytes(mps_driver)}")
            print(" | ".join(parts))
        else:
            print(f"Batch {b}: {elapsed:.3f}s total | {per_seq_ms:.3f} ms/seq")
        results.append((b, elapsed, peak))
        if args.clear_between:
            clear_device_cache(device)

    if not results:
        print("No successful batch sizes.")
        return

    max_ok = results[-1][0]
    safe = max(int(max_ok * args.margin), args.min_batch)
    safe = (safe // args.step) * args.step or args.step

    print("\nSummary:")
    print(f"Max successful batch: {max_ok}")
    print(f"Suggested safe batch (~{int(args.margin*100)}%): {safe}")


if __name__ == "__main__":
    main()
