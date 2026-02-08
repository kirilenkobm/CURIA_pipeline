#!/usr/bin/env python3
"""Simple sample to exercise GPU executor with logging."""

import multiprocessing as mp
import os
import time

from gpu_executor import ExecutorConfig, run_gpu_executor


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [pid {os.getpid()}] {msg}", flush=True)


def drain_results(output_q, expected: int, label: str) -> int:
    log(f"{label}: waiting for {expected} result(s)")
    received = 0
    deadline = time.time() + 30
    while received < expected and time.time() < deadline:
        try:
            worker_id, sequence_id, emb = output_q.get(timeout=1)
        except Exception:
            log(f"{label}: timeout waiting for results")
            break
        received += 1
        log(f"{label}: result {received}/{expected} -> {worker_id} {sequence_id} shape={emb.shape}")
    return received


def main():
    log("parent starting")
    ctx = mp.get_context("spawn")
    input_q = ctx.Queue()
    output_q = ctx.Queue()

    cfg = ExecutorConfig(max_batch=16, min_batch=2, collect_timeout=0.01, max_wait=0.1)
    proc = ctx.Process(target=run_gpu_executor, args=(input_q, output_q, cfg), name="gpu_executor")
    proc.daemon = True
    proc.start()
    log(f"spawned gpu_executor pid={proc.pid} alive={proc.is_alive()}")

    log("idle check: waiting 1s before submitting jobs")
    time.sleep(1)
    log(f"executor still alive={proc.is_alive()}")

    # Burst 1: 10 jobs
    burst1 = [
        ("w1", "s1", "ACGUACGUACGU", False),
        ("w1", "s2", "GGGAAAUUUCC", {"mean_pool": True}),
        ("w1", "s3", "AUCGAUCGAUCG", False),
        ("w1", "s4", "ACGUGCUA", False),
        ("w1", "s5", "UUGGAAUU", {"mean_pool": True}),
        ("w1", "s6", "AUAUAUAUAU", False),
        ("w1", "s7", "CCCGGGAAA", False),
        ("w1", "s8", "GGAUCCGA", {"mean_pool": True}),
        ("w1", "s9", "ACGUACGU", False),
        ("w1", "s10", "UGCAUGCA", False),
    ]
    log(f"submitting burst1: {len(burst1)} jobs")
    for job in burst1:
        input_q.put(job)

    drain_results(output_q, expected=len(burst1), label="burst1")

    log("idle check: waiting 1s after burst1")
    time.sleep(1)
    log(f"executor still alive={proc.is_alive()}")

    # Burst 2: 5 jobs, then immediately add 6 more
    burst2a = [
        ("w2", "s11", "ACGUACGUACGU", False),
        ("w2", "s12", "GGGAAAUUUCC", {"mean_pool": True}),
        ("w2", "s13", "AUCGAUCGAUCG", False),
        ("w2", "s14", "ACGUGCUA", False),
        ("w2", "s15", "UUGGAAUU", {"mean_pool": True}),
    ]
    burst2b = [
        ("w2", "s16", "AUAUAUAUAU", False),
        ("w2", "s17", "CCCGGGAAA", False),
        ("w2", "s18", "GGAUCCGA", {"mean_pool": True}),
        ("w2", "s19", "ACGUACGU", False),
        ("w2", "s20", "UGCAUGCA", False),
        ("w2", "s21", "AUGCAUGCAU", False),
    ]

    log(f"submitting burst2a: {len(burst2a)} jobs")
    for job in burst2a:
        input_q.put(job)

    log(f"immediately submitting burst2b: {len(burst2b)} jobs")
    for job in burst2b:
        input_q.put(job)

    drain_results(output_q, expected=len(burst2a) + len(burst2b), label="burst2")

    log("idle check: waiting 0.5s before shutdown")
    time.sleep(0.5)

    # Signal shutdown after processing current jobs
    input_q.put(None)
    log("sent shutdown sentinel")

    proc.join(timeout=10)
    log(f"executor joined: alive={proc.is_alive()}")
    if proc.is_alive():
        proc.terminate()
        log("executor terminated")


if __name__ == "__main__":
    main()
