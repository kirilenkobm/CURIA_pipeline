#!/usr/bin/env python3
"""Simple sample to exercise GPU executor with a few sequences."""

import multiprocessing as mp
import time

from gpu_executor import ExecutorConfig, run_gpu_executor


def main():
    ctx = mp.get_context("spawn")
    input_q = ctx.Queue()
    output_q = ctx.Queue()

    cfg = ExecutorConfig(max_batch=16, min_batch=2, collect_timeout=0.01, max_wait=0.1)
    proc = ctx.Process(target=run_gpu_executor, args=(input_q, output_q, cfg))
    proc.start()

    jobs = [
        ("w1", "s1", "ACGUACGUACGU", False),
        ("w2", "s2", "GGGAAAUUUCC", {"mean_pool": True}),
        ("w3", "s3", "AUCGAUCGAUCG", False),
    ]

    for job in jobs:
        input_q.put(job)

    # Signal shutdown after processing current jobs
    input_q.put(None)

    results = []
    deadline = time.time() + 30
    while len(results) < len(jobs) and time.time() < deadline:
        try:
            results.append(output_q.get(timeout=1))
        except Exception:
            break

    for worker_id, sequence_id, emb in results:
        print(worker_id, sequence_id, emb.shape)

    proc.join(timeout=10)
    if proc.is_alive():
        proc.terminate()


if __name__ == "__main__":
    main()
