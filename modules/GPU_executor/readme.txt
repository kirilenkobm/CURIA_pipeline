GPU executor

Purpose
- Spawn a dedicated process that runs RNA-FM on GPU/CPU with micro-batching and PCA projection.

How it works
- Parent process creates two multiprocessing queues: input_q, output_q.
- Start the executor with run_gpu_executor(input_q, output_q, cfg).
- Jobs are consumed asynchronously in micro-batches; results are pushed to output_q.

Input message format (tuple)
- (worker_id, sequence_id, sequence, flags)
- flags: bool (mean_pool) or dict {"mean_pool": bool}

Output message format (tuple)
- (worker_id, sequence_id, embedding)
- embedding shape:
  - mean_pool=True  -> (16,)
  - mean_pool=False -> (L, 16)

Shutdown
- Send None to input_q to request a graceful stop after draining current jobs.
