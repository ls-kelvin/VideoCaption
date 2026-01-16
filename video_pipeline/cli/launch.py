# video_pipeline/cli/launch.py
from __future__ import annotations

import argparse
import multiprocessing as mp
import threading
from typing import List, Optional

from ..config.loader import load_config
from ..utils.mp import find_free_port, make_gpu_groups, parse_visible_gpu_ids, spawn
from ..utils.progress import ProgressMonitor
from ..data.jsonl_reader import iter_jsonl


def _parse_gpu_ids(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--gpu-ids", type=str, default=None)
    ap.add_argument("--dp", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tp = cfg.vllm.tensor_parallel_size

    gpu_ids = _parse_gpu_ids(args.gpu_ids) or parse_visible_gpu_ids()
    if gpu_ids is None:
        raise ValueError("Please provide --gpu-ids or set CUDA_VISIBLE_DEVICES")

    gpu_groups = make_gpu_groups(tp=tp, dp=args.dp, gpu_ids=gpu_ids)
    world_size = len(gpu_groups)
    master_port = find_free_port()

    # total videos (all samples)
    total = sum(1 for _ in iter_jsonl(cfg.data.input_jsonl))

    # Queue for progress updates
    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=10000)

    # Progress monitor thread in parent (terminal only shows this)
    monitor = ProgressMonitor(total=total, desc="Completed", unit="video")
    t = threading.Thread(target=monitor.run, kwargs=dict(q=q, stop_token="__STOP__"), daemon=True)
    t.start()

    from .worker import worker_main

    worker_kwargs_list = []
    for rank, group in enumerate(gpu_groups):
        worker_kwargs_list.append(
            dict(
                rank=rank,
                world_size=world_size,
                gpu_group=group,
                master_addr="127.0.0.1",
                master_port=master_port,
                config_path=args.config,
                extra_env={"TOKENIZERS_PARALLELISM": "false"},
                progress_queue=q,          # <--- NEW
            )
        )

    try:
        spawn(worker_fn=worker_main, worker_kwargs_list=worker_kwargs_list)
    finally:
        q.put("__STOP__")
        t.join(timeout=5)


if __name__ == "__main__":
    main()
