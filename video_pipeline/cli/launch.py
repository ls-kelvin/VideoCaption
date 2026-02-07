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
from ..io.resume import load_done_keys


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
    
    # Calculate already completed count if resume is enabled
    completed_count = 0
    if cfg.data.resume:
        import os
        out_path = cfg.data.output_jsonl
        root, ext = os.path.splitext(out_path)
        
        # Count completed items across all rank files
        for rank in range(len(gpu_groups)):
            rank_out_path = f"{root}.rank{rank}{ext}"
            done_keys = load_done_keys(rank_out_path)
            completed_count += len(done_keys)

    # Queue for progress updates
    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=10000)

    # Progress monitor thread in parent (terminal only shows this)
    monitor = ProgressMonitor(total=total, desc="Completed", unit="video", initial=completed_count)
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
    
    # Consolidate rank-sharded JSONL files
    from ..io.jsonl_consolidator import consolidate_jsonl
    try:
        consolidate_jsonl(cfg.data.output_jsonl, world_size=world_size, keep_rank_files=False)
    except Exception as e:
        print(f"⚠️  JSONL consolidation failed: {e}")


if __name__ == "__main__":
    main()
