# video_pipeline/utils/dist.py
from __future__ import annotations
import os
import torch
import torch.distributed as dist

def dist_init() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)  # local_rank 应该是 0
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank

def is_main(rank: int) -> bool:
    return rank == 0

def dist_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def dist_destroy():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
