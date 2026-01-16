# video_pipeline/utils/mp.py
from __future__ import annotations

import os
import socket
import multiprocessing as mp
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def parse_visible_gpu_ids() -> Optional[List[int]]:
    """
    If CUDA_VISIBLE_DEVICES is set, return those ids (as integers).
    Otherwise return None.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.strip():
        return None
    return [int(x) for x in visible.split(",") if x.strip() != ""]


def make_gpu_groups(
    *,
    tp: int,
    dp: Optional[int],
    gpu_ids: List[int],
) -> List[List[int]]:
    if tp <= 0:
        raise ValueError("tp must be > 0")
    if len(gpu_ids) < tp:
        raise ValueError(f"Not enough GPUs: have {len(gpu_ids)}, need tp={tp}")

    max_dp = len(gpu_ids) // tp
    if dp is None:
        dp = max_dp
    if dp <= 0 or dp > max_dp:
        raise ValueError(f"Invalid dp={dp}. With {len(gpu_ids)} GPUs and tp={tp}, max dp is {max_dp}")

    groups: List[List[int]] = []
    for i in range(dp):
        groups.append(gpu_ids[i * tp : (i + 1) * tp])
    return groups


@dataclass
class SpawnSpec:
    gpu_groups: List[List[int]]    # len=world_size, each len=tp
    master_addr: str = "127.0.0.1"
    master_port: int = 0           # 0 => auto


def spawn(
    *,
    worker_fn: Callable[..., None],
    worker_kwargs_list: List[Dict],
) -> None:
    """
    Spawn N processes (spawn method) with provided kwargs.
    """
    ctx = mp.get_context("spawn")
    procs = []
    for kwargs in worker_kwargs_list:
        p = ctx.Process(target=worker_fn, kwargs=kwargs, daemon=False)
        p.start()
        procs.append(p)

    exit_codes = []
    for p in procs:
        p.join()
        exit_codes.append(p.exitcode)

    bad = [c for c in exit_codes if c != 0]
    if bad:
        raise RuntimeError(f"Some workers exited non-zero: exit_codes={exit_codes}")
