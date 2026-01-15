# video_pipeline/cli/launch.py
from __future__ import annotations

import os
import socket
import argparse
import torch.multiprocessing as mp


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _parse_gpus(gpu_str: str | None) -> list[str]:
    if not gpu_str:
        # 默认用当前机器所有可见卡
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cvd:
            return [x.strip() for x in cvd.split(",") if x.strip()]
        # 没设置就假设 0..N-1（由用户自行保证）
        raise ValueError("Please provide --gpus (e.g. 0,1,2,3) or set CUDA_VISIBLE_DEVICES.")
    return [x.strip() for x in gpu_str.split(",") if x.strip()]


def _worker_entry(
    proc_rank: int,
    world_size: int,
    config_path: str,
    master_addr: str,
    master_port: int,
    gpu_ids: list[str],
):
    # -------- verl-like GPU isolation: each process sees exactly 1 GPU --------
    picked = gpu_ids[proc_rank % len(gpu_ids)]
    os.environ["CUDA_VISIBLE_DEVICES"] = picked

    # dist env vars (env://)
    os.environ["RANK"] = str(proc_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"  # after isolation, local rank is always 0
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    # IMPORTANT: import AFTER CUDA_VISIBLE_DEVICES is set
    from .run import main as run_main
    run_main({"config": config_path})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--gpus", type=str, default=None, help="e.g. 0,1,2,3 or 2,3,4,5")
    ap.add_argument("--nprocs", type=int, default=None, help="defaults to len(gpus)")
    ap.add_argument("--master-addr", type=str, default="127.0.0.1")
    ap.add_argument("--master-port", type=int, default=0, help="0 means auto pick")
    args = ap.parse_args()

    gpu_ids = _parse_gpus(args.gpus)
    world_size = args.nprocs or len(gpu_ids)
    master_port = args.master_port or _find_free_port()

    # Ensure spawn
    mp.set_start_method("spawn", force=True)

    mp.spawn(
        _worker_entry,
        args=(world_size, args.config, args.master_addr, master_port, gpu_ids),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
