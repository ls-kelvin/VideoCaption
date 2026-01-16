# video_pipeline/cli/worker.py
from __future__ import annotations

import os
from typing import List, Optional, Dict, Any


def worker_main(
    *,
    rank: int,
    world_size: int,
    gpu_group: List[int],
    master_addr: str,
    master_port: int,
    config_path: str,
    extra_env: Dict[str, str],
) -> None:
    # ==============================
    # 1) MUST: set env BEFORE importing torch/vllm
    # ==============================
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_group)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"  # 在该 worker 的可见设备内，0..tp-1
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    for k, v in (extra_env or {}).items():
        os.environ[k] = str(v)

    # ==============================
    # 2) Now safe to import heavy libs
    # ==============================
    import torch

    torch.cuda.set_device(0)

    from ..config.loader import load_config
    from ..utils.logging import setup_logging, LogConfig, get_logger
    from ..tasks.registry import get_task

    # 确保 datasets/tasks 注册模块被 import（触发 register）
    from ..data import qwen_video  
    # from ..tasks import describe, structured_caption   (如果你用 __init__ 聚合也可以)

    from ..data.registry import get_dataset_cls
    from ..data.jsonl_reader import iter_jsonl
    from ..data.collate import collate_batch
    from ..engine.vllm_runner import VLLMRunner
    from ..io.jsonl_writer import JsonlWriter
    from ..io.resume import load_done_keys, make_key

    cfg = load_config(config_path)

    setup_logging(LogConfig(run_name=f"{cfg.run.task}", log_dir="logs"))
    logger = get_logger("video_pipeline.worker")

    # 校验 tp 与本进程可见卡数一致
    visible_gpus = torch.cuda.device_count()
    if cfg.vllm.tensor_parallel_size > visible_gpus:
        raise ValueError(
            f"tensor_parallel_size={cfg.vllm.tensor_parallel_size} but visible_gpus={visible_gpus} "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})"
        )

    task = get_task(cfg.run.task)
    DatasetCls = get_dataset_cls(task.dataset_name)

    # 读全量，然后按 rank/world 做分片（不依赖 torch.distributed，足够稳）
    indexed = []
    for line_idx, sample in iter_jsonl(cfg.data.input_jsonl):
        if (line_idx % world_size) == rank:
            indexed.append((line_idx, sample))

    # 每 rank 单独输出文件
    out_path = cfg.data.output_jsonl
    root, ext = os.path.splitext(out_path)
    out_path = f"{root}.rank{rank}{ext}"

    done = load_done_keys(out_path) if cfg.data.resume else set()

    vision_kwargs = {"total_pixels": cfg.vision.total_pixels, "min_pixels": cfg.vision.min_pixels}
    if cfg.vision.fps is not None:
        vision_kwargs["fps"] = cfg.vision.fps

    ds = DatasetCls(
        samples=indexed,
        model_path=cfg.vllm.model,
        video_field=cfg.data.video_field,
        id_field=cfg.data.id_field,
        vision_kwargs=vision_kwargs,
        task=task,
        dataset_params=cfg.task_params.get("dataset", {}),
    )

    # DataLoader 仍可用 torch 的，但这里不要 DistributedSampler（我们已经分片了）
    from torch.utils.data import DataLoader

    dl = DataLoader(
        ds,
        batch_size=cfg.run.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_batch,
        persistent_workers=(cfg.data.num_workers > 0),
    )

    # vLLM runner（此时 vLLM 只看到 gpu_group 对应的卡）
    runner = VLLMRunner(cfg.vllm)

    logger.info(
        "Worker start rank=%d world=%d visible_gpus=%d tp=%d cuda_visible=%s",
        rank, world_size, visible_gpus, cfg.vllm.tensor_parallel_size, os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )

    n = 0
    with JsonlWriter(out_path, flush_every=cfg.run.flush_every, fsync_every=cfg.run.fsync_every) as w:
        for batch in dl:
            keys = batch["keys"]
            raws = batch["raws"]
            llm_inputs = batch["llm_inputs"]

            keep = [(k, raw, inp) for k, raw, inp in zip(keys, raws, llm_inputs) if k not in done]
            if not keep:
                continue

            keys2, raws2, inputs2 = zip(*keep)
            outputs = runner.generate_batch(list(inputs2), cfg.sampling)

            for k, raw, out in zip(keys2, raws2, outputs):
                text = out.outputs[0].text if out.outputs else ""
                parsed = task.parse(text, raw)
                record = {
                    "__key": k,
                    "__task": cfg.run.task,
                    "__model": cfg.vllm.model,
                    "__rank": rank,
                    "__world_size": world_size,
                    "input": raw,
                    "output_text": text,
                    **task.extra_output_fields(),
                    **parsed,
                }
                w.write(record)
                done.add(k)
                n += 1

            if cfg.run.log_every > 0 and (n % cfg.run.log_every == 0):
                logger.info("processed=%d", n)

    logger.info("Worker done rank=%d processed=%d out=%s", rank, n, out_path)
