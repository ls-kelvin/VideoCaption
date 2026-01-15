# video_pipeline/cli/run.py
from __future__ import annotations
import os
import argparse
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..config.loader import load_config
from ..data.jsonl_reader import iter_jsonl
from ..data.registry import get_dataset_cls
from ..data.collate import collate_batch
from ..engine.vllm_runner import VLLMRunner
from ..io.jsonl_writer import JsonlWriter
from ..io.resume import load_done_keys, make_key
from ..tasks.registry import get_task
from ..utils.dist import dist_init, dist_destroy, dist_barrier

def main(args):
    cfg = load_config(args['config'])
    rank, world_size, local_rank = dist_init()
    
    from ..utils.logging import setup_logging, LogConfig, get_logger
    # 如果你希望日志文件名跟 task 走：
    setup_logging(LogConfig(run_name=f"{cfg.run.task}", log_dir="logs"))

    logger = get_logger("video_pipeline.cli.run")
    logger.info("Starting...")

    task = get_task(cfg.run.task)

    # 1) 读入样本（所有 rank 都读同一个 input_jsonl；Sampler 会分片）
    indexed = list(iter_jsonl(cfg.data.input_jsonl))

    # 2) rank 独立的输出文件（推荐）
    #    这样写入不需要跨进程锁，也不需要 gather
    out_path = cfg.data.output_jsonl
    if world_size > 1:
        root, ext = os.path.splitext(out_path)
        out_path = f"{root}.rank{rank}{ext}"

    done = load_done_keys(out_path) if cfg.data.resume else set()

    # 3) Dataset（注意：这里不做 shard 过滤，Sampler 会做）
    vision_kwargs = {"total_pixels": cfg.vision.total_pixels, "min_pixels": cfg.vision.min_pixels}
    if cfg.vision.fps is not None:
        vision_kwargs["fps"] = cfg.vision.fps

    DatasetCls = get_dataset_cls(task.dataset_name)
    ds = DatasetCls(
        samples=indexed,
        model_path=cfg.vllm.model,
        video_field=cfg.data.video_field,
        id_field=cfg.data.id_field,
        vision_kwargs=vision_kwargs,
        task=task,
        dataset_params=cfg.task_params.get("dataset", {}),
    )

    # 4) DistributedSampler
    sampler = DistributedSampler(
        ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,     # 推理通常不 shuffle，便于复现/断点
        drop_last=False,
    )

    dl = DataLoader(
        ds,
        batch_size=cfg.run.batch_size,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_batch,
        persistent_workers=(cfg.data.num_workers > 0),
    )

    # 5) vLLM：数据并行模式下，每个 rank 用 1 GPU -> tensor_parallel_size 必须是 1
    if cfg.vllm.tensor_parallel_size != 1 and world_size > 1:
        raise ValueError(
            "torchrun data-parallel mode requires vllm.tensor_parallel_size=1. "
            "If you want TP, run 1 process and set tensor_parallel_size>1."
        )

    runner = VLLMRunner(cfg.vllm)

    # 6) 写出：rank 各写各的
    with JsonlWriter(out_path, flush_every=cfg.run.flush_every, fsync_every=cfg.run.fsync_every) as w:
        n = 0
        for batch in dl:
            keys = batch["keys"]
            raws = batch["raws"]
            llm_inputs = batch["llm_inputs"]

            # 断点：如果该 key 已经写过就跳过
            # 注意：done 来自当前 rank 的 output 文件，所以只会跳过本 rank 已完成的样本
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
                    "input": raw,
                    "output_text": text,
                    **task.extra_output_fields(),
                    **parsed,
                }
                w.write(record)
                done.add(k)  # 让本进程内立即生效
                n += 1

            if cfg.run.log_every > 0 and n % cfg.run.log_every == 0:
                print(f"[rank {rank}/{world_size}] processed={n}")

    dist_barrier()
    dist_destroy()