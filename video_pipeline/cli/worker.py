# video_pipeline/cli/worker.py
from __future__ import annotations

import os
from typing import List, Dict, Any

def worker_main(
    *,
    rank: int,
    world_size: int,
    gpu_group: List[int],
    master_addr: str,
    master_port: int,
    config_path: str,
    extra_env: Dict[str, str],
    progress_queue,  # multiprocessing.Queue
) -> None:
    # 1) set env BEFORE importing torch/vllm
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_group)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    for k, v in (extra_env or {}).items():
        os.environ[k] = str(v)

    # 2) redirect all stdout/stderr of this worker to file
    from ..utils.stdio import redirect_stdouterr
    with redirect_stdouterr(f"logs/stdout.rank{rank}.log"):

        import torch
        torch.cuda.set_device(0)

        from ..config.loader import load_config
        from ..utils.logging import setup_logging, LogConfig, get_logger
        from ..tasks.registry import get_task

        from ..data.registry import get_dataset_cls
        from ..data.jsonl_reader import iter_jsonl
        from ..data.collate import collate_batch
        from ..engine.vllm_runner import VLLMRunner
        from ..io.jsonl_writer import JsonlWriter
        from ..io.resume import load_done_keys

        cfg = load_config(config_path)

        # 3) IMPORTANT: do NOT log to console from workers; file only
        setup_logging(LogConfig(run_name=f"{cfg.run.task}", log_dir="logs", console=False, file=True))
        logger = get_logger("video_pipeline.worker")

        # 4) hard-silence vllm loggers (in case they bypass stdout redirect via handlers)
        import logging
        logging.getLogger("vllm").setLevel(logging.ERROR)
        logging.getLogger("vllm.engine").setLevel(logging.ERROR)
        logging.getLogger("vllm.worker").setLevel(logging.ERROR)

        task = get_task(cfg.run.task)
        DatasetCls = get_dataset_cls(task.dataset_name)

        # shard by line_idx % world_size
        indexed = [(i, s) for i, s in iter_jsonl(cfg.data.input_jsonl) if (i % world_size) == rank]

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

        runner = VLLMRunner(cfg.vllm)

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

                    # ✅ report progress (one video done)
                    try:
                        progress_queue.put_nowait(1)
                    except Exception:
                        # if queue is full, it's okay to drop some increments occasionally
                        pass
