# video_pipeline/cli/test_video_processing.py
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

# ✅ 注意：这里只做视频处理测试，不 import vllm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--max-samples", type=int, default=8, help="limit samples for quick test")
    ap.add_argument("--out", type=str, default="video_process_test.jsonl")
    args = ap.parse_args()

    # 延迟 import：让这个脚本尽量轻量
    from ..config.loader import load_config
    from ..data.jsonl_reader import iter_jsonl
    from ..io.resume import make_key
    from ..utils.logging import setup_logging, LogConfig, get_logger

    # 触发 task/dataset 注册（按你项目的实际 import 方式调整）
    from ..tasks.registry import get_task
    from ..data.registry import get_dataset_cls

    cfg = load_config(args.config)

    setup_logging(LogConfig(run_name="video_process_test", log_dir="logs"))
    logger = get_logger("video_pipeline.test_video_processing")

    # 这个测试仅依赖 task 的 message 构造与 dataset 的 video 处理
    task = get_task(cfg.run.task)
    DatasetCls = get_dataset_cls(task.dataset_name)

    # 简单分片：可选（方便你用 python spawn 多进程时做快速验证）
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    indexed: List[Tuple[int, Dict[str, Any]]] = []
    for line_idx, sample in iter_jsonl(cfg.data.input_jsonl):
        if (line_idx % world_size) == rank:
            indexed.append((line_idx, sample))
        if len(indexed) >= args.max_samples:
            break

    vision_kwargs = {"total_pixels": cfg.vision.total_pixels, "min_pixels": cfg.vision.min_pixels}
    if cfg.vision.fps is not None:
        vision_kwargs["fps"] = cfg.vision.fps

    ds = DatasetCls(
        samples=indexed,
        model_path=cfg.vllm.model,  # 只用于 AutoProcessor.apply_chat_template
        video_field=cfg.data.video_field,
        id_field=cfg.data.id_field,
        vision_kwargs=vision_kwargs,
        task=task,
        dataset_params=cfg.task_params.get("dataset", {}),
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_ok = 0
    n_fail = 0

    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(len(ds)):
            try:
                item = ds[i]
                key = item["__key"]
                raw = item["raw"]
                llm_input = item["llm_input"]

                mm = llm_input.get("multi_modal_data", {})
                has_video = "video" in mm and mm["video"] is not None
                has_image = "image" in mm and mm["image"] is not None

                # 尝试提取一些“轻量可见”的信息（不同版本结构可能不同，所以只做安全探测）
                video_info = {}
                v = mm.get("video", None)
                if v is not None:
                    # 常见情况下 video_inputs 可能是 list/np/tensor/或包含 shape 信息的对象
                    try:
                        video_info["type"] = type(v).__name__
                        video_info["len"] = len(v) if hasattr(v, "__len__") else None
                    except Exception:
                        video_info["type"] = type(v).__name__
                        video_info["len"] = None

                rec = {
                    "__key": key,
                    "__rank": rank,
                    "__world_size": world_size,
                    "path": raw.get(cfg.data.video_field),
                    "has_video": has_video,
                    "has_image": has_image,
                    "video_info": video_info,
                    "prompt_preview": (llm_input.get("prompt", "")[:200] + "...") if llm_input.get("prompt") else "",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

                n_ok += 1
                logger.info("OK key=%s has_video=%s video_info=%s", key, has_video, video_info)

            except Exception as e:
                # 把错误也写入 jsonl，方便定位是哪条视频
                line_idx, sample = indexed[i]
                key = make_key(line_idx, sample, cfg.data.id_field)
                err = {
                    "__key": key,
                    "__rank": rank,
                    "path": sample.get(cfg.data.video_field),
                    "error": repr(e),
                }
                f.write(json.dumps(err, ensure_ascii=False) + "\n")
                f.flush()

                n_fail += 1
                logger.exception("FAIL key=%s path=%s", key, sample.get(cfg.data.video_field))

    logger.info("Done. ok=%d fail=%d out=%s", n_ok, n_fail, args.out)


if __name__ == "__main__":
    main()
