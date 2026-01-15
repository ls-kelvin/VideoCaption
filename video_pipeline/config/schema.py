# video_pipeline/config/schema.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class DataConfig:
    input_jsonl: str
    video_field: str = "path"
    id_field: Optional[str] = None  # None => 用行号当唯一key
    num_workers: int = 0            # 建议默认0：视频解码+大对象传递更稳
    prefetch_factor: int = 2
    pin_memory: bool = False

    # 断点重启 / 分片
    output_jsonl: str = "outputs.jsonl"
    resume: bool = True
    num_shards: int = 1
    shard_id: int = 0

@dataclass
class VisionConfig:
    # 这些字段会进入 messages 的 video dict 中（Qwen2.5-VL 常用）
    total_pixels: int = 20480 * 28 * 28
    min_pixels: int = 16 * 28 * 28
    fps: Optional[float] = None  # qwen-vl-utils 是否支持显式fps取决于版本；不强依赖

@dataclass
class VLLMConfig:
    model: str
    dtype: str = "bfloat16"
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = True

    # 多卡：单进程 tensor parallel
    tensor_parallel_size: int = 1

    # 多模态限制
    limit_mm_per_prompt: Dict[str, int] = field(default_factory=lambda: {"video": 1})

    # 其他 vLLM 透传参数
    trust_remote_code: bool = True

@dataclass
class SamplingConfig:
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 512
    repetition_penalty: float = 1.0

@dataclass
class RunConfig:
    task: str
    batch_size: int = 1
    log_every: int = 20
    fsync_every: int = 1   # 每写几条 fsync 一次；=1 最安全但慢
    flush_every: int = 1   # 每写几条 flush 一次

@dataclass
class AppConfig:
    data: DataConfig
    vision: VisionConfig
    vllm: VLLMConfig
    sampling: SamplingConfig
    run: RunConfig

    # task 专用配置（不同 task 自己解释）
    task_params: Dict[str, Any] = field(default_factory=dict)
