# video_pipeline/config/loader.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from .schema import AppConfig, DataConfig, VisionConfig, VLLMConfig, SamplingConfig, RunConfig

def load_config(path: str) -> AppConfig:
    p = Path(path)
    raw: Dict[str, Any]
    if p.suffix in [".yaml", ".yml"]:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    elif p.suffix == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported config suffix: {p.suffix}")

    cfg = AppConfig(
        data=DataConfig(**raw["data"]),
        vision=VisionConfig(**raw.get("vision", {})),
        vllm=VLLMConfig(**raw["vllm"]),
        sampling=SamplingConfig(**raw.get("sampling", {})),
        run=RunConfig(**raw["run"]),
        task_params=raw.get("task_params", {}),
    )
    return cfg
