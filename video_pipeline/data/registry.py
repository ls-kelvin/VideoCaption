# video_pipeline/data/registry.py
from __future__ import annotations
from typing import Dict, Type
from .base import BaseDataset

_DATASETS: Dict[str, Type[BaseDataset]] = {}

def register_dataset(name: str):
    def deco(cls: Type[BaseDataset]):
        _DATASETS[name] = cls
        return cls
    return deco

def get_dataset_cls(name: str) -> Type[BaseDataset]:
    if name not in _DATASETS:
        raise KeyError(f"Unknown dataset: {name}. Available: {sorted(_DATASETS.keys())}")
    return _DATASETS[name]
