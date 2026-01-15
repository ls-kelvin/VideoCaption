# video_pipeline/tasks/registry.py
from __future__ import annotations
from typing import Dict, Type

from .base import Task

_TASKS: Dict[str, Type[Task]] = {}

def register_task(cls: Type[Task]) -> Type[Task]:
    _TASKS[cls.name] = cls
    return cls

def get_task(name: str) -> Task:
    if name not in _TASKS:
        raise KeyError(f"Unknown task: {name}. Available: {sorted(_TASKS.keys())}")
    return _TASKS[name]()
