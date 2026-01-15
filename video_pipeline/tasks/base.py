# video_pipeline/tasks/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class Task(ABC):
    name: str
    
    dataset_name: str = "qwen_video"

    @abstractmethod
    def build_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """返回 Qwen chat messages（含 video block）。"""

    @abstractmethod
    def parse(self, generated_text: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        """把模型输出解析为你要写回 jsonl 的字段。"""

    def extra_output_fields(self) -> Dict[str, Any]:
        return {}
