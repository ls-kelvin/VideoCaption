# video_pipeline/tasks/describe.py
from __future__ import annotations
from typing import Any, Dict, List

from .base import Task
from .registry import register_task

@register_task
class DescribeVideoTask(Task):
    name = "describe"

    def build_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        # video block 会在 dataset 里补充（total_pixels/min_pixels 等可由 config 控制）
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video in detail."},
                    {"type": "video", "video": sample["__video_uri"]},
                ],
            },
        ]

    def parse(self, generated_text: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {"description": generated_text.strip()}
