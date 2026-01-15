# video_pipeline/tasks/structured_caption.py
from __future__ import annotations
import json
from typing import Any, Dict, List

from .base import Task
from .registry import register_task

@register_task
class StructuredCaptionTask(Task):
    name = "structured_caption"

    def build_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        schema = {
            "subjects": [
                {
                    "appearance": "string",
                    "action": "string",
                    "expression": "string or empty",
                    "position": "string",
                    "TYPES": {"type": "string", "sub_type": "string"},
                    "is_main_subject": True,
                }
            ],
            "scene": {"environment": "string", "camera": "string"},
            "time": {"order": "string"},
        }
        prompt = (
            "Generate a structured caption as JSON ONLY.\n"
            f"Schema example:\n{json.dumps(schema, ensure_ascii=False)}\n"
            "Return valid JSON. No markdown."
        )

        return [
            {"role": "system", "content": "You are an expert video captioning model."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video", "video": sample["__video_uri"]},
                ],
            },
        ]

    def parse(self, generated_text: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        txt = generated_text.strip()
        # 最简单容错：截取第一个 { 到最后一个 }
        l = txt.find("{")
        r = txt.rfind("}")
        if l != -1 and r != -1 and r > l:
            txt = txt[l : r + 1]
        try:
            obj = json.loads(txt)
            return {"structured": obj}
        except Exception:
            return {"structured": None, "raw_text": generated_text}
