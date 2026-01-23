# video_pipeline/tasks/describe.py
from __future__ import annotations
from typing import Any, Dict, List

from .base import Task
from .registry import register_task

SYSTEM_PROMPT = (
    "I need you to generate a structured and detailed caption for the provided video. "
    "The structured output and the requirements for each field are as shown in the following JSON content: "
    "{\"subjects\": [{\"appearance\": \"Main subject appearance description\", \"action\": \"Main subject action\", "
    "\"expression\": \"Main subject expression  (Only for human/animal categories, empty otherwise)\", "
    "\"position\": \"Subject position in the video (Can be relative position to other objects or spatial description)\", "
    "\"TYPES\": {\"type\": \"Main category (e.g., Human)\", \"sub_type\": \"Sub-category (e.g., Man)\"}, "
    "\"is_main_subject\": true}, {\"appearance\": \"Non-main subject appearance description\", "
    "\"action\": \"Non-main subject action\", \"expression\": \"Non-main subject expression "
    "(Only for human/animal categories, empty otherwise)\", \"position\": \"Position of non-main subject 1\", "
    "\"TYPES\": {\"type\": \"Main category (e.g., Vehicles)\", \"sub_type\": \"Sub-category (e.g., Ship)\"}, "
    "\"is_main_subject\": false}], \"shot_type\": \"Shot type(Options: long_shot/full_shot/medium_shot/close_up/"
    "extreme_close_up/other)\", \"shot_angle\": \"Camera angle(Options: eye_level/high_angle/low_angle/other)\", "
    "\"shot_position\": \"Camera position(Options: front_view/back_view/side_view/over_the_shoulder/overhead_view/"
    "point_of_view/aerial_view/overlooking_view/other)\", \"camera_motion\": \"Camera movement description\", "
    "\"environment\": \"Video background/environment description\", \"lighting\": \"Lighting information in the video\"}"
)

@register_task
class SkyCaptionerTask(Task):
    name = "skycaption"

    def build_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        # video block 会在 dataset 里补充（total_pixels/min_pixels 等可由 config 控制）
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {"type": "video", "video": sample["__video_uri"]},
                ],
            },
        ]

    def parse(self, generated_text: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {"caption": generated_text.strip()}
