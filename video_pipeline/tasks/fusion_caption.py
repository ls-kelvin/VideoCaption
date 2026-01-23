from __future__ import annotations

import json
import random
from typing import Any, Dict, List

from .base import Task
from .registry import register_task


SYSTEM_PROMPT_I2V = """
You are an expert in video captioning. You are given a structured video caption and you need to compose it to be more natural and fluent in English.

## Structured Input
{structured_input}

## Notes
1. If there has an empty field, just ignore it and do not mention it in the output.
2. Do not make any semantic changes to the original fields. Please be sure to follow the original meaning.
3. If the action field is not empty, eliminate the irrelevant information in the action field that is not related to the timing action(such as wearings, background and environment information) to make a pure action field.

## Output Principles and Orders
1. First, eliminate the static information in the action field that is not related to the timing action, such as background or environment information.
2. Second, describe each subject with its pure action and expression if these fields exist.

## Output
Please directly output the final composed caption without any additional information.
""".strip()


SYSTEM_PROMPT_T2V = """
You are an expert in video captioning. You are given a structured video caption and you need to compose it to be more natural and fluent in English.

## Structured Input
{structured_input}

## Notes
1. According to the action field information, change its name field to the subject pronoun in the action.
2. If there has an empty field, just ignore it and do not mention it in the output.
3. Do not make any semantic changes to the original fields. Please be sure to follow the original meaning.
4. Do NOT use any lead-in or meta phrases such as "the video shows", "this video", "we see", "in the clip", "the scene shows", "the camera shows". Start directly with the description.

## Output Principles and Orders
1. First, declare the shot_type, then declare the shot_angle and the shot_position fields.
2. Second, eliminate information in the action field that is not related to the timing action, such as background or environment information if action is not empty.
3. Third, describe each subject with its pure action, appearance, expression, position if these fields exist.
4. Finally, declare the environment and lighting if the environment and lighting fields are not empty.

## Output
Please directly output the final composed caption without any additional information.
""".strip()


def compute_camera_movement(struct_caption: Dict[str, Any]) -> str:
    camera_movement = struct_caption.get("camera_motion", "")
    if camera_movement != "":
        camera_movement += "."
    return camera_movement.capitalize()


def clean_struct_caption(struct_caption: Dict[str, Any], task_mode: str) -> Dict[str, Any]:
    raw_subjects = struct_caption.get("subjects", [])
    subjects = []
    for subject in raw_subjects:
        subject_type = subject.get("TYPES", {}).get("type", "")
        subject_sub_type = subject.get("TYPES", {}).get("sub_type", "")

        if subject_type not in ["Human", "Animal"]:
            subject["expression"] = ""
        if subject_type == "Human" and subject_sub_type == "Accessory":
            subject["expression"] = ""

        if subject_sub_type != "":
            subject["name"] = subject_sub_type

        if "TYPES" in subject:
            del subject["TYPES"]
        if "is_main_subject" in subject:
            del subject["is_main_subject"]

        subjects.append(subject)

    to_del_subject_ids = []
    for idx, subject in enumerate(subjects):
        action = subject.get("action", "").strip()
        subject["action"] = action

        if random.random() > 0.9 and "appearance" in subject:
            del subject["appearance"]
        if random.random() > 0.9 and "position" in subject:
            del subject["position"]

        if task_mode == "i2v":
            dropped_keys = ["appearance", "position"]
            for key in dropped_keys:
                if key in subject:
                    del subject[key]
            if subject.get("action", "") == "" and subject.get("expression", "") == "":
                to_del_subject_ids.append(idx)

    for idx in sorted(to_del_subject_ids, reverse=True):
        del subjects[idx]

    new_struct_caption: Dict[str, Any] = {
        "num_subjects": len(subjects),
        "subjects": subjects,
        "shot type": struct_caption.get("shot_type", "").replace("_", " "),
        "shot angle": struct_caption.get("shot_angle", "").replace("_", " "),
        "shot position": struct_caption.get("shot_position", "").replace("_", " "),
        "environment": struct_caption.get("environment", "").replace("_", " "),
        "lighting": struct_caption.get("lighting", "").replace("_", " "),
    }

    if task_mode == "t2v" and random.random() > 0.9:
        new_struct_caption.pop("lighting", None)

    if task_mode == "i2v":
        drop_keys = ["environment", "lighting", "shot type", "shot angle", "shot position"]
        for k in drop_keys:
            new_struct_caption.pop(k, None)

    return new_struct_caption


@register_task
class FusionCaptionTask(Task):
    name = "fusion_caption"
    dataset_name = "pure_text"

    def __init__(self):
        # NOTE: task_params are injected in worker via cfg.task_params, not via ctor.
        # We keep defaults here and read overrides from sample["__task_params"] if present.
        self._default_mode = "t2v"
        self._default_input_field = "caption"
        self._default_original_text = "-"

    def _get_params(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        tp = getattr(self, "task_params", None) or {}
        # sample may override (useful for tests)
        tp = {**tp, **(sample.get("__task_params") or {})}
        return {
            "mode": tp.get("mode", self._default_mode),
            "input_field": tp.get("input_field", self._default_input_field),
            "original_text": tp.get("original_text", self._default_original_text),
        }

    def build_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        p = self._get_params(sample)
        mode = p["mode"]
        input_field = p["input_field"]

        system_prompt = SYSTEM_PROMPT_T2V if mode == "t2v" else SYSTEM_PROMPT_I2V

        struct_text = sample.get(input_field, None)
        try:
            if isinstance(struct_text, str):
                s = struct_text.strip()
                l = s.find("{")
                r = s.rfind("}")
                if l != -1 and r != -1 and r > l:
                    s = s[l : r + 1]
                struct_caption = json.loads(s)
            else:
                struct_caption = struct_text
            if not isinstance(struct_caption, dict):
                raise ValueError(f"struct_caption is not dict: {type(struct_caption)}")
        except Exception:
            struct_caption = {}

        cleaned = clean_struct_caption(struct_caption, mode)
        new_struct_caption = json.dumps(cleaned, indent=4, ensure_ascii=False)

        return [
            {
                "role": "user",
                "content": system_prompt.format(structured_input=new_struct_caption),
            }
        ]

    def parse(self, generated_text: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        p = self._get_params(sample)
        mode = p["mode"]
        input_field = p["input_field"]
        original_text = p["original_text"]

        struct_text = sample.get(input_field, None)
        try:
            if isinstance(struct_text, str):
                s = struct_text.strip()
                l = s.find("{")
                r = s.rfind("}")
                if l != -1 and r != -1 and r > l:
                    s = s[l : r + 1]
                struct_caption = json.loads(s)
            else:
                struct_caption = struct_text
            if not isinstance(struct_caption, dict):
                raise ValueError(f"struct_caption is not dict: {type(struct_caption)}")
        except Exception as e:
            return {
                "status": "error",
                "error": f"{input_field}_json_parse_error: {repr(e)}",
                "fusion_caption": original_text,
                "fusion_by_llm": False,
            }

        camera_movement = compute_camera_movement(struct_caption)
        cleaned = clean_struct_caption(struct_caption, mode)
        fusion_by_llm = cleaned.get("num_subjects", 0) > 0

        if not fusion_by_llm:
            caption = (original_text + " " + camera_movement).strip()
            return {
                "status": "ok",
                "fusion_caption": caption,
                "fusion_by_llm": False,
            }

        # mimic stop=["\n"] by truncating the first line
        txt = (generated_text or "").strip().split("\n", 1)[0].strip()
        llm_caption = (txt + " " + camera_movement).strip()
        return {
            "status": "ok",
            "fusion_caption": llm_caption,
        }
