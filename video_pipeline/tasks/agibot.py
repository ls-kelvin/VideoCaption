from __future__ import annotations
from typing import Any, Dict, List

from .base import Task
from .registry import register_task

# -------------------------- 多段动作描述 Prompt 模板 --------------------------
MULTI_ACTION_PROMPT_TEMPLATE = """
# Role
You are an expert video captioning assistant for generative text-to-video (T2V) models.

# Task
Analyze the provided video frames and the input JSON (containing an initial scene description and segmented action labels). Convert each segment's simple action label into a highly descriptive, renderable T2V prompt.

# Constraints & Requirements
1.  **Visual Specificity:** Clearly identify the **Agent** (who) and **Object** (what). Describe the **Action's process** and its **visible effect**.
2.  **Environment & Camera:** Describe any background changes, lighting shifts, or specific camera movements (e.g., "slow zoom in," "tracking shot"). If the camera is still, specify "static shot."
3.  **Renderability:** Use only concrete, objective visual descriptions. Avoid abstract concepts, subjective feelings, or meta-commentary (e.g., do not say "a touching moment").
4.  **Length:** Each segment prompt must be between **20 and 80 words**.

# Input Format
- **Initial Scene:** 
{init_scene}

- **Segments:** 
{action_segments_with_indices}

# Output Format
Return a JSON object where keys are the segment timestamps and values are the expanded prompts.

# Example Output
1. Prompt1
2. ...
""".strip()

SCENE_PROMPT_TEMPLATE = """
You are an expert prompt engineer. Analyze the provided image and generate a detailed text-to-image prompt.

Your response must adhere to the following strict guidelines:

1. **Composition & Attributes**: Mention every visible object and define their exact spatial relationships. Specify distinct physical attributes for all elements, including specific colors, textures, materials, and postures if it's unique.
2. **Technical Settings**: Explicitly state the camera viewpoint (e.g., aerial, eye-level, macro).
3. **Renderability**: Use only concrete, visual descriptors. Strictly avoid abstract concepts, emotional interpretations, subjective adjectives, or meta-commentary.
4. **Style**: Write in a cohesive, narrative natural language format using full sentences of moderate length (no more than 25 words each) rather than listing.
5. **Length:** The output must be between 80 and 250 words.
""".strip()

@register_task
class AgiRobotActionTask(Task):
    name = "agibot_action"
    dataset_name = "qwen_video"

    def build_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Extract action_config from label_info
        action_config = sample.get("label_info", {}).get("action_config", [])

        # Build segment descriptions for prompt
        segments_text = []
        for i, seg in enumerate(action_config, 1):
            raw_text = seg.get("action_text", "No description")
            start = seg["start_frame"]
            end = seg["end_frame"]
            segments_text.append(
                f"Segment {i}: Frames {start/30:.1f}s–{end/30:.1f}s — Raw: \"{raw_text}\""
            )
        
        prompt = MULTI_ACTION_PROMPT_TEMPLATE.format(
            action_segments_with_indices="\n".join(segments_text)
        )

        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video", "video": sample["__video_uri"]},
                ],
            },
        ]

    def parse(self, generated_text: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        # Split by numbered lines (robust parsing)
        lines = generated_text.strip().split('\n')
        captions = []
        current_caption = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Check if line starts a new numbered item (e.g., "1.", "2.", ..., "10.")
            if line[0].isdigit() and '.' in line.split()[0]:
                if current_caption:
                    captions.append(current_caption.strip())
                # Remove the number prefix
                parts = line.split('.', 1)
                if len(parts) > 1:
                    current_caption = parts[1].strip()
                else:
                    current_caption = ""
            else:
                current_caption += " " + line

        if current_caption:
            captions.append(current_caption.strip())

        # Ensure alignment with action_config
        action_config = sample.get("label_info", {}).get("action_config", [])
        expected_count = len(action_config)
        if len(captions) != expected_count:
            # Pad or truncate to match
            captions = (captions + ["[Caption generation failed.]"] * expected_count)[:expected_count]

        return {
            "detailed_action_captions": captions  # List[str], aligned with action_config
        }
        
@register_task
class AgiRobotSceneTask(Task):
    name = "agibot_scene"
    dataset_name = "first_frame"

    def build_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        # sample 会包含: {"path": "speedup_video_frame.jpg" (或者 mp4), "raw_text": "场景描述", "first_action_text": "..."}
        # 注意：这里我们传入图片或视频首帧。框架会自动处理 sample["__video_uri"]
        
        raw_text = sample.get("raw_text", "No description")
        
        prompt = "Please output the result as a single, continuous paragraph **without any semicolons**."
        
        msg_content = [{"type": "text", "text": prompt}]
        
        # Dataset 必然是 qwen_image (或兼容 image 的 dataset)
        # sample["__video_uri"] 指向的是单帧图片
        msg_content.insert(0, {"type": "image", "image": sample["__image_pil"]})

        return [
            {"role": "system", "content": SCENE_PROMPT_TEMPLATE},
            {"role": "user", "content": msg_content},
        ]

    def parse(self, generated_text: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {"detailed_init_scene_text": generated_text.strip()}
