from __future__ import annotations
from typing import Any, Dict, List

from .base import Task
from .registry import register_task

# -------------------------- 多段动作描述 Prompt 模板 --------------------------
MULTI_ACTION_PROMPT_TEMPLATE = """
# Role
You are a detailed Video Narrator. Your task is to generate descriptive narratives for the provided video segments based on the visual evidence.

# Input
Full Video: [Video]
Input Text Segments:
{action_segments_with_indices}

# Narrative Guidelines

1. Subject and Action
Start strictly with the main subject, such as the robotic arm or the gripper. Describe the action in the third person, focusing on the movement trajectory, speed changes, and the specific interaction with objects. Use natural language to convey the smoothness or weight of the motion.

2. Scene and Environment
Weave the environmental context into the narrative. Describe the surface texture, lighting conditions, shadows, or obstacles that surround the subject. Explain how the subject is positioned spatially relative to other items in the scene.

3. Visual Accuracy
Treat the video as the primary source of truth. If the input text conflicts with the video regarding colors, shapes, or object types, describe exactly what is visible in the pixel data to ensure the narrative is grounded in reality.

4. Cinematic Perspective
Conclude the description by identifying the visual style. Mention the shot type, such as a close-up or high-angle view, and describe any camera movements, like tracking, panning, or remaining static, that frame the action.

# Output Rules
Output a numbered list matching the input segments.
Each description must be longer than 30 words.
Do not use any parentheses.
Do not include any other text or headers.

# Output Format
1. [Narrative description for segment 1]
2. [Narrative description for segment 2]
...
{num_segments}. [Narrative description for segment {num_segments}]
"""

SCENE_PROMPT_TEMPLATE = """
# Role
You are an expert Image Generation Prompt Specialist. Your objective is to synthesize the [First Frame] and [Raw Scene Text] into a precise text-to-image generation prompt.

# Input
- First Frame: [First Frame Image]
- Raw Scene Text: {raw_text}

# Directives

1.  **Generative Visual Grounding**
    Construct a description that serves as a standalone instruction for an image synthesis model. Focus exclusively on concrete visual attributes, ensuring every noun and adjective contributes directly to constructing the visual composition.

2.  **Spatial Topology and Layout**
    Establish a clear physical structure for the scene. Describe the arrangement of objects from background to foreground and define the relative positioning of the main subject within the environment. Ensure the spatial relationships are logically mapped out to guide the generation of a cohesive space.

3.  **Atmospheric and Material Realism**
    Articulate the lighting quality and material properties with factual precision. Describe the source of light, the nature of shadows, and the specific textures and colors of surfaces visible in the frame.

4.  **Grounded Natural Language**
    Use plain, high-density English. Avoid dramatic flair, exaggerated artistic terms, or emotional embellishments. The tone must remain objective, acting as a clear set of visual specifications.

# Output Format
- A single, continuous paragraph containing only the descriptive prompt.
- Minimum 100 words.

Let's write the generation prompt:
"""

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
            action_segments_with_indices="\n".join(segments_text),
            num_segments=len(action_config)
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
        
        prompt = SCENE_PROMPT_TEMPLATE.format(
            raw_text=raw_text
        )
        
        msg_content = [{"type": "text", "text": prompt}]
        
        # Dataset 必然是 qwen_image (或兼容 image 的 dataset)
        # sample["__video_uri"] 指向的是单帧图片
        msg_content.append({"type": "image", "image": sample["__image_pil"]})

        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": msg_content},
        ]

    def parse(self, generated_text: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {"detailed_init_scene_text": generated_text.strip()}
