from __future__ import annotations
from typing import Any, Dict, List

from .base import Task
from .registry import register_task

# -------------------------- 移植的 Prompt 模板 --------------------------

ACTION_PROMPT_TEMPLATE = """
# Role
You are an expert **Robotic Kinematics Analyst** and **Physics Engine Describer**. Your task is to analyze the input [Video Clip] and reference the [Raw Action Text] to generate a **high-fidelity, physically accurate, and visually grounded** motion narrative.

# Input
- Video Clip: [Video Frames]
- Raw Action Text: {raw_text} (Use with caution; visual data takes precedence)

# Principles of Enrichment (Strict Guidelines)

1.  **Visual Ground Truth (Highest Priority)**:
    - **Video > Text**: You must strictly adhere to the visual evidence in the video. If the text claims the gripper is "red" but the video shows "black," you must describe it as black.
    - **Correction**: Correct any discrepancies regarding object shapes, colors, or positions found in the raw text.
    - **Micro-Details**: Capture visual details missing from the text, such as dangling cables, blinking status lights, or slight mechanical jitters.

2.  **Kinematic Texture & Velocity**:
    - **Speed Curves**: Reject robotic, linear descriptions. Describe the **velocity profile**: *smooth acceleration*, *consistent cruising speed*, or *precise deceleration upon approach*.
    - **Mechanical Feel**: Articulate the nature of the movement—is it *fluid and hydraulic*, or *rigid and stepper-motor driven*? Mention inertia, such as the arm swaying slightly when stopping abruptly.

3.  **Physics & Interaction**:
    - **The Moment of Contact**: Zoom in on the interaction. Describe the friction, the compression of the gripper against the object, or any slight displacement of the object before it is fully lifted.
    - **Weight Perception**: Convey the object's mass by describing how the arm reacts (e.g., a momentary lag or increased motor effort when hoisting).

4.  **Lighting & Environmental Response**:
    - Describe how the robot's movement alters the scene: casting moving shadows across the workspace, or how specular highlights shift across its metallic surfaces during rotation.

5.  **Formatting & Flow**:
    - **NO Parentheses**: Integrate all specifications into fluid sentences (e.g., "the silver arm equipped with a vacuum gripper").
    - **Fluid Narrative**: Use transitional phrasing like *"Simultaneous with the base rotation..."*, *"In coordination with..."*, or *"Leading seamlessly into..."*.
    - **Length**: Minimum 150 words.

# Output
- Output ONLY the descriptive paragraph. Focus entirely on the **mechanics, flow, and physical reality** of the action.

Let's analyze the motion dynamics step by step:
"""

SCENE_PROMPT_TEMPLATE = """
# Role
You are a Hollywood-level **Virtual Set Designer** and **Director of Photography (DoP)**. Your objective is to synthesize the [First Frame], [Raw Scene Text], and [First Action Instruction] into a **spatially precise, atmospherically rich** video generation prompt.

# Input
- First Frame: [First Frame Image]
- Raw Scene Text: {raw_text}
- First Action Instruction: {first_action_text}

# Principles of Enrichment (Strict Guidelines)

1.  **Cinematic Camera & Atmosphere**:
    - **Shot Definition**: Explicitly define the camera angle (e.g., High-angle wide shot, Eye-level close-up) and the depth of field (e.g., sharp focus on the foreground, bokeh-blurred background).
    - **Lighting Logic**: Describe the lighting source and quality (e.g., "diffused cool-tone industrial overhead lighting" or "harsh directional shadows"). Describe how light interacts with materials (glossy reflections vs. matte surfaces).

2.  **Layered Spatial Mapping**:
    - **Hierarchy**: Systematically build the scene from **Background** to **Midground** to **Foreground**.
    - **Anchoring**: Use precise terms like *Left/Right, Adjacent to, Centered, Perpendicular to*.
    - **Object Specificity**: Use specific nouns from the First Action Text (e.g., replace "bottle" with "hydrating toner bottle") to resolve any ambiguity in the scene description.

3.  **High-Fidelity Subject Definition**:
    - **Visual Correction**: If the Raw Text says "Red Gripper" but the First Frame shows "Black," you must describe it as "Black."
    - **The "Idle" State**: Describe the robot's posture just before movement—its potential energy (e.g., "poised in a ready state," "hovering statically").

4.  **The "Break of Static" (Action Initiation)**:
    - After establishing the static scene, dedicate the final sentences to the **exact moment movement begins**.
    - **Trigger**: Describe the initial break from stillness (e.g., "the servo motors engage, and the arm begins a slow, calculated extension trajectory towards the shelf").
    - Do not describe the completed action, only the **onset**.

5.  **Formatting & Flow**:
    - **NO Parentheses**: All details must be woven into the narrative structure.
    - **Visual Factuality**: Avoid abstract inference (e.g., do not guess "this is a sorting zone"). Describe the visible layout (e.g., "a workbench lined with cylindrical components").
    - **Length**: Minimum 200 words.

# Output
- A single, dense, and visually evocative paragraph. It should read like the opening script for a high-end documentary, establishing the scene and transitioning smoothly into the first second of motion.

Let's construct the scene and trigger the first movement:
"""

@register_task
class AgiRobotActionTask(Task):
    name = "agirobot_action"
    dataset_name = "qwen_video"

    def build_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        # sample 会包含: {"path": "segment.mp4", "raw_text": "原始动作描述"}
        raw_text = sample.get("raw_text", "无动作描述")
        prompt = ACTION_PROMPT_TEMPLATE.format(raw_text=raw_text)
        
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
        return {"detailed_action_text": generated_text.strip()}

@register_task
class AgiRobotSceneTask(Task):
    name = "agirobot_scene"
    dataset_name = "qwen_image"

    def build_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        # sample 会包含: {"path": "speedup_video_frame.jpg" (或者 mp4), "raw_text": "场景描述", "first_action_text": "..."}
        # 注意：这里我们传入图片或视频首帧。框架会自动处理 sample["__video_uri"]
        
        raw_text = sample.get("raw_text", "无场景描述")
        first_action_text = sample.get("first_action_text", "无首个动作描述")
        
        prompt = SCENE_PROMPT_TEMPLATE.format(
            raw_text=raw_text,
            first_action_text=first_action_text
        )
        
        msg_content = [{"type": "text", "text": prompt}]
        
        # Dataset 必然是 qwen_image (或兼容 image 的 dataset)
        # sample["__video_uri"] 指向的是单帧图片
        msg_content.append({"type": "image", "image": sample["__video_uri"]})

        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": msg_content},
        ]

    def parse(self, generated_text: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {"detailed_init_scene_text": generated_text.strip()}
