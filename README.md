# Video Captioning Pipeline (AgiRobot Edition)

This repository contains a modular video captioning pipeline designed for the **AgiRobot** project. It utilizes **Qwen2-VL** models via **vLLM** for high-throughput, multi-GPU inference. The pipeline generates high-fidelity, physically accurate descriptions of robotic arm movements (Actions) and environmental settings (Scenes).

## Workflow Overview

The pipeline operates in two main stages to ensure detailed and context-aware captions:

1.  **Action Captioning (Stage 1)**: The long video is physically cut into short segments based on action timestamps. The model describes the motion, physics, and kinematics of each segment.
2.  **Scene Captioning (Stage 2)**: The model describes the initial scene arrangement using the first frame. It incorporates the "First Action Description" from Stage 1 as context to ensure consistency between the static scene and the dynamic movement.

## Quick Start

We provide a one-click shell script to run the entire pipeline (Data Prep -> Inference -> Merge).
**Note**: Please check GPU availability with `nvidia-smi` before running, as vLLM requires significant VRAM.

```bash
# Run with default settings (Uses all available GPUs)
bash run_pipeline.sh

# Run on specific GPUs (e.g., GPUs 0, 1, 2, 3)
# Recommended to specify free GPUs to avoid OOM errors
GPUS=0,1,2,3 bash run_pipeline.sh
```

## Installation

### 1. Environment Setup

```bash
conda create -n video_caption python=3.10 -y
conda activate video_caption
```

### 2. Dependencies

```bash
# Install PyTorch (adjust CUDA version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project and core libraries
pip install -e .

# Install FFmpeg for video processing
pip install imageio-ffmpeg
```

## Detailed Pipeline Steps

The `run_pipeline.sh` orchestrates the following individual scripts:

### 1. Data Preparation
**Script**: `scripts/prepare_agirobot_data.py`
- Reads raw tasks and episodes.
- Uses `ffmpeg` to physically cut video segments corresponding to actions.
- Extracts the first frame for scene description.
- Generates:
    - `agirobot_actions.jsonl`: Inputs for Action Captioning.
    - `agirobot_scenes.jsonl`: Inputs for Scene Captioning (Initial).

### 2. Action Captioning
**Config**: `configs/agirobot_action.yaml`
- **Model**: Qwen2-VL-7B-Instruct (Video Mode)
- **Input**: Video segments from Step 1.
- **Output**: `agirobot_actions_result.rank*.jsonl`
- Describes velocity, gripper interaction, and object physics.

### 3. Context Merging
**Script**: `scripts/merge_action_to_scene.py`
- Extracts the generated description of the *first action* from Step 2.
- Injects it into the prompt for Scene Captioning in `agirobot_scenes.jsonl`.
- **Reason**: The scene description looks better if it anticipates the first movement (e.g., "The robot is poised to lift...").

### 4. Scene Captioning
**Config**: `configs/agirobot_scene.yaml`
- **Model**: Qwen2-VL-7B-Instruct (Image Mode)
- **Input**: First frame image + Context from Step 3.
- **Output**: `agirobot_scenes_result.rank*.jsonl`
- Describes lighting, object arrangement, and robot starting pose.

### 5. Final Aggregation
**Script**: `scripts/merge_final_video_captions.py`
- Combines Scene descriptions and Action descriptions into the final format.
- **Output**: `agirobot_final_captions.jsonl`

## Configuration

Inference settings are defined in YAML files under `configs/`:

- **agirobot_action.yaml**: Settings for video action inference.
- **agirobot_scene.yaml**: Settings for image scene inference.

Key parameters in YAML:
```yaml
vllm:
  model: /path/to/model  # Model checkpoint path
  gpu_memory_utilization: 0.9
  tensor_parallel_size: 1 # Number of GPUs per model instance
```

## Output Format

The final file `agirobot_final_captions.jsonl` contains:

```json
{
  "video_id": "354_660485",
  "meta": { ... },
  "scene_description": "A dense paragraph describing the initial scene...",
  "actions": [
    {
      "action_idx": 0,
      "instruction": "Retrieve the bottle.",
      "description": "The robotic arm smoothly accelerates...",
      "start_frame": 0,
      "end_frame": 125
    },
    ...
  ]
}
```