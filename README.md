# video-pipeline

A modular pipeline for:
- Reading JSONL video samples (`path` field)
- Building a Torch Dataset that reads video with `qwen_vl_utils.process_vision_info` inside `__getitem__`
- Running vLLM offline inference (batch)
- Multi-process GPU inference
- Streaming JSONL output with resume/restart

## Install

Recommended (editable install):

```bash
# ensure your torch + vllm are installed properly for your CUDA
pip install -e .
````

## Input Format

Input is JSONL. Each line is a JSON object containing a video path:

```json
{"path": "/abs/path/to/video.mp4", "meta": {...}}
{"path": "/abs/path/to/video2.mp4"}
```

## Quick Start (torchrun data-parallel)

### 1) Prepare config (YAML)

Example `configs/describe.yaml`

### 2) Run

Example `run.sh`

Output will be sharded automatically by rank:

* `output.describe.rank0.jsonl`
* `output.describe.rank1.jsonl`
* ...

Merge:

```bash
cat /path/to/output.describe.rank*.jsonl > /path/to/output.describe.all.jsonl
```

## Resume / Restart

Resume is enabled via:

```yaml
data:
  resume: true
```

Each rank reads its own output file to collect `__key` and skips completed samples.

## How to Add a New Task

1. Create a new file `video_pipeline/tasks/my_task.py`
2. Implement a `Task` class and register it.

Example:

```python
from video_pipeline.tasks.base import Task
from video_pipeline.tasks.registry import register_task

@register_task
class MyTask(Task):
    name = "my_task"

    # Choose a dataset implementation for this task
    dataset_name = "qwen_video"

    def build_messages(self, sample):
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "What happens in this video?"},
                {"type": "video", "video": sample["__video_uri"]},
            ]},
        ]

    def parse(self, generated_text, sample):
        return {"answer": generated_text.strip()}
```

3. Make sure the module is imported somewhere (so registration happens).
   A simple approach is to import tasks in `video_pipeline/tasks/__init__.py`:

```python
from . import describe, structured_caption, my_task  # noqa: F401
```

Then set:

```yaml
run:
  task: my_task
```

## How to Add a New Dataset Type

Datasets are selected by tasks via `Task.dataset_name`.
We provide a dataset registry under `video_pipeline.data.registry`.

1. Create `video_pipeline/data/dataset_frames.py`
2. Register it:

```python
from video_pipeline.data.base import BaseDataset
from video_pipeline.data.registry import register_dataset

@register_dataset("frames")
class FramesDataset(BaseDataset):
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        # return dict with __key, raw, llm_input
        ...
```

3. Import it so registry is populated (e.g. in `video_pipeline/data/__init__.py`):

```python
from . import dataset_qwen_video, dataset_frames  # noqa: F401
```

4. In your task:

```python
class MyTask(Task):
    dataset_name = "frames"
```

### Dataset Contract

A dataset item should return:

```python
{
  "__key": "unique-id",
  "raw": {... original json object ...},
  "llm_input": {
     "prompt": "...",
     "multi_modal_data": {...},
     "mm_processor_kwargs": {...}
  }
}
```

# AgiRobot Captioning Workflow

This project includes a specialized pipeline for generating hierarchical captions (Action + Scene) for AgiRobot video data.

## Pipeline Overview

1.  **Data Preparation**: Scans raw data, speeds up videos (3x), extracts action clips, and generates input JSONL files.
2.  **Stage 1 - Action Captioning**: Generates detailed descriptions for each action clip (`agirobot_action`).
3.  **Context Injection**: Merges generated action descriptions into the metadata for the scene captioning task.
4.  **Stage 2 - Scene Captioning**: Generates a detailed initial scene description, conditioned on the full list of actions (`agirobot_scene`).
5.  **Final Merge**: Consolidates all results into a single structured JSONL file.

## Step-by-Step Instructions

### 0. Environment Setup

Ensure you have the `wm_dataset` environment active or dependencies installed.

```bash
conda activate wm_dataset
```

### 1. Data Preparation

Scans the directories defined in `scripts/prepare_agirobot_data.py`, processes videos (ffmpeg), and generates:
- `agirobot_actions.jsonl`: Inputs for Stage 1.
- `agirobot_scenes.jsonl`: Inputs for Stage 2 (Draft).

```bash
python scripts/prepare_agirobot_data.py
```

### 2. Stage 1: Action Captioning

Run the inference pipeline to describe every individual action clip using Qwen2-VL.

```bash
python -m video_pipeline.cli.launch \
    --config configs/agirobot_action.yaml \
    --gpu-ids 0,1,2,3,4,5,6,7  # Adjust GPU IDs as needed
```

**Output**: `agirobot_actions_result.rank*.jsonl`

### 3. Context Merge

Injects the generated action descriptions from Stage 1 into the input file for Stage 2. This allows the scene captioner to know exactly what actions happen in the video.

```bash
# This script reads agirobot_actions_result.rank*.jsonl and updates agirobot_scenes_merged.jsonl
python scripts/merge_action_to_scene.py
```

### 4. Stage 2: Scene Captioning

Run the inference pipeline to generate the global scene description.

```bash
python -m video_pipeline.cli.launch \
    --config configs/agirobot_scene.yaml \
    --gpu-ids 0,1,2,3,4,5,6,7  # Adjust GPU IDs as needed
```

**Output**: `agirobot_scenes_result.rank*.jsonl`

### 5. Final Result Merge

Consolidates the split result files and the action descriptions into a final, clean JSONL format.

```bash
# First, merge scene results into a single file
cat agirobot_scenes_result.rank*.jsonl > agirobot_scenes_result.jsonl

# Run final unification script
python scripts/merge_final_video_captions.py
```

**Final Output**: `agirobot_final_captions.jsonl`

## Configuration Files

*   `configs/agirobot_action.yaml`: Configures the prompt and model for short-duration action clips.
*   `configs/agirobot_scene.yaml`: Configures the prompt and model for the full scene video, utilizing text context.