# video-pipeline

A modular pipeline for:
- Reading JSONL video samples (`path` field)
- Building a Torch Dataset that reads video with `qwen_vl_utils.process_vision_info` inside `__getitem__`
- Running vLLM offline inference (batch)
- Multi-process GPU inference via `torchrun` (DistributedSampler)
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

Example `configs/describe.yaml`:

```yaml
data:
  input_jsonl: /path/to/input.jsonl
  video_field: path
  id_field: null
  output_jsonl: /path/to/output.describe.jsonl
  resume: true
  num_workers: 0
  pin_memory: false

vision:
  total_pixels: 16056320   # 20480*28*28
  min_pixels: 12544        # 16*28*28
  fps: 2

vllm:
  model: Qwen/Qwen2.5-VL-7B-Instruct
  dtype: bfloat16
  max_model_len: 32768
  gpu_memory_utilization: 0.9
  enforce_eager: true
  tensor_parallel_size: 1   # IMPORTANT for torchrun data-parallel
  limit_mm_per_prompt:
    video: 1
  trust_remote_code: true

sampling:
  temperature: 0.2
  top_p: 0.9
  max_tokens: 512
  repetition_penalty: 1.0

run:
  task: describe
  batch_size: 2
  log_every: 20
  flush_every: 1
  fsync_every: 10

task_params:
  dataset: {}
```

### 2) Run

```bash
python -m video_pipeline.cli.launch --config configs/describe.yaml --gpus 0,1,2,3,4,5,6,7
```

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
     "multi_modal_data": {...}
  }
}
```