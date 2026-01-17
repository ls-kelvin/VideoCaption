from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from .base import BaseDataset
from .registry import register_dataset
from ..io.resume import make_key

@register_dataset("qwen_image")
class QwenImageJsonlDataset(BaseDataset):
    def __init__(
        self,
        *,
        samples: List[Tuple[int, Dict[str, Any]]],
        model_path: str,
        video_field: str,
        id_field: Optional[str],
        vision_kwargs: Dict[str, Any],
        task,
        dataset_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            samples=samples,
            model_path=model_path,
            video_field=video_field,
            id_field=id_field,
            vision_kwargs=vision_kwargs,
            task=task,
            dataset_params=dataset_params,
        )
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_path)
        return self._processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        line_idx, sample = self.samples[i]
        key = make_key(line_idx, sample, self.id_field)

        video_path = sample[self.video_field]
        # Treat as image URI
        video_uri = video_path if str(video_path).startswith(("http://", "https://", "file://")) else f"file://{video_path}"

        sample2 = dict(sample)
        sample2["__key"] = key
        sample2["__line_idx"] = line_idx
        sample2["__video_uri"] = video_uri

        messages = self.task.build_messages(sample2)

        # Inject vision kwargs into IMAGE items
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        # For images, we can control min/max pixels if needed
                        # Usually vision_kwargs passed from config has keys like 'min_pixels', 'max_pixels'
                        # Filter relevant keys if necessary, or just update all
                        item.update(self.vision_kwargs)

        processor = self._get_processor()
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True
        )

        mm_data: Dict[str, Any] = {}
        # Theoretically video_inputs should be None for image tasks, but handle generically
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        if image_inputs is not None:
            mm_data["image"] = image_inputs

        llm_input = {
            "prompt": prompt, 
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs
        }

        return {
            "__key": key,
            "__line_idx": line_idx,
            "raw": sample,
            "messages": messages,
            "llm_input": llm_input,
        }
