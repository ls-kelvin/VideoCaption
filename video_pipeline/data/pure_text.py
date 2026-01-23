from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

from transformers import AutoTokenizer

from .base import BaseDataset
from .registry import register_dataset
from ..io.resume import make_key


@register_dataset("pure_text")
class PureTextJsonlDataset(BaseDataset):
    """JSONL -> text-only chat prompt dataset (no multimodal inputs)."""

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
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        return self._tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        line_idx, sample = self.samples[i]
        key = make_key(line_idx, sample, self.id_field)

        sample2 = dict(sample)
        sample2["__key"] = key
        sample2["__line_idx"] = line_idx

        # expose task_params to tasks via sample (so build_messages/parse can read it)
        sample2["__task_params"] = getattr(self.task, "task_params", {})

        messages = self.task.build_messages(sample2)

        tok = self._get_tokenizer()
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        llm_input = {
            "prompt": prompt,
            "multi_modal_data": {},
            "mm_processor_kwargs": {},
        }

        return {
            "__key": key,
            "__line_idx": line_idx,
            "raw": sample,
            "messages": messages,
            "llm_input": llm_input,
        }
