# video_pipeline/data/base.py
from __future__ import annotations
from abc import ABC
from typing import Any, Dict, Optional
from torch.utils.data import Dataset

class BaseDataset(Dataset, ABC):
    """
    所有 dataset 的统一基类（可选，但推荐）。
    约定：__getitem__ 返回 dict，至少包含：
      - __key
      - raw
      - llm_input  (vLLM 需要的 {"prompt":..., "multi_modal_data":...})
    """
    def __init__(
        self,
        *,
        samples,
        model_path: str,
        video_field: str,
        id_field: Optional[str],
        vision_kwargs: Dict[str, Any],
        task,
        dataset_params: Optional[Dict[str, Any]] = None,
    ):
        self.samples = samples
        self.model_path = model_path
        self.video_field = video_field
        self.id_field = id_field
        self.vision_kwargs = vision_kwargs
        self.task = task
        self.dataset_params = dataset_params or {}
