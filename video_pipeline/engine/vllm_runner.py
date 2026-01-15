# video_pipeline/engine/vllm_runner.py
from __future__ import annotations
from typing import Any, Dict, List

from vllm import LLM, SamplingParams

from ..config.schema import VLLMConfig, SamplingConfig

class VLLMRunner:
    def __init__(self, vcfg: VLLMConfig):
        self.llm = LLM(
            model=vcfg.model,
            dtype=vcfg.dtype,
            max_model_len=vcfg.max_model_len,
            gpu_memory_utilization=vcfg.gpu_memory_utilization,
            enforce_eager=vcfg.enforce_eager,
            tensor_parallel_size=vcfg.tensor_parallel_size,
            limit_mm_per_prompt=vcfg.limit_mm_per_prompt,
            trust_remote_code=vcfg.trust_remote_code,
        )

    def generate_batch(
        self,
        llm_inputs: List[Dict[str, Any]],
        scfg: SamplingConfig,
    ):
        sp = SamplingParams(
            temperature=scfg.temperature,
            top_p=scfg.top_p,
            max_tokens=scfg.max_tokens,
            repetition_penalty=scfg.repetition_penalty,
        )
        # vLLM 支持 batch: list[{"prompt":..., "multi_modal_data":...}] :contentReference[oaicite:2]{index=2}
        return self.llm.generate(llm_inputs, sampling_params=sp)
