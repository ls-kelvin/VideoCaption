# video_pipeline/data/collate.py
from __future__ import annotations
from typing import Any, Dict, List

def collate_batch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "keys": [it["__key"] for it in items],
        "line_idxs": [it["__line_idx"] for it in items],
        "raws": [it["raw"] for it in items],
        "llm_inputs": [it["llm_input"] for it in items],
    }
