# video_pipeline/data/jsonl_reader.py
from __future__ import annotations
import json
from typing import Dict, Iterator, Tuple

def iter_jsonl(path: str) -> Iterator[Tuple[int, Dict]]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield i, json.loads(line)
