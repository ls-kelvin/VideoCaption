# video_pipeline/io/resume.py
from __future__ import annotations
import json
from typing import Optional, Set

def load_done_keys(output_jsonl: str, key_field: str = "__key") -> Set[str]:
    done: Set[str] = set()
    try:
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if key_field in obj:
                    done.add(str(obj[key_field]))
    except FileNotFoundError:
        pass
    return done

def make_key(line_idx: int, sample: dict, id_field: Optional[str]) -> str:
    if id_field and id_field in sample:
        return str(sample[id_field])
    return str(line_idx)
