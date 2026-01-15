# video_pipeline/io/jsonl_writer.py
from __future__ import annotations
import json
import os
from typing import Any, Dict, Optional

class JsonlWriter:
    def __init__(self, path: str, *, flush_every: int = 1, fsync_every: int = 1):
        self.path = path
        self.flush_every = max(1, flush_every)
        self.fsync_every = max(1, fsync_every)
        self._n = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")

    def write(self, obj: Dict[str, Any]) -> None:
        self._f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._n += 1
        if self._n % self.flush_every == 0:
            self._f.flush()
        if self._n % self.fsync_every == 0:
            os.fsync(self._f.fileno())

    def close(self) -> None:
        try:
            self._f.flush()
            os.fsync(self._f.fileno())
        finally:
            self._f.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
