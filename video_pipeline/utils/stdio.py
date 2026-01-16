# video_pipeline/utils/stdio.py
from __future__ import annotations
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

@contextmanager
def redirect_stdouterr(log_file: str) -> Iterator[None]:
    """
    Redirect stdout/stderr to a file (append). Keeps terminal clean.
    """
    Path(os.path.dirname(log_file) or ".").mkdir(parents=True, exist_ok=True)

    old_out, old_err = sys.stdout, sys.stderr
    f = open(log_file, "a", encoding="utf-8", buffering=1)
    try:
        sys.stdout = f
        sys.stderr = f
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        f.close()
