# video_pipeline/utils/progress.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import threading
import queue as queue_mod

from tqdm import tqdm


@dataclass
class ProgressMonitor:
    total: int
    desc: str = "Done"
    unit: str = "video"

    def run(self, q, stop_token: str = "__STOP__") -> None:
        """
        q: multiprocessing.Queue
        """
        pbar = tqdm(total=self.total, desc=self.desc, unit=self.unit, dynamic_ncols=True)
        done = 0
        try:
            while True:
                msg = q.get()
                if msg == stop_token:
                    break
                # msg can be int (delta)
                if isinstance(msg, int):
                    done += msg
                    pbar.update(msg)
        finally:
            # ensure progress reaches done count if needed
            pbar.close()
