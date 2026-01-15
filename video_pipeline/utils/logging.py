# video_pipeline/utils/logging.py
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Dict

# ---------- Config ----------

@dataclass
class LogConfig:
    run_name: str = "run"
    log_dir: str = "logs"

    # Console behavior
    console: bool = True
    console_level_main: str = "INFO"      # rank0
    console_level_worker: str = "WARNING" # non-rank0

    # File behavior
    file: bool = True
    file_level: str = "INFO"
    rotate: bool = True
    max_bytes: int = 50 * 1024 * 1024  # 50MB
    backup_count: int = 5

    # Formatting
    utc_time: bool = False
    include_process: bool = True
    include_thread: bool = False

    # Noisy libs
    quiet_third_party: bool = True


# ---------- Rank helpers ----------

def get_dist_rank() -> int:
    return int(os.environ.get("RANK", "0"))

def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

def is_distributed() -> bool:
    return get_world_size() > 1


# ---------- Formatter ----------

class RankFormatter(logging.Formatter):
    """
    Formatter that injects rank/world_size and supports UTC timestamps.
    """
    def __init__(self, fmt: str, datefmt: str, *, utc: bool):
        super().__init__(fmt=fmt, datefmt=datefmt)
        if utc:
            self.converter = time_gmtime  # type: ignore[attr-defined]

def time_gmtime(*args):
    import time
    return time.gmtime(*args)


def _level_from_env(default: str) -> str:
    # Allow override via LOG_LEVEL (e.g. DEBUG/INFO/WARNING)
    env = os.environ.get("LOG_LEVEL")
    return env.strip().upper() if env else default


def _make_format(cfg: LogConfig) -> str:
    # Example:
    # 2026-01-15 10:22:33.123 | INFO | r0/8 | video_pipeline.cli.run:123 | msg
    parts = [
        "%(asctime)s.%(msecs)03d",
        "%(levelname)s",
        "r%(rank)d/%(world_size)d",
        "%(name)s:%(lineno)d",
        "%(message)s",
    ]
    if cfg.include_process:
        parts.insert(3, "pid=%(process)d")
    if cfg.include_thread:
        parts.insert(4, "tid=%(thread)d")
    return " | ".join(parts)


class _RankFilter(logging.Filter):
    def __init__(self, rank: int, world_size: int):
        super().__init__()
        self.rank = rank
        self.world_size = world_size

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        record.world_size = self.world_size
        return True


# ---------- Public API ----------

def setup_logging(cfg: Optional[LogConfig] = None) -> Dict[str, str]:
    """
    Configure root logging once. Safe to call per process.
    Returns useful paths/info.
    """
    cfg = cfg or LogConfig()
    rank = get_dist_rank()
    world_size = get_world_size()

    # Root logger: start clean (avoid duplicated handlers in notebooks/restarts)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # handlers control output

    for h in list(root.handlers):
        root.removeHandler(h)

    # Format
    fmt = _make_format(cfg)
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = RankFormatter(fmt=fmt, datefmt=datefmt, utc=cfg.utc_time)
    rank_filter = _RankFilter(rank=rank, world_size=world_size)

    # Console handler
    if cfg.console:
        ch = logging.StreamHandler(stream=sys.stdout)
        lvl = cfg.console_level_main if rank == 0 else cfg.console_level_worker
        ch.setLevel(getattr(logging, _level_from_env(lvl)))
        ch.setFormatter(formatter)
        ch.addFilter(rank_filter)
        root.addHandler(ch)

    # File handler
    log_path = None
    if cfg.file:
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
        log_path = str(Path(cfg.log_dir) / f"{cfg.run_name}.rank{rank}.log")

        if cfg.rotate:
            fh: logging.Handler = RotatingFileHandler(
                log_path,
                maxBytes=cfg.max_bytes,
                backupCount=cfg.backup_count,
                encoding="utf-8",
            )
        else:
            fh = logging.FileHandler(log_path, encoding="utf-8")

        fh.setLevel(getattr(logging, _level_from_env(cfg.file_level)))
        fh.setFormatter(formatter)
        fh.addFilter(rank_filter)
        root.addHandler(fh)

    # Third-party noise control
    if cfg.quiet_third_party:
        _quiet_libs()

    # Make warnings go through logging
    logging.captureWarnings(True)

    # Helpful: show config once on rank0
    if rank == 0:
        logging.getLogger("video_pipeline").info(
            "Logging initialized (world_size=%d, log_dir=%s, run_name=%s)",
            world_size, cfg.log_dir, cfg.run_name
        )

    return {
        "log_path": log_path or "",
        "rank": str(rank),
        "world_size": str(world_size),
    }


def get_logger(name: str = "video_pipeline") -> logging.Logger:
    return logging.getLogger(name)


def _quiet_libs():
    # Keep your own logs, reduce noisy libraries
    noisy = [
        "transformers",
        "vllm",
        "urllib3",
        "httpx",
        "asyncio",
        "PIL",
        "matplotlib",
        "numba",
    ]
    for n in noisy:
        logging.getLogger(n).setLevel(logging.WARNING)

    # Some libs create too verbose sub-loggers
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    logging.getLogger("vllm.engine").setLevel(logging.WARNING)
    logging.getLogger("vllm.worker").setLevel(logging.WARNING)
