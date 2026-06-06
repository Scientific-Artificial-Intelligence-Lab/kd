
from __future__ import annotations

import logging
import os

import torch

_BYTES_PER_GB = 1e9
_ENV_FLAG = "DISCOVER_LOG_MEMORY"
_ENV_ON_VALUE = "1"



_LOG_MEMORY_ENABLED: bool = os.environ.get(_ENV_FLAG, "0") == _ENV_ON_VALUE


def _log_memory(stage: str, logger: logging.Logger) -> None:
    if not _LOG_MEMORY_ENABLED:
        return
    if not torch.cuda.is_available():
        return
    try:
        alloc = torch.cuda.memory_allocated() / _BYTES_PER_GB
        peak = torch.cuda.max_memory_allocated() / _BYTES_PER_GB
        reserved = torch.cuda.memory_reserved() / _BYTES_PER_GB
    except RuntimeError:

        return
    logger.info(
        "[MEM] %s: alloc=%.2fGB peak=%.2fGB reserved=%.2fGB",
        stage, alloc, peak, reserved,
    )


__all__ = ["_log_memory", "_LOG_MEMORY_ENABLED", "_ENV_FLAG", "_ENV_ON_VALUE"]
