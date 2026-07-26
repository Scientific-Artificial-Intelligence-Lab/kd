
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from kd.core import safety_counters
from kd.search.discover.golden.pde1d import run_burgers_mode1, run_chafee_mode1
from kd.search.discover.golden.summarise import GoldenRunResult

logger = logging.getLogger(__name__)


def run_golden(
    pde: str,
    mode: str,
    seed: int,
    *,
    data_path: Path | None = None,
) -> tuple[GoldenRunResult, dict[str, Any]]:




    if safety_counters.counters_enabled():
        safety_counters.reset()
    if mode == "mode1":
        outcome = _run_mode1(pde, seed, data_path=data_path)
        safety_counters.emit_run_summary(f"golden:{pde}:{mode}:seed={seed}", logger)
        return outcome
    raise ValueError(f"Unknown mode: {mode!r}")


def _run_mode1(
    pde: str,
    seed: int,
    *,
    data_path: Path | None,
) -> tuple[GoldenRunResult, dict[str, Any]]:
    if pde == "burgers":
        return run_burgers_mode1(seed, data_path=data_path)
    if pde == "chafee":
        return run_chafee_mode1(seed, data_path=data_path)
    raise ValueError(f"Unknown pde: {pde!r}")


__all__ = ["GoldenRunResult", "run_golden"]
