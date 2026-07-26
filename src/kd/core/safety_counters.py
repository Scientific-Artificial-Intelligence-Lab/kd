
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import Tensor

_ENV_FLAG = "KD_SAFETY_COUNTERS"
_ENV_ON_VALUE = "1"
_ENV_FILE = "KD_SAFETY_COUNTERS_FILE"




_enabled: bool = os.environ.get(_ENV_FLAG, "0") == _ENV_ON_VALUE







_div_calls = 0
_div_fired = 0
_div_fired_elems = 0
_div_total_elems = 0
_exp_calls = 0
_exp_fired = 0
_exp_fired_elems = 0
_exp_total_elems = 0
_log_calls = 0
_log_fired = 0
_log_fired_elems = 0
_log_total_elems = 0


def counters_enabled() -> bool:
    return _enabled


def enable_counters() -> None:
    global _enabled
    _enabled = True


def disable_counters() -> None:
    global _enabled
    _enabled = False


@dataclass(frozen=True)
class SafetyCounterSnapshot:

    div_calls: int
    div_guard_fired: int
    exp_calls: int
    exp_guard_fired: int
    log_calls: int
    log_guard_fired: int
    div_fired_elems: int
    div_total_elems: int
    exp_fired_elems: int
    exp_total_elems: int
    log_fired_elems: int
    log_total_elems: int


def snapshot() -> SafetyCounterSnapshot:
    return SafetyCounterSnapshot(
        div_calls=_div_calls,
        div_guard_fired=_div_fired,
        exp_calls=_exp_calls,
        exp_guard_fired=_exp_fired,
        log_calls=_log_calls,
        log_guard_fired=_log_fired,
        div_fired_elems=_div_fired_elems,
        div_total_elems=_div_total_elems,
        exp_fired_elems=_exp_fired_elems,
        exp_total_elems=_exp_total_elems,
        log_fired_elems=_log_fired_elems,
        log_total_elems=_log_total_elems,
    )


def reset() -> None:
    global _div_calls, _div_fired, _exp_calls, _exp_fired, _log_calls, _log_fired
    global _div_fired_elems, _div_total_elems, _exp_fired_elems, _exp_total_elems
    global _log_fired_elems, _log_total_elems
    _div_calls = 0
    _div_fired = 0
    _exp_calls = 0
    _exp_fired = 0
    _log_calls = 0
    _log_fired = 0
    _div_fired_elems = 0
    _div_total_elems = 0
    _exp_fired_elems = 0
    _exp_total_elems = 0
    _log_fired_elems = 0
    _log_total_elems = 0


def summary_file_path() -> Path | None:
    raw = os.environ.get(_ENV_FILE)
    return Path(raw) if raw else None


def emit_run_summary(
    label: str, logger: logging.Logger
) -> SafetyCounterSnapshot | None:
    if not _enabled:
        return None
    snap = snapshot()
    logger.info(
        "[SAFETY-COUNTERS] div_calls=%d div_guard_fired=%d "
        "exp_calls=%d exp_guard_fired=%d log_calls=%d log_guard_fired=%d "
        "div_fired_elems=%d div_total_elems=%d "
        "exp_fired_elems=%d exp_total_elems=%d "
        "log_fired_elems=%d log_total_elems=%d "
        "label=%s",
        snap.div_calls,
        snap.div_guard_fired,
        snap.exp_calls,
        snap.exp_guard_fired,
        snap.log_calls,
        snap.log_guard_fired,
        snap.div_fired_elems,
        snap.div_total_elems,
        snap.exp_fired_elems,
        snap.exp_total_elems,
        snap.log_fired_elems,
        snap.log_total_elems,
        label,
    )
    path = summary_file_path()
    if path is not None:
        record: dict[str, object] = {
            "ts": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "label": label,
            "pid": os.getpid(),
            **asdict(snap),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:

            logger.warning(
                "[SAFETY-COUNTERS] file sink write failed (%s): %s", path, exc
            )
    reset()
    return snap


def record_div(b: Tensor, eps: float) -> None:
    global _div_calls, _div_fired, _div_fired_elems, _div_total_elems
    if not _enabled:
        return
    _div_calls += 1
    try:
        with torch.no_grad():
            mask = b.abs() <= eps
            total = mask.numel()
            fired_elems = int(mask.sum().item()) if total else 0
        _div_total_elems += total
        _div_fired_elems += fired_elems
        if fired_elems:
            _div_fired += 1
    except RuntimeError:


        return


def record_exp(x: Tensor, min_val: float, max_val: float) -> None:
    global _exp_calls, _exp_fired, _exp_fired_elems, _exp_total_elems
    if not _enabled:
        return
    _exp_calls += 1
    try:
        with torch.no_grad():
            mask = (x < min_val) | (x > max_val)
            total = mask.numel()
            fired_elems = int(mask.sum().item()) if total else 0
        _exp_total_elems += total
        _exp_fired_elems += fired_elems
        if fired_elems:
            _exp_fired += 1
    except RuntimeError:
        return


def record_log(x: Tensor, eps: float) -> None:
    global _log_calls, _log_fired, _log_fired_elems, _log_total_elems
    if not _enabled:
        return
    _log_calls += 1
    try:
        with torch.no_grad():
            mask = x < eps
            total = mask.numel()
            fired_elems = int(mask.sum().item()) if total else 0
        _log_total_elems += total
        _log_fired_elems += fired_elems
        if fired_elems:
            _log_fired += 1
    except RuntimeError:
        return


__all__ = [
    "SafetyCounterSnapshot",
    "counters_enabled",
    "enable_counters",
    "disable_counters",
    "snapshot",
    "reset",
    "summary_file_path",
    "emit_run_summary",
    "record_div",
    "record_exp",
    "record_log",
    "_ENV_FLAG",
    "_ENV_FILE",
]
