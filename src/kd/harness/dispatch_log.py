
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from kd.core.jsonsafe import JSON_INDENT_SPACES
from kd.harness._dispatch_schema import SHARD_ID_RE

DISPATCH_LOG_ARTIFACT_TAG: Final[str] = "kd-dispatch-log-v1"
DISPATCH_LOG_SCHEMA_VERSION: Final[int] = 1

_DISPATCH_LOG_V1_KEYS: Final[frozenset[str]] = frozenset(
    {
        "artifact",
        "dispatch_log_schema_version",
        "plan_hash",
        "workers",
        "lost_entries",
        "heavy_peak",
        "started_at",
        "finished_at",
    }
)
_DISPATCH_LOG_V1_WORKER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "shard_id",
        "pid",
        "exit_code",
        "kill_reason",
        "sigkill_used",
        "started_at",
        "finished_at",
        "wallclock_seconds",
        "timeout_seconds",
        "env",
        "heavy",
    }
)
_DISPATCH_LOG_V1_WORKER_ENV_KEYS: Final[frozenset[str]] = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "PYTORCH_NVML_BASED_CUDA_CHECK",
    }
)
_KILL_REASON_VOCABULARY: Final[frozenset[str]] = frozenset(
    {"timeout", "spawn_failed", "dispatcher_interrupted"}
)


class DispatchLogError(ValueError):
    pass


def _strict_keys(
    data: Any, *, object_name: str, required: frozenset[str]
) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise DispatchLogError(f"{object_name} must be a JSON object")
    actual = frozenset(data)
    unknown = actual - required
    if unknown:
        keys = ", ".join(repr(key) for key in sorted(unknown))
        raise DispatchLogError(f"Unknown {object_name} field(s): {keys}")
    missing = required - actual
    if missing:
        keys = ", ".join(repr(key) for key in sorted(missing))
        raise DispatchLogError(f"Missing required {object_name} field(s): {keys}")
    return data


def _shard_sort_key(shard_id: str) -> int:
    return int(shard_id.split("-")[1])





def _require_int_or_none(value: Any, *, field: str, shard_id: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise DispatchLogError(
            f"worker {shard_id!r} field {field!r} must be an int or null "
            f"(bool rejected); got {value!r}"
        )
    return int(value)


def _require_bool(value: Any, *, field: str, shard_id: str) -> bool:
    if not isinstance(value, bool):
        raise DispatchLogError(
            f"worker {shard_id!r} field {field!r} must be a bool; got {value!r}"
        )
    return value


def _require_str_or_none(value: Any, *, field: str, shard_id: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise DispatchLogError(
            f"worker {shard_id!r} field {field!r} must be a str or null; got {value!r}"
        )
    return value


def _require_nonneg_number_or_none(
    value: Any, *, field: str, shard_id: str
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DispatchLogError(
            f"worker {shard_id!r} field {field!r} must be a number or null "
            f"(bool rejected); got {value!r}"
        )
    if not math.isfinite(value) or value < 0:
        raise DispatchLogError(
            f"worker {shard_id!r} field {field!r} must be finite and non-negative; "
            f"got {value!r}"
        )
    return float(value)


@dataclass(frozen=True, kw_only=True)
class WorkerLogRow:

    shard_id: str
    pid: int | None
    exit_code: int | None
    kill_reason: str | None
    sigkill_used: bool
    started_at: str | None
    finished_at: str | None
    wallclock_seconds: float | None
    timeout_seconds: float | None
    env: dict[str, str]
    heavy: bool

    def __post_init__(self) -> None:
        if (
            self.kill_reason is not None
            and self.kill_reason not in _KILL_REASON_VOCABULARY
        ):
            raise DispatchLogError(
                f"kill_reason {self.kill_reason!r} not in "
                f"{sorted(_KILL_REASON_VOCABULARY)!r}"
            )
        if not isinstance(self.env, dict):
            raise DispatchLogError("worker env must be a mapping")
        if frozenset(self.env) != _DISPATCH_LOG_V1_WORKER_ENV_KEYS:
            unknown = sorted(frozenset(self.env) - _DISPATCH_LOG_V1_WORKER_ENV_KEYS)
            missing = sorted(_DISPATCH_LOG_V1_WORKER_ENV_KEYS - frozenset(self.env))
            raise DispatchLogError(
                f"worker env key face mismatch (unknown={unknown!r}, "
                f"missing={missing!r})"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "pid": self.pid,
            "exit_code": self.exit_code,
            "kill_reason": self.kill_reason,
            "sigkill_used": self.sigkill_used,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wallclock_seconds": self.wallclock_seconds,
            "timeout_seconds": self.timeout_seconds,
            "env": dict(self.env),
            "heavy": self.heavy,
        }


@dataclass(frozen=True, kw_only=True)
class DispatchLog:

    plan_hash: str
    workers: tuple[WorkerLogRow, ...]
    lost_entries: tuple[int, ...]
    heavy_peak: int
    started_at: str | None
    finished_at: str | None

    def to_dict(self) -> dict[str, Any]:
        ordered_workers = sorted(
            self.workers, key=lambda row: _shard_sort_key(row.shard_id)
        )
        return {
            "artifact": DISPATCH_LOG_ARTIFACT_TAG,
            "dispatch_log_schema_version": DISPATCH_LOG_SCHEMA_VERSION,
            "plan_hash": self.plan_hash,
            "workers": [row.to_dict() for row in ordered_workers],
            "lost_entries": sorted(self.lost_entries),
            "heavy_peak": self.heavy_peak,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def _decode_worker(payload: Any) -> WorkerLogRow:
    data = _strict_keys(
        payload, object_name="worker row", required=_DISPATCH_LOG_V1_WORKER_KEYS
    )
    env = data["env"]
    _strict_keys(
        env, object_name="worker env", required=_DISPATCH_LOG_V1_WORKER_ENV_KEYS
    )
    shard_id = data["shard_id"]




    if not isinstance(shard_id, str) or SHARD_ID_RE.fullmatch(shard_id) is None:
        raise DispatchLogError(
            f"worker shard_id must match 'shard-<digits>'; got {shard_id!r}"
        )
    return WorkerLogRow(
        shard_id=shard_id,
        pid=_require_int_or_none(data["pid"], field="pid", shard_id=shard_id),
        exit_code=_require_int_or_none(
            data["exit_code"], field="exit_code", shard_id=shard_id
        ),
        kill_reason=_require_str_or_none(
            data["kill_reason"], field="kill_reason", shard_id=shard_id
        ),
        sigkill_used=_require_bool(
            data["sigkill_used"], field="sigkill_used", shard_id=shard_id
        ),
        started_at=_require_str_or_none(
            data["started_at"], field="started_at", shard_id=shard_id
        ),
        finished_at=_require_str_or_none(
            data["finished_at"], field="finished_at", shard_id=shard_id
        ),
        wallclock_seconds=_require_nonneg_number_or_none(
            data["wallclock_seconds"], field="wallclock_seconds", shard_id=shard_id
        ),
        timeout_seconds=_require_nonneg_number_or_none(
            data["timeout_seconds"], field="timeout_seconds", shard_id=shard_id
        ),
        env=dict(env),
        heavy=_require_bool(data["heavy"], field="heavy", shard_id=shard_id),
    )


def _reject_constant(token: str) -> Any:
    raise DispatchLogError(
        f"dispatch log contains a non-standard JSON numeric token {token!r} "
        "(NaN / Infinity / -Infinity are not permitted)"
    )


def decode_dispatch_log(payload: Any) -> DispatchLog:
    _strict_keys(payload, object_name="dispatch log", required=_DISPATCH_LOG_V1_KEYS)
    if payload["artifact"] != DISPATCH_LOG_ARTIFACT_TAG:
        raise DispatchLogError(
            f"artifact tag mismatch: got {payload['artifact']!r}, "
            f"expected {DISPATCH_LOG_ARTIFACT_TAG!r}"
        )
    version = payload["dispatch_log_schema_version"]
    if type(version) is not int or version != DISPATCH_LOG_SCHEMA_VERSION:
        raise DispatchLogError(
            f"unsupported dispatch_log_schema_version: got {version!r}; "
            f"supported: {[DISPATCH_LOG_SCHEMA_VERSION]!r}"
        )
    plan_hash = payload["plan_hash"]
    if not isinstance(plan_hash, str) or not plan_hash:
        raise DispatchLogError(f"plan_hash must be a non-empty str; got {plan_hash!r}")
    workers_payload = payload["workers"]
    if not isinstance(workers_payload, list):
        raise DispatchLogError("dispatch log workers must be a JSON array")
    workers = tuple(_decode_worker(item) for item in workers_payload)
    seen_shard_ids: set[str] = set()
    for row in workers:
        if row.shard_id in seen_shard_ids:
            raise DispatchLogError(f"duplicate worker shard_id {row.shard_id!r}")
        seen_shard_ids.add(row.shard_id)

    lost_payload = payload["lost_entries"]
    if not isinstance(lost_payload, list):
        raise DispatchLogError("dispatch log lost_entries must be a JSON array")
    lost_entries: list[int] = []
    for value in lost_payload:
        if isinstance(value, bool) or not isinstance(value, int):
            raise DispatchLogError(
                f"lost_entries must be ints (bool rejected); got {value!r}"
            )
        lost_entries.append(value)

    heavy_peak = payload["heavy_peak"]
    if isinstance(heavy_peak, bool) or not isinstance(heavy_peak, int):
        raise DispatchLogError(f"heavy_peak must be an int; got {heavy_peak!r}")
    return DispatchLog(
        plan_hash=plan_hash,
        workers=workers,
        lost_entries=tuple(lost_entries),
        heavy_peak=heavy_peak,
        started_at=payload["started_at"],
        finished_at=payload["finished_at"],
    )


def write_dispatch_log(log: DispatchLog, path: Path) -> Path:
    target = Path(path)
    tmp_path = target.with_name(f"{target.name}.tmp")
    _validate_numeric_finiteness(log)
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(
            log.to_dict(),
            handle,
            indent=JSON_INDENT_SPACES,
            allow_nan=False,
            sort_keys=True,
        )
    os.replace(tmp_path, target)
    return target


def _validate_numeric_finiteness(log: DispatchLog) -> None:
    for row in log.workers:
        for field in (row.wallclock_seconds, row.timeout_seconds):
            if isinstance(field, float) and not math.isfinite(field):
                raise DispatchLogError(
                    f"worker {row.shard_id!r} carries a non-finite float: {field!r}"
                )


def read_dispatch_log(path: Path) -> DispatchLog:
    text = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise DispatchLogError(f"dispatch log is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DispatchLogError("dispatch log must be a JSON object")
    return decode_dispatch_log(payload)


__all__ = [
    "DISPATCH_LOG_ARTIFACT_TAG",
    "DISPATCH_LOG_SCHEMA_VERSION",
    "_DISPATCH_LOG_V1_KEYS",
    "_DISPATCH_LOG_V1_WORKER_ENV_KEYS",
    "_DISPATCH_LOG_V1_WORKER_KEYS",
    "_KILL_REASON_VOCABULARY",
    "DispatchLog",
    "DispatchLogError",
    "WorkerLogRow",
    "decode_dispatch_log",
    "read_dispatch_log",
    "write_dispatch_log",
]
