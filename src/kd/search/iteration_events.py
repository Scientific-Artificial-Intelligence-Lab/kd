
from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TextIO

__all__ = [
    "ITEREVENT_SCHEME",
    "ITEREVENT_SCHEMA_VERSION",
    "IterationEvent",
    "IterationEventEmitter",
    "IterationEventSinkError",
]





ITEREVENT_SCHEME: Final[str] = "kd-iterevent-v1"
ITEREVENT_SCHEMA_VERSION: Final[int] = 1

_MIN_EVERY_N: Final[int] = 1








_ITEREVENT_V1_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "scheme",
    "iteration",
    "n_candidates",
    "n_invalid",
    "best_score",
    "best_expression",
    "elapsed_seconds",
    "diagnostics",
)
_ITEREVENT_V1_FIELD_SET: Final[frozenset[str]] = frozenset(_ITEREVENT_V1_FIELDS)


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


class IterationEventSinkError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class IterationEvent:

    schema_version: int
    scheme: str
    iteration: int
    n_candidates: int
    n_invalid: int
    best_score: float | None
    best_expression: str | None
    elapsed_seconds: float
    diagnostics: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in _ITEREVENT_V1_FIELDS}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IterationEvent:
        if not isinstance(data, dict):
            raise ValueError(f"IterationEvent payload must be a dict, got {type(data)}")
        keys = set(data)
        unknown = keys - _ITEREVENT_V1_FIELD_SET
        if unknown:
            raise ValueError(f"IterationEvent: unknown keys {sorted(unknown)!r}")
        missing = _ITEREVENT_V1_FIELD_SET - keys
        if missing:
            raise ValueError(f"IterationEvent: missing keys {sorted(missing)!r}")
        for int_field in ("iteration", "n_candidates", "n_invalid", "schema_version"):
            _validate_int(data[int_field], field=int_field)
        _validate_str(data["scheme"], field="scheme")
        _validate_optional_str(data["best_expression"], field="best_expression")
        _validate_optional_dict(data["diagnostics"], field="diagnostics")
        _validate_optional_finite(data["best_score"], field="best_score")
        _validate_required_finite(data["elapsed_seconds"], field="elapsed_seconds")
        if data["scheme"] != ITEREVENT_SCHEME:
            raise ValueError(
                f"IterationEvent: unsupported scheme {data['scheme']!r}; "
                f"expected {ITEREVENT_SCHEME!r}"
            )
        if data["schema_version"] != ITEREVENT_SCHEMA_VERSION:
            raise ValueError(
                f"IterationEvent: unsupported schema_version "
                f"{data['schema_version']!r}; expected {ITEREVENT_SCHEMA_VERSION!r}"
            )
        return cls(**{field: data[field] for field in _ITEREVENT_V1_FIELDS})


def _validate_int(value: object, *, field: str) -> None:
    if type(value) is not int:
        raise ValueError(f"IterationEvent.{field} must be an int, got {value!r}")


def _validate_str(value: object, *, field: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"IterationEvent.{field} must be a str, got {value!r}")


def _validate_optional_str(value: object, *, field: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(
            f"IterationEvent.{field} must be a str or None, got {value!r}"
        )


def _validate_optional_dict(value: object, *, field: str) -> None:
    if value is not None and not isinstance(value, dict):
        raise ValueError(
            f"IterationEvent.{field} must be a dict or None, got {value!r}"
        )


def _validate_optional_finite(value: object, *, field: str) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(
            f"IterationEvent.{field} must be finite or None, got {value!r}"
        )


def _validate_required_finite(value: object, *, field: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(
            f"IterationEvent.{field} must be a finite number, got {value!r}"
        )


class IterationEventEmitter:

    def __init__(
        self,
        on_event: Callable[[IterationEvent], None] | None = None,
        jsonl_path: Path | None = None,
        every_n_iterations: int = 1,
    ) -> None:
        if on_event is None and jsonl_path is None:
            raise ValueError(
                "IterationEventEmitter requires at least one of on_event / "
                "jsonl_path (a consumer-less emitter is a configuration error)"
            )
        if every_n_iterations < _MIN_EVERY_N:
            raise ValueError(
                f"every_n_iterations must be >= 1, got {every_n_iterations}"
            )
        self._on_event = on_event
        self._jsonl_path = jsonl_path
        self._every_n = every_n_iterations
        self._t0: float = time.perf_counter()
        self._sink: TextIO | None = None

    @property
    def should_stop(self) -> bool:
        return False

    def on_experiment_start(self, algorithm: Any) -> None:
        self._close_sink()
        self._t0 = time.perf_counter()

    def on_iteration_start(self, iteration: int, algorithm: Any) -> None:
        pass

    def on_iteration_end(
        self,
        iteration: int,
        algorithm: Any,
        candidates: list[str],
        results: list[Any],
    ) -> None:
        if iteration % self._every_n != 0:
            return
        event = self._build_event(iteration, algorithm, candidates, results)
        if self._jsonl_path is not None:
            self._write_sink(event)
        if self._on_event is not None:
            self._on_event(event)

    def on_experiment_end(self, algorithm: Any) -> None:
        try:
            self._close_sink()
        except OSError as exc:
            raise IterationEventSinkError(
                f"Failed to close iteration event sink {self._jsonl_path!r}: {exc}"
            ) from exc

    def _build_event(
        self,
        iteration: int,
        algorithm: Any,
        candidates: list[str],
        results: list[Any],
    ) -> IterationEvent:
        raw_expression: str = algorithm.best_expression
        if raw_expression:
            best_expression: str | None = raw_expression
            best_score = _finite_or_none(float(algorithm.best_score))
        else:
            best_expression = None
            best_score = None
        return IterationEvent(
            schema_version=ITEREVENT_SCHEMA_VERSION,
            scheme=ITEREVENT_SCHEME,
            iteration=iteration,
            n_candidates=len(candidates),
            n_invalid=sum(1 for result in results if not result.is_valid),
            best_score=best_score,
            best_expression=best_expression,
            elapsed_seconds=time.perf_counter() - self._t0,
            diagnostics=None,
        )

    def _write_sink(self, event: IterationEvent) -> None:
        if self._sink is None:
            self._open_sink()
        assert self._sink is not None
        line = json.dumps(
            event.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        try:
            self._sink.write(line + "\n")
            self._sink.flush()
        except OSError as exc:
            raise IterationEventSinkError(
                f"Failed to write iteration event to {self._jsonl_path!r}: {exc}"
            ) from exc

    def _open_sink(self) -> None:
        assert self._jsonl_path is not None
        try:
            self._sink = self._jsonl_path.open("x", encoding="utf-8")
        except OSError as exc:
            raise IterationEventSinkError(
                f"Failed to open iteration event sink {self._jsonl_path!r}: {exc}"
            ) from exc

    def _close_sink(self) -> None:
        if self._sink is not None:
            sink = self._sink
            self._sink = None
            sink.close()
