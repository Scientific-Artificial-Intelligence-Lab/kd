"""Serializable experiment result types for completed search runs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from kd.core.evaluator import EvaluationResult
from kd.search.recorder import VizRecorder, _make_json_safe, _sanitize_float

logger = logging.getLogger(__name__)

_JSON_INDENT_SPACES = 2


def _serialize_evaluation_result(result: EvaluationResult) -> dict[str, Any]:
    """Convert an ``EvaluationResult`` into a JSON-safe dictionary."""
    return {
        "mse": _sanitize_float(result.mse),
        "nmse": _sanitize_float(result.nmse),
        "r2": _sanitize_float(result.r2),
        "aic": _sanitize_float(result.aic) if result.aic is not None else None,
        "complexity": result.complexity,
        "coefficients": _make_json_safe(
            result.coefficients,
            key="final_eval.coefficients",
        ),
        "is_valid": result.is_valid,
        "error_message": result.error_message,
        "selected_indices": result.selected_indices,
        "residuals": _make_json_safe(
            result.residuals,
            key="final_eval.residuals",
        ),
        "terms": result.terms,
        "expression": result.expression,
        "lhs_name": result.lhs_name,
    }


def _deserialize_tensor(value: Any) -> Tensor | None:
    """Rebuild a CPU tensor from serialized data."""
    if value is None:
        return None
    return torch.as_tensor(value)


def _load_float(value: Any) -> float:
    """Coerce a JSON-loaded score field back to a ``float``.

    ``save`` sanitizes non-finite floats to ``None`` (RFC 8259 has no inf/NaN
    representation); on load we coerce that ``None`` back to ``NaN`` so the
    ``float``-typed fields stay type-stable — mirroring the
    ``RunResult.best_score`` F4 coercion. The mapping is lossy (inf and NaN
    both round-trip as NaN). Without this, downstream viz that does
    ``math.isfinite(r2)`` (parity / comparison plots) or ``f"{r2:.6f}"``
    (report) crashes with ``TypeError: must be real number, not NoneType``
    (AUDIT-02).
    """
    return float("nan") if value is None else value


def _deserialize_evaluation_result(data: dict[str, Any]) -> EvaluationResult:
    """Reconstruct an ``EvaluationResult`` from serialized data.

    ``mse`` / ``nmse`` / ``r2`` are typed ``float``, so a sanitized ``None``
    is coerced back to ``NaN`` (see :func:`_load_float`). ``aic`` is
    ``float | None`` — ``None`` is a legal "not computed" sentinel and is kept
    as-is (coercing it to NaN would fabricate a score; see
    ``test_save_with_nan_aic``).
    """
    return EvaluationResult(
        mse=_load_float(data["mse"]),
        nmse=_load_float(data["nmse"]),
        r2=_load_float(data["r2"]),
        aic=data["aic"],
        complexity=data["complexity"],
        coefficients=_deserialize_tensor(data["coefficients"]),
        is_valid=data["is_valid"],
        error_message=data["error_message"],
        selected_indices=data["selected_indices"],
        residuals=_deserialize_tensor(data["residuals"]),
        terms=data["terms"],
        expression=data["expression"],
        lhs_name=data.get("lhs_name"),
    )


@runtime_checkable
class ResultBuilder(Protocol):
    """Optional protocol for algorithms that build a final result directly.

    Implementations must not depend on transient evaluation caches. Restoring
    ``state`` and then calling ``build_final_result`` must return a complete
    valid result whenever a best expression is available and the algorithm has
    been prepared with the required evaluation context.
    """

    def build_final_result(self) -> EvaluationResult:
        """Return the semantically correct final evaluation result."""
        ...


@runtime_checkable
class ResultTargetProvider(Protocol):
    """Optional protocol for algorithms with a private final target domain."""

    def build_result_target(self) -> Tensor:
        """Return the target tensor used by ``build_final_result``."""
        ...


@dataclass
class RunResult:
    """Backward-compatible summary of a completed experiment."""

    best_expression: str
    best_score: float
    iterations: int
    early_stopped: bool


@dataclass
class ExperimentResult(RunResult):
    """Serializable value object for a completed experiment."""

    final_eval: EvaluationResult
    actual: Tensor
    predicted: Tensor
    dataset_name: str
    algorithm_name: str
    config: dict[str, Any]
    recorder: VizRecorder
    lhs_label: str = "u_t"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe (RFC 8259) representation of the result."""
        return {
            "best_expression": self.best_expression,
            "best_score": _sanitize_float(self.best_score),
            "iterations": self.iterations,
            "early_stopped": self.early_stopped,
            "final_eval": _serialize_evaluation_result(self.final_eval),
            "actual": _make_json_safe(self.actual, key="actual"),
            "predicted": _make_json_safe(self.predicted, key="predicted"),
            "dataset_name": self.dataset_name,
            "algorithm_name": self.algorithm_name,
            "config": _make_json_safe(self.config, key="config"),
            "recorder": self.recorder.to_dict(),
            "lhs_label": self.lhs_label,
        }

    def save(self, path: Path | str) -> None:
        """Persist the result to disk as RFC 8259 compliant JSON.

        Note: JSON serialization does not preserve tensor dtype.
        For exact dtype fidelity, use checkpoint mechanisms instead.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                self.to_dict(),
                handle,
                indent=_JSON_INDENT_SPACES,
                allow_nan=False,
            )
        logger.debug("Saved experiment result to %s", output_path)

    @classmethod
    def load(cls, path: Path | str) -> ExperimentResult:
        """Load a serialized result from disk.

        ``RunResult.best_score`` is typed ``float``; ``save`` sanitizes
        non-finite floats to ``None`` for RFC 8259 compliance, and ``load``
        coerces those Nones back to NaN to keep the type stable. The
        coercion is lossy (inf and NaN both round-trip as NaN) — record
        non-finite scores in a separate sidecar if exact reproducibility
        matters.
        """
        input_path = Path(path)
        with input_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)




        best_score_raw = data["best_score"]
        best_score = float("nan") if best_score_raw is None else best_score_raw

        result = cls(
            best_expression=data["best_expression"],
            best_score=best_score,
            iterations=data["iterations"],
            early_stopped=data["early_stopped"],
            final_eval=_deserialize_evaluation_result(data["final_eval"]),
            actual=torch.as_tensor(data["actual"]),
            predicted=torch.as_tensor(data["predicted"]),
            dataset_name=data["dataset_name"],
            algorithm_name=data["algorithm_name"],
            config=data["config"],
            recorder=VizRecorder.from_dict(data["recorder"]),
            lhs_label=data.get("lhs_label", "u_t"),
        )
        logger.debug("Loaded experiment result from %s", input_path)
        return result
