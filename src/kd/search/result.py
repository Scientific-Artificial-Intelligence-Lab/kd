"""Serializable experiment result types for completed search runs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from kd.core.equation import (
    Equation,
    LhsSpec,
    build_equation,
)
from kd.core.equation import (
    from_dict as equation_from_dict,
)
from kd.core.equation import (
    to_dict as equation_to_dict,
)
from kd.core.evaluator import EvaluationResult, Evaluator
from kd.core.expr.naming import parse_derivative_name
from kd.search.recorder import VizRecorder, _make_json_safe, _sanitize_float

logger = logging.getLogger(__name__)

_JSON_INDENT_SPACES = 2






DEFAULT_SCORE_KIND = "Score"
DEFAULT_SCORE_DIRECTION = "min"


def _serialize_evaluation_result(result: EvaluationResult) -> dict[str, Any]:
    """Convert an ``EvaluationResult`` into a JSON-safe dictionary.

    Thin delegate to ``EvaluationResult.to_dict``: the dataclass method is
    now the single serialization source. Kept as a free function so existing
    callers (result-build / save path) keep importing it unchanged.
    """
    return result.to_dict()


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
    ``RunResult.best_score`` coercion. The mapping is lossy (inf and NaN
    both round-trip as NaN). Without this, downstream viz that does
    ``math.isfinite(r2)`` (parity / comparison plots) or ``f"{r2:.6f}"``
    (report) crashes with ``TypeError: must be real number, not NoneType``.
    """
    return float("nan") if value is None else value


def _deserialize_evaluation_result(data: dict[str, Any]) -> EvaluationResult:
    """Reconstruct an ``EvaluationResult`` from serialized data.

    ``mse`` / ``nmse`` / ``r2`` are typed ``float``, so a sanitized ``None``
    is coerced back to ``NaN`` (see :func:`_load_float`). ``score`` is
    ``float | None`` — ``None`` is a legal "not computed" sentinel and is kept
    as-is (coercing it to NaN would fabricate a score; see
    ``test_save_with_nan_score``). Reads the ``"score"`` key when present;
    falls back to the legacy ``"aic"`` key otherwise so files
    written before the rename stay readable. Neither key present -> the
    natural ``KeyError`` propagates.
    """
    return EvaluationResult(
        mse=_load_float(data["mse"]),
        nmse=_load_float(data["nmse"]),
        r2=_load_float(data["r2"]),
        score=data["score"] if "score" in data else data["aic"],
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


def _derive_lhs_spec_from_name(name: object) -> LhsSpec | None:
    if not isinstance(name, str) or not name:
        return None
    parsed = parse_derivative_name(name)
    if parsed is None:
        return None
    field, axis, order = parsed
    return LhsSpec(field=field, axis=axis, order=order)


def _derive_equation_from_final_eval(
    final_eval: EvaluationResult,
    lhs_label: object,
) -> Equation | None:
    """Rebuild a default evolution equation from legacy final-eval fields."""
    lhs_spec = (
        _derive_lhs_spec_from_name(final_eval.lhs_name)
        if isinstance(final_eval.lhs_name, str) and final_eval.lhs_name
        else _derive_lhs_spec_from_name(lhs_label)
    )
    return build_equation(
        final_eval.terms,
        final_eval.coefficients,
        lhs_spec,
        is_valid=final_eval.is_valid,
    )


def _infer_legacy_score_meta(data: dict[str, Any]) -> tuple[str, str]:
    """Infer (score_kind, score_direction) for a legacy serialized payload.

    Legacy result files predate the ``score_kind`` / ``score_direction``
    fields. The algorithm id is looked up in the FROZEN fallback tables in
    ``kd.viz._labels`` — frozen because new algorithms declare their metadata
    on the plugin class (``ScoreContract``) and their files carry it inline.

    The id comes from ``config["algorithm"]`` (lowercase, e.g. ``"sga"``, set
    by each plugin's ``config`` property) — NOT from ``algorithm_name``, which
    holds the plugin *class name* (e.g. ``"SGAPlugin"``, written by
    ``runner._build_experiment_result`` via ``type(...).__name__``) and would
    miss the tables, downgrading every real legacy file to "Score"/"min".

    Imported lazily: ``kd.viz`` pulls matplotlib at package init, which the
    search layer must not load eagerly (no circular import: viz modules import
    the search layer — e.g. ``plots/comparison.py`` reads
    ``DEFAULT_SCORE_KIND`` at runtime — while search→viz stays lazy
    function-local, so the import graph is acyclic; precedent:
    ``runner._build_manifest``).
    """
    from kd.viz._labels import score_direction as _legacy_score_direction
    from kd.viz._labels import score_label as _legacy_score_label

    config = data.get("config")
    algorithm = ""
    if isinstance(config, dict):
        raw = config.get("algorithm", "")
        if isinstance(raw, str):
            algorithm = raw
    return _legacy_score_label(algorithm), _legacy_score_direction(algorithm)


def default_final_result(expression: str, evaluator: Evaluator) -> EvaluationResult:
    """Simple algorithms delegate their final evaluation to the platform.

    One-line ``SearchAlgorithm.build_final_result`` body for algorithms that
    score through the platform evaluator: the returned
    result's ``residuals`` live in the evaluator's LHS-target domain, so the
    matching ``build_result_target`` is the evaluator's ``lhs_target``
    (detach + clone it — see the protocol docstring's domain-consistency
    contract).
    """
    return evaluator.evaluate_expression(expression)


def invalid_evaluation_result(
    error_message: str,
    *,
    score: float | None,
    expression: str = "",
    terms: list[str] | None = None,
) -> EvaluationResult:
    """Build a plugin-level invalid final evaluation result.

    ``residuals=None`` follows the EQGPT/L5 adjudication: ``None`` means
    "no prediction" and the runner maps it to zero residuals where display
    surfaces need tensors. ``score`` lands in ``.score``; callers must state
    the score semantics explicitly.

    None = not computed (score-semantics batch adjudication 2, 2026-07-07):
    an invalid result computed nothing, so terms/selected_indices are None,
    matching residuals=None (EQGPT/L5).
    """
    copied_terms = list(terms) if terms else None
    return EvaluationResult(
        mse=float("inf"),
        nmse=float("inf"),
        r2=-float("inf"),
        score=score,
        complexity=len(terms) if terms else 0,
        coefficients=None,
        is_valid=False,
        error_message=error_message,
        selected_indices=None,
        residuals=None,
        terms=copied_terms,
        expression=expression,
    )


@dataclass
class RunManifest:
    """Minimal reproducibility facts captured for a completed search run.

    Records exactly the facts needed to *reproduce* a run, with no
    non-deterministic content (no timestamp, no object id / memory address):

    - ``dataset_fingerprint``: content+meta fingerprint of the dataset.
    - ``kd_version``: installed kd package version (e.g. ``"0.1.0"``).
    - ``seed``: RNG seed for the run; ``None`` if the algorithm exposes none.
      How faithfully a seed replays a run is algorithm-dependent -- e.g.
      PySR under its default parallelism is only weakly deterministic (see
      ``PySRConfig.seed``), so for such algorithms this field records
      intent, not a bit-identical replay guarantee.
    - ``terms``: term library (only term-library algorithms fill it); else
      ``None``.
    - ``artifacts``: reserved for artifact-bearing algorithms (weights / data)
      to record ``{sha256, size, ...}`` per artifact; else ``None``.

    The round-trip contract is ``RunManifest.from_dict(m.to_dict()) == m``.
    """

    dataset_fingerprint: str
    kd_version: str
    seed: int | None
    terms: list[str] | None = None
    artifacts: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of this manifest.

        The scalar fields (str / int / None / list of str) are JSON-native by
        construction — unlike ``ExperimentResult`` which holds tensors and
        non-finite floats. ``artifacts`` is the one open-typed field
        (``dict[str, Any]``) and is emitted verbatim: today it is always
        ``None`` (reserved for artifact-bearing algorithms, e.g. EqGPT), and
        whichever code starts filling it owns its JSON safety — keep values
        JSON-native or route them through ``_make_json_safe`` as
        ``ExperimentResult`` does.
        """
        return {
            "dataset_fingerprint": self.dataset_fingerprint,
            "kd_version": self.kd_version,
            "seed": self.seed,
            "terms": self.terms,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        """Reconstruct a ``RunManifest`` from a JSON-safe dictionary.

        ``terms`` / ``artifacts`` are read with ``.get`` so a manifest dict
        predating those fields (or hand-built) decodes with them as ``None`` —
        mirroring ``ExperimentResult.load``'s ``data.get`` backward-compat.
        """
        return cls(
            dataset_fingerprint=data["dataset_fingerprint"],
            kd_version=data["kd_version"],
            seed=data["seed"],
            terms=data.get("terms"),
            artifacts=data.get("artifacts"),
        )


@dataclass
class RunResult:
    """Backward-compatible summary of a completed experiment."""

    best_expression: str
    best_score: float
    iterations: int
    early_stopped: bool


@dataclass
class ExperimentResult(RunResult):
    """Serializable value object for a completed experiment.

    ``score_kind`` / ``score_direction`` carry the algorithm's
    ``ScoreContract`` declaration (what quantity ``best_score`` is and which
    direction is better) so viz consumers read the result instead of keeping
    per-algorithm lookup tables. The defaults mirror the historical fallback
    semantics ("Score" / "min") so existing direct construction sites keep
    working; correctness for the built-in algorithms is enforced by the
    facade contract tests, not by these defaults. Typed plain ``str`` (not
    ``Literal``) because the values round-trip through JSON.
    """

    final_eval: EvaluationResult
    actual: Tensor
    predicted: Tensor
    dataset_name: str
    algorithm_name: str
    config: dict[str, Any]
    recorder: VizRecorder
    lhs_label: str = "u_t"
    equation: Equation | None = None
    manifest: RunManifest | None = None
    score_kind: str = DEFAULT_SCORE_KIND
    score_direction: str = DEFAULT_SCORE_DIRECTION

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
            "equation": (
                equation_to_dict(self.equation) if self.equation is not None else None
            ),


            "manifest": self.manifest.to_dict() if self.manifest is not None else None,
            "score_kind": self.score_kind,
            "score_direction": self.score_direction,
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



        manifest_data = data.get("manifest")
        manifest = (
            RunManifest.from_dict(manifest_data) if manifest_data is not None else None
        )

        final_eval = _deserialize_evaluation_result(data["final_eval"])
        equation: Equation | None
        if "equation" in data:
            eq_data = data["equation"]
            equation = equation_from_dict(eq_data) if eq_data is not None else None
        else:
            equation = _derive_equation_from_final_eval(
                final_eval,
                data.get("lhs_label"),
            )






        score_kind = data.get("score_kind")
        score_direction = data.get("score_direction")
        if score_kind is None or score_direction is None:
            score_kind, score_direction = _infer_legacy_score_meta(data)

        result = cls(
            best_expression=data["best_expression"],
            best_score=best_score,
            iterations=data["iterations"],
            early_stopped=data["early_stopped"],
            final_eval=final_eval,
            actual=torch.as_tensor(data["actual"]),
            predicted=torch.as_tensor(data["predicted"]),
            dataset_name=data["dataset_name"],
            algorithm_name=data["algorithm_name"],
            config=data["config"],
            recorder=VizRecorder.from_dict(data["recorder"]),
            lhs_label=data.get("lhs_label", "u_t"),
            equation=equation,
            manifest=manifest,
            score_kind=score_kind,
            score_direction=score_direction,
        )
        logger.debug("Loaded experiment result from %s", input_path)
        return result
