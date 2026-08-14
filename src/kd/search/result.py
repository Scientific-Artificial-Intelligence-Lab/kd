"""Serializable experiment result types for completed search runs."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import torch
from torch import Tensor

from kd.core.equation import (
    DEFAULT_LHS_LABEL,
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
from kd.core.jsonsafe import JSON_INDENT_SPACES
from kd.search.recorder import VizRecorder, _make_json_safe, _sanitize_float
from kd.search.records import RunRecord, validate_invalid_reason

if TYPE_CHECKING:
    from kd.search.sketch_outcome import SketchOutcome

logger = logging.getLogger(__name__)

RESULT_SCHEMA_VERSION: Final[int] = 1






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


def _load_condition_number(data: dict[str, Any]) -> float | None:
    """Restore ``condition_number`` from its two-key JSON encoding.

    The payload splits the diagnostic in two so that a measured non-finite
    value stays distinguishable from an unmeasured one after ``sanitize_float``
    turns both into ``null``::

        (value, computed)
        (float, True) -> measured, finite -> that float
        (None, True) -> measured, non-finite -> ``float("inf")``
        (None, False) -> not measured -> ``None``

    ``inf`` is the only non-finite the producing solvers emit (an all-zero
    matrix is pre-guarded to ``inf``, and a LAPACK failure returns ``inf``), so
    the restoration is exact rather than a best guess. Both keys are read with
    ``.get``: results written before this field existed carry neither, and must
    keep loading as "not measured".
    """
    if not data.get("condition_number_computed", False):
        return None
    value = data.get("condition_number")
    return float("inf") if value is None else float(value)


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
        invalid_reason=data.get("invalid_reason"),
        selected_indices=data["selected_indices"],
        residuals=_deserialize_tensor(data["residuals"]),
        terms=data["terms"],
        expression=data["expression"],
        lhs_name=data.get("lhs_name"),
        condition_number=_load_condition_number(data),
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
        active_indices=final_eval.selected_indices,
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
    reason: str = "unclassified",
) -> EvaluationResult:
    """Build a plugin-level invalid final evaluation result.

    ``residuals=None`` follows the EQGPT/L5 adjudication: ``None`` means
    "no prediction" and the runner maps it to zero residuals where display
    surfaces need tensors. ``score`` lands in ``.score``; callers must state
    the score semantics explicitly.

    None = not computed (score-semantics batch adjudication 2, 2026-07-07):
    an invalid result computed nothing, so terms/selected_indices are None,
    matching residuals=None (EQGPT/L5).
    ``reason`` is validated against the record-schema vocabulary at the
    producer boundary and defaults to the explicit ``unclassified`` fallback.
    """
    declared_reason = validate_invalid_reason(reason)
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
        invalid_reason=declared_reason,
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

    - ``dataset_cache_fingerprint``: content+meta fingerprint of the dataset.
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
    - ``resumed``: whether this invocation consumed a runner checkpoint restore.
    - ``resume_source``: resume provenance (M4 lineage) — the exact
      ``_record_schema.LINEAGE_FIELDS`` key set (``resume_from`` path plus the
      source checkpoint's run id / config hash / final status / iteration, each
      nullable when the source predates the recording layout); ``None`` for a
      fresh run. The path is an operational pointer, not a content identity —
      it never enters ``RunSpec`` or any sealed hash.

    The round-trip contract is ``RunManifest.from_dict(m.to_dict()) == m``.
    """

    dataset_cache_fingerprint: str
    kd_version: str
    seed: int | None
    terms: list[str] | None = None
    artifacts: dict[str, Any] | None = None
    resumed: bool = False
    resume_source: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of this manifest.

        The scalar fields (str / int / None / list of str) are JSON-native by
        construction — unlike ``ExperimentResult`` which holds tensors and
        non-finite floats. ``artifacts`` is the one open-typed field
        (``dict[str, Any]``) and is emitted verbatim. Its producer owns JSON
        safety; RunSpec construction subsequently enforces the strict
        JSON-native identity whitelist and rejects unsupported values.
        """
        return {
            "dataset_cache_fingerprint": self.dataset_cache_fingerprint,
            "kd_version": self.kd_version,
            "seed": self.seed,
            "terms": self.terms,
            "artifacts": self.artifacts,
            "resumed": self.resumed,
            "resume_source": self.resume_source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        """Reconstruct a ``RunManifest`` from a JSON-safe dictionary.

        ``terms`` / ``artifacts`` are read with ``.get`` so a manifest dict
        predating those fields (or hand-built) decodes with them as ``None`` —
        mirroring ``ExperimentResult.load``'s ``data.get`` backward-compat.
        """
        return cls(
            dataset_cache_fingerprint=data["dataset_cache_fingerprint"],
            kd_version=data["kd_version"],
            seed=data["seed"],
            terms=data.get("terms"),
            artifacts=data.get("artifacts"),
            resumed=data.get("resumed", False),
            resume_source=data.get("resume_source"),
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
    per-algorithm lookup tables. ``score_kind`` is a STABLE metric identifier
    consumed by comparison logic (it decides score commensurability), not a
    display label; display formatting may derive a label from it but must not
    replace it. The defaults mirror the historical fallback semantics
    ("Score" / "min") so existing direct construction sites keep working;
    correctness for the built-in algorithms is enforced by the facade contract
    tests, not by these defaults. Typed plain ``str`` (not ``Literal``) because
    the values round-trip through JSON.
    """

    final_eval: EvaluationResult
    actual: Tensor
    predicted: Tensor
    dataset_name: str
    algorithm_name: str
    config: dict[str, Any]
    recorder: VizRecorder
    lhs_label: str = DEFAULT_LHS_LABEL
    equation: Equation | None = None
    manifest: RunManifest | None = None
    run_record: RunRecord | None = None
    score_kind: str = DEFAULT_SCORE_KIND
    score_direction: str = DEFAULT_SCORE_DIRECTION










    finalize_failures: tuple[str, ...] = ()


    sketch_outcome: SketchOutcome | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe (RFC 8259) representation of the result."""
        return {
            "result_schema_version": RESULT_SCHEMA_VERSION,
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


            "run_record": (
                self.run_record.to_dict() if self.run_record is not None else None
            ),
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


        tmp_path = output_path.with_name(f"{output_path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(
                self.to_dict(),
                handle,
                indent=JSON_INDENT_SPACES,
                allow_nan=False,
            )
        os.replace(tmp_path, output_path)
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

        result_schema_version = data.get("result_schema_version")
        if "result_schema_version" in data and (
            type(result_schema_version) is not int
            or not (1 <= result_schema_version <= RESULT_SCHEMA_VERSION)
        ):
            raise ValueError(
                "Unsupported result_schema_version: "
                f"got {result_schema_version!r}; "
                f"supported range is 1..{RESULT_SCHEMA_VERSION}"
            )




        best_score_raw = data["best_score"]
        best_score = float("nan") if best_score_raw is None else best_score_raw



        manifest_data = data.get("manifest")
        manifest = (
            RunManifest.from_dict(manifest_data) if manifest_data is not None else None
        )


        record_data = data.get("run_record")
        run_record = (
            RunRecord.from_dict(record_data) if record_data is not None else None
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
            lhs_label=data.get("lhs_label", DEFAULT_LHS_LABEL),
            equation=equation,
            manifest=manifest,
            run_record=run_record,
            score_kind=score_kind,
            score_direction=score_direction,
        )
        logger.debug("Loaded experiment result from %s", input_path)
        return result
