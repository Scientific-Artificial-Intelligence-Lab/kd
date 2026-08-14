
from __future__ import annotations

import logging
from datetime import datetime, timezone

from torch import Tensor

from kd.core.equation import Equation, Scalar
from kd.core.equation import to_dict as equation_to_dict
from kd.core.equation.library import TermLibrarySpec
from kd.core.equation.projection import active_law
from kd.core.evaluator import EvaluationResult
from kd.core.jsonsafe import sanitize_float
from kd.search.records import (
    EVIDENCE_HASH_SCHEME,
    RECORD_HASH_SCHEME,
    RUN_RECORD_SCHEMA_VERSION,
    EvidenceRecord,
    RecordSchemaError,
    ResidualStats,
    RunCost,
    RunRecord,
    seal_record_hash,
)
from kd.search.run_spec import RUN_SPEC_HASH_SCHEME, RunSpec

logger = logging.getLogger(__name__)


def cpu_seconds_now() -> float | None:
    try:
        import resource

        self_usage = resource.getrusage(resource.RUSAGE_SELF)
        child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    except (ImportError, AttributeError, OSError):
        return None
    return (
        self_usage.ru_utime
        + self_usage.ru_stime
        + child_usage.ru_utime
        + child_usage.ru_stime
    )


def residual_stats_from(residuals: Tensor | None) -> ResidualStats | None:
    if residuals is None:
        return None
    values = residuals.detach().cpu()
    if values.numel() == 0:




        return ResidualStats(mean=None, std=None, max_abs=None, n=0)
    return ResidualStats(
        mean=sanitize_float(float(values.mean().item())),
        std=sanitize_float(float(values.std().item())),
        max_abs=sanitize_float(float(values.abs().max().item())),
        n=values.numel(),
    )


def active_support_and_coefficients(
    final_eval: EvaluationResult,
) -> tuple[list[str] | None, list[float | None] | None]:
    terms = final_eval.terms
    if terms is None:
        return None, None
    coeff_values: list[float | None] | None = None
    if final_eval.coefficients is not None:
        coeff_values = [
            sanitize_float(float(value))
            for value in final_eval.coefficients.detach().cpu().flatten().tolist()
        ]
        if len(coeff_values) != len(terms):
            logger.warning(
                "Dropping evidence coefficients: %d values do not align with "
                "%d terms",
                len(coeff_values),
                len(terms),
            )
            coeff_values = None
    selected = final_eval.selected_indices
    if selected is None:
        indices = list(range(len(terms)))
    else:
        indices = list(selected)
        if any(not 0 <= index < len(terms) for index in indices):
            logger.warning(
                "Dropping evidence support/coefficients: selected_indices "
                "contain out-of-range entries for %d terms",
                len(terms),
            )
            return None, None
    support = [terms[i] for i in indices]
    coefficients: list[float | None] | None = None
    if coeff_values is not None:
        coefficients = [coeff_values[i] for i in indices]
    return support, coefficients


def active_support_and_coefficients_from_equation(
    equation: Equation | None,
) -> tuple[list[str] | None, list[float | None] | None]:
    if equation is None:
        return None, None
    projected = active_law(equation)
    support = [term_ir for term_ir, _coefficient in projected.terms]
    coefficients = [
        sanitize_float(_scalar_coefficient(term_ir, coefficient))
        for term_ir, coefficient in projected.terms
    ]
    return support, coefficients


def _scalar_coefficient(term_ir: str, coefficient: object) -> float:
    if not isinstance(coefficient, Scalar):
        raise NotImplementedError(
            f"{type(coefficient).__name__} coefficient for {term_ir!r} is reserved"
        )
    return float(coefficient.value)


def _assert_catalog_fingerprint(
    run_spec: RunSpec,
    manifest_terms: list[str] | None,
) -> None:
    if run_spec.library_fingerprint is None or manifest_terms is None:
        return
    actual_fingerprint = TermLibrarySpec.from_terms(manifest_terms).fingerprint
    if actual_fingerprint != run_spec.library_fingerprint:
        raise RecordSchemaError(
            "library_fingerprint mismatch: config declares "
            f"{run_spec.library_fingerprint!r}, manifest terms yield "
            f"{actual_fingerprint!r}"
        )


def _build_evidence(
    *,
    instrument: str,
    dataset_name: str,
    dataset_cache_fingerprint: str,
    seed: int | None,
    final_eval: EvaluationResult,
    equation: Equation | None,
    best_expression: str,
    best_score: float,
    score_kind: str,
    score_direction: str,
    headline_coefficient_source: str,
    support_from_equation: bool,
) -> EvidenceRecord:


    if support_from_equation:
        support, coefficients = active_support_and_coefficients_from_equation(
            equation
        )
    else:
        support, coefficients = active_support_and_coefficients(final_eval)
    return EvidenceRecord(
        instrument=instrument,
        dataset_name=dataset_name,
        dataset_cache_fingerprint=dataset_cache_fingerprint,
        seed=seed,
        is_valid=final_eval.is_valid,
        expression=best_expression,
        score_kind=score_kind,
        score_direction=score_direction,
        headline_coefficient_source=headline_coefficient_source,
        catalog_fit=equation_to_dict(equation) if equation is not None else None,
        support=support,
        coefficients=coefficients,



        complexity=final_eval.complexity if final_eval.is_valid else None,



        mse=sanitize_float(float(final_eval.mse)),
        nmse=sanitize_float(float(final_eval.nmse)),
        r2=sanitize_float(float(final_eval.r2)),
        score=sanitize_float(float(best_score)),
        residual_stats=residual_stats_from(final_eval.residuals),
        invalid_reason=None if final_eval.is_valid else final_eval.invalid_reason,
        error_detail=None if final_eval.is_valid else final_eval.error_message,
    )


def assemble_run_record(
    *,
    instrument: str,
    dataset_name: str,
    dataset_cache_fingerprint: str,
    seed: int | None,
    final_eval: EvaluationResult,
    equation: Equation | None,
    best_expression: str,
    best_score: float,
    score_kind: str,
    score_direction: str,
    headline_coefficient_source: str,
    run_spec: RunSpec,
    manifest_terms: list[str] | None = None,
    cost: RunCost,
    support_from_equation: bool = False,
) -> RunRecord:
    _assert_catalog_fingerprint(run_spec, manifest_terms)
    evidence = _build_evidence(
        instrument=instrument,
        dataset_name=dataset_name,
        dataset_cache_fingerprint=dataset_cache_fingerprint,
        seed=seed,
        final_eval=final_eval,
        equation=equation,
        best_expression=best_expression,
        best_score=best_score,
        score_kind=score_kind,
        score_direction=score_direction,
        headline_coefficient_source=headline_coefficient_source,
        support_from_equation=support_from_equation,
    )


    record = RunRecord(
        schema_version=RUN_RECORD_SCHEMA_VERSION,
        evidence_hash_scheme=EVIDENCE_HASH_SCHEME,
        created_at=datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        cost=cost,
        evidence=evidence,
        evidence_hash=evidence.content_hash(),
        run_spec=run_spec,
        run_spec_hash=run_spec.run_spec_hash,
        run_spec_hash_scheme=RUN_SPEC_HASH_SCHEME,
        record_hash="",
        record_hash_scheme=RECORD_HASH_SCHEME,
    )
    return seal_record_hash(record)
