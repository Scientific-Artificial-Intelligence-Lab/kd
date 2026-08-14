
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Final, Literal, cast

from kd.core.equation.construct import build_equation
from kd.core.equation.signature import law_term_entry
from kd.core.equation.sketch import Sketch
from kd.core.equation.types import Equation
from kd.core.evaluator import EvaluationResult

logger = logging.getLogger(__name__)

SKETCH_CONFIG_KEY: Final[str] = "sketch"

SketchClause = Literal[
    "fixed_terms",
    "anchors",
    "hole_count",
    "derivative_order",
    "operator_set",
    "field_axis_set",
]
SKETCH_CLAUSES: Final[tuple[SketchClause, ...]] = (
    "fixed_terms",
    "anchors",
    "hole_count",
    "derivative_order",
    "operator_set",
    "field_axis_set",
)
EnforcementLevel = Literal[
    "lowered",
    "generation_enforced",
    "fit_enforced",
    "exit_checked",
    "unsupported",
]
_ENFORCEMENT_LEVELS: Final[tuple[EnforcementLevel, ...]] = (
    "lowered",
    "generation_enforced",
    "fit_enforced",
    "exit_checked",
    "unsupported",
)


@dataclass(frozen=True, kw_only=True)
class SketchClauseLevels:

    fixed_terms: EnforcementLevel = "unsupported"
    anchors: EnforcementLevel = "unsupported"
    hole_count: EnforcementLevel = "unsupported"
    derivative_order: EnforcementLevel = "unsupported"
    operator_set: EnforcementLevel = "unsupported"
    field_axis_set: EnforcementLevel = "unsupported"

    def __post_init__(self) -> None:
        for clause in SKETCH_CLAUSES:
            declared = getattr(self, clause)
            if declared not in _ENFORCEMENT_LEVELS:
                raise ValueError(
                    f"{clause} has unknown enforcement level {declared!r}"
                )

    def level(self, clause: SketchClause) -> EnforcementLevel:
        return cast("EnforcementLevel", getattr(self, clause))


@dataclass(frozen=True, kw_only=True)
class CompileReport:

    levels: SketchClauseLevels
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class CompiledSketch:

    sketch: Sketch
    report: CompileReport

    @property
    def closed(self) -> bool:
        return not self.sketch.anchored and not self.sketch.holes

    def lift(self, final_eval: EvaluationResult | None) -> Equation | None:
        if final_eval is None:
            if not self.closed:
                raise ValueError("final_eval is required for an open sketch")
            candidate_aggregates: dict[str, float] | None = {}
        else:
            if not final_eval.is_valid:
                return None
            active = _active_candidate_entries(final_eval)
            if active is None:
                return None
            candidate_aggregates = _aggregate_candidate_values(active)
        if candidate_aggregates is None:
            return None

        policy = self.sketch.match_policy
        entries: list[tuple[str, float]] = []
        for pin in self.sketch.pinned:
            key, expected = law_term_entry(pin.term_ir, pin.value)
            residue = candidate_aggregates.pop(key, 0.0)
            band = policy.coeff_atol + policy.coeff_rtol * abs(expected)
            value = expected if abs(residue) <= band else expected + residue
            if value != 0.0:
                entries.append((key, value))
        entries.extend(
            (key, candidate_aggregates[key])
            for key in sorted(candidate_aggregates)
            if candidate_aggregates[key] != 0.0
        )
        return build_equation(
            [term_ir for term_ir, _value in entries],
            [value for _term_ir, value in entries],
            self.sketch.lhs_spec,
            active_indices=None,
            is_valid=True,
        )


def _active_candidate_entries(
    final_eval: EvaluationResult,
) -> list[tuple[str, float]] | None:
    terms = final_eval.terms
    coefficients = final_eval.coefficients
    if terms is None or coefficients is None:
        return None
    values = [float(value) for value in coefficients.detach().flatten().tolist()]
    if len(terms) != len(values):
        logger.warning(
            "Skipping sketch lift: term/coefficient length mismatch "
            "(terms=%d, coefficients=%d)",
            len(terms),
            len(values),
        )
        return None
    indices = (
        list(range(len(terms)))
        if final_eval.selected_indices is None
        else list(final_eval.selected_indices)
    )
    if any(not 0 <= index < len(terms) for index in indices):
        logger.warning(
            "Skipping sketch lift: selected_indices contain out-of-range "
            "entries for %d terms",
            len(terms),
        )
        return None
    return [(terms[index], values[index]) for index in indices]


def _aggregate_candidate_values(
    entries: list[tuple[str, float]],
) -> dict[str, float] | None:
    grouped: dict[str, list[float]] = {}
    for term_ir, coefficient in entries:
        if not math.isfinite(coefficient):
            logger.warning(
                "Skipping sketch lift: non-finite candidate coefficient "
                "for %r",
                term_ir,
            )
            return None
        try:
            key, signed_value = law_term_entry(term_ir, coefficient)
        except ValueError as exc:
            logger.warning(
                "Skipping sketch lift: candidate term %r has no law key (%s)",
                term_ir,
                exc,
            )
            return None
        grouped.setdefault(key, []).append(signed_value)
    try:
        return {key: math.fsum(values) for key, values in grouped.items()}
    except (OverflowError, ValueError) as exc:
        logger.warning(
            "Skipping sketch lift: candidate aggregation failed (%s)", exc
        )
        return None


def used_clauses(sketch: Sketch) -> frozenset[SketchClause]:
    used: set[SketchClause] = set()
    if sketch.pinned:
        used.add("fixed_terms")
    if sketch.anchored:
        used.add("anchors")
    if sketch.holes:
        used.add("hole_count")
    constraints = tuple(hole.constraint for hole in sketch.holes)
    if any(constraint.max_deriv_order is not None for constraint in constraints):
        used.add("derivative_order")
    if any(constraint.operators is not None for constraint in constraints):
        used.add("operator_set")
    if any(
        constraint.fields is not None or constraint.axes is not None
        for constraint in constraints
    ):
        used.add("field_axis_set")
    return frozenset(used)


def compile_sketch(sketch: Sketch) -> CompiledSketch:
    return CompiledSketch(
        sketch=sketch,
        report=CompileReport(
            levels=SketchClauseLevels(
                fixed_terms="lowered",
                anchors="exit_checked",
                hole_count="exit_checked",
                derivative_order="exit_checked",
                operator_set="exit_checked",
                field_axis_set="exit_checked",
            )
        ),
    )


__all__ = [
    "SKETCH_CLAUSES",
    "SKETCH_CONFIG_KEY",
    "CompileReport",
    "CompiledSketch",
    "EnforcementLevel",
    "SketchClause",
    "SketchClauseLevels",
    "compile_sketch",
    "used_clauses",
]
