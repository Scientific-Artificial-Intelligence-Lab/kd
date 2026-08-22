
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import cast

from kd.core.equation.projection import active_law




from kd.core.equation.signature import (
    LAWSIG_DOMAIN,
    _scalar_value,
    law_term_entry,
    law_term_key,
)
from kd.core.equation.types import Equation, Evolution, LhsSpec
from kd.core.expr.term_features import (
    ColumnFingerprint,
    TermFeatures,
    TermVocabulary,
    analyze_term,
    column_fingerprint,
)

SKETCH_SCHEMA_TAG = "kd-sketch-v1"


@dataclass(frozen=True)
class PinnedTerm:

    term_ir: str
    value: float

    def __post_init__(self) -> None:
        value = float(self.value)
        object.__setattr__(self, "value", value)
        if not math.isfinite(value):
            raise ValueError("PinnedTerm value must be finite")
        if value == 0.0:
            raise ValueError("PinnedTerm value must not be zero")
        law_term_key(self.term_ir)


@dataclass(frozen=True)
class AnchoredTerm:

    term_ir: str

    def __post_init__(self) -> None:
        law_term_key(self.term_ir)


def _validate_name_dimension(dimension: str, values: frozenset[str] | None) -> None:
    if values is None:
        return
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{dimension} contains an empty or non-string name")


@dataclass(frozen=True, kw_only=True)
class TermConstraint:

    max_deriv_order: int | None = None
    operators: frozenset[str] | None = None
    fields: frozenset[str] | None = None
    axes: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.max_deriv_order is not None and self.max_deriv_order < 0:
            raise ValueError("max_deriv_order must be non-negative or None")
        _validate_name_dimension("operators", self.operators)
        _validate_name_dimension("fields", self.fields)
        _validate_name_dimension("axes", self.axes)
        if self.fields is not None and not self.fields:
            raise ValueError("fields must not be empty")


@dataclass(frozen=True, kw_only=True)
class TermHole:

    id: str
    min_count: int
    max_count: int
    constraint: TermConstraint

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("TermHole id must be non-empty")
        if self.max_count < 1:
            raise ValueError("TermHole max_count must be at least 1")
        if self.min_count < 0 or self.min_count > self.max_count:
            raise ValueError("TermHole min_count must be within [0, max_count]")


def _coerce_policy_value(value: float, name: str) -> float:
    coerced = float(value)
    if not math.isfinite(coerced) or coerced < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return coerced


@dataclass(frozen=True, kw_only=True)
class SketchMatchPolicy:

    coeff_atol: float
    coeff_rtol: float
    support_threshold: float
    term_identity: str = LAWSIG_DOMAIN
    hole_assignment: str = "disjoint"
    derivative_order: str = "total-effective"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "coeff_atol", _coerce_policy_value(self.coeff_atol, "coeff_atol")
        )
        object.__setattr__(
            self, "coeff_rtol", _coerce_policy_value(self.coeff_rtol, "coeff_rtol")
        )
        object.__setattr__(
            self,
            "support_threshold",
            _coerce_policy_value(self.support_threshold, "support_threshold"),
        )


        if self.term_identity != LAWSIG_DOMAIN:
            raise ValueError(f"term_identity must be {LAWSIG_DOMAIN!r}")
        if self.hole_assignment != "disjoint":
            raise ValueError("hole_assignment must be 'disjoint'")
        if self.derivative_order != "total-effective":
            raise ValueError("derivative_order must be 'total-effective'")


@dataclass(frozen=True, kw_only=True)
class PinnedVerdict:

    term_ir: str
    law_key: str
    expected: float
    observed: float | None
    matched: bool


@dataclass(frozen=True, kw_only=True)
class AnchoredVerdict:

    term_ir: str
    law_key: str
    observed: float | None
    matched: bool


@dataclass(frozen=True, kw_only=True)
class HoleVerdict:

    hole_id: str
    assigned: tuple[str, ...]
    matched: bool


@dataclass(frozen=True, kw_only=True)
class UnassignedTerm:

    law_key: str
    reason: str


@dataclass(frozen=True, kw_only=True)
class SketchVerdict:

    lhs_matched: bool
    pinned: tuple[PinnedVerdict, ...]
    anchored: tuple[AnchoredVerdict, ...]
    holes: tuple[HoleVerdict, ...]
    unassigned: tuple[UnassignedTerm, ...]
    overall: bool
    policy: SketchMatchPolicy


def _validate_lhs(sketch: Sketch) -> None:
    lhs = sketch.lhs_spec
    if lhs.field not in sketch.vocabulary.fields:
        raise ValueError(f"LHS field {lhs.field!r} is outside the vocabulary")
    if lhs.axis not in sketch.vocabulary.coordinates:
        raise ValueError(f"LHS axis {lhs.axis!r} is outside the vocabulary")
    if len(lhs.axis) != 1:
        raise ValueError("LHS axis must be a single letter")
    if lhs.order < 1:
        raise ValueError("LHS order must be at least 1")


def _validate_clauses(sketch: Sketch) -> None:
    term_irs = tuple(clause.term_ir for clause in sketch.pinned) + tuple(
        clause.term_ir for clause in sketch.anchored
    )
    keys = tuple(law_term_key(term_ir) for term_ir in term_irs)
    if len(keys) != len(set(keys)):
        raise ValueError("pinned and anchored clauses collide by law-identity")
    for term_ir in term_irs:
        analyze_term(term_ir, sketch.vocabulary)


def _outside(values: frozenset[str] | None, allowed: frozenset[str]) -> set[str]:
    if values is None:
        return set()
    return set(values - allowed)


def _validate_holes(sketch: Sketch) -> None:
    ids = tuple(hole.id for hole in sketch.holes)
    if len(ids) != len(set(ids)):
        raise ValueError("Sketch hole id values must be unique")
    for hole in sketch.holes:
        unknown_fields = _outside(hole.constraint.fields, sketch.vocabulary.fields)
        if unknown_fields:
            raise ValueError(f"Hole fields outside vocabulary: {unknown_fields!r}")
        unknown_axes = _outside(hole.constraint.axes, sketch.vocabulary.coordinates)
        if unknown_axes:
            raise ValueError(f"Hole axes outside vocabulary: {unknown_axes!r}")
    for first, second in combinations(sketch.holes, 2):
        first_fields = first.constraint.fields
        second_fields = second.constraint.fields
        if (
            first_fields is None
            or second_fields is None
            or not first_fields.isdisjoint(second_fields)
        ):
            raise ValueError(
                "v1 hole constraints require provably disjoint fields sets"
            )


def _validate_pin_support(sketch: Sketch) -> None:
    policy = sketch.match_policy
    for pin in sketch.pinned:
        _key, wanted = law_term_entry(pin.term_ir, pin.value)
        band = policy.coeff_atol + policy.coeff_rtol * abs(wanted)
        if policy.support_threshold > abs(wanted) + band:
            raise ValueError(
                "support_threshold makes a pinned tolerance band unsatisfiable"
            )


@dataclass(frozen=True, kw_only=True)
class Sketch:

    lhs_spec: LhsSpec
    vocabulary: TermVocabulary
    pinned: tuple[PinnedTerm, ...]
    anchored: tuple[AnchoredTerm, ...]
    holes: tuple[TermHole, ...]
    match_policy: SketchMatchPolicy

    def __post_init__(self) -> None:
        _validate_lhs(self)
        _validate_clauses(self)
        _validate_holes(self)
        if not self.pinned and not self.anchored and not self.holes:
            raise ValueError("Sketch must not be fully empty")
        _validate_pin_support(self)

    def matches(self, eq: Equation) -> SketchVerdict:
        return _match_sketch(self, eq)


def pinned_fingerprints(sketch: Sketch) -> frozenset[ColumnFingerprint]:
    seen: dict[ColumnFingerprint, str] = {}
    for pin in sketch.pinned:
        fp = column_fingerprint(analyze_term(pin.term_ir, sketch.vocabulary))
        if fp in seen:
            raise ValueError(
                f"pinned terms {seen[fp]!r} and {pin.term_ir!r} are alias "
                "spellings of one physical column; pin it once"
            )
        seen[fp] = pin.term_ir
    for anchor in sketch.anchored:
        fp = column_fingerprint(analyze_term(anchor.term_ir, sketch.vocabulary))
        if fp in seen:
            raise ValueError(
                f"anchored term {anchor.term_ir!r} is an alias spelling of the "
                f"pinned column {seen[fp]!r}; a pinned column is never "
                "refit"
            )
    return frozenset(seen)


def constraint_admits(constraint: TermConstraint, features: TermFeatures) -> bool:
    if (
        constraint.max_deriv_order is not None
        and features.max_total_derivative_order > constraint.max_deriv_order
    ):
        return False
    if constraint.operators is not None and not (
        features.operators <= constraint.operators
    ):
        return False
    if constraint.fields is not None and (
        not features.base_fields or not features.base_fields <= constraint.fields
    ):
        return False
    axes_used = set(features.coordinate_dependencies)
    for _field, multiindex in features.derivative_multiindices:
        axes_used.update(axis for axis, _order in multiindex)
    return constraint.axes is None or axes_used <= constraint.axes


def _aggregate_law(eq: Evolution, support_threshold: float) -> dict[str, float]:
    projected = cast(Evolution, active_law(eq))
    grouped: dict[str, list[float]] = {}
    for term_ir, coefficient in projected.terms:
        scalar = _scalar_value(coefficient)




        if not math.isfinite(scalar):
            raise ValueError(f"law coefficient for term {term_ir!r} must be finite")
        key, value = law_term_entry(term_ir, scalar)
        grouped.setdefault(key, []).append(value)
    aggregated = {key: math.fsum(values) for key, values in grouped.items()}




    return {
        key: value
        for key, value in aggregated.items()
        if value != 0.0 and abs(value) >= support_threshold
    }


def _pinned_verdicts(
    sketch: Sketch, active: dict[str, float]
) -> tuple[PinnedVerdict, ...]:
    verdicts: list[PinnedVerdict] = []
    for pin in sketch.pinned:
        key, expected = law_term_entry(pin.term_ir, pin.value)
        observed = active.get(key)
        band = sketch.match_policy.coeff_atol + (
            sketch.match_policy.coeff_rtol * abs(expected)
        )
        matched = observed is not None and abs(observed - expected) <= band
        verdicts.append(
            PinnedVerdict(
                term_ir=pin.term_ir,
                law_key=key,
                expected=expected,
                observed=observed,
                matched=matched,
            )
        )
    return tuple(verdicts)


def _anchored_verdicts(
    sketch: Sketch, active: dict[str, float]
) -> tuple[AnchoredVerdict, ...]:
    verdicts: list[AnchoredVerdict] = []
    for anchor in sketch.anchored:
        key = law_term_key(anchor.term_ir)
        observed = active.get(key)
        verdicts.append(
            AnchoredVerdict(
                term_ir=anchor.term_ir,
                law_key=key,
                observed=observed,
                matched=observed is not None,
            )
        )
    return tuple(verdicts)


def _assign_holes(
    sketch: Sketch, remaining: tuple[str, ...]
) -> tuple[tuple[HoleVerdict, ...], tuple[UnassignedTerm, ...]]:
    assignments: dict[str, list[str]] = {hole.id: [] for hole in sketch.holes}
    unassigned: list[UnassignedTerm] = []
    for key in remaining:
        try:
            features = analyze_term(key, sketch.vocabulary)
        except ValueError as exc:
            unassigned.append(UnassignedTerm(law_key=key, reason=str(exc)))
            continue
        admitted = [
            hole
            for hole in sketch.holes
            if constraint_admits(hole.constraint, features)
        ]
        if not admitted:
            unassigned.append(
                UnassignedTerm(law_key=key, reason="no sketch hole admits this term")
            )
            continue
        assignments[admitted[0].id].append(key)
    verdicts = tuple(
        HoleVerdict(
            hole_id=hole.id,
            assigned=tuple(sorted(assignments[hole.id])),
            matched=hole.min_count <= len(assignments[hole.id]) <= hole.max_count,
        )
        for hole in sketch.holes
    )
    return verdicts, tuple(unassigned)


def _match_sketch(sketch: Sketch, eq: Equation) -> SketchVerdict:
    if not isinstance(eq, Evolution):
        raise TypeError("kd-sketch-v1 matches EVOLUTION equations only")
    active = _aggregate_law(eq, sketch.match_policy.support_threshold)
    pinned = _pinned_verdicts(sketch, active)
    anchored = _anchored_verdicts(sketch, active)
    reserved = {entry.law_key for entry in pinned} | {
        entry.law_key for entry in anchored
    }
    remaining = tuple(sorted(set(active) - reserved))
    holes, unassigned = _assign_holes(sketch, remaining)
    lhs_matched = eq.lhs_spec == sketch.lhs_spec
    overall = (
        lhs_matched
        and all(entry.matched for entry in pinned)
        and all(entry.matched for entry in anchored)
        and all(entry.matched for entry in holes)
        and not unassigned
    )
    return SketchVerdict(
        lhs_matched=lhs_matched,
        pinned=pinned,
        anchored=anchored,
        holes=holes,
        unassigned=unassigned,
        overall=overall,
        policy=sketch.match_policy,
    )


def sketch_to_dict(sketch: Sketch) -> dict[str, object]:
    from kd.core.equation._sketch_serialize import sketch_to_dict_impl

    return sketch_to_dict_impl(sketch)


def sketch_from_dict(payload: Mapping[str, object]) -> Sketch:
    from kd.core.equation._sketch_serialize import sketch_from_dict_impl

    return sketch_from_dict_impl(payload)


__all__ = [
    "AnchoredTerm",
    "AnchoredVerdict",
    "HoleVerdict",
    "PinnedTerm",
    "PinnedVerdict",
    "SKETCH_SCHEMA_TAG",
    "Sketch",
    "SketchMatchPolicy",
    "SketchVerdict",
    "TermConstraint",
    "TermHole",
    "UnassignedTerm",
    "constraint_admits",
    "pinned_fingerprints",
    "sketch_from_dict",
    "sketch_to_dict",
]
