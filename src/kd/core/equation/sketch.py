
from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
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
from kd.core.expr.jet import JetExpansion, Monomial, jet_expansion, jet_monomials
from kd.core.expr.term_features import (
    ColumnFingerprint,
    TermFeatures,
    TermVocabulary,
    analyze_term,
    column_fingerprint,
)

SKETCH_SCHEMA_TAG = "kd-sketch-v2"

SKETCH_V1_SCHEMA_TAG = "kd-sketch-v1"


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



    for anchor in sketch.anchored:
        jet_expansion(law_term_key(anchor.term_ir), sketch.vocabulary)
    _monomial_owners(
        [anchor.term_ir for anchor in sketch.anchored],
        sketch.vocabulary,
        _anchor_overlap,
    )


def _anchor_overlap(first: str, second: str, shared: list[str]) -> str:
    return (
        f"anchored terms {first!r} and {second!r} share the jet monomials "
        f"{shared!r}; each anchor has one independently fitted coefficient, "
        "so a column is anchored once"
    )


def _pin_overlap(first: str, second: str, shared: list[str]) -> str:
    return (
        f"pinned terms {first!r} and {second!r} are alias spellings of one "
        f"physical column (shared monomials {shared!r}); pin it once"
    )


def _monomial_owners(
    term_irs: Sequence[str],
    vocabulary: TermVocabulary,
    overlap: Callable[[str, str, list[str]], str],
) -> dict[Monomial, str]:
    owners: dict[Monomial, str] = {}
    for term_ir in term_irs:
        monomials = jet_monomials(law_term_key(term_ir), vocabulary)
        shared = sorted(str(monomial) for monomial in monomials & owners.keys())
        if shared:
            first = next(owners[m] for m in monomials if m in owners)
            raise ValueError(overlap(first, term_ir, shared))
        owners.update(dict.fromkeys(monomials, term_ir))
    return owners


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

    def save(self, path: str | os.PathLike[str]) -> None:

        from kd.core.equation.sketch_input import save_sketch

        save_sketch(self, path)


def _assert_pins_disjoint(sketch: Sketch) -> dict[Monomial, str]:
    return _monomial_owners(
        [pin.term_ir for pin in sketch.pinned], sketch.vocabulary, _pin_overlap
    )


def pinned_monomials(sketch: Sketch) -> frozenset[Monomial]:
    if not sketch.pinned:
        return frozenset()
    owners = _assert_pins_disjoint(sketch)
    for anchor in sketch.anchored:
        monomials = jet_monomials(law_term_key(anchor.term_ir), sketch.vocabulary)
        pinned = next((owners[m] for m in monomials if m in owners), None)
        if pinned is not None:
            raise ValueError(
                f"anchored term {anchor.term_ir!r} is an alias spelling of the "
                f"pinned column {pinned!r}; a pinned column is never refit"
            )
    return frozenset(owners)


@dataclass(frozen=True)
class _AnchorColumn:

    term_ir: str
    fingerprint: ColumnFingerprint
    expansion: JetExpansion


@lru_cache(maxsize=256)
def _anchor_columns(sketch: Sketch) -> tuple[_AnchorColumn, ...]:
    columns: list[_AnchorColumn] = []
    for anchor in sketch.anchored:
        key = law_term_key(anchor.term_ir)
        columns.append(
            _AnchorColumn(
                term_ir=anchor.term_ir,
                fingerprint=column_fingerprint(analyze_term(key, sketch.vocabulary)),
                expansion=jet_expansion(key, sketch.vocabulary),
            )
        )
    return tuple(columns)


def _spelled(key: str, features: TermFeatures, sketch: Sketch) -> _AnchorColumn | None:
    fingerprint = column_fingerprint(features)
    candidates = [
        column
        for column in _anchor_columns(sketch)
        if column.fingerprint == fingerprint
    ]
    if not candidates:
        return None
    expansion = jet_expansion(key, sketch.vocabulary)
    return next(
        (column for column in candidates if column.expansion == expansion), None
    )


def _carried(key: str, sketch: Sketch) -> _AnchorColumn | None:
    monomials = jet_monomials(key, sketch.vocabulary)
    return next(
        (
            column
            for column in _anchor_columns(sketch)
            if any(monomial in monomials for monomial, _ in column.expansion)
        ),
        None,
    )


def anchor_spelled_by(term_ir: str, sketch: Sketch) -> str | None:
    key = law_term_key(term_ir)
    column = _spelled(key, analyze_term(key, sketch.vocabulary), sketch)
    return None if column is None else column.term_ir


def anchor_carried_by(term_ir: str, sketch: Sketch) -> str | None:
    key = law_term_key(term_ir)
    if _spelled(key, analyze_term(key, sketch.vocabulary), sketch) is not None:
        return None
    column = _carried(key, sketch)
    return None if column is None else column.term_ir


def anchor_basis(term_irs: Sequence[str], sketch: Sketch) -> tuple[int, ...]:
    kept: list[int] = []
    seen: set[tuple[ColumnFingerprint, JetExpansion]] = set()
    for index, term_ir in enumerate(term_irs):
        try:
            key = law_term_key(term_ir)
            column = _spelled(key, analyze_term(key, sketch.vocabulary), sketch)
        except ValueError:
            column = None
        if column is not None:
            identity = (column.fingerprint, column.expansion)
            if identity in seen:
                continue
            seen.add(identity)
        kept.append(index)
    return tuple(kept)


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


def term_admission(
    term_ir: str,
    sketch: Sketch,
    *,
    pinned_keys: frozenset[str],
    pinned_monomials: frozenset[Monomial],
    anchor_fingerprints: frozenset[ColumnFingerprint],
) -> str | None:
    try:
        key = law_term_key(term_ir)
        features = analyze_term(key, sketch.vocabulary)
        pinned = key in pinned_keys or bool(
            pinned_monomials
            and jet_monomials(key, sketch.vocabulary) & pinned_monomials
        )
        spelled = (
            column_fingerprint(features) in anchor_fingerprints
            and _spelled(key, features, sketch) is not None
        )
        carried = None if spelled or not sketch.anchored else _carried(key, sketch)
    except ValueError:
        return f"term {term_ir!r} is outside the sketch vocabulary"
    if pinned:
        return f"term {term_ir!r} regenerates a pinned column"
    if spelled:
        return None
    if carried is not None:
        return (
            f"term {term_ir!r} carries the column of anchored term {carried.term_ir!r}"
        )
    if any(constraint_admits(hole.constraint, features) for hole in sketch.holes):
        return None
    return f"no sketch hole admits term {term_ir!r}"


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


def _anchor_spellings(sketch: Sketch, active: dict[str, float]) -> dict[str, list[str]]:
    spellings: dict[str, list[str]] = {anchor.term_ir: [] for anchor in sketch.anchored}
    for key in active:
        try:
            features = analyze_term(key, sketch.vocabulary)
            if _spelled(key, features, sketch) is None:
                continue
        except ValueError:

            continue
        expansion = jet_expansion(key, sketch.vocabulary)
        for column in _anchor_columns(sketch):
            if column.fingerprint == column_fingerprint(features) and (
                column.expansion == expansion
            ):
                spellings[column.term_ir].append(key)
    return spellings


def _anchored_verdicts(
    sketch: Sketch, active: dict[str, float]
) -> tuple[tuple[AnchoredVerdict, ...], set[str], list[UnassignedTerm]]:
    spellings = _anchor_spellings(sketch, active)
    verdicts: list[AnchoredVerdict] = []
    reserved: set[str] = set()
    repeated: list[UnassignedTerm] = []
    for anchor in sketch.anchored:
        keys = spellings[anchor.term_ir]
        reserved.update(keys)
        if len(keys) > 1:
            reason = (
                f"one of {len(keys)} spellings of anchored term "
                f"{anchor.term_ir!r}; an anchor is one fitted column"
            )
            repeated.extend(UnassignedTerm(law_key=key, reason=reason) for key in keys)
        observed = active[keys[0]] if len(keys) == 1 else None
        verdicts.append(
            AnchoredVerdict(
                term_ir=anchor.term_ir,
                law_key=law_term_key(anchor.term_ir),
                observed=observed,
                matched=observed is not None,
            )
        )
    return tuple(verdicts), reserved, repeated


def _hole_refusal(key: str, sketch: Sketch) -> tuple[TermFeatures | None, str | None]:
    try:
        features = analyze_term(key, sketch.vocabulary)
        carried = _carried(key, sketch) if sketch.anchored else None
    except ValueError as exc:
        return None, str(exc)
    if carried is not None:
        return None, f"carries the column of anchored term {carried.term_ir!r}"
    return features, None


def _assign_holes(
    sketch: Sketch, remaining: tuple[str, ...]
) -> tuple[tuple[HoleVerdict, ...], tuple[UnassignedTerm, ...]]:
    assignments: dict[str, list[str]] = {hole.id: [] for hole in sketch.holes}
    unassigned: list[UnassignedTerm] = []
    for key in remaining:
        features, refusal = _hole_refusal(key, sketch)
        if features is None:
            assert refusal is not None
            unassigned.append(UnassignedTerm(law_key=key, reason=refusal))
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
        raise TypeError(f"{SKETCH_SCHEMA_TAG} matches EVOLUTION equations only")
    active = _aggregate_law(eq, sketch.match_policy.support_threshold)
    pinned = _pinned_verdicts(sketch, active)
    anchored, anchored_keys, repeated = _anchored_verdicts(sketch, active)
    reserved = {entry.law_key for entry in pinned} | anchored_keys
    remaining = tuple(sorted(set(active) - reserved))
    holes, outside = _assign_holes(sketch, remaining)
    unassigned = tuple(sorted((*repeated, *outside), key=lambda entry: entry.law_key))
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


def require_complete_sketch(sketch: object, consumer: str) -> None:
    if not isinstance(sketch, Sketch):
        raise TypeError(
            f"{consumer} takes a complete Sketch, got {type(sketch).__name__}; "
            "bind a shorthand to its dataset with spec.resolve(dataset) first"
        )


def sketch_to_dict(sketch: Sketch) -> dict[str, object]:
    from kd.core.equation._sketch_serialize import sketch_to_dict_impl

    require_complete_sketch(sketch, "sketch_to_dict")

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
    "SKETCH_V1_SCHEMA_TAG",
    "Sketch",
    "SketchMatchPolicy",
    "SketchVerdict",
    "TermConstraint",
    "TermHole",
    "UnassignedTerm",
    "anchor_basis",
    "anchor_carried_by",
    "anchor_spelled_by",
    "constraint_admits",
    "pinned_monomials",
    "require_complete_sketch",
    "sketch_from_dict",
    "sketch_to_dict",
    "term_admission",
]
