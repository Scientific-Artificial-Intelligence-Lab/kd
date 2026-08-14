
from __future__ import annotations

from dataclasses import dataclass

from kd.core.equation.signature import law_term_key
from kd.core.equation.sketch import Sketch, constraint_admits
from kd.core.expr.term_features import analyze_term
from kd.core.platform.sketch_compile import CompileReport, SketchClauseLevels

_LEVELS = SketchClauseLevels(
    fixed_terms="lowered",
    anchors="exit_checked",
    hole_count="exit_checked",
    derivative_order="generation_enforced",
    operator_set="generation_enforced",
    field_axis_set="generation_enforced",
)

_PINNED = "pinned"
_OUTSIDE_VOCABULARY = "outside-vocabulary"
_HOLE_FILTERED = "hole-filtered"


@dataclass(frozen=True, kw_only=True)
class PySINDyCompiled:

    effective_terms: tuple[str, ...]
    report: CompileReport
    dropped: tuple[tuple[str, str], ...]


def _closed_compilation(
    sketch: Sketch, config_terms: tuple[str, ...]
) -> PySINDyCompiled:
    pins = ", ".join(pin.term_ir for pin in sketch.pinned)
    return PySINDyCompiled(
        effective_terms=config_terms,
        report=CompileReport(
            levels=_LEVELS,
            notes=(
                "closed sketch: configured library remains unfiltered because "
                "the search result is discarded",
                f"closed sketch pinned terms remain in the library: {pins}",
            ),
        ),
        dropped=(),
    )


def _config_law_keys(config_terms: tuple[str, ...]) -> tuple[str | None, ...]:
    keys: list[str | None] = []
    for term in config_terms:
        try:
            keys.append(law_term_key(term))
        except ValueError:
            keys.append(None)
    return tuple(keys)


def _assert_anchors_representable(
    sketch: Sketch, config_keys: tuple[str | None, ...]
) -> frozenset[str]:
    available = frozenset(key for key in config_keys if key is not None)
    anchor_entries = tuple(
        (anchor, law_term_key(anchor.term_ir)) for anchor in sketch.anchored
    )
    for anchor, key in anchor_entries:
        if key not in available:
            raise ValueError(
                f"PySINDy anchors require configured term {anchor.term_ir!r}, "
                "but its law identity is absent from config.terms"
            )
    return frozenset(key for _anchor, key in anchor_entries)


def _exclude_pinned(
    sketch: Sketch,
    config_terms: tuple[str, ...],
    config_keys: tuple[str | None, ...],
) -> tuple[list[tuple[str, str | None]], list[tuple[str, str]]]:
    pinned_keys = {law_term_key(pin.term_ir) for pin in sketch.pinned}
    remaining: list[tuple[str, str | None]] = []
    dropped: list[tuple[str, str]] = []
    for term, key in zip(config_terms, config_keys, strict=True):
        if key in pinned_keys:
            dropped.append((term, _PINNED))
        else:
            remaining.append((term, key))
    return remaining, dropped


def _filter_for_holes(
    sketch: Sketch,
    remaining: list[tuple[str, str | None]],
    anchor_keys: frozenset[str],
    dropped: list[tuple[str, str]],
) -> tuple[str, ...]:
    effective: list[str] = []
    for term, key in remaining:
        if key in anchor_keys:
            effective.append(term)
            continue
        try:
            features = analyze_term(term, sketch.vocabulary)
        except ValueError:
            dropped.append((term, _OUTSIDE_VOCABULARY))
            continue
        if any(
            constraint_admits(hole.constraint, features) for hole in sketch.holes
        ):
            effective.append(term)
        else:
            dropped.append((term, _HOLE_FILTERED))
    return tuple(effective)


def _assert_non_empty(
    sketch: Sketch, effective_terms: tuple[str, ...]
) -> None:
    if effective_terms:
        return
    pinned = tuple(pin.term_ir for pin in sketch.pinned)
    holes = tuple(hole.id for hole in sketch.holes)
    raise ValueError(
        "PySINDy effective library is empty after pinned exclusion and hole "
        f"constraints; pinned={pinned!r}, holes={holes!r}"
    )


def _assert_hole_feasibility(
    sketch: Sketch,
    effective_terms: tuple[str, ...],
    anchor_keys: frozenset[str],
) -> None:




    assignable = {
        key: analyze_term(term, sketch.vocabulary)
        for term, key in (
            (term, law_term_key(term)) for term in effective_terms
        )
        if key not in anchor_keys
    }
    for hole in sketch.holes:
        if hole.min_count < 1:
            continue
        admitted = sum(
            constraint_admits(hole.constraint, term_features)
            for term_features in assignable.values()
        )
        if admitted < hole.min_count:
            raise ValueError(
                f"PySINDy hole {hole.id!r} requires min_count={hole.min_count}, "
                f"but only {admitted} assignable law keys in the effective "
                "library are admissible (anchor-reserved columns and alias "
                "duplicates cannot fill a hole)"
            )


def _compile_notes(
    sketch: Sketch, dropped: tuple[tuple[str, str], ...]
) -> tuple[str, ...]:
    anchors = ", ".join(anchor.term_ir for anchor in sketch.anchored) or "none"
    pins = ", ".join(pin.term_ir for pin in sketch.pinned) or "none"
    notes = [
        f"anchors represented by configured law identities: {anchors}",
        f"pinned terms are owned by lower/lift: {pins}",
    ]
    notes.extend(f"dropped {term!r}: {reason}" for term, reason in dropped)
    return tuple(notes)


def compile_for_pysindy(
    sketch: Sketch, config_terms: tuple[str, ...]
) -> PySINDyCompiled:
    if not sketch.anchored and not sketch.holes:
        return _closed_compilation(sketch, config_terms)

    config_keys = _config_law_keys(config_terms)
    anchor_keys = _assert_anchors_representable(sketch, config_keys)
    remaining, dropped = _exclude_pinned(sketch, config_terms, config_keys)
    effective_terms = _filter_for_holes(
        sketch, remaining, anchor_keys, dropped
    )
    _assert_non_empty(sketch, effective_terms)
    _assert_hole_feasibility(sketch, effective_terms, anchor_keys)
    frozen_dropped = tuple(dropped)
    return PySINDyCompiled(
        effective_terms=effective_terms,
        report=CompileReport(
            levels=_LEVELS,
            notes=_compile_notes(sketch, frozen_dropped),
        ),
        dropped=frozen_dropped,
    )


__all__ = ["PySINDyCompiled", "compile_for_pysindy"]
