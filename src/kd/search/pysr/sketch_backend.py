
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations, permutations

import sympy

from kd.core.equation.signature import law_term_key
from kd.core.equation.sketch import (
    Sketch,
    TermHole,
    anchor_basis,
    anchor_spelled_by,
    constraint_admits,
    pinned_monomials,
    term_admission,
)
from kd.core.expr.jet import Monomial, jet_monomials
from kd.core.expr.registry import FunctionRegistry
from kd.core.expr.term_features import (
    ColumnFingerprint,
    TermFeatures,
    TermVocabulary,
    analyze_term,
    column_fingerprint,
)
from kd.core.expr.terms import split_terms
from kd.core.platform.sketch_compile import CompileReport, SketchClauseLevels
from kd.search.pysr.config import PySRConfig
from kd.search.pysr.convert import build_feature_names, pysr_sympy_to_kd_terms
from kd.search.term_utils import fold_add

_LEVELS = SketchClauseLevels(
    fixed_terms="lowered",
    anchors="exit_checked",
    hole_count="exit_checked",
    derivative_order="generation_enforced",
    operator_set="exit_checked",
    field_axis_set="generation_enforced",
)

_PINNED = "pinned"
_PINNED_ALIAS = "pinned-alias"
_OUTSIDE_VOCABULARY = "outside-vocabulary"
_HOLE_FILTERED = "hole-filtered"
_NO_LAW_TERM = "no-law-term"





_BINARY_ALIASES: dict[str, str] = {"plus": "+", "sub": "-", "mult": "*", "div": "/"}
_SEPARATORS = frozenset({"+", "-"})










_BINARY_UNITS: dict[str, tuple[frozenset[str], ...]] = {
    "*": (frozenset({"mul"}), frozenset({"n2"}), frozenset({"n3"})),
    "/": (
        frozenset({"recip"}),
        frozenset({"mul", "recip"}),
        frozenset({"mul"}),
        frozenset({"n2"}),
        frozenset({"n3"}),
    ),
}
_UNARY_FEATURES: dict[str, str] = {
    "sin": "sin",
    "cos": "cos",
    "exp": "exp",
    "log": "log",
    "square": "n2",
    "cube": "n3",
    "inv": "recip",
    "sqrt": "sqrt",
}






_ADD_KEEPERS = frozenset({"/", "inv", "sin", "cos", "log", "sqrt"})



_POWERS = frozenset({"n2", "n3"})

_GENERATION_NOTE = (
    "derivative_order and field_axis_set are enforced by the kept columns, "
    "which bound the Julia search: no PySR term can exceed the highest kept "
    "derivative order or use a field or axis outside the kept columns; "
    "per-hole membership (a product joining two holes, a product with an "
    "anchor, a recomposed pin) is enforced by the pre-score gate on every "
    "candidate before kd scores it, after the Julia search, which the gate "
    "does not steer per hole"
)
_FILTER_BOUNDARY_NOTE = (
    "column filtering and the required-hole check read each column as the "
    "law terms a PySR fit of that column alone converts to; only the "
    "required-hole check credits an anchored column's compositions with "
    "itself, and it credits neither a column that converts to a pinned "
    "column nor that column's compositions; both are exact for "
    "cancellation-free PySR compositions of field-bearing, power-free "
    "columns; otherwise (a quotient that cancels a factor, a power-bearing "
    "column whose power sympy merges with another factor, a bare coordinate "
    "column judged without a field, a column that converts to a pinned "
    "column) both err toward refusal, except that a product of two column "
    "terms sharing a base is judged on its literal IR while PySR merges it "
    "into a power, and an anchored sqrt-bearing column's square is credited "
    "as n2(sqrt(.)) while PySR converts c0*c0 to the column under the root; "
    "there the required-hole check errs toward admission"
)
_OPERATOR_NOTE = (
    "operator_set is exit-checked: the law features a PySR operator emits "
    "depend on its operands ('*' emits mul, n2 or n3; '/' emits recip, "
    "mul+recip, mul, n2 or n3; '+' or '-' leaves a nested add only under "
    "'/', inv, sin, cos, log or sqrt, and '-' only when it cancels its own sign, "
    "since c0-c1 leaves mul(-1,.) outside the vocabulary), so no configured "
    "operator set bounds a hole's operator clause; the pre-score gate judges "
    "operators as well"
)
_PIN_NOTES = (
    "pin reservation refuses, in the library and at the pre-score gate, a "
    "term on a pin's law key or whose expansion into jet monomials (total "
    "derivatives by the chain rule, symbol-free factors stripped as "
    "coefficients, division by or log of an identically zero column read as "
    "the constant kd's protected operators produce) contains a "
    "pinned monomial, so a composite that carries a pinned monomial beside "
    "another one is refused by design",
    "pin reservation leaves open: the expansion cancels no common factor "
    "between a numerator and a denominator, so a rational respelling of a "
    "pin is admitted; a subtraction inside a denominator converts to an IR "
    "outside the vocabulary, so the reachable form is a sum of two admitted "
    "terms: terms are judged one at a time, so u*u*u_x/(u+x) + x*u*u_x/(u+x) "
    "adds up to the pinned column with '/' and '+'; function identities "
    "(sin or cos squared, log of exp) respell a pin when unary_operators "
    "configures sin, cos or log (cos alone or log alone suffices); sqrt of a "
    "square keeps a monomial of its own although its column is |u|, so with "
    "sqrt configured, sqrt(c0*c0)*c1 respells the pin u*u_x on a field that "
    "is never negative",
)
_SPELLING_MAP = (
    "PySR's square, cube and inv arrive as n2, n3 and recip; '*' as mul, n2 "
    "or n3; '/' as recip, mul+recip, mul, n2 or n3 (write division as "
    "mul+recip)"
)




_Parts = tuple[tuple[str, TermFeatures], ...]
_Converted = _Parts | None


@dataclass(frozen=True, kw_only=True)
class PySRCompiled:

    effective_terms: tuple[str, ...]
    pinned_keys: frozenset[str]
    pinned_monomials: frozenset[Monomial]
    anchor_fingerprints: frozenset[ColumnFingerprint]
    report: CompileReport
    dropped: tuple[tuple[str, str], ...]


def _closed_compilation(sketch: Sketch, config_terms: tuple[str, ...]) -> PySRCompiled:
    pins = ", ".join(pin.term_ir for pin in sketch.pinned)
    return PySRCompiled(
        effective_terms=config_terms,
        pinned_keys=frozenset(),
        pinned_monomials=frozenset(),
        anchor_fingerprints=frozenset(),
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


def _convert_column(term: str, vocabulary: TermVocabulary) -> _Converted:
    try:
        (name,) = build_feature_names([term])
        converted = pysr_sympy_to_kd_terms(sympy.Symbol(name), [term], [name])
        return tuple((part, analyze_term(part, vocabulary)) for part in converted)
    except ValueError:
        return None


def _binary_symbols(config: PySRConfig) -> frozenset[str]:
    return frozenset(
        _BINARY_ALIASES.get(name, name) for name in config.binary_operators
    )


def _assert_anchors_representable(
    sketch: Sketch, columns: dict[str, _Converted]
) -> frozenset[ColumnFingerprint]:
    converted = {
        column_fingerprint(features)
        for parts in columns.values()
        if parts is not None
        for _part, features in parts
    }
    fingerprints: list[ColumnFingerprint] = []
    for anchor in sketch.anchored:
        fingerprint = column_fingerprint(
            analyze_term(anchor.term_ir, sketch.vocabulary)
        )
        if fingerprint not in converted:
            raise ValueError(
                f"PySR anchors require a configured term that converts to the "
                f"column of {anchor.term_ir!r} (any spelling with its column "
                "fingerprint), but no term in config.terms does (PySR "
                "compositions of columns are not credited)"
            )
        fingerprints.append(fingerprint)
    return frozenset(fingerprints)


def _pysr_units(binary: frozenset[str], unary: tuple[str, ...]) -> list[frozenset[str]]:
    units = [unit for name in binary for unit in _BINARY_UNITS.get(name, ())]
    units.extend(
        frozenset({_UNARY_FEATURES[name]}) for name in unary if name in _UNARY_FEATURES
    )
    return units


def _hole_units(
    hole: TermHole,
    remaining: list[tuple[str, _Converted]],
    pysr_units: list[frozenset[str]],
    *,
    pysr_sum: bool,
    keeper: bool,
) -> tuple[list[frozenset[str]], frozenset[str]]:
    shape = replace(hole.constraint, operators=None)
    column_units = [
        features.operators
        for _term, parts in remaining
        if parts is not None
        for _part, features in parts
        if features.operators and constraint_admits(shape, features)
    ]
    units = [*column_units, *pysr_units]
    generable = frozenset[str]().union(*units)
    split_source = any(
        parts is not None
        and len(parts) > 1
        and all(constraint_admits(shape, features) for _part, features in parts)
        for _term, parts in remaining
    )
    if keeper and (pysr_sum or split_source):
        generable |= {"add"}
    return units, generable


def _assert_operator_clauses_generable(
    sketch: Sketch, config: PySRConfig, remaining: list[tuple[str, _Converted]]
) -> None:
    binary = _binary_symbols(config)
    pysr_units = _pysr_units(binary, config.unary_operators)
    pysr_sum = bool(binary & _SEPARATORS)
    keeper = bool(_ADD_KEEPERS & {*binary, *config.unary_operators})
    operators = sorted({*config.binary_operators, *config.unary_operators})
    for hole in sketch.holes:
        clause = hole.constraint.operators
        if not clause:
            continue
        units, generable = _hole_units(
            hole, remaining, pysr_units, pysr_sum=pysr_sum, keeper=keeper
        )
        unknown = clause - generable
        if unknown:
            raise ValueError(
                f"PySR hole {hole.id!r} operator clause names {sorted(unknown)!r}, "
                "which no column left in config.terms after pin exclusion "
                "converts to in a term this hole admits by derivative order, "
                "fields and axes, and the configured PySR operators "
                f"{operators!r} cannot emit (features that can reach the hole "
                f"{sorted(generable)!r}; {_SPELLING_MAP}); an empty clause "
                "means terminal-only"
            )
        if not any(unit <= clause for unit in units):
            raise ValueError(
                f"PySR hole {hole.id!r} operator clause {sorted(clause)!r} admits "
                "neither an operator-bearing column term left in config.terms "
                "after pin exclusion that the hole admits by derivative order, "
                "fields and axes, nor a PySR composition as a whole"
            )


def _on_a_pin(
    term: str, vocabulary: TermVocabulary, pinned: frozenset[Monomial]
) -> bool:
    return bool(pinned and jet_monomials(term, vocabulary) & pinned)


def _exclude_pinned(
    sketch: Sketch,
    config_terms: tuple[str, ...],
    pinned_keys: frozenset[str],
    pin_monomials: frozenset[Monomial],
) -> tuple[list[str], list[tuple[str, str]]]:
    remaining: list[str] = []
    dropped: list[tuple[str, str]] = []
    for term, key in zip(config_terms, _config_law_keys(config_terms), strict=True):
        if key in pinned_keys:
            dropped.append((term, _PINNED))
            continue
        if pin_monomials:
            try:
                monomials = jet_monomials(term, sketch.vocabulary)
            except ValueError:
                dropped.append((term, _OUTSIDE_VOCABULARY))
                continue
            if monomials & pin_monomials:
                dropped.append((term, _PINNED_ALIAS))
                continue
        remaining.append(term)
    return remaining, dropped


def _filter_for_holes(
    sketch: Sketch,
    remaining: list[tuple[str, _Converted]],
    anchor_fingerprints: frozenset[ColumnFingerprint],
    dropped: list[tuple[str, str]],
) -> list[tuple[str, _Parts]]:
    effective: list[tuple[str, _Parts]] = []
    for term, parts in remaining:
        if parts is None:
            dropped.append((term, _OUTSIDE_VOCABULARY))
            continue
        if len(parts) == 0:
            dropped.append((term, _NO_LAW_TERM))
            continue
        if all(
            column_fingerprint(features) in anchor_fingerprints
            or any(
                constraint_admits(hole.constraint, features) for hole in sketch.holes
            )
            for _part, features in parts
        ):
            effective.append((term, parts))
        else:
            dropped.append((term, _HOLE_FILTERED))
    return effective


def _assert_non_empty(
    sketch: Sketch,
    effective: list[tuple[str, _Parts]],
    dropped: list[tuple[str, str]],
) -> None:
    if effective:
        return
    pinned = tuple(pin.term_ir for pin in sketch.pinned)
    holes = tuple(hole.id for hole in sketch.holes)
    raise ValueError(
        "PySR effective library is empty after pinned exclusion and hole "
        f"constraints; pinned={pinned!r}, holes={holes!r}, "
        f"dropped={tuple(dropped)!r}"
    )


def _self_compositions(part: str, units: list[frozenset[str]]) -> list[str]:
    heads = sorted({head for unit in units if len(unit) == 1 for head in unit})
    return [f"{head}({part})" for head in heads if head != "mul"]


def _pair_products(parts: list[str], units: list[frozenset[str]]) -> list[str]:
    distinct = sorted(set(parts))
    products: list[str] = []
    if frozenset({"mul"}) in units:
        products.extend(f"mul({p},{q})" for p, q in combinations(distinct, 2))
    if frozenset({"mul", "recip"}) in units:
        products.extend(f"mul({p},recip({q}))" for p, q in permutations(distinct, 2))
    return products


def _composed_filler(
    composed: str, sketch: Sketch, pin_monomials: frozenset[Monomial]
) -> TermFeatures | None:
    if _on_a_pin(composed, sketch.vocabulary, pin_monomials):
        return None
    if anchor_spelled_by(composed, sketch) is not None:
        return None
    return analyze_term(composed, sketch.vocabulary)


def _hole_fillers(
    sketch: Sketch,
    effective: list[tuple[str, _Parts]],
    pin_monomials: frozenset[Monomial],
    units: list[frozenset[str]],
) -> list[TermFeatures]:
    fillers: list[TermFeatures] = []
    composed: list[str] = []
    for _term, parts in effective:
        for part, features in parts:
            if _on_a_pin(part, sketch.vocabulary, pin_monomials):
                continue
            if anchor_spelled_by(part, sketch) is None:
                fillers.append(features)
            elif not features.operators & _POWERS:
                composed.extend(_self_compositions(part, units))
    column_terms = [part for _term, column in effective for part, _ in column]
    composed.extend(_pair_products(column_terms, units))
    for candidate in composed:
        filler = _composed_filler(candidate, sketch, pin_monomials)
        if filler is not None:
            fillers.append(filler)
    return fillers


def _assert_required_holes_live(
    sketch: Sketch,
    config: PySRConfig,
    effective: list[tuple[str, _Parts]],
    pin_monomials: frozenset[Monomial],
) -> None:
    units = _pysr_units(_binary_symbols(config), config.unary_operators)
    fillers = _hole_fillers(sketch, effective, pin_monomials, units)
    for hole in sketch.holes:
        if hole.min_count < 1:
            continue
        if not any(constraint_admits(hole.constraint, item) for item in fillers):
            terms = tuple(term for term, _parts in effective)
            raise ValueError(
                f"PySR hole {hole.id!r} admits no column of the effective "
                f"library {terms!r}, no PySR composition of an anchored "
                "column with itself and no PySR product or quotient of two "
                "columns (a column the exit reserves for an anchor or a term "
                "that shares a monomial with a pin fills no hole, and a PySR "
                "composition inherits its factors' derivative order, fields "
                "and axes)"
            )


def _assert_separator_capacity(
    sketch: Sketch,
    config: PySRConfig,
    anchor_fingerprints: frozenset[ColumnFingerprint],
) -> None:
    if _binary_symbols(config) & _SEPARATORS:
        return
    hole_terms = sum(hole.min_count for hole in sketch.holes)
    required = len(anchor_fingerprints) + hole_terms
    if required < 2:
        return
    raise ValueError(
        f"PySR binary_operators {config.binary_operators!r} carry neither '+' "
        f"nor '-', so every fit yields one law term, but the sketch requires "
        f"{required} ({len(anchor_fingerprints)} distinct anchor columns plus "
        f"hole min_count total {hole_terms}); configure '+' or '-'"
    )


def _column_notes(columns: dict[str, _Converted]) -> list[str]:
    notes: list[str] = []
    for term, parts in columns.items():
        if parts is None:
            continue
        if len(parts) == 0:
            notes.append(
                f"configured column {term!r} converts to no law term "
                "(sympy.expand leaves no non-constant part), so a PySR fit "
                "of it is identically zero or a constant and it is dropped"
            )
        elif len(parts) > 1:
            notes.append(
                f"configured column {term!r} converts into {len(parts)} law "
                "terms: it carries add only under '/', inv, sin, cos, log or sqrt, "
                "and the one-law-term check without '+' or '-' does not count "
                "its split, so that check errs toward refusal"
            )
    return notes


def _compile_notes(
    sketch: Sketch,
    columns: dict[str, _Converted],
    dropped: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    anchors = ", ".join(anchor.term_ir for anchor in sketch.anchored) or "none"
    pins = ", ".join(pin.term_ir for pin in sketch.pinned) or "none"
    notes = [
        "anchors and hole counts are exit-checked; anchors represented by a "
        f"configured column with their column fingerprint: {anchors} (PySR "
        "compositions of columns are not credited)",
        _GENERATION_NOTE,
        _FILTER_BOUNDARY_NOTE,
        _OPERATOR_NOTE,
        f"pinned terms are owned by lower/lift: {pins}",
    ]
    if sketch.pinned:
        notes.extend(_PIN_NOTES)
    notes.extend(
        f"hole {hole.id!r} min_count={hole.min_count} is exit-checked; the "
        "compile checks only that one term is admitted, because PySR "
        "composes new law keys from the columns"
        for hole in sketch.holes
        if hole.min_count > 1
    )
    notes.extend(_column_notes(columns))
    notes.extend(f"dropped {term!r}: {reason}" for term, reason in dropped)
    return tuple(notes)


def compile_for_pysr(sketch: Sketch, config: PySRConfig) -> PySRCompiled:
    if not sketch.anchored and not sketch.holes:
        return _closed_compilation(sketch, config.terms)
    pin_monomials = pinned_monomials(sketch)
    pinned_keys = frozenset(law_term_key(pin.term_ir) for pin in sketch.pinned)
    columns = {term: _convert_column(term, sketch.vocabulary) for term in config.terms}
    anchor_fingerprints = _assert_anchors_representable(sketch, columns)
    remaining_terms, dropped = _exclude_pinned(
        sketch, config.terms, pinned_keys, pin_monomials
    )
    remaining = [(term, columns[term]) for term in remaining_terms]
    _assert_operator_clauses_generable(sketch, config, remaining)
    effective = _filter_for_holes(sketch, remaining, anchor_fingerprints, dropped)
    _assert_non_empty(sketch, effective, dropped)
    _assert_required_holes_live(sketch, config, effective, pin_monomials)
    _assert_separator_capacity(sketch, config, anchor_fingerprints)
    frozen_dropped = tuple(dropped)
    return PySRCompiled(
        effective_terms=tuple(term for term, _parts in effective),
        pinned_keys=pinned_keys,
        pinned_monomials=pin_monomials,
        anchor_fingerprints=anchor_fingerprints,
        report=CompileReport(
            levels=_LEVELS,
            notes=_compile_notes(sketch, columns, frozen_dropped),
        ),
        dropped=frozen_dropped,
    )


def candidate_refusal(
    expression: str,
    *,
    sketch: Sketch,
    compiled: PySRCompiled,
    registry: FunctionRegistry,
) -> str | None:
    try:
        terms = split_terms(expression, registry)
    except ValueError:
        return None
    for term in terms:
        refusal = term_admission(
            term,
            sketch,
            pinned_keys=compiled.pinned_keys,
            pinned_monomials=compiled.pinned_monomials,
            anchor_fingerprints=compiled.anchor_fingerprints,
        )
        if refusal is not None:
            return refusal
    return None


def fit_expression(
    expression: str, *, sketch: Sketch, registry: FunctionRegistry
) -> str:
    try:
        terms = split_terms(expression, registry)
    except ValueError:
        return expression
    kept = anchor_basis(terms, sketch)
    if len(kept) == len(terms):
        return expression
    return fold_add([terms[index] for index in kept])


__all__ = ["PySRCompiled", "candidate_refusal", "compile_for_pysr", "fit_expression"]
