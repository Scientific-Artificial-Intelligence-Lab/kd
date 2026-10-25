
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Protocol

from torch import Tensor

from kd.core.equation.signature import law_term_key
from kd.core.equation.sketch import (
    Sketch,
    TermConstraint,
    anchor_basis,
    constraint_admits,
    pinned_monomials,
    term_admission,
)
from kd.core.evaluator import EvaluationResult
from kd.core.expr.executor import DIFF_OPERATOR_PATTERN
from kd.core.expr.jet import Monomial
from kd.core.expr.registry import FunctionRegistry
from kd.core.expr.term_features import (
    ColumnFingerprint,
    TermFeatures,
    analyze_term,
    column_fingerprint,
)
from kd.core.expr.terms import split_terms
from kd.core.platform.sketch_compile import CompileReport, SketchClauseLevels
from kd.search.discover.config import DiscoverConfig
from kd.search.discover.tokens.library import (
    _DIFF_OPERATORS,
    _OPERATOR_ARITIES,
    _SPECIAL_DIFF_OPERATORS,
    LibraryConfig,
)
from kd.search.result import invalid_evaluation_result
from kd.search.term_utils import fold_add

_LEVELS = SketchClauseLevels(
    fixed_terms="lowered",
    anchors="exit_checked",
    hole_count="exit_checked",
    derivative_order="generation_enforced",
    operator_set="generation_enforced",
    field_axis_set="generation_enforced",
)



_SEPARATORS = frozenset({"add", "sub"})


_LAW_FEATURES = {"div": frozenset({"mul", "recip"})}

_OUTSIDE_VOCABULARY = "outside-vocabulary"
_CONSTRAINT_FILTERED = "constraint-filtered"


class _ExpressionEvaluator(Protocol):

    @property
    def lhs_target(self) -> Tensor:
        pass

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        pass

    def invalidate_term_cache(self) -> None:
        pass


@dataclass(frozen=True, kw_only=True)
class DiscoverCompiled:

    library: LibraryConfig
    max_diff_order: int | None
    repeat_tokens: tuple[str, ...]
    pinned_keys: frozenset[str]
    pinned_monomials: frozenset[Monomial]
    anchor_fingerprints: frozenset[ColumnFingerprint]
    report: CompileReport
    dropped: tuple[tuple[str, str], ...]


def _law_features(name: str) -> frozenset[str]:
    return _LAW_FEATURES.get(name, frozenset({name}))


def _is_derivative_token(name: str) -> bool:
    return name in _DIFF_OPERATORS or name in _SPECIAL_DIFF_OPERATORS


def _feature_tokens(operators: list[str]) -> list[str]:
    return [name for name in operators if not _is_derivative_token(name)]


def _axis_diff_tokens(operators: list[str], axis: str) -> dict[str, int]:
    tokens: dict[str, int] = {}
    for name in operators:
        match = DIFF_OPERATOR_PATTERN.fullmatch(name)
        if match is not None and match.group(2) == axis:
            tokens[name] = int(match.group(1) or "1")
    return tokens


def _composable(order: int, parts: list[int]) -> bool:
    reachable = {0}
    for total in range(1, order + 1):
        if any(total - part in reachable for part in parts):
            reachable.add(total)
    return order in reachable


def _term_axes(features: TermFeatures) -> frozenset[str]:
    axes = set(features.coordinate_dependencies)
    for _field, multiindex in features.derivative_multiindices:
        axes.update(axis for axis, _order in multiindex)
    return frozenset(axes)


def _uncovered_features(required: frozenset[str], operators: list[str]) -> set[str]:
    covered: set[str] = set()
    for name in _feature_tokens(operators):
        token_features = _law_features(name)
        if token_features <= required:
            covered.update(token_features)
    return set(required) - covered


def _assert_operator_clauses_generable(sketch: Sketch, operators: list[str]) -> None:
    for hole in sketch.holes:
        clause = hole.constraint.operators
        if not clause:
            continue
        uncovered = _uncovered_features(clause, operators)
        if uncovered:
            raise ValueError(
                f"DISCOVER hole {hole.id!r} operator clause {sorted(clause)!r} "
                f"names {sorted(uncovered)!r}, which no configured token "
                f"(config.library.operators={operators!r}) builds without adding "
                "a law feature outside the clause; configure a token for each "
                "name or change the clause (division is written mul+recip, div "
                "brings both; an empty clause means terminal-only)"
            )


def _effective_cap(
    sketch: Sketch, anchors: tuple[TermFeatures, ...], config_cap: int | None
) -> tuple[int | None, int | None]:
    caps = [hole.constraint.max_deriv_order for hole in sketch.holes]
    finite = [cap for cap in caps if cap is not None]
    if len(finite) < len(caps):
        return config_cap, None
    finite.extend(item.max_total_derivative_order for item in anchors)
    sketch_cap = max(finite, default=0)
    if config_cap is None:
        return sketch_cap, sketch_cap
    return min(config_cap, sketch_cap), sketch_cap


def _assert_anchor_symbols(
    term: str, features: TermFeatures, library: LibraryConfig
) -> None:
    fields = sorted(features.base_fields - set(library.state_vars))
    if fields:
        raise ValueError(
            f"DISCOVER anchor {term!r} requires fields {fields!r} absent from "
            "config.library.state_vars"
        )
    coordinates = sorted(features.coordinate_dependencies - set(library.coord_vars))
    if coordinates:
        raise ValueError(
            f"DISCOVER anchor {term!r} requires coordinates {coordinates!r} absent "
            "from config.library.coord_vars"
        )


def _assert_anchor_orders(
    term: str, features: TermFeatures, operators: list[str], effective: int | None
) -> None:
    for _field, multiindex in sorted(features.derivative_multiindices):
        for axis, order in multiindex:
            tokens = _axis_diff_tokens(operators, axis)
            if not _composable(order, list(tokens.values())):
                raise ValueError(
                    f"DISCOVER anchor {term!r} requires derivative order {order} "
                    f"on axis {axis!r}, which the configured diff tokens "
                    f"{list(tokens)!r} cannot compose"
                )
    order = features.max_total_derivative_order
    if effective is not None and order > effective:
        raise ValueError(
            f"DISCOVER anchor {term!r} requires derivative order {order} but "
            f"max_diff_order={effective} caps generation"
        )


def _assert_anchors_generable(
    sketch: Sketch,
    anchors: tuple[TermFeatures, ...],
    config: DiscoverConfig,
    effective: int | None,
) -> None:
    for anchor, features in zip(sketch.anchored, anchors, strict=True):
        _assert_anchor_symbols(anchor.term_ir, features, config.library)
        missing = _uncovered_features(features.operators, config.library.operators)
        if missing:
            raise ValueError(
                f"DISCOVER anchor {anchor.term_ir!r} requires operator features "
                f"{sorted(missing)!r} that no configured token builds without "
                "adding a feature the anchor lacks (div brings mul and recip "
                "together)"
            )
        _assert_anchor_orders(
            anchor.term_ir, features, config.library.operators, effective
        )


def _admitted_names(
    sketch: Sketch, anchors: tuple[TermFeatures, ...]
) -> tuple[frozenset[str], frozenset[str]]:
    vocabulary = sketch.vocabulary
    fields: set[str] = set()
    axes: set[str] = set()
    for hole in sketch.holes:
        constraint = hole.constraint
        fields.update(
            vocabulary.fields if constraint.fields is None else constraint.fields
        )
        axes.update(
            vocabulary.coordinates if constraint.axes is None else constraint.axes
        )
    for item in anchors:
        fields.update(item.base_fields)
        axes.update(_term_axes(item))
    return frozenset(fields), frozenset(axes)


def _keep_symbol(
    name: str,
    known: frozenset[str],
    admitted: frozenset[str],
    dropped: list[tuple[str, str]],
) -> bool:
    if name not in known:
        dropped.append((name, _OUTSIDE_VOCABULARY))
        return False
    if name not in admitted:
        dropped.append((name, _CONSTRAINT_FILTERED))
        return False
    return True


def _keep_operator(
    name: str,
    sketch: Sketch,
    anchors: tuple[TermFeatures, ...],
    axes: frozenset[str],
    effective: int | None,
) -> str | None:
    if name in _SEPARATORS:
        return None
    if _is_derivative_token(name):
        match = DIFF_OPERATOR_PATTERN.fullmatch(name)

        if match is None or match.group(2) not in sketch.vocabulary.coordinates:
            return _OUTSIDE_VOCABULARY
        order = int(match.group(1) or "1")
        within_cap = effective is None or order <= effective
        return None if within_cap and match.group(2) in axes else _CONSTRAINT_FILTERED
    features = _law_features(name)

    if any(
        hole.constraint.operators is None or features <= hole.constraint.operators
        for hole in sketch.holes
    ) or any(features <= item.operators for item in anchors):
        return None
    return _CONSTRAINT_FILTERED


def _narrow_library(
    sketch: Sketch,
    anchors: tuple[TermFeatures, ...],
    library: LibraryConfig,
    admitted: tuple[frozenset[str], frozenset[str]],
    effective: int | None,
    dropped: list[tuple[str, str]],
) -> LibraryConfig:
    fields, axes = admitted
    vocabulary = sketch.vocabulary
    coord_vars = [
        name
        for name in library.coord_vars
        if _keep_symbol(name, vocabulary.coordinates, axes, dropped)
    ]
    state_vars = [
        name
        for name in library.state_vars
        if _keep_symbol(name, vocabulary.fields, fields, dropped)
    ]
    operators: list[str] = []
    for name in library.operators:
        reason = _keep_operator(name, sketch, anchors, axes, effective)
        if reason is None:
            operators.append(name)
        else:
            dropped.append((name, reason))
    return LibraryConfig(
        operators=operators, state_vars=state_vars, coord_vars=coord_vars
    )


def _assert_state_variable(
    sketch: Sketch,
    library: LibraryConfig,
    configured: LibraryConfig,
    fields: frozenset[str],
) -> None:
    if library.state_vars:
        return
    holes = tuple(hole.id for hole in sketch.holes)
    raise ValueError(
        f"DISCOVER holes {holes!r} admit fields {sorted(fields)!r}, none of which "
        "is a configured state variable "
        f"(config.library.state_vars={configured.state_vars!r})"
    )


def _assert_rootable(
    sketch: Sketch, library: LibraryConfig, configured: LibraryConfig
) -> None:
    if library.operators:
        return
    holes = tuple(hole.id for hole in sketch.holes)
    widen = (
        f"; holes {holes!r}: configure add or sub so bare terminals become "
        "summands, or widen a hole's operators or max_deriv_order"
        if holes
        else "; configure add or sub, or an operator the anchors need"
    )
    raise ValueError(
        "DISCOVER effective vocabulary keeps no operator token after sketch "
        f"constraints (configured operators {configured.operators!r}), and a "
        "DISCOVER expression cannot be a lone terminal (LengthConstraint forbids "
        f"a terminal root){widen}"
    )


def _root_admitted(
    hole_constraint: TermConstraint,
    name: str,
    leaves: list[str],
    library: LibraryConfig,
    sketch: Sketch,
) -> bool:
    children = (
        [leaf for leaf in leaves if leaf in library.state_vars]
        if _is_derivative_token(name)
        else leaves
    )
    arity = _OPERATOR_ARITIES[name]
    return any(
        constraint_admits(
            hole_constraint,
            analyze_term(f"{name}({','.join([leaf] * arity)})", sketch.vocabulary),
        )
        for leaf in children
    )


def _assert_required_holes_live(sketch: Sketch, library: LibraryConfig) -> None:
    separated = any(name in _SEPARATORS for name in library.operators)
    roots = [name for name in library.operators if name not in _SEPARATORS]
    for hole in sketch.holes:
        if hole.min_count < 1:
            continue
        leaves = [
            name
            for name in (*library.state_vars, *library.coord_vars)
            if constraint_admits(hole.constraint, analyze_term(name, sketch.vocabulary))
        ]
        if not leaves:
            raise ValueError(
                f"DISCOVER hole {hole.id!r} admits no generable term after "
                f"narrowing (kept state_vars={library.state_vars!r}, "
                f"coord_vars={library.coord_vars!r})"
            )
        if separated or any(
            _root_admitted(hole.constraint, name, leaves, library, sketch)
            for name in roots
        ):
            continue
        raise ValueError(
            f"DISCOVER hole {hole.id!r} admits no generable term after "
            "narrowing: no separator (add or sub) is configured, so every "
            f"expression is one term rooted at a kept operator {roots!r}, and "
            "none of them builds a term this hole admits; configure add or sub "
            "so bare terminals become summands, or widen the hole's constraint"
        )


def _assert_single_term_capacity(
    sketch: Sketch,
    library: LibraryConfig,
    anchor_fingerprints: frozenset[ColumnFingerprint],
) -> None:
    if any(name in _SEPARATORS for name in library.operators):
        return
    hole_terms = sum(hole.min_count for hole in sketch.holes)
    required = len(anchor_fingerprints) + hole_terms
    if required < 2:
        return
    raise ValueError(
        f"DISCOVER sketch requires {required} terms ({len(anchor_fingerprints)} "
        f"distinct anchor columns plus hole min_count total {hole_terms}), but "
        "neither add nor sub is configured (kept operators "
        f"{library.operators!r}), so every DISCOVER expression is a single "
        "term; configure add or sub"
    )


def _prior_token_notes(
    config: DiscoverConfig, dropped_names: frozenset[str]
) -> list[str]:
    notes: list[str] = []
    if config.token_bias_weight != 0.0:
        lost = [name for name in config.token_bias_tokens if name in dropped_names]
        if lost:
            notes.append(
                f"token_bias_tokens {lost!r} were dropped by narrowing, so their "
                "bias does not apply"
            )
    if config.diagnostic_scaffold:
        scaffold = (
            *config.diagnostic_scaffold_diffusion_tokens,
            *config.diagnostic_scaffold_reaction_tokens,
            *config.diagnostic_scaffold_root_tokens,
            *config.diagnostic_scaffold_neutral_tokens,
        )
        lost = [name for name in scaffold if name in dropped_names]
        if lost:
            notes.append(
                f"diagnostic scaffold tokens {lost!r} were dropped by narrowing "
                "and are skipped by the scaffold prior"
            )
    return notes


def _boundary_notes(sketch: Sketch) -> list[str]:
    notes: list[str] = []
    if sketch.pinned:
        notes.append(
            "pin reservation refuses a term whose expansion into jet monomials "
            "(total derivatives by the chain rule, symbol-free factors stripped "
            "as coefficients, division by or log of an identically zero column "
            "read as the constant kd's protected operators produce) contains a "
            "pinned monomial, so a composite that carries a pinned monomial "
            "beside another one is refused by design"
        )
        notes.append(
            "pin reservation leaves open: the expansion cancels no common factor "
            "between a numerator and a denominator, so a rational respelling of "
            "a pin ((u*u*u_x - u*u_x*u_x)/(u - u_x) for a pinned u*u_x) is "
            "admitted, reachable with div and add or sub; terms are judged one "
            "at a time, so several admitted terms can add up to the pinned "
            "column; function identities (sin or cos squared, log of exp) "
            "respell a pin when config.library.operators holds sin, cos or log "
            "(cos alone or log alone suffices)"
        )
    if sketch.anchored:
        notes.append(
            "anchor generability is judged on the token grammar (fields, "
            "coordinates, operator features, per-axis orders); configured priors "
            "such as TrigConstraint or DiffChildConstraint can still keep an "
            "anchor out of reach"
        )
    return notes


def _notes(
    sketch: Sketch,
    config: DiscoverConfig,
    *,
    effective: int | None,
    sketch_cap: int | None,
    repeat_tokens: tuple[str, ...],
    dropped: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    pins = ", ".join(pin.term_ir for pin in sketch.pinned) or "none"
    notes = [
        "anchors and hole counts are exit-checked",
        "token narrowing bounds generation; the evaluator gate refuses, before "
        "scoring, any term outside the sketch vocabulary, any term on a pinned "
        "law key or sharing a jet monomial with a pin, any term carrying an "
        "anchor's jet monomial without spelling the anchor, and any term no "
        "hole admits unless it spells an anchor (column fingerprint and jet "
        "expansion); a candidate's later spellings of one anchor are left out "
        "of its fit",
        f"max_diff_order effective={effective} (config={config.max_diff_order}, "
        f"sketch={sketch_cap})",
    ]
    notes.extend(
        f"hole {hole.id!r} admits derivative order {hole.constraint.max_deriv_order} "
        f"but generation is capped at {effective}"
        for hole in sketch.holes
        if effective is not None
        and hole.constraint.max_deriv_order is not None
        and hole.constraint.max_deriv_order > effective
    )
    notes.append(f"pinned terms are owned by lower/lift: {pins}")
    notes.extend(_boundary_notes(sketch))
    if repeat_tokens != tuple(config.repeat_tokens):
        notes.append(f"repeat_tokens narrowed to {list(repeat_tokens)!r}")
    notes.extend(
        _prior_token_notes(config, frozenset(name for name, _reason in dropped))
    )
    notes.extend(f"dropped {name!r}: {reason}" for name, reason in dropped)
    return tuple(notes)


def _closed_compilation(config: DiscoverConfig) -> DiscoverCompiled:
    return DiscoverCompiled(
        library=config.library,
        max_diff_order=config.max_diff_order,
        repeat_tokens=tuple(config.repeat_tokens),
        pinned_keys=frozenset(),
        pinned_monomials=frozenset(),
        anchor_fingerprints=frozenset(),
        report=CompileReport(
            levels=_LEVELS,
            notes=(
                "closed sketch: vocabulary and derivative cap remain unchanged "
                "because the search result is discarded; lift from pins alone",
            ),
        ),
        dropped=(),
    )


def compile_for_discover(sketch: Sketch, config: DiscoverConfig) -> DiscoverCompiled:
    if not sketch.anchored and not sketch.holes:
        return _closed_compilation(config)
    anchors = tuple(
        analyze_term(item.term_ir, sketch.vocabulary) for item in sketch.anchored
    )
    pin_monomials = pinned_monomials(sketch)
    _assert_operator_clauses_generable(sketch, config.library.operators)
    effective, sketch_cap = _effective_cap(sketch, anchors, config.max_diff_order)
    _assert_anchors_generable(sketch, anchors, config, effective)
    admitted = _admitted_names(sketch, anchors)
    dropped: list[tuple[str, str]] = []
    library = _narrow_library(
        sketch, anchors, config.library, admitted, effective, dropped
    )
    _assert_state_variable(sketch, library, config.library, admitted[0])
    _assert_rootable(sketch, library, config.library)
    _assert_required_holes_live(sketch, library)
    anchor_fingerprints = frozenset(column_fingerprint(item) for item in anchors)
    _assert_single_term_capacity(sketch, library, anchor_fingerprints)


    removed = {name for name, _reason in dropped}
    repeat_tokens = tuple(name for name in config.repeat_tokens if name not in removed)
    frozen_dropped = tuple(dropped)
    return DiscoverCompiled(
        library=library,
        max_diff_order=effective,
        repeat_tokens=repeat_tokens,
        pinned_keys=frozenset(law_term_key(pin.term_ir) for pin in sketch.pinned),
        pinned_monomials=pin_monomials,
        anchor_fingerprints=anchor_fingerprints,
        report=CompileReport(
            levels=_LEVELS,
            notes=_notes(
                sketch,
                config,
                effective=effective,
                sketch_cap=sketch_cap,
                repeat_tokens=repeat_tokens,
                dropped=frozen_dropped,
            ),
        ),
        dropped=frozen_dropped,
    )


def narrowed_config(
    config: DiscoverConfig, compiled: DiscoverCompiled
) -> DiscoverConfig:
    return dataclasses.replace(
        config,
        library=compiled.library,
        max_diff_order=compiled.max_diff_order,
        repeat_tokens=list(compiled.repeat_tokens),
    )


class SketchGatedEvaluator:

    def __init__(
        self,
        evaluator: _ExpressionEvaluator,
        *,
        sketch: Sketch,
        compiled: DiscoverCompiled,
        registry: FunctionRegistry,
    ) -> None:
        self._inner = evaluator
        self._sketch = sketch
        self._compiled = compiled
        self._registry = registry

    @property
    def lhs_target(self) -> Tensor:
        return self._inner.lhs_target

    def invalidate_term_cache(self) -> None:
        self._inner.invalidate_term_cache()

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        try:
            terms = split_terms(expr, self._registry)
        except ValueError:


            return self._inner.evaluate_expression(expr)
        for term in terms:
            refusal = self._refusal(term)
            if refusal is not None:
                return invalid_evaluation_result(
                    refusal,
                    score=None,
                    expression=expr,
                    terms=terms,
                    reason="structural_reject",
                )
        kept = anchor_basis(terms, self._sketch)
        if len(kept) == len(terms):
            return self._inner.evaluate_expression(expr)
        fitted = fold_add([terms[index] for index in kept])
        return dataclasses.replace(
            self._inner.evaluate_expression(fitted), expression=expr
        )

    def _refusal(self, term: str) -> str | None:
        compiled = self._compiled
        return term_admission(
            term,
            self._sketch,
            pinned_keys=compiled.pinned_keys,
            pinned_monomials=compiled.pinned_monomials,
            anchor_fingerprints=compiled.anchor_fingerprints,
        )


__all__ = [
    "DiscoverCompiled",
    "SketchGatedEvaluator",
    "compile_for_discover",
    "narrowed_config",
]
