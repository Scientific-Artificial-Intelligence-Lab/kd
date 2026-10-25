
from __future__ import annotations

import ast
from dataclasses import dataclass

from kd.core.equation.signature import law_term_key
from kd.core.equation.sketch import Sketch, constraint_admits, pinned_monomials
from kd.core.expr.jet import Monomial, jet_monomials
from kd.core.expr.term_features import (
    ColumnFingerprint,
    TermFeatures,
    analyze_term,
    column_fingerprint,
)
from kd.core.platform.sketch_compile import CompileReport, SketchClauseLevels
from kd.search.dlga.config import DLGAConfig

_LEVELS = SketchClauseLevels(
    fixed_terms="lowered",
    anchors="fit_enforced",
    hole_count="exit_checked",
    derivative_order="generation_enforced",
    operator_set="generation_enforced",
    field_axis_set="generation_enforced",
)
_PRODUCT_OPERATORS = frozenset({"mul"})


@dataclass(frozen=True, kw_only=True)
class DLGACompiled:

    library: tuple[str, ...]
    allow_products: bool
    gated: bool
    pinned_keys: frozenset[str]
    reserved_monomials: frozenset[Monomial]
    anchor_terms: tuple[str, ...]
    anchor_fingerprints: frozenset[ColumnFingerprint]
    report: CompileReport
    dropped: tuple[tuple[str, str], ...]


def _product_factors(node: ast.expr) -> list[str]:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "mul"
    ):
        return [
            factor for argument in node.args for factor in _product_factors(argument)
        ]
    return [ast.unparse(node)]


def _assert_anchor_factors_configured(
    sketch: Sketch, available: frozenset[ColumnFingerprint]
) -> None:
    for anchor in sketch.anchored:
        node = ast.parse(law_term_key(anchor.term_ir), mode="eval").body
        for factor in _product_factors(node):
            fingerprint = column_fingerprint(analyze_term(factor, sketch.vocabulary))
            if fingerprint not in available:
                raise ValueError(
                    f"DLGA anchor {anchor.term_ir!r} requires factor {factor!r} "
                    "absent from config.library"
                )


def _assert_operator_clauses_generable(sketch: Sketch) -> None:
    for hole in sketch.holes:
        clause = hole.constraint.operators
        if not clause:
            continue
        unknown = clause - _PRODUCT_OPERATORS
        if unknown:
            raise ValueError(
                f"DLGA hole {hole.id!r} operator clause names {sorted(unknown)!r}; "
                "DLGA generates only mul; an empty clause means terminal-only"
            )


def _narrow_library(
    sketch: Sketch, config: DLGAConfig, dropped: list[tuple[str, str]]
) -> tuple[str, ...]:
    features: dict[str, TermFeatures] = {}
    for token in config.library:
        try:
            features[token] = analyze_term(token, sketch.vocabulary)
        except ValueError:
            dropped.append((token, "outside-vocabulary"))
    _assert_anchor_factors_configured(
        sketch, frozenset(column_fingerprint(item) for item in features.values())
    )
    kept: list[str] = []
    for token in config.library:
        if token not in features:
            continue
        if any(
            constraint_admits(hole.constraint, features[token]) for hole in sketch.holes
        ):
            kept.append(token)
        else:
            dropped.append((token, "constraint-filtered"))
    if not kept:
        raise ValueError(
            "DLGA effective library is empty after sketch constraints: no hole "
            "admits a configured token (anchors are supplied to the fit, not "
            "generated)"
        )
    return tuple(kept)


def _closed_compilation(config: DLGAConfig) -> DLGACompiled:
    return DLGACompiled(
        library=tuple(config.library),
        allow_products=True,
        gated=False,
        pinned_keys=frozenset(),
        reserved_monomials=frozenset(),
        anchor_terms=(),
        anchor_fingerprints=frozenset(),
        report=CompileReport(
            levels=_LEVELS,
            notes=(
                "closed sketch: library, grammar and target remain unchanged "
                "because the search result is discarded; lift from pins alone",
            ),
        ),
        dropped=(),
    )


def _reserved_monomials(sketch: Sketch) -> frozenset[Monomial]:
    pins = pinned_monomials(sketch)
    owners: dict[Monomial, str] = {}
    for anchor in sketch.anchored:
        monomials = jet_monomials(law_term_key(anchor.term_ir), sketch.vocabulary)
        first = next((owners[m] for m in monomials if m in owners), None)
        if first is not None:
            raise ValueError(
                f"anchored terms {first!r} and {anchor.term_ir!r} are alias "
                "spellings of one column (shared jet monomials); anchor it once"
            )
        owners.update(dict.fromkeys(monomials, anchor.term_ir))
    return pins.union(owners)


def _notes(
    allow_products: bool, dropped: tuple[tuple[str, str], ...]
) -> tuple[str, ...]:
    return (
        "pinned terms are deducted from the platform's lowered target and "
        "added back at their exact values by the exit lift",
        "anchor columns are appended to every fit and kept in the fitted law; "
        "the exit still requires |coefficient| >= support_threshold, so with "
        "support_threshold=0 an anchored clause fails only on an exactly zero "
        "coefficient",
        "pre-score membership refuses a term sharing a jet monomial with a pin "
        "or an anchor, so no generated term spells or carries an anchored "
        "column; any other term must be admitted by one hole; hole counts are "
        "exit-checked",
        f"product modules enabled={allow_products}",
        *(f"dropped {token!r}: {reason}" for token, reason in dropped),
    )


def compile_for_dlga(sketch: Sketch, config: DLGAConfig) -> DLGACompiled:
    if not sketch.anchored and not sketch.holes:
        return _closed_compilation(config)
    _assert_operator_clauses_generable(sketch)
    required = sum(hole.min_count for hole in sketch.holes)
    if required > config.max_modules:
        raise ValueError(
            f"DLGA sketch holes require {required} terms "
            f"but config.max_modules={config.max_modules}"
        )
    dropped: list[tuple[str, str]] = []
    narrowed = _narrow_library(sketch, config, dropped)
    allow_products = any(
        hole.constraint.operators is None
        or hole.constraint.operators >= _PRODUCT_OPERATORS
        for hole in sketch.holes
    )
    if not allow_products:
        dropped.append(("mul", "constraint-filtered"))
    frozen_dropped = tuple(dropped)
    return DLGACompiled(
        library=narrowed,
        allow_products=allow_products,
        gated=True,
        pinned_keys=frozenset(law_term_key(pin.term_ir) for pin in sketch.pinned),
        reserved_monomials=_reserved_monomials(sketch),
        anchor_terms=tuple(anchor.term_ir for anchor in sketch.anchored),
        anchor_fingerprints=frozenset(
            column_fingerprint(analyze_term(anchor.term_ir, sketch.vocabulary))
            for anchor in sketch.anchored
        ),
        report=CompileReport(
            levels=_LEVELS, notes=_notes(allow_products, frozen_dropped)
        ),
        dropped=frozen_dropped,
    )
