
from __future__ import annotations

import ast
from dataclasses import dataclass

from kd.core.equation.signature import law_term_key
from kd.core.equation.sketch import Sketch, constraint_admits
from kd.core.expr.term_features import (
    ColumnFingerprint,
    TermFeatures,
    analyze_term,
    column_fingerprint,
)
from kd.core.platform.sketch_compile import CompileReport, SketchClauseLevels
from kd.search.dlga.config import DLGAConfig

_LEVELS = SketchClauseLevels(
    fixed_terms="unsupported",
    anchors="exit_checked",
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


def _anchor_tokens(
    sketch: Sketch, available: frozenset[ColumnFingerprint], max_module_length: int
) -> frozenset[ColumnFingerprint]:
    required: set[ColumnFingerprint] = set()
    for anchor in sketch.anchored:
        node = ast.parse(law_term_key(anchor.term_ir), mode="eval").body
        factors = _product_factors(node)
        if len(factors) > max_module_length:
            raise ValueError(
                f"DLGA anchor {anchor.term_ir!r} requires {len(factors)} factors "
                f"but config.max_module_length={max_module_length}"
            )
        for factor in factors:
            fingerprint = column_fingerprint(analyze_term(factor, sketch.vocabulary))
            if fingerprint not in available:
                raise ValueError(
                    f"DLGA anchor {anchor.term_ir!r} requires factor {factor!r} "
                    "absent from config.library"
                )
            required.add(fingerprint)
    return frozenset(required)


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
    required = _anchor_tokens(
        sketch,
        frozenset(column_fingerprint(item) for item in features.values()),
        config.max_module_length,
    )
    kept: list[str] = []
    for token in config.library:
        if token not in features:
            continue
        item = features[token]
        if column_fingerprint(item) in required or any(
            constraint_admits(hole.constraint, item) for hole in sketch.holes
        ):
            kept.append(token)
        else:
            dropped.append((token, "constraint-filtered"))
    if not kept:
        raise ValueError("DLGA effective library is empty after sketch constraints")
    return tuple(kept)


def compile_for_dlga(sketch: Sketch, config: DLGAConfig) -> DLGACompiled:
    _assert_operator_clauses_generable(sketch)
    required = sum(hole.min_count for hole in sketch.holes) + len(sketch.anchored)
    if required > config.max_modules:
        raise ValueError(
            f"DLGA sketch requires {required} terms "
            f"but config.max_modules={config.max_modules}"
        )
    dropped: list[tuple[str, str]] = []
    narrowed = _narrow_library(sketch, config, dropped)
    allow_products = any(
        hole.constraint.operators is None
        or hole.constraint.operators >= _PRODUCT_OPERATORS
        for hole in sketch.holes
    ) or any(
        "mul" in analyze_term(anchor.term_ir, sketch.vocabulary).operators
        for anchor in sketch.anchored
    )
    if not allow_products:
        dropped.append(("mul", "constraint-filtered"))
    notes = (
        "anchors and hole counts are exit-checked; anchor factors remain generable",
        "token filtering narrows generation; pre-score membership requires "
        "each term to match an anchor fingerprint or be admitted by one hole",
        f"product modules enabled={allow_products}",
        *(f"dropped {token!r}: {reason}" for token, reason in dropped),
    )
    return DLGACompiled(
        library=narrowed,
        allow_products=allow_products,
        report=CompileReport(levels=_LEVELS, notes=notes),
        dropped=tuple(dropped),
    )
