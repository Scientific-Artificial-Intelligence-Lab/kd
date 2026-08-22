
from __future__ import annotations

import logging
from collections.abc import Iterable

import sympy

from kd.core.expr.sympy_bridge import from_sympy, to_sympy

logger = logging.getLogger(__name__)



_FEATURE_PREFIXES: tuple[str, ...] = ("c", "feat", "col", "z", "v")


def _term_collision_names(terms: list[str], reserved: Iterable[str]) -> set[str]:
    collisions: set[str] = set(reserved)
    for term in terms:
        collisions |= {symbol.name for symbol in to_sympy(term).free_symbols}
    return collisions


def build_feature_names(
    terms: list[str],
    *,
    reserved: Iterable[str] = (),
) -> list[str]:
    if not terms:
        return []

    collisions = _term_collision_names(terms, reserved)
    for prefix in _FEATURE_PREFIXES:
        candidate = [f"{prefix}{index}" for index in range(len(terms))]
        if collisions.isdisjoint(candidate):
            return candidate
        logger.debug("Feature prefix %r collides; trying next prefix", prefix)

    raise ValueError(
        f"Could not build collision-free feature names for {len(terms)} terms; "
        f"all candidate prefixes {_FEATURE_PREFIXES} collide with {collisions}"
    )


def _build_substitution(
    terms: list[str],
    feature_names: list[str],
) -> dict[sympy.Symbol, sympy.Expr]:
    return {
        sympy.Symbol(feature_names[index]): to_sympy(terms[index])
        for index in range(len(terms))
    }


def _check_undeclared_features(
    pysr_expr: sympy.Expr,
    feature_names: list[str],
    terms: list[str],
) -> None:
    feature_symbols = {sympy.Symbol(name) for name in feature_names}
    stray = pysr_expr.free_symbols - feature_symbols
    if stray:
        raise ValueError(
            f"PySR expression references undeclared columns {stray}; only the "
            f"declared features {feature_names} (for terms {terms}) are allowed"
        )


def _structural_ir(term: sympy.Expr) -> str | None:
    _coeff, rest = term.as_coeff_Mul()
    if rest == sympy.Integer(1):
        return None
    return from_sympy(rest)


def pysr_sympy_to_kd_terms(
    pysr_expr: sympy.Expr,
    terms: list[str],
    feature_names: list[str],
) -> list[str]:
    if len(feature_names) != len(terms):
        raise ValueError(
            "feature_names and terms must have the same length: "
            f"{len(feature_names)} != {len(terms)}"
        )

    _check_undeclared_features(pysr_expr, feature_names, terms)
    subs_map = _build_substitution(terms, feature_names)
    expr = sympy.expand(pysr_expr.subs(subs_map, simultaneous=True))

    result: list[str] = []
    seen: set[sympy.Expr] = set()
    for additive_term in expr.as_ordered_terms():
        ir = _structural_ir(additive_term)
        if ir is None:
            continue
        key = to_sympy(ir)
        if key in seen:
            continue
        seen.add(key)
        result.append(ir)
    return result


def pysr_sympy_to_tabular_term(
    pysr_expr: sympy.Expr,
    terms: list[str],
    feature_names: list[str],
) -> str:
    if len(feature_names) != len(terms):
        raise ValueError(
            "feature_names and terms must have the same length: "
            f"{len(feature_names)} != {len(terms)}"
        )
    _check_undeclared_features(pysr_expr, feature_names, terms)
    declared = {sympy.Symbol(name) for name in feature_names}
    if not (pysr_expr.free_symbols & declared):
        raise ValueError(
            "PySR tabular expression is a constant model with no feature "
            "symbols; kd has no intercept-only semantics"
        )
    subs_map = _build_substitution(terms, feature_names)
    return from_sympy(pysr_expr.subs(subs_map, simultaneous=True))
