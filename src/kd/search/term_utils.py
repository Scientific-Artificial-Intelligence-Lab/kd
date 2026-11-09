
from __future__ import annotations

from collections.abc import Sequence

from kd.core.equation.rendering import render_lhs_label
from kd.core.equation.types import LhsSpec
from kd.core.expr.naming import parse_compound_derivative
from kd.core.expr.sympy_bridge import to_sympy


def refuse_target_derivatives(terms: Sequence[str], lhs: LhsSpec) -> None:
    field, axis = lhs.field, lhs.axis
    label = render_lhs_label(lhs)
    aliases = {label, f"diff{lhs.order}_{axis}({field})"}
    first_calls = (f"diff_{axis}", f"diff1_{axis}")
    if lhs.order == 1:
        aliases.add(f"diff_{axis}({field})")
    elif lhs.order == 2:
        first_terms = (f"{field}_{axis}", *(f"{fn}({field})" for fn in first_calls))
        aliases.add(f"{field}_{axis}_{axis}")
        aliases.update(f"{fn}({term})" for fn in first_calls for term in first_terms)
    for term in terms:
        if term in aliases:
            raise ValueError(
                f"terms contains LHS {label!r} as {term!r}: tautology; remove that term"
            )


def fold_add(kd_terms: list[str]) -> str:
    if not kd_terms:
        raise ValueError("fold_add requires at least one term")
    if len(kd_terms) == 1:
        return kd_terms[0]
    return _build_add_chain(kd_terms)


def _build_add_chain(kd_terms: list[str]) -> str:
    result = kd_terms[-1]
    for term in reversed(kd_terms[:-1]):
        result = f"add({term}, {result})"
    return result


def infer_max_atomic_order(terms: list[str]) -> int:
    max_order = 0
    for term in terms:
        for symbol in to_sympy(term).free_symbols:
            parsed = parse_compound_derivative(symbol.name)
            if parsed is None:
                continue
            _field, segments = parsed
            for _axis, order in segments:
                max_order = max(max_order, order)
    return max_order
