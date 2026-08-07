
from __future__ import annotations

from kd.core.expr.naming import parse_compound_derivative
from kd.core.expr.sympy_bridge import to_sympy


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
