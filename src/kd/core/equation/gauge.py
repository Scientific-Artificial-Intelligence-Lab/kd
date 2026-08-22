
from __future__ import annotations

import ast

from kd.core.equation.canonical import skeletonize_constants
from kd.core.expr.term_key import numeric_scalar_value, split_scalar_factor


_CONST = "const"

__all__ = ["regression_term_gauge"]


def regression_term_gauge(term_ir: str) -> tuple[str, float]:
    try:
        parsed = ast.parse(term_ir.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid IR syntax: {exc}") from exc
    factor, residual = _split_div_numerator(*split_scalar_factor(parsed.body))
    constant = numeric_scalar_value(residual)
    if constant is not None:
        return _CONST, factor * constant
    return skeletonize_constants(ast.unparse(residual)), factor


def _split_div_numerator(
    factor: float, residual: ast.expr
) -> tuple[float, ast.expr]:
    if not (
        isinstance(residual, ast.Call)
        and isinstance(residual.func, ast.Name)
        and residual.func.id == "div"
        and len(residual.args) == 2
        and not residual.keywords
    ):
        return factor, residual
    numerator, denominator = residual.args
    value = numeric_scalar_value(numerator)
    if value is None:
        return factor, residual
    return factor * value, ast.Call(
        func=ast.Name(id="recip"), args=[denominator], keywords=[]
    )
