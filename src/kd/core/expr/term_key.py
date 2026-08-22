
from __future__ import annotations

import ast
import math

__all__ = ["numeric_scalar_value", "split_scalar_factor", "structure_term_key"]


def structure_term_key(term: str) -> str:
    try:





        tree = ast.parse(term.strip(), mode="eval")
    except (SyntaxError, ValueError):
        return "".join(term.split())
    unparsed = ast.unparse(split_scalar_factor(tree.body)[1])






    from kd.core.equation.canonical import canonicalize_expression

    try:
        return canonicalize_expression(unparsed)
    except ValueError:






        return "".join(unparsed.split())


def split_scalar_factor(node: ast.expr) -> tuple[float, ast.expr]:
    if isinstance(node, ast.Call) and _is_call(node, "neg", 1):
        factor, residual = split_scalar_factor(node.args[0])
        return -factor, residual
    if isinstance(node, ast.Call) and _is_call(node, "mul", 2):
        return _split_mul_scalar(node)
    if isinstance(node, ast.Call) and _is_call(node, "div", 2):
        divisor = numeric_scalar_value(node.args[1])
        if divisor is not None:
            factor, residual = split_scalar_factor(node.args[0])





            return (factor * math.inf if divisor == 0.0 else factor / divisor), residual
    return 1.0, node


def _split_mul_scalar(node: ast.Call) -> tuple[float, ast.expr]:
    left, right = node.args
    left_value = numeric_scalar_value(left)
    if left_value is not None:
        factor, residual = split_scalar_factor(right)
        return left_value * factor, residual
    right_value = numeric_scalar_value(right)
    if right_value is not None:
        factor, residual = split_scalar_factor(left)
        return right_value * factor, residual
    return 1.0, node


def _is_call(node: ast.expr, name: str, arity: int) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        and len(node.args) == arity
    )


def numeric_scalar_value(node: ast.expr) -> float | None:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float) and not isinstance(node.value, bool):
            try:
                return float(node.value)
            except OverflowError:





                return math.inf if node.value > 0 else -math.inf
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        value = numeric_scalar_value(node.operand)
        if value is None:
            return None
        return -value if isinstance(node.op, ast.USub) else value
    return None
