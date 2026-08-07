
from __future__ import annotations

import ast

__all__ = ["structure_term_key"]


def structure_term_key(term: str) -> str:
    try:





        tree = ast.parse(term.strip(), mode="eval")
    except (SyntaxError, ValueError):
        return "".join(term.split())
    unparsed = ast.unparse(_strip_sign_and_scalar(tree.body))





    from kd.core.equation.canonical import canonicalize_expression

    try:
        return canonicalize_expression(unparsed)
    except ValueError:






        return "".join(unparsed.split())


def _strip_sign_and_scalar(node: ast.expr) -> ast.expr:
    if isinstance(node, ast.Call) and _is_call(node, "neg", 1):
        return _strip_sign_and_scalar(node.args[0])
    if isinstance(node, ast.Call) and _is_call(node, "mul", 2):
        return _strip_mul_scalar(node)
    if (
        isinstance(node, ast.Call)
        and _is_call(node, "div", 2)
        and _is_numeric_scalar(node.args[1])
    ):
        return _strip_sign_and_scalar(node.args[0])
    return node


def _strip_mul_scalar(node: ast.Call) -> ast.expr:
    left, right = node.args
    if _is_numeric_scalar(left):
        return _strip_sign_and_scalar(right)
    if _is_numeric_scalar(right):
        return _strip_sign_and_scalar(left)
    return node


def _is_call(node: ast.expr, name: str, arity: int) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        and len(node.args) == arity
    )


def _is_numeric_scalar(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int | float) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        return _is_numeric_scalar(node.operand)
    return False
