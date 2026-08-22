
from __future__ import annotations

import ast
from typing import Final



_COMMUTATIVE_OPS: Final[frozenset[str]] = frozenset({"add", "mul"})

_PARSE_MODE: Final[str] = "eval"


def canonicalize_expression(expression: str) -> str:
    stripped = expression.strip()
    if not stripped:
        raise ValueError("Expression string cannot be empty.")

    try:
        parsed = ast.parse(stripped, mode=_PARSE_MODE)
    except SyntaxError as exc:
        raise ValueError(f"Invalid IR syntax: {exc}") from exc

    if not isinstance(parsed, ast.Expression):
        raise ValueError("Expression must parse as a Python expression.")

    return _canonicalize_node(parsed.body)


def skeletonize_constants(expression: str) -> str:
    stripped = expression.strip()
    if not stripped:
        raise ValueError("Expression string cannot be empty.")
    try:
        parsed = ast.parse(stripped, mode=_PARSE_MODE)
    except SyntaxError as exc:
        raise ValueError(f"Invalid IR syntax: {exc}") from exc
    if not isinstance(parsed, ast.Expression):
        raise ValueError("Expression must parse as a Python expression.")
    has_numeric_literal = any(
        isinstance(node, ast.expr) and _is_numeric_literal(node)
        for node in ast.walk(parsed.body)
    )
    if not has_numeric_literal:
        return expression
    return _skeletonize_node(parsed.body)


def _skeletonize_node(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        if node.keywords or not isinstance(node.func, ast.Name):
            raise ValueError(
                "Regression term calls require bare names and no keywords."
            )
        children = [_skeletonize_node(arg) for arg in node.args]
        return f"{node.func.id}({','.join(children)})"
    if _is_numeric_literal(node):
        return "const"
    raise ValueError(f"Unsupported regression term syntax: {type(node).__name__}.")


def _is_numeric_literal(node: ast.expr) -> bool:
    literal = node
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        literal = node.operand
    return (
        isinstance(literal, ast.Constant)
        and not isinstance(literal.value, bool)
        and isinstance(literal.value, (int, float))
    )


def _canonicalize_node(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return _canonicalize_call(node)
    if isinstance(node, ast.Constant):
        raise ValueError("Numeric constants are not supported in expression IR.")
    raise ValueError(f"Unsupported IR syntax: {type(node).__name__}.")


def _canonicalize_call(node: ast.Call) -> str:
    if node.keywords:
        raise ValueError("Keyword arguments are not allowed in IR.")
    if not isinstance(node.func, ast.Name):
        raise ValueError("IR calls must target bare token names.")

    op_name = node.func.id
    child_strs = [_canonicalize_node(arg) for arg in node.args]

    if op_name in _COMMUTATIVE_OPS:
        child_strs = sorted(child_strs)

    return f"{op_name}({','.join(child_strs)})"


__all__ = ["canonicalize_expression", "skeletonize_constants"]
