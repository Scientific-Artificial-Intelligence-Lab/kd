
from __future__ import annotations

import ast

from kd.core.equation.canonical import canonicalize_expression

_INFIX_REFUSALS: dict[type[ast.operator], str] = {
    ast.Add: "use separate terms or explicit add(...)",
    ast.Sub: "use separate terms or explicit sub(...)",
    ast.Div: "infix '/' is not supported; use div(...)",
    ast.BitXor: "'^' is not supported; use '**'",
}


def lower_term(text: str) -> str:
    if "*" not in text:
        return text
    tree = _parse(text)
    if not any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult | ast.Pow)
        for node in ast.walk(tree)
    ):
        return text
    return _ir(tree)


def shorthand_term(text: str) -> str:
    tree = _parse(text)
    term = _ir(tree)
    _refuse_literals(tree)
    return canonicalize_expression(term)


def _parse(text: str) -> ast.expr:
    try:
        return ast.parse(text.strip(), mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"not a term expression ({exc.msg})") from exc


def _refuse_literals(node: ast.expr) -> None:
    if isinstance(node, ast.Constant):
        raise ValueError(
            "numbers appear only as power exponents; coefficients belong in "
            "pinned values and the constant column is 'one'"
        )
    if isinstance(node, ast.UnaryOp):
        raise ValueError("unary signs are not supported; use neg(...)")
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        _refuse_literals(node.left)
        return
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr):
            _refuse_literals(child)


def _ir(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if _is_number(node):
        return ast.unparse(node)
    if isinstance(node, ast.Call):
        return _call_ir(node)
    if isinstance(node, ast.BinOp):
        return _binary_ir(node)
    if isinstance(node, ast.UnaryOp):
        raise ValueError("a unary sign applies only to a number; use neg(...)")
    raise ValueError(f"{type(node).__name__} syntax is not supported")


def _is_number(node: ast.expr) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        node = node.operand
    return isinstance(node, ast.Constant) and type(node.value) in (int, float)


def _call_ir(node: ast.Call) -> str:
    if not isinstance(node.func, ast.Name):
        raise ValueError("a call must name its operator directly")
    if node.keywords:
        raise ValueError("keyword arguments are not supported")
    arguments = ",".join(_ir(argument) for argument in node.args)
    return f"{node.func.id}({arguments})"


def _binary_ir(node: ast.BinOp) -> str:
    if isinstance(node.op, ast.Mult):
        return f"mul({_ir(node.left)},{_ir(node.right)})"
    if isinstance(node.op, ast.Pow):
        return _power_ir(_ir(node.left), node.right)
    reason = _INFIX_REFUSALS.get(type(node.op))
    if reason is None:
        reason = f"operator {type(node.op).__name__} is not supported"
    raise ValueError(reason)


def _power_ir(base: str, exponent: ast.expr) -> str:
    if not (
        isinstance(exponent, ast.Constant)
        and type(exponent.value) is int
        and exponent.value >= 1
    ):
        raise ValueError("power exponent must be a positive integer")
    product = base
    for _ in range(exponent.value - 1):
        product = f"mul({product},{base})"
    return product


__all__ = ["lower_term", "shorthand_term"]
