
from __future__ import annotations

import ast
import logging

import sympy as sp

logger = logging.getLogger(__name__)

_DIFFERENTIAL_OPS: frozenset[str] = frozenset(
    {"diff_x", "diff_y", "diff_t", "diff2_x", "diff2_y", "diff2_t", "diff3_x"}
)
_POWER_OPS: dict[str, int] = {"n2": 2, "n3": 3}
_LAPLACIAN_TERMS: tuple[str, str] = ("diff2_x", "diff2_y")


def _node_to_sympy(node: ast.AST) -> sp.Expr:
    if isinstance(node, ast.Name):
        return sp.Symbol(node.id)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise ValueError(f"unsupported boolean: {node.value!r}")
        if isinstance(node.value, int):
            return sp.Integer(node.value)
        if isinstance(node.value, float):
            return sp.Float(node.value)
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_node_to_sympy(node.operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _node_to_sympy(node.operand)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        args = [_node_to_sympy(a) for a in node.args]
        if name == "neg" and len(args) == 1:
            return -args[0]
        if name == "add" and len(args) == 2:
            return args[0] + args[1]
        if name == "sub" and len(args) == 2:
            return args[0] - args[1]
        if name == "mul" and len(args) == 2:
            return args[0] * args[1]
        if name == "div" and len(args) == 2:
            return args[0] / args[1]
        if name in _POWER_OPS and len(args) == 1:
            return args[0] ** _POWER_OPS[name]
        if name in _DIFFERENTIAL_OPS:
            return sp.Function(name)(*args)
    raise ValueError(
        f"unsupported expression node: {ast.dump(node, annotate_fields=False)}"
    )


def parse_to_sympy(term: str) -> sp.Expr:
    try:
        tree = ast.parse(term, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"could not parse term {term!r}: {exc}") from exc
    return _node_to_sympy(tree)


def _laplacian_coefficient(
    expanded: sp.Expr, op_name: str, arg: sp.Symbol
) -> float | None:
    func = sp.Function(op_name)(arg)
    coef = expanded.coeff(func)
    if coef == 0:
        return None
    if not coef.is_number:



        return None
    try:
        return float(coef)
    except (TypeError, ValueError):
        return None


def check_diffusion_sign_consistency(expression: str) -> tuple[bool, str]:
    try:
        symbolic = parse_to_sympy(expression)
    except ValueError as exc:
        return (False, f"could not parse expression: {exc}")
    expanded = sp.expand(symbolic)
    u_sym = sp.Symbol("u")
    coef_xx = _laplacian_coefficient(expanded, _LAPLACIAN_TERMS[0], u_sym)
    coef_yy = _laplacian_coefficient(expanded, _LAPLACIAN_TERMS[1], u_sym)
    if coef_xx is None or coef_yy is None:




        return (True, "")
    if (coef_xx > 0 and coef_yy > 0) or (coef_xx < 0 and coef_yy < 0):
        return (True, "")
    return (
        False,
        f"diff2_x and diff2_y have opposite signs in expression "
        f"(coef_xx={coef_xx:+g}, coef_yy={coef_yy:+g}) — "
        f"sign-flip cheating signature; the SR engine wrote the "
        f"Laplacian as a difference rather than a sum.",
    )
