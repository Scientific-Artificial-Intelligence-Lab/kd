
from __future__ import annotations

import re
import warnings

from kd.search.sga.pde import PDE
from kd.search.sga.tree import Node, Tree



_NAME_MAP: dict[str, str] = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "^2": "n2",
    "^3": "n3",
}

_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


class KdExpression(str):

    def split(
        self,
        sep: str | None = None,
        maxsplit: int = -1,
    ) -> list[str]:
        if sep is not None:
            return super().split(sep, maxsplit)
        return _TOKEN_PATTERN.findall(self)


def _map_name(name: str) -> str:
    return _NAME_MAP.get(name, name)


def _axis_name(axis_node: Node) -> str:
    if not axis_node.is_leaf:
        raise ValueError("Derivative axis must be a leaf node")
    return _map_name(axis_node.name)


def _derivative_funcall(name: str, node: Node) -> str:
    if len(node.children) != 2:
        raise ValueError(f"Operator '{node.name}' requires exactly 2 children")
    axis = _axis_name(node.children[1])
    inner = _node_to_funcall(node.children[0])
    return f"{name}_{axis}({inner})"


def _node_to_funcall(node: Node) -> str:
    if node.name == "d" and len(node.children) == 2:
        return _derivative_funcall("diff", node)
    if node.name == "d^2" and len(node.children) == 2:
        return _derivative_funcall("diff2", node)
    if not node.children:
        return _map_name(node.name)
    children = ", ".join(_node_to_funcall(child) for child in node.children)
    return f"{_map_name(node.name)}({children})"


def tree_to_kd_expr(tree: Tree) -> str:
    return KdExpression(_node_to_funcall(tree.root))







def pde_to_kd_expr(
    pde: PDE,
    coefficients: list[float] | None = None,
) -> str:
    if coefficients is not None:
        warnings.warn(
            "coefficients is deprecated because it emits mul(<float>, term) "
            "executable IR outside the canonicalizable subset of "
            "kd.core.equation.canonical; equation coefficients belong in the "
            "separate Equation IR (Scalar).",
            DeprecationWarning,
            stacklevel=2,
        )

    if pde.width == 0:
        return KdExpression("")

    if coefficients is not None and len(coefficients) != pde.width:
        raise ValueError(
            f"Coefficient count ({len(coefficients)}) does not match "
            f"term count ({pde.width})"
        )

    term_strs = [tree_to_kd_expr(tree) for tree in pde.terms]

    if coefficients is not None:



        term_strs = [
            f"mul({float(coefficient)}, {term})"
            for coefficient, term in zip(coefficients, term_strs, strict=True)
        ]

    if len(term_strs) == 1:
        return KdExpression(term_strs[0])

    result = term_strs[-1]
    for term in reversed(term_strs[:-1]):
        result = f"add({term}, {result})"

    return KdExpression(result)
