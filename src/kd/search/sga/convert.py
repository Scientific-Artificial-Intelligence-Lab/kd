
from __future__ import annotations

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
    return _node_to_funcall(tree.root)







def pde_to_kd_expr(pde: PDE) -> str:
    if pde.width == 0:
        return ""

    term_strs = [tree_to_kd_expr(tree) for tree in pde.terms]

    if len(term_strs) == 1:
        return term_strs[0]

    result = term_strs[-1]
    for term in reversed(term_strs[:-1]):
        result = f"add({term}, {result})"

    return result
