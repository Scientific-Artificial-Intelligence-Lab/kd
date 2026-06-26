
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from kd.viz.extension import PlotInfo
from kd.viz.tree_layout import RenderNode, draw_tree, forest_to_render

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.sga.pde import PDE
    from kd.search.sga.tree import Node


_DERIV_NAMES = frozenset({"d", "d^2"})

_UNAVAILABLE_TEXT = "Native genome unavailable (see Expression Tree)"

GENOME_TREE_INFO = PlotInfo(
    name="genome_tree",
    title="Genome Tree",
    description=(
        "Raw GP genome of the best individual — operator names (*, ^2, d) as "
        "the genetic operators produced them, pre-canonicalization. Shows all "
        "evolved terms before STRidge pruning, so its term count may exceed the "
        "discovered equation's (Expression Tree). Unavailable for results loaded "
        "from disk (no live population)."
    ),
)


def genome_tree_info() -> PlotInfo:
    return dataclasses.replace(GENOME_TREE_INFO)


def sga_node_to_render(node: Node) -> RenderNode:
    if node.is_leaf:
        return RenderNode(node.name, kind="var")
    kind = "deriv" if node.name in _DERIV_NAMES else "op"
    children = tuple(sga_node_to_render(child) for child in node.children)
    return RenderNode(node.name, children, kind=kind)


def _pde_to_render(pde: PDE) -> RenderNode:
    roots = [sga_node_to_render(tree.root) for tree in pde.terms]
    return forest_to_render(roots, op="+")


def _draw_unavailable(ax: Axes) -> list[str]:
    ax.axis("off")
    ax.text(
        0.5, 0.5, _UNAVAILABLE_TEXT, transform=ax.transAxes, ha="center", va="center"
    )
    ax.set_title("Genome Tree")
    return [_UNAVAILABLE_TEXT]


def render_genome_tree(ax: Axes, pde: PDE | None) -> list[str]:
    if pde is None or pde.width == 0:
        return _draw_unavailable(ax)
    warnings = draw_tree(_pde_to_render(pde), ax)
    ax.set_title("Genome Tree")
    return warnings


def _render_node_to_dict(node: RenderNode) -> dict[str, Any]:
    return {
        "label": node.label,
        "kind": node.kind,
        "children": [_render_node_to_dict(child) for child in node.children],
    }


def genome_tree_data(pde: PDE | None) -> dict[str, Any]:
    if pde is None or pde.width == 0:
        return {"available": False, "tree": None}
    return {"available": True, "tree": _render_node_to_dict(_pde_to_render(pde))}


__all__ = [
    "GENOME_TREE_INFO",
    "genome_tree_data",
    "genome_tree_info",
    "render_genome_tree",
    "sga_node_to_render",
]
