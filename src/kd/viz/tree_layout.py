
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from matplotlib.colors import to_rgba

from kd.viz.style import AXIS, GRID, SERIES

if TYPE_CHECKING:
    from matplotlib.axes import Axes


_MAX_NODES = 60


_LEAF_FILL_ALPHA = 0.18


def _leaf_style(colour: str) -> dict[str, Any]:
    return {"facecolor": to_rgba(colour, _LEAF_FILL_ALPHA), "edgecolor": colour}



_KIND_STYLE: dict[str, dict[str, Any]] = {
    "op": {"facecolor": GRID, "edgecolor": AXIS},
    "var": _leaf_style(SERIES[0]),
    "const": _leaf_style(SERIES[3]),
    "deriv": _leaf_style(SERIES[2]),
}

_EDGE_COLOR = AXIS
_FONT_SIZE = 9
_X_MARGIN = 0.6
_Y_MARGIN = 0.4


@dataclass(frozen=True)
class RenderNode:

    label: str
    children: tuple[RenderNode, ...] = ()
    kind: str = "op"


@dataclass(frozen=True)
class _Placed:

    label: str
    kind: str
    x: float
    y: float
    depth: int



_Edge = tuple[float, float, float, float]


@dataclass
class _Layout:

    placed: list[_Placed] = field(default_factory=list)
    edges: list[_Edge] = field(default_factory=list)
    next_leaf_x: int = 0


def layout(root: RenderNode) -> tuple[list[_Placed], list[_Edge]]:
    state = _Layout()
    _visit(root, 0, state)
    return state.placed, state.edges


def _visit(node: RenderNode, depth: int, state: _Layout) -> _Placed:
    child_placements = [_visit(child, depth + 1, state) for child in node.children]
    if child_placements:
        x = sum(cp.x for cp in child_placements) / len(child_placements)
    else:
        x = float(state.next_leaf_x)
        state.next_leaf_x += 1
    y = float(-depth)
    placed = _Placed(label=node.label, kind=node.kind, x=x, y=y, depth=depth)
    state.placed.append(placed)
    for cp in child_placements:
        state.edges.append((x, y, cp.x, cp.y))
    return placed


def forest_to_render(roots: Sequence[RenderNode], op: str = "+") -> RenderNode:
    if not roots:
        return RenderNode("0", kind="const")
    if len(roots) == 1:
        return roots[0]
    return RenderNode(op, tuple(roots), kind="op")


def draw_tree(root: RenderNode, ax: Axes) -> list[str]:
    warnings: list[str] = []
    placed, edges = layout(root)
    if len(placed) > _MAX_NODES:
        warnings.append(
            f"Tree has {len(placed)} nodes (>{_MAX_NODES}); layout may be crowded."
        )

    for x0, y0, x1, y1 in edges:
        ax.plot([x0, x1], [y0, y1], color=_EDGE_COLOR, linewidth=0.8, zorder=1)

    for node in placed:
        style = _KIND_STYLE.get(node.kind, _KIND_STYLE["op"])
        ax.text(
            node.x,
            node.y,
            node.label,
            ha="center",
            va="center",
            fontsize=_FONT_SIZE,
            zorder=2,
            bbox={"boxstyle": "round,pad=0.3", **style},
        )

    xs = [p.x for p in placed]
    ys = [p.y for p in placed]
    ax.set_xlim(min(xs) - _X_MARGIN, max(xs) + _X_MARGIN)
    ax.set_ylim(min(ys) - _Y_MARGIN, max(ys) + _Y_MARGIN)
    ax.axis("off")
    return warnings


__all__ = ["RenderNode", "draw_tree", "forest_to_render", "layout"]
