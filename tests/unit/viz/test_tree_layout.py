
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from kd.viz.tree_layout import (
    RenderNode,
    draw_tree,
    forest_to_render,
    layout,
)


class TestLayoutCoordinates:

    def test_single_leaf(self) -> None:
        placed, edges = layout(RenderNode("u", kind="var"))
        assert len(placed) == 1
        assert len(edges) == 0
        node = placed[0]
        assert node.label == "u"
        assert (node.x, node.y, node.depth) == (0.0, 0.0, 0)

    def test_balanced_binary(self) -> None:
        a = RenderNode("a", kind="var")
        b = RenderNode("b", kind="var")
        root = RenderNode("+", (a, b))
        placed, edges = layout(root)

        assert len(placed) == 3
        by_label = {p.label: p for p in placed}

        assert (by_label["a"].x, by_label["a"].depth) == (0.0, 1)
        assert (by_label["b"].x, by_label["b"].depth) == (1.0, 1)
        assert (by_label["+"].x, by_label["+"].depth) == (0.5, 0)

        assert by_label["+"].y == 0.0
        assert by_label["a"].y == -1.0

        assert len(edges) == 2
        assert (0.5, 0.0, 0.0, -1.0) in edges
        assert (0.5, 0.0, 1.0, -1.0) in edges

    def test_unbalanced_left_deep(self) -> None:

        inner = RenderNode(
            "*", (RenderNode("u", kind="var"), RenderNode("u_x", kind="var"))
        )
        root = RenderNode("+", (inner, RenderNode("c", kind="const")))
        placed, _ = layout(root)
        by_label = {p.label: p for p in placed}
        assert by_label["u"].x == 0.0
        assert by_label["u_x"].x == 1.0
        assert by_label["*"].x == 0.5
        assert by_label["c"].x == 2.0
        assert by_label["+"].x == 1.25
        assert max(p.depth for p in placed) == 2

    def test_shared_equal_subtree_not_collapsed(self) -> None:
        leaf = RenderNode("u", kind="var")
        root = RenderNode("*", (leaf, leaf))
        placed, edges = layout(root)
        assert len(placed) == 3
        xs = sorted(p.x for p in placed if p.label == "u")
        assert xs == [0.0, 1.0]
        assert len(edges) == 2


        assert edges[0] != edges[1]


class TestForestToRender:

    def test_single_root_no_wrap(self) -> None:
        only = RenderNode("u_x", kind="var")
        assert forest_to_render([only]) is only

    def test_multi_root_wraps_in_virtual_op(self) -> None:
        roots = [RenderNode("u_x", kind="var"), RenderNode("u_xx", kind="var")]
        combined = forest_to_render(roots, op="+")
        assert combined.label == "+"
        assert combined.kind == "op"
        assert combined.children == tuple(roots)

    def test_empty_forest_is_zero(self) -> None:
        node = forest_to_render([])
        assert node.label == "0"
        assert node.children == ()
        assert node.kind == "const"


class TestDrawTree:

    def test_draws_nodes_and_edges(self) -> None:
        a = RenderNode("a", kind="var")
        b = RenderNode("b", kind="var")
        fig, ax = plt.subplots()
        warnings = draw_tree(RenderNode("+", (a, b)), ax)
        assert warnings == []

        assert len(ax.texts) == 3
        assert len(ax.get_lines()) >= 2
        assert not ax.axison
        plt.close(fig)

    def test_single_leaf_no_edges(self) -> None:
        fig, ax = plt.subplots()
        warnings = draw_tree(RenderNode("u", kind="var"), ax)
        assert warnings == []
        assert len(ax.texts) == 1
        assert len(ax.get_lines()) == 0
        plt.close(fig)

    def test_deep_chain_renders_without_recursion_error(self) -> None:
        node = RenderNode("u", kind="var")
        for _ in range(40):
            node = RenderNode("^2", (node,), kind="op")
        fig, ax = plt.subplots()
        warnings = draw_tree(node, ax)
        assert warnings == []
        assert len(ax.texts) == 41
        plt.close(fig)

    def test_large_tree_warns_but_draws(self) -> None:

        leaves = tuple(RenderNode(f"x{i}", kind="var") for i in range(80))
        fig, ax = plt.subplots()
        warnings = draw_tree(RenderNode("+", leaves), ax)
        assert any("node" in w.lower() for w in warnings)
        assert len(ax.texts) == 81
        plt.close(fig)


@pytest.mark.smoke
def test_smoke_import_and_draw() -> None:
    fig, ax = plt.subplots()
    draw_tree(RenderNode("u", kind="var"), ax)
    plt.close(fig)
