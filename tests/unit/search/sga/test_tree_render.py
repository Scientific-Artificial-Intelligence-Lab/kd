
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from kd.search.sga.pde import PDE
from kd.search.sga.tree import Node, Tree
from kd.search.sga.tree_render import (
    GENOME_TREE_INFO,
    genome_tree_data,
    genome_tree_info,
    render_genome_tree,
    sga_node_to_render,
)


def _mul_term() -> Tree:
    return Tree(
        Node("*", 2, [Node("u", 0), Node("d", 2, [Node("u", 0), Node("x", 0)])])
    )


def _square_term() -> Tree:
    return Tree(Node("^2", 1, [Node("u", 0)]))


def _labels(ax) -> set[str]:
    return {t.get_text() for t in ax.texts}


class TestSgaNodeToRender:
    def test_maps_name_and_children(self) -> None:
        node = sga_node_to_render(_mul_term().root)
        assert node.label == "*"
        assert node.kind == "op"
        assert {c.label for c in node.children} == {"u", "d"}

    def test_leaf_is_var(self) -> None:
        node = sga_node_to_render(Node("u", 0))
        assert node.label == "u"
        assert node.kind == "var"
        assert node.children == ()

    def test_preserves_raw_genome_names(self) -> None:
        sq = sga_node_to_render(_square_term().root)
        assert sq.label == "^2"
        assert [c.label for c in sq.children] == ["u"]
        d_node = sga_node_to_render(Node("d", 2, [Node("u", 0), Node("x", 0)]))
        assert d_node.label == "d"
        assert d_node.kind == "deriv"


class TestRenderGenomeTree:
    def test_multi_term_virtual_root(self) -> None:
        pde = PDE([_mul_term(), _square_term()])
        fig, ax = plt.subplots()
        warnings = render_genome_tree(ax, pde)
        assert warnings == []

        assert _labels(ax) == {"+", "*", "u", "d", "x", "^2"}


        assert len(ax.texts) == 8
        plt.close(fig)

    def test_single_term_no_virtual_root(self) -> None:
        pde = PDE([_mul_term()])
        fig, ax = plt.subplots()
        render_genome_tree(ax, pde)
        assert "+" not in _labels(ax)
        plt.close(fig)

    def test_none_pde_degrades_gracefully(self) -> None:
        fig, ax = plt.subplots()
        warnings = render_genome_tree(ax, None)
        assert len(warnings) > 0
        assert len(ax.get_lines()) == 0
        plt.close(fig)

    def test_empty_pde_degrades_gracefully(self) -> None:
        fig, ax = plt.subplots()
        warnings = render_genome_tree(ax, PDE([]))
        assert len(warnings) > 0
        plt.close(fig)


class TestGenomeTreeData:
    def test_nested_dict_is_json_safe(self) -> None:
        import json

        data = genome_tree_data(PDE([_mul_term()]))
        assert data["available"] is True

        json.dumps(data)
        tree = data["tree"]
        assert tree["label"] == "*"
        assert {c["label"] for c in tree["children"]} == {"u", "d"}

    def test_none_pde_marks_unavailable(self) -> None:
        data = genome_tree_data(None)
        assert data["available"] is False
        assert data["tree"] is None


def test_genome_tree_info_descriptor() -> None:
    assert GENOME_TREE_INFO.name == "genome_tree"
    assert GENOME_TREE_INFO.title
    assert GENOME_TREE_INFO.description


def test_genome_tree_info_returns_fresh_copies() -> None:
    first = genome_tree_info()
    first.title = "MUTATED"
    assert genome_tree_info().title != "MUTATED"
    assert GENOME_TREE_INFO.title != "MUTATED"


@pytest.mark.smoke
def test_smoke_render_genome_tree() -> None:
    fig, ax = plt.subplots()
    render_genome_tree(ax, PDE([_mul_term()]))
    plt.close(fig)
