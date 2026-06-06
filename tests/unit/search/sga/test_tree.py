
from __future__ import annotations

import pytest




from kd.search.sga.tree import Node, Tree






class TestNodeConstruction:

    @pytest.mark.smoke
    def test_leaf_node_creation(self) -> None:
        leaf = Node(name="u", arity=0, children=[])
        assert leaf.name == "u"
        assert leaf.arity == 0
        assert leaf.children == []

    @pytest.mark.smoke
    def test_unary_node_creation(self) -> None:
        child = Node(name="u", arity=0, children=[])
        unary = Node(name="^2", arity=1, children=[child])
        assert unary.name == "^2"
        assert unary.arity == 1
        assert len(unary.children) == 1

    @pytest.mark.smoke
    def test_binary_node_creation(self) -> None:
        left = Node(name="u", arity=0, children=[])
        right = Node(name="x", arity=0, children=[])
        binary = Node(name="*", arity=2, children=[left, right])
        assert binary.name == "*"
        assert binary.arity == 2
        assert len(binary.children) == 2







class TestNodeIsLeaf:

    def test_leaf_returns_true(self) -> None:
        leaf = Node(name="x", arity=0, children=[])
        assert leaf.is_leaf is True

    def test_unary_returns_false(self) -> None:
        child = Node(name="u", arity=0, children=[])
        unary = Node(name="^2", arity=1, children=[child])
        assert unary.is_leaf is False

    def test_binary_returns_false(self) -> None:
        left = Node(name="u", arity=0, children=[])
        right = Node(name="x", arity=0, children=[])
        binary = Node(name="+", arity=2, children=[left, right])
        assert binary.is_leaf is False







class TestNodeDepth:

    def test_single_leaf_depth_zero(self) -> None:
        leaf = Node(name="u", arity=0, children=[])
        assert leaf.depth == 0

    def test_unary_chain_depth(self) -> None:
        u = Node(name="u", arity=0, children=[])
        cube = Node(name="^3", arity=1, children=[u])
        sq = Node(name="^2", arity=1, children=[cube])
        assert sq.depth == 2

    def test_binary_balanced_depth(self) -> None:
        left = Node(name="u", arity=0, children=[])
        right = Node(name="x", arity=0, children=[])
        add = Node(name="+", arity=2, children=[left, right])
        assert add.depth == 1

    def test_binary_unbalanced_depth(self) -> None:
        u = Node(name="u", arity=0, children=[])
        x = Node(name="x", arity=0, children=[])
        sq = Node(name="^2", arity=1, children=[u])
        mul = Node(name="*", arity=2, children=[sq, x])
        assert mul.depth == 2







class TestNodeSize:

    def test_single_leaf_size_one(self) -> None:
        leaf = Node(name="u", arity=0, children=[])
        assert leaf.size == 1

    def test_unary_size(self) -> None:
        u = Node(name="u", arity=0, children=[])
        sq = Node(name="^2", arity=1, children=[u])
        assert sq.size == 2

    def test_binary_size(self) -> None:
        left = Node(name="u", arity=0, children=[])
        right = Node(name="x", arity=0, children=[])
        add = Node(name="+", arity=2, children=[left, right])
        assert add.size == 3

    def test_nested_size(self) -> None:
        u1 = Node(name="u", arity=0, children=[])
        u2 = Node(name="u", arity=0, children=[])
        x = Node(name="x", arity=0, children=[])
        sq = Node(name="^2", arity=1, children=[u1])
        add = Node(name="+", arity=2, children=[x, u2])
        mul = Node(name="*", arity=2, children=[sq, add])
        assert mul.size == 6







class TestNodeStr:

    def test_leaf_str(self) -> None:
        leaf = Node(name="u", arity=0, children=[])
        assert str(leaf) == "u"

    def test_unary_str(self) -> None:
        u = Node(name="u", arity=0, children=[])
        sq = Node(name="^2", arity=1, children=[u])
        assert str(sq) == "^2 u"

    def test_binary_str(self) -> None:
        u = Node(name="u", arity=0, children=[])
        x = Node(name="x", arity=0, children=[])
        mul = Node(name="*", arity=2, children=[u, x])
        assert str(mul) == "* u x"

    def test_nested_str(self) -> None:
        u = Node(name="u", arity=0, children=[])
        x = Node(name="x", arity=0, children=[])
        sq = Node(name="^2", arity=1, children=[u])
        add = Node(name="+", arity=2, children=[sq, x])
        assert str(add) == "+ ^2 u x"







class TestNodeEq:

    def test_equal_leaves(self) -> None:
        a = Node(name="u", arity=0, children=[])
        b = Node(name="u", arity=0, children=[])
        assert a == b

    def test_different_name(self) -> None:
        a = Node(name="u", arity=0, children=[])
        b = Node(name="x", arity=0, children=[])
        assert a != b

    def test_equal_nested(self) -> None:

        def make_tree() -> Node:
            u = Node(name="u", arity=0, children=[])
            x = Node(name="x", arity=0, children=[])
            return Node(name="*", arity=2, children=[u, x])

        assert make_tree() == make_tree()

    def test_different_structure(self) -> None:
        u = Node(name="u", arity=0, children=[])
        x = Node(name="x", arity=0, children=[])
        mul_ux = Node(name="*", arity=2, children=[u, x])
        mul_xu = Node(name="*", arity=2, children=[x, u])
        assert mul_ux != mul_xu

    def test_not_equal_to_non_node(self) -> None:
        leaf = Node(name="u", arity=0, children=[])
        assert leaf != "u"
        assert leaf != 42







class TestNodePurity:

    def test_no_cache_attribute(self) -> None:
        leaf = Node(name="u", arity=0, children=[])
        assert not hasattr(leaf, "cache")
        assert not hasattr(leaf, "var")

    def test_only_expected_public_attributes(self) -> None:
        leaf = Node(name="u", arity=0, children=[])
        public_attrs = {a for a in dir(leaf) if not a.startswith("_")}

        expected_min = {"name", "arity", "children", "is_leaf", "depth", "size"}
        assert expected_min.issubset(public_attrs)







class TestTreeBasics:

    @pytest.mark.smoke
    def test_tree_creation(self) -> None:
        root = Node(name="u", arity=0, children=[])
        tree = Tree(root=root)
        assert tree.root is root

    def test_tree_depth_delegates(self) -> None:
        u = Node(name="u", arity=0, children=[])
        x = Node(name="x", arity=0, children=[])
        mul = Node(name="*", arity=2, children=[u, x])
        tree = Tree(root=mul)
        assert tree.depth == mul.depth

    def test_tree_size_delegates(self) -> None:
        u = Node(name="u", arity=0, children=[])
        x = Node(name="x", arity=0, children=[])
        mul = Node(name="*", arity=2, children=[u, x])
        tree = Tree(root=mul)
        assert tree.size == mul.size

    def test_tree_str_delegates(self) -> None:
        u = Node(name="u", arity=0, children=[])
        x = Node(name="x", arity=0, children=[])
        mul = Node(name="*", arity=2, children=[u, x])
        tree = Tree(root=mul)
        assert str(tree) == str(mul)







class TestTreeCopy:

    def test_copy_equals_original(self) -> None:
        u = Node(name="u", arity=0, children=[])
        x = Node(name="x", arity=0, children=[])
        mul = Node(name="*", arity=2, children=[u, x])
        original = Tree(root=mul)
        copied = original.copy()
        assert copied == original

    def test_copy_is_independent(self) -> None:
        u = Node(name="u", arity=0, children=[])
        x = Node(name="x", arity=0, children=[])
        mul = Node(name="*", arity=2, children=[u, x])
        original = Tree(root=mul)
        copied = original.copy()


        copied.root.children[0] = Node(name="t", arity=0, children=[])


        assert original.root.children[0].name == "u"

    def test_copy_root_is_different_object(self) -> None:
        leaf = Node(name="u", arity=0, children=[])
        original = Tree(root=leaf)
        copied = original.copy()
        assert copied.root is not original.root







class TestTreeEq:

    def test_equal_trees(self) -> None:
        def make() -> Tree:
            u = Node(name="u", arity=0, children=[])
            x = Node(name="x", arity=0, children=[])
            mul = Node(name="*", arity=2, children=[u, x])
            return Tree(root=mul)

        assert make() == make()

    def test_different_trees(self) -> None:
        u = Node(name="u", arity=0, children=[])
        x = Node(name="x", arity=0, children=[])
        tree_u = Tree(root=u)
        tree_x = Tree(root=x)
        assert tree_u != tree_x

    def test_not_equal_to_non_tree(self) -> None:
        leaf = Node(name="u", arity=0, children=[])
        tree = Tree(root=leaf)
        assert tree != "not a tree"







class TestNodeInvariants:

    def test_depth_geq_zero(self) -> None:
        leaf = Node(name="u", arity=0, children=[])
        assert leaf.depth >= 0

    def test_size_geq_one(self) -> None:
        leaf = Node(name="u", arity=0, children=[])
        assert leaf.size >= 1

    def test_size_equals_one_plus_children_sizes(self) -> None:
        u = Node(name="u", arity=0, children=[])
        x = Node(name="x", arity=0, children=[])
        sq = Node(name="^2", arity=1, children=[u])
        add = Node(name="+", arity=2, children=[sq, x])

        expected = 1 + sq.size + x.size
        assert add.size == expected

    def test_depth_is_one_plus_max_child_depth(self) -> None:
        u = Node(name="u", arity=0, children=[])
        x = Node(name="x", arity=0, children=[])
        sq = Node(name="^2", arity=1, children=[u])
        add = Node(name="+", arity=2, children=[sq, x])

        expected = 1 + max(sq.depth, x.depth)
        assert add.depth == expected

    def test_leaf_depth_zero_size_one(self) -> None:
        for name in ("u", "x", "t", "ux", "0"):
            leaf = Node(name=name, arity=0, children=[])
            assert leaf.depth == 0
            assert leaf.size == 1
