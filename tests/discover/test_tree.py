
from __future__ import annotations

import numpy as np
import pytest


from kd.search.discover.core.tree import (
    ExpressionTree,
    TreeNode,
    finish_tokens,
    max_diff_order,
    natural_length,
    trim_to_natural,
)
from kd.search.discover.tokens.library import Library, Token, TokenType






def _make_burgers_full_library() -> Library:
    tokens = [
        Token(name="x1", arity=0, token_type=TokenType.COORDINATE),
        Token(name="u1", arity=0, token_type=TokenType.TERMINAL),
        Token(name="add", arity=2, token_type=TokenType.OPERATOR),
        Token(name="mul", arity=2, token_type=TokenType.OPERATOR),
        Token(name="div", arity=2, token_type=TokenType.OPERATOR),
        Token(name="diff", arity=1, token_type=TokenType.OPERATOR),
        Token(name="diff2", arity=1, token_type=TokenType.OPERATOR),
        Token(name="diff3", arity=1, token_type=TokenType.OPERATOR),
        Token(name="n2", arity=1, token_type=TokenType.OPERATOR),
        Token(name="n3", arity=1, token_type=TokenType.OPERATOR),
    ]
    return Library(tokens)


def _make_library_with_sub() -> Library:
    tokens = [
        Token(name="x1", arity=0, token_type=TokenType.COORDINATE),
        Token(name="u1", arity=0, token_type=TokenType.TERMINAL),
        Token(name="add", arity=2, token_type=TokenType.OPERATOR),
        Token(name="sub", arity=2, token_type=TokenType.OPERATOR),
        Token(name="mul", arity=2, token_type=TokenType.OPERATOR),
    ]
    return Library(tokens)



X1, U1, ADD, MUL, DIV, DIFF, DIFF2, DIFF3, N2, N3 = range(10)







class TestTreeCrossValidation:

    TREE_CASES = [
        "single_terminal",
        "unary_n2",
        "binary_add",
        "burgers_gt",
        "deeply_nested",
        "nested_add",
    ]

    @pytest.mark.equivalence
    @pytest.mark.parametrize("case_name", TREE_CASES)
    def test_preorder_roundtrip(
        self, tree_fixture: dict[str, np.ndarray], case_name: str
    ) -> None:
        lib = _make_burgers_full_library()
        tokens = tree_fixture[f"{case_name}__tokens"].tolist()
        expected_rt = tree_fixture[f"{case_name}__roundtrip"]

        tree = ExpressionTree.from_preorder(tokens, lib)
        result = tree.to_preorder()

        np.testing.assert_array_equal(result, expected_rt)

    @pytest.mark.equivalence
    @pytest.mark.parametrize("case_name", TREE_CASES)
    def test_depth_matches_reference(
        self, tree_fixture: dict[str, np.ndarray], case_name: str
    ) -> None:
        lib = _make_burgers_full_library()
        tokens = tree_fixture[f"{case_name}__tokens"].tolist()
        expected_depth = int(tree_fixture[f"{case_name}__depth"])

        tree = ExpressionTree.from_preorder(tokens, lib)

        assert tree.depth() == expected_depth

    @pytest.mark.equivalence
    @pytest.mark.parametrize("case_name", TREE_CASES)
    def test_n_terms_matches_reference(
        self, tree_fixture: dict[str, np.ndarray], case_name: str
    ) -> None:
        lib = _make_burgers_full_library()
        tokens = tree_fixture[f"{case_name}__tokens"].tolist()
        expected_n_terms = int(tree_fixture[f"{case_name}__n_terms"])

        tree = ExpressionTree.from_preorder(tokens, lib)

        assert tree.n_terms() == expected_n_terms


class TestFinishTokensCrossValidation:

    FT_CASES = [
        "complete",
        "add_only",
        "add_one_child",
        "overlong",
        "unary_incomplete",
        "nested_incomplete",
    ]

    @pytest.mark.equivalence
    @pytest.mark.parametrize("case_name", FT_CASES)
    def test_finish_tokens_matches_reference(
        self, tree_fixture: dict[str, np.ndarray], case_name: str
    ) -> None:
        lib = _make_burgers_full_library()
        inp = tree_fixture[f"ft_{case_name}__input"].tolist()
        expected = tree_fixture[f"ft_{case_name}__output"]

        result = finish_tokens(inp, lib)

        np.testing.assert_array_equal(result, expected)







class TestTreeNode:

    @pytest.mark.unit
    def test_single_terminal(self) -> None:
        lib = _make_burgers_full_library()
        tree = ExpressionTree.from_preorder([U1], lib)

        assert tree.root.token == lib[U1]
        assert tree.root.children == []
        assert tree.root.parent is None

    @pytest.mark.unit
    def test_unary_operator(self) -> None:
        lib = _make_burgers_full_library()

        tree = ExpressionTree.from_preorder([N2, U1], lib)

        assert tree.root.token.name == "n2"
        assert len(tree.root.children) == 1
        assert tree.root.children[0].token.name == "u1"
        assert tree.root.children[0].parent is tree.root

    @pytest.mark.unit
    def test_binary_operator(self) -> None:
        lib = _make_burgers_full_library()
        tree = ExpressionTree.from_preorder([ADD, X1, U1], lib)

        assert tree.root.token.name == "add"
        assert len(tree.root.children) == 2
        assert tree.root.children[0].token.name == "x1"
        assert tree.root.children[1].token.name == "u1"

    @pytest.mark.unit
    def test_nested_burgers(self) -> None:
        lib = _make_burgers_full_library()
        tree = ExpressionTree.from_preorder(
            [ADD, MUL, U1, DIFF, U1, DIFF2, U1], lib
        )


        assert tree.root.token.name == "add"
        assert len(tree.root.children) == 2


        mul_node = tree.root.children[0]
        assert mul_node.token.name == "mul"
        assert len(mul_node.children) == 2
        assert mul_node.children[0].token.name == "u1"


        diff_node = mul_node.children[1]
        assert diff_node.token.name == "diff"
        assert len(diff_node.children) == 1
        assert diff_node.children[0].token.name == "u1"


        diff2_node = tree.root.children[1]
        assert diff2_node.token.name == "diff2"
        assert len(diff2_node.children) == 1
        assert diff2_node.children[0].token.name == "u1"







class TestExpressionTree:

    @pytest.mark.unit
    def test_depth_single(self) -> None:
        lib = _make_burgers_full_library()
        tree = ExpressionTree.from_preorder([U1], lib)
        assert tree.depth() == 1

    @pytest.mark.unit
    def test_n_nodes_matches_tokens(self) -> None:
        lib = _make_burgers_full_library()
        tokens = [ADD, MUL, U1, DIFF, U1, DIFF2, U1]
        tree = ExpressionTree.from_preorder(tokens, lib)
        assert tree.n_nodes() == len(tokens)

    @pytest.mark.unit
    def test_to_tokens_returns_token_objects(self) -> None:
        lib = _make_burgers_full_library()
        tokens = [ADD, X1, U1]
        tree = ExpressionTree.from_preorder(tokens, lib)

        result = tree.to_tokens()

        assert len(result) == 3
        assert all(isinstance(t, Token) for t in result)
        assert result[0].name == "add"
        assert result[1].name == "x1"
        assert result[2].name == "u1"

    @pytest.mark.unit
    def test_is_complete_true(self) -> None:
        lib = _make_burgers_full_library()
        tree = ExpressionTree.from_preorder([ADD, X1, U1], lib)
        assert tree.is_complete()

    @pytest.mark.unit
    def test_is_complete_false_missing_children(self) -> None:
        lib = _make_burgers_full_library()

        node = TreeNode(token=lib["add"], children=[], parent=None)
        tree = ExpressionTree(root=node)
        assert not tree.is_complete()

    @pytest.mark.unit
    def test_n_terms_no_add_or_sub(self) -> None:
        lib = _make_burgers_full_library()

        tree = ExpressionTree.from_preorder([MUL, U1, X1], lib)
        assert tree.n_terms() == 1

    @pytest.mark.unit
    def test_n_terms_with_sub(self) -> None:
        lib = _make_library_with_sub()
        sub_idx = lib.name_to_index("sub")
        x1_idx = lib.name_to_index("x1")
        u1_idx = lib.name_to_index("u1")

        tree = ExpressionTree.from_preorder([sub_idx, u1_idx, x1_idx], lib)
        assert tree.n_terms() == 2

    @pytest.mark.unit
    def test_n_terms_add_nested_under_mul(self) -> None:
        lib = _make_burgers_full_library()

        tree = ExpressionTree.from_preorder([MUL, ADD, X1, U1, U1], lib)
        assert tree.n_terms() == 1

    @pytest.mark.unit
    def test_n_terms_nested_sub(self) -> None:
        lib = _make_library_with_sub()
        sub_idx = lib.name_to_index("sub")
        x1_idx = lib.name_to_index("x1")
        u1_idx = lib.name_to_index("u1")

        tree = ExpressionTree.from_preorder(
            [sub_idx, sub_idx, x1_idx, u1_idx, u1_idx], lib
        )
        assert tree.n_terms() == 3

    @pytest.mark.unit
    def test_is_complete_false_too_many_children(self) -> None:
        lib = _make_burgers_full_library()

        child = TreeNode(token=lib["x1"], children=[], parent=None)
        node = TreeNode(token=lib["u1"], children=[child], parent=None)
        tree = ExpressionTree(root=node)
        assert not tree.is_complete()

    @pytest.mark.unit
    def test_from_preorder_invalid_index_raises(self) -> None:
        lib = _make_burgers_full_library()
        with pytest.raises((KeyError, IndexError, ValueError)):
            ExpressionTree.from_preorder([999], lib)

    @pytest.mark.unit
    def test_from_preorder_incomplete_raises(self) -> None:
        lib = _make_burgers_full_library()
        with pytest.raises((ValueError, IndexError)):
            ExpressionTree.from_preorder([ADD], lib)







def _build_unary_chain_manual(lib: Library, depth: int) -> TreeNode:
    n2_tok = lib["n2"]
    leaf = TreeNode(token=lib["u1"])
    node = leaf
    for _ in range(depth):
        new_node = TreeNode(token=n2_tok)
        new_node.children = [node]
        node.parent = new_node
        node = new_node
    return node


def _build_left_leaning_add_chain_manual(lib: Library, depth: int) -> TreeNode:
    add_tok = lib["add"]
    u1_tok = lib["u1"]
    node: TreeNode = TreeNode(token=u1_tok)
    for _ in range(depth):
        right = TreeNode(token=u1_tok)
        new_node = TreeNode(token=add_tok)
        new_node.children = [node, right]
        node.parent = new_node
        right.parent = new_node
        node = new_node
    return node




_DEEP_CHAIN = 2000


class TestTreeDepthGuards:

    @pytest.mark.unit
    def test_depth_handles_deep_unary_chain(self) -> None:
        lib = _make_burgers_full_library()
        root = _build_unary_chain_manual(lib, _DEEP_CHAIN)
        tree = ExpressionTree(root=root, _library=lib)
        assert tree.depth() == _DEEP_CHAIN + 1

    @pytest.mark.unit
    def test_is_complete_handles_deep_unary_chain(self) -> None:
        lib = _make_burgers_full_library()
        root = _build_unary_chain_manual(lib, _DEEP_CHAIN)
        tree = ExpressionTree(root=root, _library=lib)
        assert tree.is_complete()

    @pytest.mark.unit
    def test_n_terms_handles_deep_add_chain(self) -> None:
        lib = _make_burgers_full_library()
        root = _build_left_leaning_add_chain_manual(lib, _DEEP_CHAIN)
        tree = ExpressionTree(root=root, _library=lib)

        assert tree.n_terms() == _DEEP_CHAIN + 1

    @pytest.mark.unit
    def test_n_nodes_handles_deep_unary_chain(self) -> None:
        lib = _make_burgers_full_library()
        root = _build_unary_chain_manual(lib, _DEEP_CHAIN)
        tree = ExpressionTree(root=root, _library=lib)
        assert tree.n_nodes() == _DEEP_CHAIN + 1

    @pytest.mark.unit
    def test_from_preorder_handles_deep_unary_chain(self) -> None:
        lib = _make_burgers_full_library()
        tokens = [N2] * _DEEP_CHAIN + [U1]
        tree = ExpressionTree.from_preorder(tokens, lib)
        assert tree.is_complete()
        assert tree.depth() == _DEEP_CHAIN + 1

    @pytest.mark.unit
    def test_from_preorder_handles_deep_add_chain(self) -> None:
        lib = _make_burgers_full_library()


        tokens = [ADD] * _DEEP_CHAIN + [U1] * (_DEEP_CHAIN + 1)
        tree = ExpressionTree.from_preorder(tokens, lib)
        assert tree.is_complete()
        assert tree.n_terms() == _DEEP_CHAIN + 1







class TestCountTermsMalformedAdditive:

    @pytest.mark.unit
    def test_add_with_zero_children_raises(self) -> None:
        lib = _make_burgers_full_library()
        root = TreeNode(token=lib["add"], children=[])
        tree = ExpressionTree(root=root, _library=lib)
        with pytest.raises(ValueError, match="add"):
            tree.n_terms()

    @pytest.mark.unit
    def test_add_with_one_child_raises(self) -> None:
        lib = _make_burgers_full_library()
        root = TreeNode(token=lib["add"])
        leaf = TreeNode(token=lib["u1"], parent=root)
        root.children = [leaf]
        tree = ExpressionTree(root=root, _library=lib)
        with pytest.raises(ValueError, match="add"):
            tree.n_terms()

    @pytest.mark.unit
    def test_add_with_three_children_raises(self) -> None:
        lib = _make_burgers_full_library()
        root = TreeNode(token=lib["add"])
        children = [TreeNode(token=lib["u1"], parent=root) for _ in range(3)]
        root.children = children
        tree = ExpressionTree(root=root, _library=lib)
        with pytest.raises(ValueError, match="add"):
            tree.n_terms()

    @pytest.mark.unit
    def test_sub_with_one_child_raises(self) -> None:
        lib = _make_library_with_sub()
        root = TreeNode(token=lib["sub"])
        leaf = TreeNode(token=lib["u1"], parent=root)
        root.children = [leaf]
        tree = ExpressionTree(root=root, _library=lib)
        with pytest.raises(ValueError, match="sub"):
            tree.n_terms()

    @pytest.mark.unit
    def test_non_additive_arity_mismatch_is_ignored(self) -> None:
        lib = _make_burgers_full_library()
        root = TreeNode(token=lib["mul"], children=[])
        tree = ExpressionTree(root=root, _library=lib)

        assert tree.n_terms() == 1







class TestFinishTokens:

    @pytest.mark.unit
    def test_already_complete_unchanged(self) -> None:
        lib = _make_burgers_full_library()
        result = finish_tokens([ADD, X1, U1], lib)
        np.testing.assert_array_equal(result, [ADD, X1, U1])

    @pytest.mark.unit
    def test_empty_returns_single_terminal(self) -> None:
        lib = _make_burgers_full_library()
        result = finish_tokens([], lib)
        np.testing.assert_array_equal(result, [X1])

    @pytest.mark.unit
    def test_completed_forms_valid_expression(self) -> None:
        lib = _make_burgers_full_library()

        completed = finish_tokens([ADD], lib)

        tree = ExpressionTree.from_preorder(list(completed), lib)
        assert tree.is_complete()

    @pytest.mark.unit
    def test_truncation_preserves_first_expression(self) -> None:
        lib = _make_burgers_full_library()

        result = finish_tokens([U1, X1, ADD], lib)
        np.testing.assert_array_equal(result, [U1])







class TestTreeInvariants:

    VALID_TOKEN_SEQS = [
        [U1],
        [N2, U1],
        [ADD, X1, U1],
        [DIV, X1, U1],
        [ADD, MUL, U1, DIFF, U1, DIFF2, U1],
        [MUL, N2, DIFF, U1, DIFF3, N3, U1],
        [ADD, ADD, X1, U1, U1],
        [N2, N3, DIFF, DIFF2, U1],
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("tokens", VALID_TOKEN_SEQS)
    def test_preorder_roundtrip_identity(self, tokens: list[int]) -> None:
        lib = _make_burgers_full_library()
        tree = ExpressionTree.from_preorder(tokens, lib)
        np.testing.assert_array_equal(tree.to_preorder(), tokens)

    @pytest.mark.unit
    @pytest.mark.parametrize("tokens", VALID_TOKEN_SEQS)
    def test_n_nodes_equals_preorder_length(self, tokens: list[int]) -> None:
        lib = _make_burgers_full_library()
        tree = ExpressionTree.from_preorder(tokens, lib)
        assert tree.n_nodes() == len(tree.to_preorder())

    @pytest.mark.unit
    @pytest.mark.parametrize("tokens", VALID_TOKEN_SEQS)
    def test_parent_references_consistent(self, tokens: list[int]) -> None:
        lib = _make_burgers_full_library()
        tree = ExpressionTree.from_preorder(tokens, lib)

        def check_parent(node: TreeNode, expected_parent: TreeNode | None) -> None:
            assert node.parent is expected_parent, (
                f"Node {node.token.name}: parent should be "
                f"{expected_parent.token.name if expected_parent else None}, "
                f"got {node.parent.token.name if node.parent else None}"
            )
            for child in node.children:
                check_parent(child, node)

        check_parent(tree.root, None)

    @pytest.mark.unit
    @pytest.mark.parametrize("tokens", VALID_TOKEN_SEQS)
    def test_all_complete(self, tokens: list[int]) -> None:
        lib = _make_burgers_full_library()
        tree = ExpressionTree.from_preorder(tokens, lib)
        assert tree.is_complete()







class TestNaturalLength:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_complete_expression(self) -> None:
        lib = _make_burgers_full_library()

        tokens = np.array([MUL, U1, U1], dtype=np.int32)
        assert natural_length(tokens, lib) == 3

    @pytest.mark.unit
    def test_incomplete_expression(self) -> None:
        lib = _make_burgers_full_library()

        tokens = np.array([ADD, U1], dtype=np.int32)
        assert natural_length(tokens, lib) == -1

    @pytest.mark.unit
    def test_padded_expression(self) -> None:
        lib = _make_burgers_full_library()
        ea = lib.EMPTY_ACTION
        tokens = np.array([MUL, U1, U1, ea, ea], dtype=np.int32)
        assert natural_length(tokens, lib) == 3

    @pytest.mark.unit
    def test_single_terminal(self) -> None:
        lib = _make_burgers_full_library()
        tokens = np.array([U1], dtype=np.int32)
        assert natural_length(tokens, lib) == 1

    @pytest.mark.unit
    def test_empty_array(self) -> None:
        lib = _make_burgers_full_library()
        tokens = np.array([], dtype=np.int32)
        assert natural_length(tokens, lib) == -1

    @pytest.mark.unit
    def test_out_of_range_token(self) -> None:
        lib = _make_burgers_full_library()
        tokens = np.array([999], dtype=np.int32)
        assert natural_length(tokens, lib) == -1


class TestTrimToNatural:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_trims_padding(self) -> None:
        lib = _make_burgers_full_library()
        ea = lib.EMPTY_ACTION
        tokens = np.array([MUL, U1, U1, ea, ea, ea], dtype=np.int32)
        trimmed = trim_to_natural(tokens, lib)
        np.testing.assert_array_equal(trimmed, [MUL, U1, U1])

    @pytest.mark.unit
    def test_already_exact(self) -> None:
        lib = _make_burgers_full_library()
        tokens = np.array([MUL, U1, U1], dtype=np.int32)
        trimmed = trim_to_natural(tokens, lib)
        np.testing.assert_array_equal(trimmed, [MUL, U1, U1])

    @pytest.mark.unit
    def test_trailing_real_tokens_trimmed(self) -> None:
        lib = _make_burgers_full_library()

        tokens = np.array([MUL, U1, U1, ADD, U1], dtype=np.int32)
        trimmed = trim_to_natural(tokens, lib)
        np.testing.assert_array_equal(trimmed, [MUL, U1, U1])

    @pytest.mark.unit
    def test_incomplete_raises(self) -> None:
        lib = _make_burgers_full_library()
        tokens = np.array([ADD, U1], dtype=np.int32)
        with pytest.raises(ValueError, match="[Ii]ncomplete"):
            trim_to_natural(tokens, lib)

    @pytest.mark.unit
    def test_empty_raises(self) -> None:
        lib = _make_burgers_full_library()
        tokens = np.array([], dtype=np.int32)
        with pytest.raises(ValueError, match="[Ii]ncomplete|[Ee]mpty"):
            trim_to_natural(tokens, lib)







def _make_diff_library() -> Library:
    tokens = [
        Token(name="x", arity=0, token_type=TokenType.COORDINATE),
        Token(name="u", arity=0, token_type=TokenType.TERMINAL),
        Token(name="add", arity=2, token_type=TokenType.OPERATOR),
        Token(name="mul", arity=2, token_type=TokenType.OPERATOR),
        Token(name="diff_x", arity=1, token_type=TokenType.OPERATOR),
        Token(name="diff2_x", arity=1, token_type=TokenType.OPERATOR),
    ]
    return Library(tokens)


class TestMaxDiffOrder:

    @pytest.mark.unit
    def test_no_diff_operator_returns_zero(self) -> None:
        lib = _make_diff_library()
        x, u, add, mul, _, _ = range(6)
        tokens = np.array([add, u, u], dtype=np.int32)
        assert max_diff_order(tokens, lib) == 0

    @pytest.mark.unit
    def test_single_diff_returns_one(self) -> None:
        lib = _make_diff_library()
        _, u, _, _, diff_x, _ = range(6)
        tokens = np.array([diff_x, u], dtype=np.int32)
        assert max_diff_order(tokens, lib) == 1

    @pytest.mark.unit
    def test_single_diff2_returns_two(self) -> None:
        lib = _make_diff_library()
        _, u, _, _, _, diff2_x = range(6)
        tokens = np.array([diff2_x, u], dtype=np.int32)
        assert max_diff_order(tokens, lib) == 2

    @pytest.mark.unit
    def test_nested_diff_accumulates(self) -> None:
        lib = _make_diff_library()
        _, u, _, _, diff_x, diff2_x = range(6)
        tokens = np.array([diff2_x, diff_x, u], dtype=np.int32)
        assert max_diff_order(tokens, lib) == 3

    @pytest.mark.unit
    def test_mixed_diff_through_mul_returns_two(self) -> None:
        lib = _make_diff_library()
        _, u, _, mul, diff_x, _ = range(6)
        tokens = np.array([diff_x, mul, u, diff_x, u], dtype=np.int32)
        assert max_diff_order(tokens, lib) == 2

    @pytest.mark.unit
    def test_mixed_diff_through_add_returns_two(self) -> None:
        lib = _make_diff_library()
        _, u, add, _, diff_x, _ = range(6)
        tokens = np.array([diff_x, add, diff_x, u, u], dtype=np.int32)
        assert max_diff_order(tokens, lib) == 2

    @pytest.mark.unit
    def test_mixed_diff_through_mul_with_diff2_returns_three(self) -> None:
        lib = _make_diff_library()
        _, u, _, mul, diff_x, diff2_x = range(6)
        tokens = np.array([diff2_x, mul, diff_x, u, u], dtype=np.int32)
        assert max_diff_order(tokens, lib) == 3

    @pytest.mark.unit
    def test_sibling_diffs_do_not_accumulate(self) -> None:
        lib = _make_diff_library()
        _, u, _, mul, diff_x, _ = range(6)
        tokens = np.array([mul, diff_x, u, diff_x, u], dtype=np.int32)
        assert max_diff_order(tokens, lib) == 1

    @pytest.mark.unit
    def test_diff_in_only_one_branch(self) -> None:
        lib = _make_diff_library()
        _, u, add, _, diff_x, _ = range(6)
        tokens = np.array([add, diff_x, u, u], dtype=np.int32)
        assert max_diff_order(tokens, lib) == 1

    @pytest.mark.unit
    def test_out_of_range_token_raises_value_error(self) -> None:
        lib = _make_diff_library()
        n_tokens = len(lib.tokens)
        tokens = np.array([0, n_tokens, 0], dtype=np.int32)
        with pytest.raises(ValueError, match="out of range"):
            max_diff_order(tokens, lib)

    @pytest.mark.unit
    def test_negative_token_raises_value_error(self) -> None:
        lib = _make_diff_library()
        tokens = np.array([0, -1, 0], dtype=np.int32)
        with pytest.raises(ValueError, match="out of range"):
            max_diff_order(tokens, lib)
