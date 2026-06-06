
from __future__ import annotations

import ast
import re
from typing import Any

import pytest

from kd.search.discover.core.tree import ExpressionTree
from kd.search.discover.ir.conversion import ir_to_tree, tokens_to_ir, tree_to_ir
from kd.search.discover.tokens.library import Library, LibraryConfig


_KD_DIFF_PATTERN = re.compile(r"^diff([0-9]*)_([a-z]+)$")







def _make_legacy_library() -> Library:
    return Library.from_config(
        LibraryConfig(
            coord_vars=["x1"],
            state_vars=["u1"],
            operators=[
                "add", "mul", "div", "diff", "diff2", "diff3", "n2", "n3",
            ],
        )
    )


def _make_kd_library() -> Library:
    return Library.from_config(
        LibraryConfig(
            coord_vars=["x"],
            state_vars=["u"],
            operators=[
                "add", "mul", "div", "diff_x", "diff2_x", "diff3_x",
                "n2", "n3", "sub", "sin", "cos", "neg",
            ],
        )
    )


@pytest.fixture
def legacy_lib() -> Library:
    return _make_legacy_library()


@pytest.fixture
def kd_lib() -> Library:
    return _make_kd_library()







class TestTreeToIRCrossValidation:

    @pytest.mark.equivalence
    @pytest.mark.parametrize(
        "case_name",
        [
            "single_terminal",
            "coord_terminal",
            "unary_n2",
            "unary_diff",
            "binary_add",
            "binary_mul",
            "binary_div",
            "burgers_gt",
            "deeply_nested",
            "nested_add",
            "chain_unary",
            "div_nested",
        ],
    )
    def test_legacy_vocab_matches_reference(
        self, ir_fixture: dict[str, Any], legacy_lib: Library, case_name: str
    ) -> None:
        case = ir_fixture["legacy_vocab"]["cases"][case_name]
        tokens = case["tokens"]
        expected_ir = case["expected_ir"]

        tree = ExpressionTree.from_preorder(tokens, legacy_lib)
        actual_ir = tree_to_ir(tree)
        assert actual_ir == expected_ir, (
            f"Case {case_name}: expected {expected_ir!r}, got {actual_ir!r}"
        )

    @pytest.mark.equivalence
    @pytest.mark.parametrize(
        "case_name",
        [
            "single_terminal",
            "coord_terminal",
            "unary_n2",
            "unary_diff",
            "binary_add",
            "binary_mul",
            "binary_div",
            "burgers_gt",
            "deeply_nested",
            "nested_add",
            "chain_unary",
            "div_nested",
            "sub_simple",
            "sin_u",
            "cos_x",
            "neg_mul",
            "burgers_with_sub",
            "right_nested_add",
        ],
    )
    def test_kd_vocab_matches_reference(
        self, ir_fixture: dict[str, Any], kd_lib: Library, case_name: str
    ) -> None:
        case = ir_fixture["kd_vocab"]["cases"][case_name]
        tokens = case["tokens"]
        expected_ir = case["expected_ir"]

        tree = ExpressionTree.from_preorder(tokens, kd_lib)
        actual_ir = tree_to_ir(tree)
        assert actual_ir == expected_ir

    @pytest.mark.equivalence
    @pytest.mark.parametrize(
        "case_name",
        ["single_terminal", "burgers_gt", "deeply_nested", "burgers_with_sub"],
    )
    def test_tokens_to_ir_matches_reference(
        self, ir_fixture: dict[str, Any], kd_lib: Library, case_name: str
    ) -> None:
        case = ir_fixture["kd_vocab"]["cases"][case_name]
        tokens = case["tokens"]
        expected_ir = case["expected_ir"]

        actual_ir = tokens_to_ir(tokens, kd_lib)
        assert actual_ir == expected_ir







class TestTreeToIRUnit:

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_single_terminal_no_parens(self, kd_lib: Library) -> None:
        tokens = [kd_lib.name_to_index("u")]
        tree = ExpressionTree.from_preorder(tokens, kd_lib)
        assert tree_to_ir(tree) == "u"

    @pytest.mark.unit
    def test_unary_operator_format(self, kd_lib: Library) -> None:
        tokens = [kd_lib.name_to_index(n) for n in ["sin", "u"]]
        tree = ExpressionTree.from_preorder(tokens, kd_lib)
        assert tree_to_ir(tree) == "sin(u)"

    @pytest.mark.unit
    def test_binary_operator_format(self, kd_lib: Library) -> None:
        tokens = [kd_lib.name_to_index(n) for n in ["add", "x", "u"]]
        tree = ExpressionTree.from_preorder(tokens, kd_lib)
        assert tree_to_ir(tree) == "add(x,u)"

    @pytest.mark.unit
    def test_diff_x_format(self, kd_lib: Library) -> None:
        tokens = [kd_lib.name_to_index(n) for n in ["diff_x", "u"]]
        tree = ExpressionTree.from_preorder(tokens, kd_lib)
        assert tree_to_ir(tree) == "diff_x(u)"

    @pytest.mark.unit
    def test_nested_expression(self, kd_lib: Library) -> None:
        names = ["mul", "n2", "diff_x", "u", "diff3_x", "n3", "u"]
        tokens = [kd_lib.name_to_index(n) for n in names]
        tree = ExpressionTree.from_preorder(tokens, kd_lib)
        assert tree_to_ir(tree) == "mul(n2(diff_x(u)),diff3_x(n3(u)))"


class TestIRToTreeUnit:

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_parse_terminal(self, kd_lib: Library) -> None:
        tree = ir_to_tree("u", kd_lib)
        assert tree.to_preorder() == [kd_lib.name_to_index("u")]

    @pytest.mark.unit
    def test_parse_unary(self, kd_lib: Library) -> None:
        tree = ir_to_tree("sin(u)", kd_lib)
        expected = [
            kd_lib.name_to_index("sin"),
            kd_lib.name_to_index("u"),
        ]
        assert tree.to_preorder() == expected

    @pytest.mark.unit
    def test_parse_binary(self, kd_lib: Library) -> None:
        tree = ir_to_tree("add(x,u)", kd_lib)
        expected = [
            kd_lib.name_to_index("add"),
            kd_lib.name_to_index("x"),
            kd_lib.name_to_index("u"),
        ]
        assert tree.to_preorder() == expected

    @pytest.mark.unit
    def test_parse_nested_burgers(self, kd_lib: Library) -> None:
        ir = "add(mul(u,diff_x(u)),diff2_x(u))"
        tree = ir_to_tree(ir, kd_lib)
        expected = [
            kd_lib.name_to_index("add"),
            kd_lib.name_to_index("mul"),
            kd_lib.name_to_index("u"),
            kd_lib.name_to_index("diff_x"),
            kd_lib.name_to_index("u"),
            kd_lib.name_to_index("diff2_x"),
            kd_lib.name_to_index("u"),
        ]
        assert tree.to_preorder() == expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "ir_with_space",
        [
            "add(x, u)",
            "add( x,u)",
            "add(x,u )",
            " add(x,u) ",
        ],
    )
    def test_parse_tolerates_whitespace(
        self, kd_lib: Library, ir_with_space: str
    ) -> None:
        tree = ir_to_tree(ir_with_space, kd_lib)
        expected = [
            kd_lib.name_to_index("add"),
            kd_lib.name_to_index("x"),
            kd_lib.name_to_index("u"),
        ]
        assert tree.to_preorder() == expected

    @pytest.mark.unit
    def test_parse_sets_parent_pointers(self, kd_lib: Library) -> None:
        tree = ir_to_tree("add(mul(u,x),diff_x(u))", kd_lib)
        root = tree.root
        assert root.parent is None

        assert root.children[0].parent is root
        assert root.children[1].parent is root

        assert root.children[0].children[0].parent is root.children[0]
        assert root.children[0].children[1].parent is root.children[0]

        assert root.children[1].children[0].parent is root.children[1]

    @pytest.mark.unit
    def test_parse_unknown_token_raises(self, kd_lib: Library) -> None:
        with pytest.raises(ValueError):
            ir_to_tree("add(u,unknown_var)", kd_lib)

    @pytest.mark.unit
    def test_parse_empty_raises(self, kd_lib: Library) -> None:
        with pytest.raises(ValueError):
            ir_to_tree("", kd_lib)

    @pytest.mark.unit
    def test_parse_infix_raises(self, kd_lib: Library) -> None:
        with pytest.raises(ValueError):
            ir_to_tree("u + x", kd_lib)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "bad_ir,description",
        [
            ("add(u)", "binary op with too few args"),
            ("sin(u,x)", "unary op with too many args"),
            ("add(u,x,u)", "binary op with too many args"),
        ],
    )
    def test_parse_arity_mismatch_raises(
        self, kd_lib: Library, bad_ir: str, description: str
    ) -> None:
        with pytest.raises(ValueError):
            ir_to_tree(bad_ir, kd_lib)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "bad_ir",
        [
            "add(u,x",
            "add(u,x))",
            "add()",
            "add(u,,x)",
            "add(u,x)extra",
        ],
    )
    def test_parse_malformed_raises(
        self, kd_lib: Library, bad_ir: str
    ) -> None:
        with pytest.raises(ValueError):
            ir_to_tree(bad_ir, kd_lib)


class TestTokensToIRUnit:

    @pytest.mark.unit
    def test_shortcut_matches_two_step(self, kd_lib: Library) -> None:
        names = ["add", "mul", "u", "diff_x", "u", "diff2_x", "u"]
        tokens = [kd_lib.name_to_index(n) for n in names]
        shortcut = tokens_to_ir(tokens, kd_lib)
        two_step = tree_to_ir(ExpressionTree.from_preorder(tokens, kd_lib))
        assert shortcut == two_step







class TestIRInvariants:

    @pytest.mark.equivalence
    @pytest.mark.parametrize(
        "case_name",
        [
            "single_terminal",
            "coord_terminal",
            "unary_n2",
            "unary_diff",
            "binary_add",
            "binary_mul",
            "binary_div",
            "burgers_gt",
            "deeply_nested",
            "nested_add",
            "chain_unary",
            "div_nested",
            "sub_simple",
            "sin_u",
            "cos_x",
            "neg_mul",
            "burgers_with_sub",
            "right_nested_add",
        ],
    )
    def test_roundtrip_tree_to_ir_to_tree(
        self, ir_fixture: dict[str, Any], kd_lib: Library, case_name: str
    ) -> None:
        case = ir_fixture["kd_vocab"]["cases"][case_name]
        tokens = case["tokens"]

        tree = ExpressionTree.from_preorder(tokens, kd_lib)
        ir_str = tree_to_ir(tree)
        reconstructed = ir_to_tree(ir_str, kd_lib)
        assert reconstructed.to_preorder() == tokens

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "ir_str",
        [
            "u",
            "x",
            "add(u,x)",
            "mul(u,x)",
            "div(u,x)",
            "sin(u)",
            "n2(u)",
            "diff_x(u)",
            "diff2_x(u)",
            "add(mul(u,diff_x(u)),diff2_x(u))",
            "mul(n2(diff_x(u)),diff3_x(n3(u)))",
            "sub(neg(mul(u,diff_x(u))),diff2_x(u))",
            "add(add(u,x),mul(u,x))",
            "div(mul(u,x),diff2_x(u))",
            "neg(mul(u,x))",

            "add(u,add(x,u))",
            "mul(u,div(x,u))",

            "add(u,add(x,add(u,add(x,u))))",
        ],
    )
    def test_roundtrip_ir_to_tree_to_ir(
        self, kd_lib: Library, ir_str: str
    ) -> None:
        tree = ir_to_tree(ir_str, kd_lib)
        result = tree_to_ir(tree)
        assert result == ir_str

    @pytest.mark.parametrize(
        "case_name",
        [
            "single_terminal",
            "burgers_gt",
            "deeply_nested",
            "sub_simple",
            "neg_mul",
            "burgers_with_sub",
            "right_nested_add",
        ],
    )
    def test_tokens_to_ir_equals_tree_to_ir(
        self, ir_fixture: dict[str, Any], kd_lib: Library, case_name: str
    ) -> None:
        case = ir_fixture["kd_vocab"]["cases"][case_name]
        tokens = case["tokens"]

        via_shortcut = tokens_to_ir(tokens, kd_lib)
        via_tree = tree_to_ir(ExpressionTree.from_preorder(tokens, kd_lib))
        assert via_shortcut == via_tree







class TestKdCompatibility:

    @pytest.mark.parametrize(
        "case_name",
        [
            "single_terminal",
            "burgers_gt",
            "deeply_nested",
            "sub_simple",
            "sin_u",
            "neg_mul",
            "burgers_with_sub",
            "right_nested_add",
        ],
    )
    def test_ir_is_valid_python_expression(
        self, ir_fixture: dict[str, Any], kd_lib: Library, case_name: str
    ) -> None:
        case = ir_fixture["kd_vocab"]["cases"][case_name]
        tokens = case["tokens"]

        ir_str = tokens_to_ir(tokens, kd_lib)

        parsed = ast.parse(ir_str, mode="eval")
        assert parsed is not None

    @pytest.mark.parametrize(
        "case_name",
        [
            "single_terminal",
            "burgers_gt",
            "deeply_nested",
            "sub_simple",
            "neg_mul",
            "burgers_with_sub",
        ],
    )
    def test_ir_has_no_infix_operators(
        self, ir_fixture: dict[str, Any], kd_lib: Library, case_name: str
    ) -> None:
        case = ir_fixture["kd_vocab"]["cases"][case_name]
        tokens = case["tokens"]

        ir_str = tokens_to_ir(tokens, kd_lib)
        parsed = ast.parse(ir_str, mode="eval")

        for node in ast.walk(parsed):
            assert not isinstance(node, ast.BinOp), (
                f"IR contains infix BinOp: {ir_str}"
            )
            assert not isinstance(node, ast.UnaryOp), (
                f"IR contains UnaryOp: {ir_str}"
            )

    @pytest.mark.parametrize(
        "case_name",
        ["burgers_gt", "deeply_nested", "neg_mul", "burgers_with_sub"],
    )
    def test_ir_uses_function_call_syntax(
        self, ir_fixture: dict[str, Any], kd_lib: Library, case_name: str
    ) -> None:
        case = ir_fixture["kd_vocab"]["cases"][case_name]
        tokens = case["tokens"]

        ir_str = tokens_to_ir(tokens, kd_lib)
        parsed = ast.parse(ir_str, mode="eval")


        call_names = set()
        name_nodes = set()
        for node in ast.walk(parsed):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                call_names.add(node.func.id)
            elif isinstance(node, ast.Name):
                name_nodes.add(node.id)


        bare_names = name_nodes - call_names
        for name in bare_names:
            token = kd_lib[name]
            assert token.arity == 0, (
                f"Non-terminal {name!r} appears as bare name in IR: {ir_str}"
            )

    def test_kd_diff_names_match_executor_regex(
        self, kd_lib: Library
    ) -> None:
        for token in kd_lib.tokens:
            if token.name.startswith("diff"):
                assert _KD_DIFF_PATTERN.match(token.name), (
                    f"Diff token {token.name!r} does not match kd regex "
                    f"r'^diff([0-9]*)_([a-z]+)$'"
                )

    def test_legacy_diff_names_do_not_match_kd_regex(
        self, legacy_lib: Library
    ) -> None:
        for token in legacy_lib.tokens:
            if token.name.startswith("diff"):
                assert not _KD_DIFF_PATTERN.match(token.name), (
                    f"Legacy diff {token.name!r} unexpectedly matches kd "
                    f"regex — kd requires axis suffix (e.g., diff_x)"
                )









class TestDeepTreeIterative:

    DEEP_DEPTH = 1500

    @pytest.mark.unit
    def test_tree_to_ir_deep_unary_chain(self, kd_lib: Library) -> None:
        sin_idx = kd_lib.name_to_index("sin")
        u_idx = kd_lib.name_to_index("u")
        tokens = [sin_idx] * self.DEEP_DEPTH + [u_idx]
        tree = ExpressionTree.from_preorder(tokens, kd_lib)

        ir = tree_to_ir(tree)

        assert ir.startswith("sin(")
        assert ir.endswith("u" + ")" * self.DEEP_DEPTH)

    @pytest.mark.unit
    def test_tree_to_ir_deep_left_binary_chain(self, kd_lib: Library) -> None:
        add_idx = kd_lib.name_to_index("add")
        u_idx = kd_lib.name_to_index("u")

        tokens: list[int] = []
        for _ in range(self.DEEP_DEPTH - 1):
            tokens.append(add_idx)
        tokens.append(u_idx)
        for _ in range(self.DEEP_DEPTH - 1):
            tokens.append(u_idx)
        tree = ExpressionTree.from_preorder(tokens, kd_lib)

        ir = tree_to_ir(tree)


        assert ir.startswith("add(")
        assert ir.endswith(",u)")

    @pytest.mark.unit
    def test_tree_to_ir_deep_preserves_structure(self, kd_lib: Library) -> None:

        shallow_depth = 150
        sin_idx = kd_lib.name_to_index("sin")
        u_idx = kd_lib.name_to_index("u")
        tokens = [sin_idx] * shallow_depth + [u_idx]
        tree = ExpressionTree.from_preorder(tokens, kd_lib)

        ir = tree_to_ir(tree)
        reconstructed = ir_to_tree(ir, kd_lib)

        assert reconstructed.to_preorder() == tokens







class TestDefensiveGuards:

    @pytest.mark.unit
    def test_ir_to_tree_wraps_recursion_error_as_value_error(
        self,
        kd_lib: Library,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.discover.ir import conversion



        real_parse = conversion.ast.parse

        def _fake_parse(source: str, *args: object, **kwargs: object) -> object:


            if source.strip() == "__TD_082_8_TRIGGER__":
                raise RecursionError("simulated ast.parse recursion limit")
            return real_parse(source, *args, **kwargs)

        monkeypatch.setattr(conversion.ast, "parse", _fake_parse)

        with pytest.raises(ValueError, match="too deeply nested"):
            ir_to_tree("__TD_082_8_TRIGGER__", kd_lib)

    @pytest.mark.unit
    def test_node_to_ir_rejects_childless_operator(
        self, kd_lib: Library,
    ) -> None:
        from kd.search.discover.core.tree import TreeNode

        add_token = kd_lib["add"]
        malformed_root = TreeNode(token=add_token, children=[])
        malformed_tree = ExpressionTree(
            root=malformed_root, _library=kd_lib,
        )

        with pytest.raises(ValueError, match="childless"):
            tree_to_ir(malformed_tree)

    @pytest.mark.unit
    def test_node_to_ir_rejects_arity_mismatch_internal_node(
        self, kd_lib: Library,
    ) -> None:
        from kd.search.discover.core.tree import TreeNode

        u_token = kd_lib["u"]
        add_token = kd_lib["add"]


        single_child = TreeNode(token=u_token, children=[])
        malformed_root = TreeNode(token=add_token, children=[single_child])
        malformed_tree = ExpressionTree(
            root=malformed_root, _library=kd_lib,
        )

        with pytest.raises(ValueError, match="arity"):
            tree_to_ir(malformed_tree)
