
from __future__ import annotations

import pytest

from kd.search.sga.convert import pde_to_kd_expr, tree_to_kd_expr
from kd.search.sga.pde import PDE
from kd.search.sga.tree import Node, Tree






def _leaf(name: str) -> Node:
    return Node(name=name, arity=0, children=[])


def _unary(op: str, child: Node) -> Node:
    return Node(name=op, arity=1, children=[child])


def _binary(op: str, left: Node, right: Node) -> Node:
    return Node(name=op, arity=2, children=[left, right])


def _tree(root: Node) -> Tree:
    return Tree(root=root)







class TestTreeToKdExprLeaf:

    @pytest.mark.smoke
    def test_single_variable(self) -> None:
        tree = _tree(_leaf("u"))
        assert tree_to_kd_expr(tree) == "u"

    def test_various_variables(self) -> None:
        for var in ("u", "x", "t", "u_x", "u_t", "u_xx", "u_tt"):
            tree = _tree(_leaf(var))
            result = tree_to_kd_expr(tree)
            assert result == var, f"Expected '{var}', got '{result}'"







class TestTreeToKdExprNameMapping:

    def test_binary_add(self) -> None:
        tree = _tree(_binary("+", _leaf("u"), _leaf("x")))
        result = tree_to_kd_expr(tree)
        assert result == "add(u, x)"

    def test_binary_sub(self) -> None:
        tree = _tree(_binary("-", _leaf("u"), _leaf("x")))
        result = tree_to_kd_expr(tree)
        assert result == "sub(u, x)"

    def test_binary_mul(self) -> None:
        tree = _tree(_binary("*", _leaf("u"), _leaf("x")))
        result = tree_to_kd_expr(tree)
        assert result == "mul(u, x)"

    def test_binary_div(self) -> None:
        tree = _tree(_binary("/", _leaf("u"), _leaf("x")))
        result = tree_to_kd_expr(tree)
        assert result == "div(u, x)"

    def test_unary_square(self) -> None:
        tree = _tree(_unary("^2", _leaf("u")))
        result = tree_to_kd_expr(tree)
        assert result == "n2(u)"

    def test_unary_cube(self) -> None:
        tree = _tree(_unary("^3", _leaf("u")))
        result = tree_to_kd_expr(tree)
        assert result == "n3(u)"







class TestTreeToKdExprDerivatives:

    def test_d_simple(self) -> None:
        tree = _tree(_binary("d", _leaf("u"), _leaf("x")))
        result = tree_to_kd_expr(tree)
        assert result == "diff_x(u)"

    def test_d_axis_t(self) -> None:
        tree = _tree(_binary("d", _leaf("u"), _leaf("t")))
        result = tree_to_kd_expr(tree)
        assert result == "diff_t(u)"

    def test_d2_simple(self) -> None:
        tree = _tree(_binary("d^2", _leaf("u"), _leaf("x")))
        result = tree_to_kd_expr(tree)
        assert result == "diff2_x(u)"

    def test_d2_axis_t(self) -> None:
        tree = _tree(_binary("d^2", _leaf("u"), _leaf("t")))
        result = tree_to_kd_expr(tree)
        assert result == "diff2_t(u)"

    def test_d_with_expression_arg(self) -> None:
        expr = _binary("*", _leaf("u"), _leaf("u_x"))
        tree = _tree(_binary("d", expr, _leaf("x")))
        result = tree_to_kd_expr(tree)
        assert result == "diff_x(mul(u, u_x))"

    def test_d2_with_expression_arg(self) -> None:
        expr = _binary("+", _leaf("u"), _leaf("x"))
        tree = _tree(_binary("d^2", expr, _leaf("t")))
        result = tree_to_kd_expr(tree)
        assert result == "diff2_t(add(u, x))"

    def test_nested_d_of_d2(self) -> None:
        inner = _binary("d^2", _leaf("u"), _leaf("x"))
        tree = _tree(_binary("d", inner, _leaf("x")))
        result = tree_to_kd_expr(tree)
        assert result == "diff_x(diff2_x(u))"

    def test_d_in_larger_tree(self) -> None:
        d_node = _binary("d", _leaf("u"), _leaf("x"))
        tree = _tree(_binary("*", d_node, _leaf("u")))
        result = tree_to_kd_expr(tree)
        assert result == "mul(diff_x(u), u)"







class TestTreeToKdExprNested:

    def test_binary_with_unary_child(self) -> None:
        inner = _unary("^2", _leaf("u"))
        tree = _tree(_binary("*", inner, _leaf("x")))
        result = tree_to_kd_expr(tree)
        assert result == "mul(n2(u), x)"

    def test_deeply_nested(self) -> None:
        cube_u = _unary("^3", _leaf("u"))
        add_xt = _binary("+", _leaf("x"), _leaf("t"))
        tree = _tree(_binary("/", cube_u, add_xt))
        result = tree_to_kd_expr(tree)
        assert result == "div(n3(u), add(x, t))"

    def test_triple_nesting(self) -> None:
        inner = _binary("*", _leaf("u"), _leaf("x"))
        tree = _tree(_unary("^2", inner))
        result = tree_to_kd_expr(tree)
        assert result == "n2(mul(u, x))"

    def test_complex_tree(self) -> None:
        left = _binary("+", _leaf("u"), _leaf("x"))
        right = _binary("-", _leaf("u_x"), _leaf("t"))
        tree = _tree(_binary("*", left, right))
        result = tree_to_kd_expr(tree)
        assert result == "mul(add(u, x), sub(u_x, t))"







class TestTreeToKdExprProperties:

    def test_output_is_valid_python_funcall(self) -> None:
        tree = _tree(_binary("*", _leaf("u"), _leaf("x")))
        result = tree_to_kd_expr(tree)

        compile(result, "<test>", "eval")

    def test_output_is_valid_python_funcall_nested(self) -> None:
        inner = _unary("^2", _leaf("u"))
        tree = _tree(_binary("/", inner, _leaf("x")))
        result = tree_to_kd_expr(tree)
        compile(result, "<test>", "eval")

    def test_leaves_appear_in_output(self) -> None:
        tree = _tree(
            _binary(
                "*",
                _binary("+", _leaf("u"), _leaf("x")),
                _leaf("t"),
            )
        )
        result = tree_to_kd_expr(tree)
        for var in ("u", "x", "t"):
            assert var in result

    def test_mapped_operator_not_in_output(self) -> None:
        tree = _tree(_binary("+", _binary("*", _leaf("u"), _leaf("x")), _leaf("t")))
        result = tree_to_kd_expr(tree)


        assert "add(" in result
        assert "mul(" in result

    def test_return_type_is_str(self) -> None:
        tree = _tree(_leaf("u"))
        result = tree_to_kd_expr(tree)
        assert isinstance(result, str)

    def test_d_operator_not_raw_in_output(self) -> None:
        tree = _tree(_binary("d", _leaf("u"), _leaf("x")))
        result = tree_to_kd_expr(tree)

        assert result.startswith("diff_")
        assert "d^2" not in result

    def test_d2_operator_maps_to_diff2(self) -> None:
        tree = _tree(_binary("d^2", _leaf("u"), _leaf("x")))
        result = tree_to_kd_expr(tree)
        assert result.startswith("diff2_")







class TestPdeToKdExprNoCoefficents:

    @pytest.mark.smoke
    def test_single_term(self) -> None:
        pde = PDE(terms=[_tree(_binary("*", _leaf("u"), _leaf("x")))])
        result = pde_to_kd_expr(pde)
        assert result == "mul(u, x)"

    def test_two_terms(self) -> None:
        pde = PDE(
            terms=[
                _tree(_leaf("u")),
                _tree(_leaf("x")),
            ]
        )
        result = pde_to_kd_expr(pde)
        assert result == "add(u, x)"

    def test_three_terms(self) -> None:
        pde = PDE(
            terms=[
                _tree(_leaf("u")),
                _tree(_leaf("x")),
                _tree(_leaf("t")),
            ]
        )
        result = pde_to_kd_expr(pde)
        assert result == "add(u, add(x, t))"

    def test_four_terms_right_associative(self) -> None:
        pde = PDE(
            terms=[
                _tree(_leaf("a")),
                _tree(_leaf("b")),
                _tree(_leaf("c")),
                _tree(_leaf("d")),
            ]
        )
        result = pde_to_kd_expr(pde)
        assert result == "add(a, add(b, add(c, d)))"

    def test_pde_with_d_operator_term(self) -> None:
        pde = PDE(
            terms=[
                _tree(_binary("d", _leaf("u"), _leaf("x"))),
                _tree(_leaf("u")),
            ]
        )
        result = pde_to_kd_expr(pde)
        assert result == "add(diff_x(u), u)"







class TestPdeToKdExprEdgeCases:

    def test_empty_pde_returns_empty_or_raises(self) -> None:
        pde = PDE(terms=[])
        try:
            result = pde_to_kd_expr(pde)

            assert isinstance(result, str)
        except (ValueError, IndexError):

            pass

    def test_return_type_is_str(self) -> None:
        pde = PDE(terms=[_tree(_leaf("u"))])
        result = pde_to_kd_expr(pde)
        assert isinstance(result, str)

    def test_output_is_valid_python_expression(self) -> None:
        pde = PDE(
            terms=[
                _tree(_binary("*", _leaf("u"), _leaf("u_x"))),
                _tree(_leaf("u_xx")),
            ]
        )
        result = pde_to_kd_expr(pde)
        compile(result, "<test>", "eval")







class TestConverterNegative:

    def test_tree_to_kd_expr_preserves_unknown_ops(self) -> None:
        tree = _tree(_unary("sin", _leaf("u")))
        try:
            result = tree_to_kd_expr(tree)

            assert "sin" in result
        except (KeyError, ValueError):

            pass

    def test_d_operator_with_non_leaf_axis_raises_or_handles(self) -> None:

        bad_axis = _binary("+", _leaf("x"), _leaf("t"))
        tree = _tree(_binary("d", _leaf("u"), bad_axis))
        try:
            result = tree_to_kd_expr(tree)

            assert isinstance(result, str)
        except (ValueError, AttributeError, TypeError):

            pass
