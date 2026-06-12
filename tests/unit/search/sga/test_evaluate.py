
from __future__ import annotations

import pytest
import torch
from torch import Tensor


from kd.search.sga.evaluate import DiffContext, build_theta, execute_pde, execute_tree
from kd.search.sga.pde import PDE
from kd.search.sga.tree import Node, Tree



N_SAMPLES = 50
RTOL = 1e-5
ATOL = 1e-8





def _leaf(name: str) -> Node:
    return Node(name=name, arity=0, children=[])


def _unary(op: str, child: Node) -> Node:
    return Node(name=op, arity=1, children=[child])


def _binary(op: str, left: Node, right: Node) -> Node:
    return Node(name=op, arity=2, children=[left, right])


def _make_data(seed: int = 42) -> dict[str, Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {
        "u": torch.randn(N_SAMPLES, generator=gen),
        "x": torch.randn(N_SAMPLES, generator=gen),
        "t": torch.randn(N_SAMPLES, generator=gen),
    }







class TestSmoke:

    @pytest.mark.smoke
    def test_execute_tree_callable(self) -> None:
        tree = Tree(root=_leaf("u"))
        data = {"u": torch.ones(5)}
        result = execute_tree(tree, data)
        assert isinstance(result, Tensor)

    @pytest.mark.smoke
    def test_execute_pde_callable(self) -> None:
        tree = Tree(root=_leaf("u"))
        pde = PDE(terms=[tree])
        data = {"u": torch.ones(5)}
        valid_terms, valid_indices = execute_pde(pde, data)
        assert isinstance(valid_terms, Tensor)
        assert isinstance(valid_indices, list)

    @pytest.mark.smoke
    def test_build_theta_callable(self) -> None:
        vt = torch.randn(10, 3)
        result = build_theta(vt)
        assert isinstance(result, Tensor)







class TestExecuteTreeLeaf:

    def test_single_leaf_returns_data(self) -> None:
        data = _make_data()
        tree = Tree(root=_leaf("u"))
        result = execute_tree(tree, data)
        torch.testing.assert_close(result, data["u"], rtol=RTOL, atol=ATOL)

    def test_different_variable(self) -> None:
        data = _make_data()
        tree_x = Tree(root=_leaf("x"))
        result = execute_tree(tree_x, data)
        torch.testing.assert_close(result, data["x"], rtol=RTOL, atol=ATOL)

    def test_missing_variable_raises(self) -> None:
        data = {"u": torch.ones(5)}
        tree = Tree(root=_leaf("nonexistent"))
        with pytest.raises(KeyError):
            execute_tree(tree, data)







class TestExecuteTreeBinaryOps:

    def test_add(self) -> None:
        data = _make_data()
        tree = Tree(root=_binary("+", _leaf("u"), _leaf("x")))
        result = execute_tree(tree, data)
        expected = data["u"] + data["x"]
        torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)

    def test_sub(self) -> None:
        data = _make_data()
        tree = Tree(root=_binary("-", _leaf("u"), _leaf("x")))
        result = execute_tree(tree, data)
        expected = data["u"] - data["x"]
        torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)

    def test_mul(self) -> None:
        data = _make_data()
        tree = Tree(root=_binary("*", _leaf("u"), _leaf("x")))
        result = execute_tree(tree, data)
        expected = data["u"] * data["x"]
        torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)

    def test_div_nonzero_denominator(self) -> None:
        data = _make_data()

        data["x"] = data["x"].clamp(min=0.1)
        tree = Tree(root=_binary("/", _leaf("u"), _leaf("x")))
        result = execute_tree(tree, data)
        expected = data["u"] / data["x"]

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)







class TestExecuteTreeUnaryOps:

    def test_square(self) -> None:
        data = _make_data()
        tree = Tree(root=_unary("^2", _leaf("u")))
        result = execute_tree(tree, data)
        expected = data["u"] ** 2
        torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)

    def test_cube(self) -> None:
        data = _make_data()
        tree = Tree(root=_unary("^3", _leaf("u")))
        result = execute_tree(tree, data)
        expected = data["u"] ** 3
        torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)







class TestExecuteTreeNested:

    def test_add_of_products(self) -> None:
        data = _make_data()
        mul_ux = _binary("*", _leaf("u"), _leaf("x"))
        sq_u = _unary("^2", _leaf("u"))
        root = _binary("+", mul_ux, sq_u)
        tree = Tree(root=root)

        result = execute_tree(tree, data)
        expected = data["u"] * data["x"] + data["u"] ** 2
        torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)

    def test_deeply_nested(self) -> None:
        data = _make_data()
        cube_u = _unary("^3", _leaf("u"))
        sub_xt = _binary("-", _leaf("x"), _leaf("t"))
        root = _binary("*", cube_u, sub_xt)
        tree = Tree(root=root)

        result = execute_tree(tree, data)
        expected = data["u"] ** 3 * (data["x"] - data["t"])
        torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)

    def test_chained_unary(self) -> None:
        data = _make_data()

        data["u"] = torch.linspace(-1.0, 1.0, N_SAMPLES)
        cube_u = _unary("^3", _leaf("u"))
        sq_cube = _unary("^2", cube_u)
        tree = Tree(root=sq_cube)

        result = execute_tree(tree, data)
        expected = data["u"] ** 6
        torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)

    def test_div_in_nested_context(self) -> None:
        data = _make_data()
        data["x"] = data["x"].clamp(min=0.5)
        div_ux = _binary("/", _leaf("u"), _leaf("x"))
        root = _binary("+", div_ux, _leaf("t"))
        tree = Tree(root=root)

        result = execute_tree(tree, data)
        expected = data["u"] / data["x"] + data["t"]
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)







class TestExecuteTreeStateless:

    def test_tree_unchanged_after_execution(self) -> None:
        data = _make_data()
        root = _binary("*", _unary("^2", _leaf("u")), _leaf("x"))
        tree = Tree(root=root)

        str_before = str(tree)
        _ = execute_tree(tree, data)
        str_after = str(tree)

        assert str_before == str_after

    def test_tree_structurally_equal_after_execution(self) -> None:
        data = _make_data()
        root = _binary("+", _leaf("u"), _leaf("x"))
        tree = Tree(root=root)
        tree_copy = tree.copy()

        _ = execute_tree(tree, data)
        assert tree == tree_copy







class TestExecuteTreeShape:

    def test_output_is_1d(self) -> None:
        data = _make_data()
        tree = Tree(root=_binary("*", _leaf("u"), _leaf("x")))
        result = execute_tree(tree, data)
        assert result.ndim == 1

    def test_output_length_matches_input(self) -> None:
        data = _make_data()
        tree = Tree(root=_leaf("u"))
        result = execute_tree(tree, data)
        assert result.shape[0] == N_SAMPLES







class TestExecutePdeFiltering:

    def test_all_valid_terms_kept(self) -> None:
        data = _make_data()
        t1 = Tree(root=_leaf("u"))
        t2 = Tree(root=_leaf("x"))
        t3 = Tree(root=_binary("*", _leaf("u"), _leaf("x")))
        pde = PDE(terms=[t1, t2, t3])

        valid_terms, valid_indices = execute_pde(pde, data)
        assert valid_terms.shape[1] == 3
        assert valid_indices == [0, 1, 2]

    def test_nan_column_filtered(self) -> None:

        data = _make_data()
        data["z"] = torch.full((N_SAMPLES,), float("nan"))
        t_valid = Tree(root=_leaf("u"))
        t_nan = Tree(root=_leaf("z"))
        pde = PDE(terms=[t_valid, t_nan])

        valid_terms, valid_indices = execute_pde(pde, data)
        assert valid_terms.shape[1] == 1
        assert 0 in valid_indices
        assert 1 not in valid_indices

    def test_inf_column_filtered(self) -> None:
        data = _make_data()
        data["inf_var"] = torch.full((N_SAMPLES,), float("inf"))
        t_valid = Tree(root=_leaf("u"))
        t_inf = Tree(root=_leaf("inf_var"))
        pde = PDE(terms=[t_valid, t_inf])

        valid_terms, valid_indices = execute_pde(pde, data)
        assert valid_terms.shape[1] == 1
        assert 0 in valid_indices
        assert 1 not in valid_indices

    def test_zero_column_filtered(self) -> None:
        data = _make_data()
        data["zeros"] = torch.zeros(N_SAMPLES)
        t_valid = Tree(root=_leaf("u"))
        t_zero = Tree(root=_leaf("zeros"))
        pde = PDE(terms=[t_valid, t_zero])

        valid_terms, valid_indices = execute_pde(pde, data)
        assert valid_terms.shape[1] == 1
        assert 0 in valid_indices
        assert 1 not in valid_indices

    def test_valid_indices_map_back_correctly(self) -> None:
        data = _make_data()
        data["zeros"] = torch.zeros(N_SAMPLES)

        pde = PDE(
            terms=[
                Tree(root=_leaf("u")),
                Tree(root=_leaf("zeros")),
                Tree(root=_leaf("x")),
                Tree(root=_leaf("zeros")),
                Tree(root=_binary("*", _leaf("u"), _leaf("x"))),
            ]
        )

        valid_terms, valid_indices = execute_pde(pde, data)

        assert valid_indices == [0, 2, 4]
        assert valid_terms.shape[1] == 3







class TestExecutePdeShape:

    def test_output_shape(self) -> None:
        data = _make_data()
        pde = PDE(
            terms=[
                Tree(root=_leaf("u")),
                Tree(root=_leaf("x")),
            ]
        )
        valid_terms, valid_indices = execute_pde(pde, data)
        assert valid_terms.shape == (N_SAMPLES, 2)

    def test_output_is_2d(self) -> None:
        data = _make_data()
        pde = PDE(terms=[Tree(root=_leaf("u"))])
        valid_terms, _ = execute_pde(pde, data)
        assert valid_terms.ndim == 2







class TestExecutePdeEdgeCases:

    def test_all_terms_filtered_returns_empty(self) -> None:
        data = _make_data()
        data["nan_var"] = torch.full((N_SAMPLES,), float("nan"))
        data["inf_var"] = torch.full((N_SAMPLES,), float("inf"))
        pde = PDE(
            terms=[
                Tree(root=_leaf("nan_var")),
                Tree(root=_leaf("inf_var")),
            ]
        )

        valid_terms, valid_indices = execute_pde(pde, data)
        assert valid_terms.shape[1] == 0
        assert valid_indices == []

    def test_single_valid_term(self) -> None:
        data = _make_data()
        pde = PDE(terms=[Tree(root=_leaf("u"))])

        valid_terms, valid_indices = execute_pde(pde, data)
        assert valid_terms.shape == (N_SAMPLES, 1)
        assert valid_indices == [0]
        torch.testing.assert_close(valid_terms[:, 0], data["u"], rtol=RTOL, atol=ATOL)

    def test_empty_pde(self) -> None:
        data = _make_data()
        pde = PDE(terms=[])

        valid_terms, valid_indices = execute_pde(pde, data)
        assert valid_terms.shape[1] == 0
        assert valid_indices == []

    def test_pde_not_modified(self) -> None:
        data = _make_data()
        data["zeros"] = torch.zeros(N_SAMPLES)
        pde = PDE(
            terms=[
                Tree(root=_leaf("u")),
                Tree(root=_leaf("zeros")),
            ]
        )
        original_width = pde.width
        original_str = str(pde)

        _ = execute_pde(pde, data)

        assert pde.width == original_width
        assert str(pde) == original_str







class TestExecutePdeDivisionGuard:

    def test_div_by_zero_does_not_crash(self) -> None:
        data = _make_data()
        data["zeros"] = torch.zeros(N_SAMPLES)
        tree = Tree(root=_binary("/", _leaf("u"), _leaf("zeros")))
        pde = PDE(terms=[tree])


        valid_terms, valid_indices = execute_pde(pde, data)




        if len(valid_indices) == 0:
            pytest.fail(
                "Premise not met: /(u, zeros) term was guard-filtered; "
                "expected the safe_div path (term survives, finite). "
                "If filtering is now intended, update this test."
            )

        assert torch.isfinite(valid_terms).all()

    def test_div_by_near_zero_result_is_finite(self) -> None:
        data = _make_data()
        data["small"] = torch.full((N_SAMPLES,), 1e-15)
        tree = Tree(root=_binary("/", _leaf("u"), _leaf("small")))

        result = execute_tree(tree, data)

        assert torch.isfinite(result).all()







class TestBuildTheta:

    def test_no_defaults(self) -> None:
        vt = torch.randn(N_SAMPLES, 3)
        result = build_theta(vt)
        torch.testing.assert_close(result, vt, rtol=RTOL, atol=ATOL)

    def test_no_defaults_explicit_none(self) -> None:
        vt = torch.randn(N_SAMPLES, 3)
        result = build_theta(vt, default_terms=None)
        torch.testing.assert_close(result, vt, rtol=RTOL, atol=ATOL)

    def test_with_defaults_prepended(self) -> None:
        gen = torch.Generator().manual_seed(99)
        defaults = torch.randn(N_SAMPLES, 2, generator=gen)
        valid = torch.randn(N_SAMPLES, 3, generator=gen)

        result = build_theta(valid, default_terms=defaults)

        assert result.shape == (N_SAMPLES, 5)

        torch.testing.assert_close(result[:, :2], defaults, rtol=RTOL, atol=ATOL)

        torch.testing.assert_close(result[:, 2:], valid, rtol=RTOL, atol=ATOL)

    def test_output_shape_correct(self) -> None:
        n_defaults = 4
        n_valid = 7
        defaults = torch.randn(N_SAMPLES, n_defaults)
        valid = torch.randn(N_SAMPLES, n_valid)

        result = build_theta(valid, default_terms=defaults)
        assert result.shape == (N_SAMPLES, n_defaults + n_valid)

    def test_with_empty_valid_terms(self) -> None:
        defaults = torch.randn(N_SAMPLES, 3)
        empty_valid = torch.empty(N_SAMPLES, 0)

        result = build_theta(empty_valid, default_terms=defaults)
        assert result.shape == (N_SAMPLES, 3)
        torch.testing.assert_close(result, defaults, rtol=RTOL, atol=ATOL)







class TestExecutePdeRuntimeErrorContainment:

    def test_runtime_error_term_skipped_keeps_siblings(self) -> None:

        data: dict[str, Tensor] = {
            "u": torch.randn(N_SAMPLES),
            "x": torch.randn(N_SAMPLES),

            "v_short": torch.randn(N_SAMPLES // 2),
        }
        good = Tree(root=_leaf("u"))
        bad = Tree(root=_binary("+", _leaf("u"), _leaf("v_short")))
        also_good = Tree(root=_leaf("x"))
        pde = PDE(terms=[good, bad, also_good])

        valid_terms, valid_indices = execute_pde(pde, data)


        assert valid_indices == [0, 2]
        assert valid_terms.shape == (N_SAMPLES, 2)

    def test_runtime_error_only_returns_empty(self) -> None:
        data: dict[str, Tensor] = {
            "u": torch.randn(N_SAMPLES),
            "v_short": torch.randn(N_SAMPLES // 2),
        }
        bad1 = Tree(root=_binary("+", _leaf("u"), _leaf("v_short")))
        bad2 = Tree(root=_binary("*", _leaf("u"), _leaf("v_short")))
        pde = PDE(terms=[bad1, bad2])

        valid_terms, valid_indices = execute_pde(pde, data)

        assert valid_indices == []
        assert valid_terms.shape[1] == 0







class TestNumericalStability:

    @pytest.mark.numerical
    def test_nan_input_in_data_dict(self) -> None:
        data = {"u": torch.tensor([float("nan"), 1.0, 2.0])}
        tree = Tree(root=_leaf("u"))
        result = execute_tree(tree, data)

        assert result.shape[0] == 3

    @pytest.mark.numerical
    def test_inf_input_in_data_dict(self) -> None:
        data = {"u": torch.tensor([float("inf"), 1.0, 2.0])}
        tree = Tree(root=_unary("^2", _leaf("u")))
        result = execute_tree(tree, data)
        assert result.shape[0] == 3

    @pytest.mark.numerical
    def test_large_values_no_crash(self) -> None:
        data = {"u": torch.full((10,), 1e30)}
        tree = Tree(root=_unary("^2", _leaf("u")))

        result = execute_tree(tree, data)
        assert result.shape[0] == 10

    @pytest.mark.numerical
    def test_execute_pde_mixed_nan_inf_valid(self) -> None:
        n = 20
        data = {
            "good": torch.randn(n),
            "nan_data": torch.full((n,), float("nan")),
            "inf_data": torch.full((n,), float("inf")),
        }
        pde = PDE(
            terms=[
                Tree(root=_leaf("good")),
                Tree(root=_leaf("nan_data")),
                Tree(root=_leaf("inf_data")),
            ]
        )
        valid_terms, valid_indices = execute_pde(pde, data)

        assert valid_indices == [0]
        assert valid_terms.shape == (n, 1)
        assert torch.isfinite(valid_terms).all()







class TestAlgebraicProperties:

    def test_mul_commutativity_in_result(self) -> None:
        data = _make_data()
        tree_ux = Tree(root=_binary("*", _leaf("u"), _leaf("x")))
        tree_xu = Tree(root=_binary("*", _leaf("x"), _leaf("u")))

        r_ux = execute_tree(tree_ux, data)
        r_xu = execute_tree(tree_xu, data)
        torch.testing.assert_close(r_ux, r_xu, rtol=RTOL, atol=ATOL)

    def test_add_commutativity_in_result(self) -> None:
        data = _make_data()
        tree_ux = Tree(root=_binary("+", _leaf("u"), _leaf("x")))
        tree_xu = Tree(root=_binary("+", _leaf("x"), _leaf("u")))

        r_ux = execute_tree(tree_ux, data)
        r_xu = execute_tree(tree_xu, data)
        torch.testing.assert_close(r_ux, r_xu, rtol=RTOL, atol=ATOL)

    def test_sub_anticommutativity(self) -> None:
        data = _make_data()
        tree_ux = Tree(root=_binary("-", _leaf("u"), _leaf("x")))
        tree_xu = Tree(root=_binary("-", _leaf("x"), _leaf("u")))

        r_ux = execute_tree(tree_ux, data)
        r_xu = execute_tree(tree_xu, data)
        torch.testing.assert_close(r_ux, -r_xu, rtol=RTOL, atol=ATOL)

    def test_square_is_nonnegative(self) -> None:
        data = _make_data()
        tree = Tree(root=_unary("^2", _leaf("u")))
        result = execute_tree(tree, data)
        assert (result >= 0).all()

    def test_cube_preserves_sign(self) -> None:
        data = _make_data()

        data["u"] = data["u"].clamp(min=0.01)
        tree = Tree(root=_unary("^3", _leaf("u")))
        result = execute_tree(tree, data)
        assert (result > 0).all()

    def test_identity_via_div_self(self) -> None:
        data = _make_data()

        data["u"] = data["u"].abs().clamp(min=0.1)
        tree = Tree(root=_binary("/", _leaf("u"), _leaf("u")))
        result = execute_tree(tree, data)
        expected = torch.ones_like(result)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)







class TestAllOperatorsCovered:

    @pytest.mark.parametrize("op", ["+", "-", "*", "/"])
    def test_binary_op_executes(self, op: str) -> None:
        data = _make_data()
        data["u"] = data["u"].abs().clamp(min=0.1)
        data["x"] = data["x"].abs().clamp(min=0.1)
        tree = Tree(root=_binary(op, _leaf("u"), _leaf("x")))
        result = execute_tree(tree, data)
        assert result.shape == (N_SAMPLES,)
        assert torch.isfinite(result).all()

    @pytest.mark.parametrize("op", ["^2", "^3"])
    def test_unary_op_executes(self, op: str) -> None:
        data = _make_data()
        tree = Tree(root=_unary(op, _leaf("u")))
        result = execute_tree(tree, data)
        assert result.shape == (N_SAMPLES,)
        assert torch.isfinite(result).all()







class TestErrorHandling:

    def test_unknown_operator_raises(self) -> None:
        data = _make_data()
        tree = Tree(root=_unary("sin", _leaf("u")))
        with pytest.raises((KeyError, ValueError)):
            execute_tree(tree, data)

    def test_unknown_binary_operator_raises(self) -> None:
        data = _make_data()
        tree = Tree(root=_binary("mod", _leaf("u"), _leaf("x")))
        with pytest.raises((KeyError, ValueError)):
            execute_tree(tree, data)

    def test_arity_mismatch_unary_with_two_children(self) -> None:
        data = _make_data()

        bad_node = Node(name="^2", arity=1, children=[_leaf("u"), _leaf("x")])
        tree = Tree(root=bad_node)


        try:
            result = execute_tree(tree, data)

            expected = data["u"] ** 2
            torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)
        except (IndexError, ValueError, TypeError):
            pass

    def test_build_theta_mismatched_rows_raises(self) -> None:

        ok_result = build_theta(torch.randn(10, 2), default_terms=torch.randn(10, 1))
        assert ok_result.shape == (10, 3)


        defaults = torch.randn(10, 2)
        valid = torch.randn(20, 3)
        with pytest.raises((RuntimeError, ValueError)):
            build_theta(valid, default_terms=defaults)







class TestZeroColumnThreshold:

    def test_near_zero_column_filtered(self) -> None:
        data = _make_data()
        data["tiny"] = torch.full((N_SAMPLES,), 1e-15)
        pde = PDE(
            terms=[
                Tree(root=_leaf("u")),
                Tree(root=_leaf("tiny")),
            ]
        )

        valid_terms, valid_indices = execute_pde(pde, data)

        assert 1 not in valid_indices
        assert 0 in valid_indices

    def test_small_but_nonzero_column_kept(self) -> None:
        data = _make_data()
        data["small_meaningful"] = torch.full((N_SAMPLES,), 0.01)
        pde = PDE(
            terms=[
                Tree(root=_leaf("small_meaningful")),
            ]
        )

        valid_terms, valid_indices = execute_pde(pde, data)
        assert valid_indices == [0]







def _make_grid_data(
    nx: int = 50,
    dx: float = 0.1,
) -> dict[str, Tensor]:
    x = torch.linspace(0.0, (nx - 1) * dx, nx)
    u = torch.sin(x)
    return {
        "u": u,
        "x": x,
    }


def _make_diff_ctx(
    nx: int = 50,
    dx: float = 0.1,
    lhs_axis: str | None = None,
) -> DiffContext:
    delta = {"x": dx}
    if lhs_axis is not None:
        delta[lhs_axis] = 0.02
    return DiffContext(
        field_shape=(nx,),
        axis_map={"x": 0},
        delta=delta,
        lhs_axis=lhs_axis,
    )


class TestDerivativeExecution:

    def test_d_dispatch_exists(self) -> None:



        data = _make_grid_data()
        diff_ctx = _make_diff_ctx(nx=data["u"].shape[0])
        d_node = _binary("d", _leaf("u"), _leaf("x"))
        tree = Tree(root=d_node)

        result = execute_tree(tree, data, diff_ctx=diff_ctx)
        assert isinstance(result, Tensor)

    def test_d2_dispatch_exists(self) -> None:
        data = _make_grid_data()
        diff_ctx = _make_diff_ctx(nx=data["u"].shape[0])
        d2_node = _binary("d^2", _leaf("u"), _leaf("x"))
        tree = Tree(root=d2_node)
        result = execute_tree(tree, data, diff_ctx=diff_ctx)
        assert isinstance(result, Tensor)

    def test_d_simple_leaf(self) -> None:
        nx = 50
        dx = 0.1
        data = _make_grid_data(nx=nx, dx=dx)
        diff_ctx = _make_diff_ctx(nx=nx, dx=dx)

        tree = Tree(root=_binary("d", _leaf("u"), _leaf("x")))
        result = execute_tree(tree, data, diff_ctx=diff_ctx)



        x = data["x"]
        expected = torch.cos(x)

        torch.testing.assert_close(result, expected, rtol=0.05, atol=0.02)

    def test_d2_simple_leaf(self) -> None:
        nx = 50
        dx = 0.1
        data = _make_grid_data(nx=nx, dx=dx)
        diff_ctx = _make_diff_ctx(nx=nx, dx=dx)

        tree = Tree(root=_binary("d^2", _leaf("u"), _leaf("x")))
        result = execute_tree(tree, data, diff_ctx=diff_ctx)


        x = data["x"]
        expected = -torch.sin(x)

        torch.testing.assert_close(result, expected, rtol=0.1, atol=0.05)

    def test_d_composite_expr(self) -> None:
        nx = 50
        dx = 0.1
        data = _make_grid_data(nx=nx, dx=dx)
        diff_ctx = _make_diff_ctx(nx=nx, dx=dx)


        mul_node = _binary("*", _leaf("u"), _leaf("u"))
        d_node = _binary("d", mul_node, _leaf("x"))
        tree = Tree(root=d_node)
        result = execute_tree(tree, data, diff_ctx=diff_ctx)

        x = data["x"]
        expected = torch.sin(2.0 * x)
        torch.testing.assert_close(result, expected, rtol=0.1, atol=0.05)

    def test_d_nested_derivative(self) -> None:
        nx = 100
        dx = 0.05
        data = _make_grid_data(nx=nx, dx=dx)
        diff_ctx = _make_diff_ctx(nx=nx, dx=dx)


        inner_d = _binary("d", _leaf("u"), _leaf("x"))
        outer_d = _binary("d", inner_d, _leaf("x"))
        tree = Tree(root=outer_d)
        result = execute_tree(tree, data, diff_ctx=diff_ctx)

        x = data["x"]
        expected = -torch.sin(x)

        torch.testing.assert_close(result, expected, rtol=0.2, atol=0.1)

    def test_d_output_shape(self) -> None:
        data = _make_grid_data()
        diff_ctx = _make_diff_ctx(nx=data["u"].shape[0])
        tree = Tree(root=_binary("d", _leaf("u"), _leaf("x")))
        result = execute_tree(tree, data, diff_ctx=diff_ctx)
        assert result.ndim == 1
        assert result.shape[0] == data["u"].shape[0]

    def test_d_stateless(self) -> None:
        data = _make_grid_data()
        diff_ctx = _make_diff_ctx(nx=data["u"].shape[0])
        tree = Tree(root=_binary("d", _leaf("u"), _leaf("x")))
        str_before = str(tree)
        _ = execute_tree(tree, data, diff_ctx=diff_ctx)
        assert str(tree) == str_before


class TestDerivativeLHSRejection:

    def test_d_along_lhs_axis_is_filtered(self) -> None:
        data = _make_grid_data()
        data["t"] = torch.linspace(0.0, 1.0, data["u"].shape[0])
        diff_ctx = _make_diff_ctx(nx=data["u"].shape[0], lhs_axis="t")

        tree_ok = Tree(root=_binary("d", _leaf("u"), _leaf("x")))
        tree_bad = Tree(root=_binary("d", _leaf("u"), _leaf("t")))
        pde = PDE(terms=[tree_ok, tree_bad])

        valid_terms, valid_indices = execute_pde(pde, data, diff_ctx=diff_ctx)

        assert 0 in valid_indices
        assert 1 not in valid_indices

    def test_d2_along_lhs_axis_is_filtered(self) -> None:
        data = _make_grid_data()
        data["t"] = torch.linspace(0.0, 1.0, data["u"].shape[0])
        diff_ctx = _make_diff_ctx(nx=data["u"].shape[0], lhs_axis="t")

        tree_ok = Tree(root=_leaf("u"))
        tree_bad = Tree(root=_binary("d^2", _leaf("u"), _leaf("t")))
        pde = PDE(terms=[tree_ok, tree_bad])

        valid_terms, valid_indices = execute_pde(pde, data, diff_ctx=diff_ctx)
        assert 0 in valid_indices
        assert 1 not in valid_indices







class TestPruneInvalidTerms:

    def test_prune_removes_nan_terms(self) -> None:
        from kd.search.sga.evaluate import prune_invalid_terms

        data = _make_data()
        data["z"] = torch.full((N_SAMPLES,), float("nan"))
        t_valid = Tree(root=_leaf("u"))
        t_nan = Tree(root=_leaf("z"))
        pde = PDE(terms=[t_valid, t_nan])

        pruned_pde, valid_terms, valid_indices = prune_invalid_terms(pde, data)
        assert pruned_pde.width == 1
        assert str(pruned_pde.terms[0]) == str(t_valid)
        assert valid_terms.shape[1] == 1
        assert valid_indices == [0]

    def test_prune_removes_zero_terms(self) -> None:
        from kd.search.sga.evaluate import prune_invalid_terms

        data = _make_data()
        data["zeros"] = torch.zeros(N_SAMPLES)
        t_valid = Tree(root=_leaf("u"))
        t_zero = Tree(root=_leaf("zeros"))
        pde = PDE(terms=[t_valid, t_zero])

        pruned_pde, valid_terms, valid_indices = prune_invalid_terms(pde, data)
        assert pruned_pde.width == 1
        assert str(pruned_pde.terms[0]) == str(t_valid)

    def test_prune_removes_inf_terms(self) -> None:
        from kd.search.sga.evaluate import prune_invalid_terms

        data = _make_data()
        data["inf_var"] = torch.full((N_SAMPLES,), float("inf"))
        t_valid = Tree(root=_leaf("u"))
        t_inf = Tree(root=_leaf("inf_var"))
        pde = PDE(terms=[t_valid, t_inf])

        pruned_pde, valid_terms, valid_indices = prune_invalid_terms(pde, data)
        assert pruned_pde.width == 1
        assert str(pruned_pde.terms[0]) == str(t_valid)

    def test_prune_preserves_valid_term_order(self) -> None:
        from kd.search.sga.evaluate import prune_invalid_terms

        data = _make_data()
        data["zeros"] = torch.zeros(N_SAMPLES)

        t_u = Tree(root=_leaf("u"))
        t_z1 = Tree(root=_leaf("zeros"))
        t_x = Tree(root=_leaf("x"))
        t_z2 = Tree(root=_leaf("zeros"))
        t_ux = Tree(root=_binary("*", _leaf("u"), _leaf("x")))
        pde = PDE(terms=[t_u, t_z1, t_x, t_z2, t_ux])

        pruned_pde, valid_terms, valid_indices = prune_invalid_terms(pde, data)
        assert pruned_pde.width == 3
        assert valid_indices == [0, 2, 4]
        assert str(pruned_pde.terms[0]) == str(t_u)
        assert str(pruned_pde.terms[1]) == str(t_x)
        assert str(pruned_pde.terms[2]) == str(t_ux)

    def test_prune_returns_aligned_theta(self) -> None:
        from kd.search.sga.evaluate import prune_invalid_terms

        data = _make_data()
        data["zeros"] = torch.zeros(N_SAMPLES)
        pde = PDE(
            terms=[
                Tree(root=_leaf("u")),
                Tree(root=_leaf("zeros")),
                Tree(root=_leaf("x")),
            ]
        )

        pruned_pde, valid_terms, valid_indices = prune_invalid_terms(pde, data)

        assert valid_terms.shape[1] == pruned_pde.width
        torch.testing.assert_close(valid_terms[:, 0], data["u"], rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(valid_terms[:, 1], data["x"], rtol=RTOL, atol=ATOL)

    def test_prune_does_not_mutate_original(self) -> None:
        from kd.search.sga.evaluate import prune_invalid_terms

        data = _make_data()
        data["zeros"] = torch.zeros(N_SAMPLES)
        pde = PDE(
            terms=[
                Tree(root=_leaf("u")),
                Tree(root=_leaf("zeros")),
            ]
        )
        original_width = pde.width
        original_str = str(pde)

        pruned_pde, _, _ = prune_invalid_terms(pde, data)


        assert pde.width == original_width
        assert str(pde) == original_str

        assert pruned_pde.width == 1

    def test_prune_all_invalid_returns_empty(self) -> None:
        from kd.search.sga.evaluate import prune_invalid_terms

        data = _make_data()
        data["nan_var"] = torch.full((N_SAMPLES,), float("nan"))
        data["inf_var"] = torch.full((N_SAMPLES,), float("inf"))
        pde = PDE(
            terms=[
                Tree(root=_leaf("nan_var")),
                Tree(root=_leaf("inf_var")),
            ]
        )

        pruned_pde, valid_terms, valid_indices = prune_invalid_terms(pde, data)
        assert pruned_pde.width == 0
        assert valid_terms.shape[1] == 0
        assert valid_indices == []

    def test_prune_empty_pde(self) -> None:
        from kd.search.sga.evaluate import prune_invalid_terms

        data = _make_data()
        pde = PDE(terms=[])

        pruned_pde, valid_terms, valid_indices = prune_invalid_terms(pde, data)
        assert pruned_pde.width == 0
        assert valid_indices == []

    def test_prune_with_lhs_axis_filtering(self) -> None:
        from kd.search.sga.evaluate import prune_invalid_terms

        data = _make_grid_data()
        data["t"] = torch.linspace(0.0, 1.0, data["u"].shape[0])
        diff_ctx = _make_diff_ctx(nx=data["u"].shape[0], lhs_axis="t")

        tree_ok = Tree(root=_binary("d", _leaf("u"), _leaf("x")))
        tree_bad = Tree(root=_binary("d", _leaf("u"), _leaf("t")))
        pde = PDE(terms=[tree_ok, tree_bad])

        pruned_pde, valid_terms, valid_indices = prune_invalid_terms(
            pde, data, diff_ctx=diff_ctx
        )
        assert pruned_pde.width == 1
        assert str(pruned_pde.terms[0]) == str(tree_ok)
        assert valid_indices == [0]
