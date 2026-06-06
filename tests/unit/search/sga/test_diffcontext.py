
from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import Tensor

from kd.search.sga.evaluate import (
    DiffContext,
    _finite_diff_torch,
    execute_pde,
    execute_tree,
    prune_invalid_terms,
)
from kd.search.sga.pde import PDE
from kd.search.sga.tree import Node, Tree



RTOL = 1e-4
ATOL = 1e-6

BOUNDARY_RTOL = 5e-3
BOUNDARY_ATOL = 5e-3





def _leaf(name: str) -> Node:
    return Node(name=name, arity=0, children=[])


def _unary(op: str, child: Node) -> Node:
    return Node(name=op, arity=1, children=[child])


def _binary(op: str, left: Node, right: Node) -> Node:
    return Node(name=op, arity=2, children=[left, right])


def _make_1d_grid(n: int, x_min: float, x_max: float) -> tuple[Tensor, float]:
    x = torch.linspace(x_min, x_max, n, dtype=torch.float64)
    dx = (x_max - x_min) / (n - 1)
    return x, dx


def _make_2d_grid(
    nx: int,
    nt: int,
    x_range: tuple[float, float] = (0.0, 2.0 * math.pi),
    t_range: tuple[float, float] = (0.0, math.pi),
) -> tuple[Tensor, Tensor, float, float]:
    x = torch.linspace(x_range[0], x_range[1], nx, dtype=torch.float64)
    t = torch.linspace(t_range[0], t_range[1], nt, dtype=torch.float64)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    dx = (x_range[1] - x_range[0]) / (nx - 1)
    dt = (t_range[1] - t_range[0]) / (nt - 1)
    return xg, tg, dx, dt


def _make_3d_grid(
    nx: int,
    ny: int,
    nt: int,
) -> tuple[Tensor, Tensor, Tensor, float, float, float]:
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    y = torch.linspace(0.0, 1.0, ny, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    xg, yg, tg = torch.meshgrid(x, y, t, indexing="ij")
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    dt = 1.0 / (nt - 1)
    return xg, yg, tg, dx, dy, dt







class TestDiffContextSmoke:

    @pytest.mark.smoke
    def test_import(self) -> None:
        assert DiffContext is not None

    @pytest.mark.smoke
    def test_construct_minimal(self) -> None:
        ctx = DiffContext(field_shape=(100,))
        assert ctx.field_shape == (100,)

    @pytest.mark.smoke
    def test_construct_full(self) -> None:
        ctx = DiffContext(
            field_shape=(256, 100),
            axis_map={"x": 0, "t": 1},
            delta={"x": 0.01, "t": 0.001},
            lhs_axis="t",
        )
        assert ctx.field_shape == (256, 100)
        assert ctx.lhs_axis == "t"

    @pytest.mark.smoke
    def test_finite_diff_torch_callable(self) -> None:
        assert callable(_finite_diff_torch)







class TestDiffContextFields:

    def test_field_shape_stored(self) -> None:
        ctx = DiffContext(field_shape=(64, 32))
        assert ctx.field_shape == (64, 32)

    def test_axis_map_default_empty(self) -> None:
        ctx = DiffContext(field_shape=(10,))
        assert ctx.axis_map == {}

    def test_delta_default_empty(self) -> None:
        ctx = DiffContext(field_shape=(10,))
        assert ctx.delta == {}

    def test_lhs_axis_default_none(self) -> None:
        ctx = DiffContext(field_shape=(10,))
        assert ctx.lhs_axis is None

    def test_axis_map_access(self) -> None:
        ctx = DiffContext(
            field_shape=(256, 100),
            axis_map={"x": 0, "t": 1},
        )
        assert ctx.axis_map["x"] == 0
        assert ctx.axis_map["t"] == 1

    def test_delta_access(self) -> None:
        ctx = DiffContext(
            field_shape=(256, 100),
            delta={"x": 0.01, "t": 0.001},
        )
        assert ctx.delta["x"] == pytest.approx(0.01)
        assert ctx.delta["t"] == pytest.approx(0.001)

    def test_no_mutable_default_sharing(self) -> None:
        ctx1 = DiffContext(field_shape=(10,))
        ctx2 = DiffContext(field_shape=(10,))
        ctx1.axis_map["x"] = 0
        assert "x" not in ctx2.axis_map







class TestFiniteDiffTorch1D:



    def test_order1_sin(self) -> None:
        n = 200
        x, dx = _make_1d_grid(n, 0.0, 2.0 * math.pi)
        f = torch.sin(x)
        expected = torch.cos(x)
        result = _finite_diff_torch(f, dx, axis=0, order=1)
        assert result.shape == f.shape

        torch.testing.assert_close(result[1:-1], expected[1:-1], rtol=RTOL, atol=ATOL)

    def test_order1_polynomial(self) -> None:
        n = 300
        x, dx = _make_1d_grid(n, -1.0, 1.0)
        f = x**3
        expected = 3.0 * x**2
        result = _finite_diff_torch(f, dx, axis=0, order=1)


        torch.testing.assert_close(result[1:-1], expected[1:-1], rtol=1e-3, atol=1e-3)

    def test_order1_preserves_shape(self) -> None:
        f = torch.randn(50, dtype=torch.float64)
        result = _finite_diff_torch(f, dx=0.1, axis=0, order=1)
        assert result.shape == f.shape

    def test_order1_constant_is_zero(self) -> None:
        f = torch.full((100,), 7.0, dtype=torch.float64)
        result = _finite_diff_torch(f, dx=0.01, axis=0, order=1)
        torch.testing.assert_close(
            result, torch.zeros_like(result), rtol=RTOL, atol=ATOL
        )

    def test_order1_linear_is_slope(self) -> None:
        n = 100
        x, dx = _make_1d_grid(n, 0.0, 1.0)
        slope = 3.5
        f = slope * x
        result = _finite_diff_torch(f, dx=dx, axis=0, order=1)
        expected = torch.full_like(f, slope)
        torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)

    def test_order1_boundary_formula(self) -> None:
        n = 200
        x, dx = _make_1d_grid(n, 0.0, 2.0 * math.pi)
        f = torch.sin(x)
        expected = torch.cos(x)
        result = _finite_diff_torch(f, dx, axis=0, order=1)

        torch.testing.assert_close(
            result[0:1], expected[0:1], rtol=BOUNDARY_RTOL, atol=BOUNDARY_ATOL
        )
        torch.testing.assert_close(
            result[-1:], expected[-1:], rtol=BOUNDARY_RTOL, atol=BOUNDARY_ATOL
        )

    def test_order2_boundary_formula(self) -> None:
        n = 300
        x, dx = _make_1d_grid(n, 0.0, 2.0 * math.pi)
        f = torch.sin(x)
        expected = -torch.sin(x)
        result = _finite_diff_torch(f, dx, axis=0, order=2)
        torch.testing.assert_close(
            result[0:1], expected[0:1], rtol=BOUNDARY_RTOL, atol=BOUNDARY_ATOL
        )
        torch.testing.assert_close(
            result[-1:], expected[-1:], rtol=BOUNDARY_RTOL, atol=BOUNDARY_ATOL
        )



    def test_order2_sin(self) -> None:
        n = 300
        x, dx = _make_1d_grid(n, 0.0, 2.0 * math.pi)
        f = torch.sin(x)
        expected = -torch.sin(x)
        result = _finite_diff_torch(f, dx, axis=0, order=2)
        assert result.shape == f.shape
        torch.testing.assert_close(result[1:-1], expected[1:-1], rtol=RTOL, atol=ATOL)

    def test_order2_quadratic(self) -> None:
        n = 100
        x, dx = _make_1d_grid(n, -1.0, 1.0)
        a = 5.0
        f = a * x**2
        expected = torch.full_like(f, 2.0 * a)
        result = _finite_diff_torch(f, dx, axis=0, order=2)

        torch.testing.assert_close(result[1:-1], expected[1:-1], rtol=RTOL, atol=ATOL)

    def test_order2_constant_is_zero(self) -> None:
        f = torch.full((100,), 3.14, dtype=torch.float64)
        result = _finite_diff_torch(f, dx=0.01, axis=0, order=2)
        torch.testing.assert_close(
            result, torch.zeros_like(result), rtol=RTOL, atol=ATOL
        )

    def test_order2_linear_is_zero(self) -> None:
        n = 100
        x, dx = _make_1d_grid(n, 0.0, 1.0)
        f = 2.5 * x
        result = _finite_diff_torch(f, dx=dx, axis=0, order=2)
        torch.testing.assert_close(
            result, torch.zeros_like(result), rtol=RTOL, atol=ATOL
        )

    def test_order2_preserves_shape(self) -> None:
        f = torch.randn(50, dtype=torch.float64)
        result = _finite_diff_torch(f, dx=0.1, axis=0, order=2)
        assert result.shape == f.shape







class TestFiniteDiffTorchND:



    def test_2d_du_dx(self) -> None:
        nx, nt = 100, 80
        xg, tg, dx, dt = _make_2d_grid(nx, nt)
        u = torch.sin(xg) * torch.cos(tg)
        expected = torch.cos(xg) * torch.cos(tg)
        result = _finite_diff_torch(u, dx, axis=0, order=1)
        assert result.shape == (nx, nt)

        torch.testing.assert_close(
            result[1:-1,:], expected[1:-1,:], rtol=RTOL, atol=ATOL
        )

    def test_2d_du_dt(self) -> None:
        nx, nt = 100, 80
        xg, tg, dx, dt = _make_2d_grid(nx, nt)
        u = torch.sin(xg) * torch.cos(tg)
        expected = -torch.sin(xg) * torch.sin(tg)
        result = _finite_diff_torch(u, dt, axis=1, order=1)
        assert result.shape == (nx, nt)
        torch.testing.assert_close(
            result[:, 1:-1], expected[:, 1:-1], rtol=RTOL, atol=ATOL
        )

    def test_2d_d2u_dx2(self) -> None:
        nx, nt = 150, 80
        xg, tg, dx, dt = _make_2d_grid(nx, nt)
        u = torch.sin(xg) * torch.cos(tg)
        expected = -torch.sin(xg) * torch.cos(tg)
        result = _finite_diff_torch(u, dx, axis=0, order=2)
        assert result.shape == (nx, nt)
        torch.testing.assert_close(
            result[1:-1,:], expected[1:-1,:], rtol=RTOL, atol=ATOL
        )

    def test_2d_shape_preserved(self) -> None:
        f = torch.randn(30, 20, dtype=torch.float64)
        for axis in (0, 1):
            result = _finite_diff_torch(f, dx=0.1, axis=axis, order=1)
            assert result.shape == f.shape



    def test_3d_du_dx(self) -> None:
        nx, ny, nt = 30, 20, 15
        xg, yg, _tg, dx, dy, dt = _make_3d_grid(nx, ny, nt)
        u = xg**2 + yg
        expected = 2.0 * xg
        result = _finite_diff_torch(u, dx, axis=0, order=1)
        assert result.shape == (nx, ny, nt)
        torch.testing.assert_close(
            result[1:-1, :,:], expected[1:-1, :,:], rtol=RTOL, atol=ATOL
        )

    def test_3d_d2u_dx2(self) -> None:
        nx, ny, nt = 30, 20, 15
        xg, yg, _tg, dx, dy, dt = _make_3d_grid(nx, ny, nt)
        u = xg**2 + yg
        expected = torch.full_like(u, 2.0)
        result = _finite_diff_torch(u, dx, axis=0, order=2)
        assert result.shape == (nx, ny, nt)
        torch.testing.assert_close(
            result[1:-1, :,:], expected[1:-1, :,:], rtol=RTOL, atol=ATOL
        )

    def test_3d_du_dy(self) -> None:
        nx, ny, nt = 30, 20, 15
        xg, yg, _tg, dx, dy, dt = _make_3d_grid(nx, ny, nt)
        u = xg**2 + yg
        expected = torch.ones_like(u)
        result = _finite_diff_torch(u, dy, axis=1, order=1)
        assert result.shape == (nx, ny, nt)
        torch.testing.assert_close(
            result[:, 1:-1,:], expected[:, 1:-1,:], rtol=RTOL, atol=ATOL
        )

    def test_3d_du_dt_zero(self) -> None:
        nx, ny, nt = 30, 20, 15
        xg, yg, _tg, dx, dy, dt = _make_3d_grid(nx, ny, nt)
        u = xg**2 + yg
        result = _finite_diff_torch(u, dt, axis=2, order=1)
        assert result.shape == (nx, ny, nt)
        torch.testing.assert_close(
            result, torch.zeros_like(result), rtol=RTOL, atol=ATOL
        )

    def test_3d_shape_preserved(self) -> None:
        f = torch.randn(10, 12, 8, dtype=torch.float64)
        for axis in (0, 1, 2):
            result = _finite_diff_torch(f, dx=0.1, axis=axis, order=1)
            assert result.shape == f.shape







class TestFiniteDiffTorchEdgeCases:



    def test_order1_min_3_points_pass(self) -> None:
        f = torch.tensor([0.0, 1.0, 4.0], dtype=torch.float64)
        result = _finite_diff_torch(f, dx=1.0, axis=0, order=1)
        assert result.shape == (3,)

    def test_order1_fewer_than_3_points_raises(self) -> None:
        f = torch.tensor([1.0, 2.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="[Pp]oint|[Mm]in"):
            _finite_diff_torch(f, dx=1.0, axis=0, order=1)

    def test_order2_min_4_points_pass(self) -> None:
        f = torch.tensor([0.0, 1.0, 4.0, 9.0], dtype=torch.float64)
        result = _finite_diff_torch(f, dx=1.0, axis=0, order=2)
        assert result.shape == (4,)

    def test_order2_fewer_than_4_points_raises(self) -> None:
        f = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="[Pp]oint|[Mm]in"):
            _finite_diff_torch(f, dx=1.0, axis=0, order=2)

    def test_nd_min_points_on_target_axis(self) -> None:

        f = torch.randn(2, 50, dtype=torch.float64)
        with pytest.raises(ValueError, match="[Pp]oint|[Mm]in"):
            _finite_diff_torch(f, dx=0.1, axis=0, order=1)

    def test_nd_min_points_other_axis_ok(self) -> None:

        f = torch.randn(2, 50, dtype=torch.float64)
        result = _finite_diff_torch(f, dx=0.1, axis=1, order=1)
        assert result.shape == (2, 50)



    def test_dx_zero_raises(self) -> None:
        f = torch.randn(10, dtype=torch.float64)
        with pytest.raises(ValueError, match="[Zz]ero|[Ff]inite|dx"):
            _finite_diff_torch(f, dx=0.0, axis=0, order=1)

    def test_dx_nan_raises(self) -> None:
        f = torch.randn(10, dtype=torch.float64)
        with pytest.raises(ValueError):
            _finite_diff_torch(f, dx=float("nan"), axis=0, order=1)

    def test_dx_inf_raises(self) -> None:
        f = torch.randn(10, dtype=torch.float64)
        with pytest.raises(ValueError):
            _finite_diff_torch(f, dx=float("inf"), axis=0, order=1)

    def test_dx_negative_inf_raises(self) -> None:
        f = torch.randn(10, dtype=torch.float64)
        with pytest.raises(ValueError):
            _finite_diff_torch(f, dx=float("-inf"), axis=0, order=1)



    def test_order_zero_raises(self) -> None:
        f = torch.randn(10, dtype=torch.float64)
        with pytest.raises((ValueError, NotImplementedError)):
            _finite_diff_torch(f, dx=0.1, axis=0, order=0)

    def test_order_three_raises(self) -> None:
        f = torch.randn(10, dtype=torch.float64)
        with pytest.raises((ValueError, NotImplementedError)):
            _finite_diff_torch(f, dx=0.1, axis=0, order=3)



    def test_axis_out_of_range_raises(self) -> None:
        f = torch.randn(10, dtype=torch.float64)
        with pytest.raises((IndexError, ValueError)):
            _finite_diff_torch(f, dx=0.1, axis=1, order=1)

    def test_negative_axis(self) -> None:
        f = torch.randn(10, 20, dtype=torch.float64)

        try:
            result = _finite_diff_torch(f, dx=0.1, axis=-1, order=1)
            assert result.shape == f.shape
        except (ValueError, IndexError):
            pass







class TestFiniteDiffTorchProperties:

    @given(
        n=st.integers(min_value=5, max_value=200),
        c=st.floats(
            min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=30)
    def test_order1_constant_property(self, n: int, c: float) -> None:
        f = torch.full((n,), c, dtype=torch.float64)
        dx = 1.0 / (n - 1)
        result = _finite_diff_torch(f, dx=dx, axis=0, order=1)
        assert result.shape == (n,)
        assert torch.isfinite(result).all()
        assert result.abs().max().item() < 1e-8

    @given(
        n=st.integers(min_value=5, max_value=200),
        slope=st.floats(
            min_value=-100,
            max_value=100,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=30)
    def test_order2_linear_property(self, n: int, slope: float) -> None:
        x = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
        f = slope * x
        dx = 1.0 / (n - 1)
        result = _finite_diff_torch(f, dx=dx, axis=0, order=2)
        assert result.shape == (n,)
        assert torch.isfinite(result).all()
        assert result.abs().max().item() < 1e-6

    @given(
        n=st.integers(min_value=10, max_value=200),
        a=st.floats(
            min_value=0.01,
            max_value=50.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=20)
    def test_order1_output_finite(self, n: int, a: float) -> None:
        x = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
        f = a * torch.sin(x)
        dx = 1.0 / (n - 1)
        result = _finite_diff_torch(f, dx=dx, axis=0, order=1)
        assert torch.isfinite(result).all()

    @given(
        nx=st.integers(min_value=5, max_value=30),
        nt=st.integers(min_value=5, max_value=30),
    )
    @settings(max_examples=15)
    def test_nd_shape_invariant(self, nx: int, nt: int) -> None:
        f = torch.randn(nx, nt, dtype=torch.float64)
        for axis in (0, 1):
            result = _finite_diff_torch(f, dx=0.1, axis=axis, order=1)
            assert result.shape == (nx, nt)

    @given(
        n=st.integers(min_value=10, max_value=200),
        slope=st.floats(
            min_value=0.1,
            max_value=50.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=20)
    def test_order1_linear_equals_slope(self, n: int, slope: float) -> None:
        x = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
        f = slope * x
        dx = 1.0 / (n - 1)
        result = _finite_diff_torch(f, dx=dx, axis=0, order=1)

        expected = torch.full((n,), slope, dtype=torch.float64)
        torch.testing.assert_close(result[1:-1], expected[1:-1], rtol=1e-4, atol=1e-6)







class TestFiniteDiffTorchNumerical:

    @pytest.mark.numerical
    def test_nan_in_input_propagates(self) -> None:
        f = torch.ones(10, dtype=torch.float64)
        f[5] = float("nan")
        result = _finite_diff_torch(f, dx=0.1, axis=0, order=1)

        assert torch.isnan(result).any() or not torch.isfinite(result).all()

    @pytest.mark.numerical
    def test_inf_in_input_propagates(self) -> None:
        f = torch.ones(10, dtype=torch.float64)
        f[5] = float("inf")
        result = _finite_diff_torch(f, dx=0.1, axis=0, order=1)

        has_nonfinite = not torch.isfinite(result).all()
        assert has_nonfinite

    @pytest.mark.numerical
    def test_very_small_dx(self) -> None:
        n = 100
        x = torch.linspace(0.0, 1e-10, n, dtype=torch.float64)
        f = x**2
        dx = 1e-10 / (n - 1)
        result = _finite_diff_torch(f, dx=dx, axis=0, order=1)

        assert torch.isfinite(result).all()

    @pytest.mark.numerical
    def test_negative_dx_raises_or_handles(self) -> None:
        f = torch.randn(10, dtype=torch.float64)



        try:
            result = _finite_diff_torch(f, dx=-0.1, axis=0, order=1)

            assert torch.isfinite(result).all()
        except ValueError:
            pass







class TestExecuteTreeWithDiffCtx:

    def test_non_derivative_tree_no_diffctx(self) -> None:
        tree = Tree(root=_binary("*", _leaf("u"), _leaf("x")))
        data = {
            "u": torch.ones(10),
            "x": torch.full((10,), 2.0),
        }

        result = execute_tree(tree, data, diff_ctx=None)
        expected = torch.full((10,), 2.0)
        torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)

    def test_derivative_tree_requires_diffctx(self) -> None:
        tree = Tree(root=_binary("d", _leaf("u"), _leaf("x")))
        data = {"u": torch.randn(50), "x": torch.randn(50)}
        with pytest.raises((ValueError, TypeError)):
            execute_tree(tree, data, diff_ctx=None)

    def test_1d_derivative_with_diffctx(self) -> None:
        n = 100
        x, dx = _make_1d_grid(n, 0.0, 2.0 * math.pi)
        u = torch.sin(x)
        data = {"u": u, "x": x}
        ctx = DiffContext(
            field_shape=(n,),
            axis_map={"x": 0},
            delta={"x": dx},
        )
        tree = Tree(root=_binary("d", _leaf("u"), _leaf("x")))
        result = execute_tree(tree, data, diff_ctx=ctx)

        assert result.numel() == n

    def test_2d_derivative_du_dx(self) -> None:
        nx, nt = 80, 60
        xg, tg, dx, dt = _make_2d_grid(nx, nt)
        u = torch.sin(xg) * torch.cos(tg)
        data = {"u": u.reshape(-1), "x": xg.reshape(-1)}
        ctx = DiffContext(
            field_shape=(nx, nt),
            axis_map={"x": 0, "t": 1},
            delta={"x": dx, "t": dt},
        )
        tree = Tree(root=_binary("d", _leaf("u"), _leaf("x")))
        result = execute_tree(tree, data, diff_ctx=ctx)

        assert result.numel() == nx * nt

        result_2d = result.reshape(nx, nt)
        expected = torch.cos(xg) * torch.cos(tg)
        torch.testing.assert_close(
            result_2d[1:-1,:], expected[1:-1,:], rtol=RTOL, atol=ATOL
        )

    def test_2d_second_derivative_d2u_dx2(self) -> None:
        nx, nt = 120, 60
        xg, tg, dx, dt = _make_2d_grid(nx, nt)
        u = torch.sin(xg) * torch.cos(tg)
        data = {"u": u.reshape(-1)}
        ctx = DiffContext(
            field_shape=(nx, nt),
            axis_map={"x": 0, "t": 1},
            delta={"x": dx, "t": dt},
        )
        tree = Tree(root=_binary("d^2", _leaf("u"), _leaf("x")))
        result = execute_tree(tree, data, diff_ctx=ctx)
        result_2d = result.reshape(nx, nt)
        expected = -torch.sin(xg) * torch.cos(tg)
        torch.testing.assert_close(
            result_2d[2:-2,:], expected[2:-2,:], rtol=RTOL, atol=ATOL
        )

    def test_missing_axis_in_diffctx_raises(self) -> None:
        n = 50
        data = {"u": torch.randn(n)}
        ctx = DiffContext(
            field_shape=(n,),
            axis_map={"x": 0},
            delta={"x": 0.1},
        )

        tree = Tree(root=_binary("d", _leaf("u"), _leaf("t")))
        with pytest.raises((KeyError, ValueError)):
            execute_tree(tree, data, diff_ctx=ctx)

    def test_field_shape_data_mismatch_raises(self) -> None:
        n = 50
        data = {"u": torch.randn(n)}
        ctx = DiffContext(
            field_shape=(30, 20),
            axis_map={"x": 0},
            delta={"x": 0.1},
        )
        tree = Tree(root=_binary("d", _leaf("u"), _leaf("x")))
        with pytest.raises((RuntimeError, ValueError)):
            execute_tree(tree, data, diff_ctx=ctx)

    def test_2d_derivative_du_dt(self) -> None:
        nx, nt = 80, 60
        xg, tg, dx, dt = _make_2d_grid(nx, nt)
        u = torch.sin(xg) * torch.cos(tg)
        data = {"u": u.reshape(-1), "t": tg.reshape(-1)}
        ctx = DiffContext(
            field_shape=(nx, nt),
            axis_map={"x": 0, "t": 1},
            delta={"x": dx, "t": dt},
        )
        tree = Tree(root=_binary("d", _leaf("u"), _leaf("t")))
        result = execute_tree(tree, data, diff_ctx=ctx)
        assert result.numel() == nx * nt
        result_2d = result.reshape(nx, nt)
        expected = -torch.sin(xg) * torch.sin(tg)
        torch.testing.assert_close(
            result_2d[:, 1:-1], expected[:, 1:-1], rtol=RTOL, atol=ATOL
        )







class TestExecutePdeWithDiffCtx:

    def test_basic_pde_with_diffctx(self) -> None:
        n = 50
        data = {"u": torch.randn(n), "x": torch.randn(n)}
        ctx = DiffContext(
            field_shape=(n,),
            axis_map={"x": 0},
            delta={"x": 0.1},
        )
        pde = PDE(
            terms=[
                Tree(root=_leaf("u")),
                Tree(root=_leaf("x")),
            ]
        )
        valid_terms, valid_indices = execute_pde(pde, data, diff_ctx=ctx)
        assert valid_terms.shape[1] == 2
        assert valid_indices == [0, 1]

    def test_lhs_axis_filtering_via_diffctx(self) -> None:
        nx, nt = 50, 20
        xg, tg, dx, dt = _make_2d_grid(nx, nt, x_range=(0.0, 1.0), t_range=(0.0, 1.0))
        u = torch.sin(xg) * torch.cos(tg)
        data = {"u": u.reshape(-1), "x": xg.reshape(-1), "t": tg.reshape(-1)}
        ctx = DiffContext(
            field_shape=(nx, nt),
            axis_map={"x": 0, "t": 1},
            delta={"x": dx, "t": dt},
            lhs_axis="t",
        )
        pde = PDE(
            terms=[
                Tree(root=_leaf("u")),
                Tree(root=_binary("d", _leaf("u"), _leaf("t"))),
                Tree(root=_binary("d", _leaf("u"), _leaf("x"))),
            ]
        )
        valid_terms, valid_indices = execute_pde(pde, data, diff_ctx=ctx)

        assert 1 not in valid_indices

        assert 0 in valid_indices
        assert 2 in valid_indices

    def test_no_lhs_axis_keeps_all_valid(self) -> None:
        nx, nt = 50, 20
        xg, tg, dx, dt = _make_2d_grid(nx, nt, x_range=(0.0, 1.0), t_range=(0.0, 1.0))
        u = torch.sin(xg) * torch.cos(tg)
        data = {"u": u.reshape(-1), "x": xg.reshape(-1), "t": tg.reshape(-1)}
        ctx = DiffContext(
            field_shape=(nx, nt),
            axis_map={"x": 0, "t": 1},
            delta={"x": dx, "t": dt},
            lhs_axis=None,
        )
        pde = PDE(
            terms=[
                Tree(root=_leaf("u")),
                Tree(root=_binary("d", _leaf("u"), _leaf("x"))),
            ]
        )
        valid_terms, valid_indices = execute_pde(pde, data, diff_ctx=ctx)
        assert len(valid_indices) == 2

    def test_empty_pde_with_diffctx(self) -> None:
        n = 20
        data = {"u": torch.randn(n)}
        ctx = DiffContext(
            field_shape=(n,),
            axis_map={"x": 0},
            delta={"x": 0.1},
        )
        pde = PDE(terms=[])
        valid_terms, valid_indices = execute_pde(pde, data, diff_ctx=ctx)
        assert valid_terms.shape == (n, 0)
        assert valid_indices == []







class TestPruneInvalidTermsWithDiffCtx:

    def test_prune_with_diffctx(self) -> None:
        nx, nt = 50, 20
        xg, tg, dx, dt = _make_2d_grid(nx, nt, x_range=(0.0, 1.0), t_range=(0.0, 1.0))
        u = torch.sin(xg) * torch.cos(tg)
        data = {"u": u.reshape(-1), "x": xg.reshape(-1), "t": tg.reshape(-1)}
        ctx = DiffContext(
            field_shape=(nx, nt),
            axis_map={"x": 0, "t": 1},
            delta={"x": dx, "t": dt},
            lhs_axis="t",
        )
        pde = PDE(
            terms=[
                Tree(root=_leaf("u")),
                Tree(root=_binary("d", _leaf("u"), _leaf("t"))),
                Tree(root=_binary("d", _leaf("u"), _leaf("x"))),
            ]
        )
        pruned_pde, valid_terms, valid_indices = prune_invalid_terms(
            pde, data, diff_ctx=ctx
        )

        assert 1 not in valid_indices

        assert pruned_pde.width <= pde.width

    def test_prune_preserves_original_pde(self) -> None:
        n = 50
        data = {"u": torch.randn(n)}
        ctx = DiffContext(
            field_shape=(n,),
            axis_map={"x": 0},
            delta={"x": 0.1},
        )
        pde = PDE(terms=[Tree(root=_leaf("u"))])
        original_width = pde.width
        _ = prune_invalid_terms(pde, data, diff_ctx=ctx)
        assert pde.width == original_width







class TestEvaluateCandidateWithDiffCtx:

    def test_evaluate_candidate_accepts_diffctx(self) -> None:
        from kd.search.sga.config import SGAConfig
        from kd.search.sga.train import evaluate_candidate

        n = 100
        x, dx = _make_1d_grid(n, 0.0, 1.0)
        u = torch.sin(x)
        data = {"u": u, "x": x}
        ctx = DiffContext(
            field_shape=(n,),
            axis_map={"x": 0},
            delta={"x": dx},
        )
        pde = PDE(
            terms=[
                Tree(root=_leaf("u")),
                Tree(root=_binary("*", _leaf("u"), _leaf("x"))),
            ]
        )
        y = torch.randn(n, dtype=torch.float64)
        config = SGAConfig()

        result = evaluate_candidate(
            pde,
            data,
            default_terms=None,
            y=y,
            config=config,
            diff_ctx=ctx,
        )
        assert hasattr(result, "aic_score")
        assert hasattr(result, "coefficients")

    def test_evaluate_candidate_diffctx_replaces_delta_lhs(self) -> None:
        from kd.search.sga.config import SGAConfig
        from kd.search.sga.train import evaluate_candidate

        n = 50
        data = {"u": torch.randn(n)}
        ctx = DiffContext(field_shape=(n,), axis_map={"x": 0}, delta={"x": 0.1})
        pde = PDE(terms=[Tree(root=_leaf("u"))])
        y = torch.randn(n)
        config = SGAConfig()





        with pytest.raises(TypeError):
            evaluate_candidate(
                pde,
                data,
                default_terms=None,
                y=y,
                config=config,
                diff_ctx=ctx,
                delta={"x": 0.1},
            )
