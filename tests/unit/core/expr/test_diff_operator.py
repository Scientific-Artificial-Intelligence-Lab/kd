
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from kd.core.executor.context import ExecutionContext
from kd.core.expr.executor import PythonExecutor
from kd.core.expr.registry import FunctionRegistry
from kd.data.derivatives.autograd import AutogradProvider
from kd.data.derivatives.finite_diff import FiniteDiffProvider
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)






class SinProductModel(nn.Module):

    def forward(self, *, x: Tensor, t: Tensor) -> Tensor:
        return torch.sin(x) * torch.exp(-t)


class PolyModel(nn.Module):

    def forward(self, *, x: Tensor, t: Tensor) -> Tensor:
        return x**3 * t**2







_N_X = 30
_N_T = 20


def _make_coords_2d(n_x: int = _N_X, n_t: int = _N_T) -> dict[str, Tensor]:
    x_1d = torch.linspace(0.1, 2 * math.pi - 0.1, n_x, dtype=torch.float64)
    t_1d = torch.linspace(0.1, 1.0, n_t, dtype=torch.float64)
    X, T = torch.meshgrid(x_1d, t_1d, indexing="ij")
    x = X.reshape(-1).clone().detach().requires_grad_(True)
    t = T.reshape(-1).clone().detach().requires_grad_(True)
    return {"x": x, "t": t}


def _make_dataset_2d(n_x: int = _N_X, n_t: int = _N_T) -> PDEDataset:
    x = torch.linspace(0.1, 2 * math.pi - 0.1, n_x, dtype=torch.float64)
    t = torch.linspace(0.1, 1.0, n_t, dtype=torch.float64)
    X, T = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(X) * torch.exp(-T)

    return PDEDataset(
        name="test_diff_operator",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def _make_poly_coords(n_x: int = _N_X, n_t: int = _N_T) -> dict[str, Tensor]:
    x_1d = torch.linspace(0.1, 3.0, n_x, dtype=torch.float64)
    t_1d = torch.linspace(0.1, 2.0, n_t, dtype=torch.float64)
    X, T = torch.meshgrid(x_1d, t_1d, indexing="ij")
    x = X.reshape(-1).clone().detach().requires_grad_(True)
    t = T.reshape(-1).clone().detach().requires_grad_(True)
    return {"x": x, "t": t}


def _make_poly_dataset(n_x: int = _N_X, n_t: int = _N_T) -> PDEDataset:
    x = torch.linspace(0.1, 3.0, n_x, dtype=torch.float64)
    t = torch.linspace(0.1, 2.0, n_t, dtype=torch.float64)
    X, T = torch.meshgrid(x, t, indexing="ij")
    u = X**3 * T**2

    return PDEDataset(
        name="test_poly",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def _make_grid_dataset(n_x: int = 50, n_t: int = 30) -> PDEDataset:
    x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
    t = torch.linspace(0, 1, n_t, dtype=torch.float64)
    X, T = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(X) * torch.exp(-T)

    return PDEDataset(
        name="test_grid",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )







@pytest.fixture
def coords() -> dict[str, Tensor]:
    return _make_coords_2d()


@pytest.fixture
def sin_product_provider(coords: dict[str, Tensor]) -> AutogradProvider:
    model = SinProductModel().double()
    dataset = _make_dataset_2d()
    return AutogradProvider(model=model, coords=coords, dataset=dataset)


@pytest.fixture
def poly_provider() -> AutogradProvider:
    coords = _make_poly_coords()
    model = PolyModel().double()
    dataset = _make_poly_dataset()
    return AutogradProvider(model=model, coords=coords, dataset=dataset)


@pytest.fixture
def executor() -> PythonExecutor:
    return PythonExecutor(FunctionRegistry.create_default())


@pytest.fixture
def autograd_context(
    sin_product_provider: AutogradProvider,
) -> ExecutionContext:
    return ExecutionContext(
        dataset=sin_product_provider.dataset,
        derivative_provider=sin_product_provider,
    )







@pytest.mark.smoke
class TestAutogradProviderDiffSmoke:

    def test_diff_returns_tensor(self, sin_product_provider: AutogradProvider) -> None:
        u = sin_product_provider.get_field("u")
        result = sin_product_provider.diff(u, "x", 1)
        assert isinstance(result, Tensor)

    def test_diff_preserves_shape(self, sin_product_provider: AutogradProvider) -> None:
        u = sin_product_provider.get_field("u")
        result = sin_product_provider.diff(u, "x", 1)
        assert result.shape == u.shape


@pytest.mark.unit
class TestDiffFirstOrder:

    def test_diff_x_matches_analytical(
        self, sin_product_provider: AutogradProvider, coords: dict[str, Tensor]
    ) -> None:
        u = sin_product_provider.get_field("u")
        u_x = sin_product_provider.diff(u, "x", 1)

        x = coords["x"]
        t = coords["t"]
        expected = torch.cos(x) * torch.exp(-t)
        torch.testing.assert_close(u_x, expected, rtol=1e-4, atol=1e-6)

    def test_diff_t_matches_analytical(
        self, sin_product_provider: AutogradProvider, coords: dict[str, Tensor]
    ) -> None:
        u = sin_product_provider.get_field("u")
        u_t = sin_product_provider.diff(u, "t", 1)

        x = coords["x"]
        t = coords["t"]
        expected = -torch.sin(x) * torch.exp(-t)
        torch.testing.assert_close(u_t, expected, rtol=1e-4, atol=1e-6)

    def test_diff_x_matches_get_derivative(
        self, sin_product_provider: AutogradProvider
    ) -> None:
        u = sin_product_provider.get_field("u")
        diff_result = sin_product_provider.diff(u, "x", 1)
        get_deriv_result = sin_product_provider.get_derivative("u", "x", 1)
        torch.testing.assert_close(diff_result, get_deriv_result, rtol=1e-5, atol=1e-8)

    def test_diff_t_matches_get_derivative(
        self, sin_product_provider: AutogradProvider
    ) -> None:
        u = sin_product_provider.get_field("u")
        diff_result = sin_product_provider.diff(u, "t", 1)
        get_deriv_result = sin_product_provider.get_derivative("u", "t", 1)
        torch.testing.assert_close(diff_result, get_deriv_result, rtol=1e-5, atol=1e-8)


@pytest.mark.unit
class TestDiffSecondOrder:

    def test_diff2_x_matches_analytical(
        self, sin_product_provider: AutogradProvider, coords: dict[str, Tensor]
    ) -> None:
        u = sin_product_provider.get_field("u")
        u_xx = sin_product_provider.diff(u, "x", 2)

        x = coords["x"]
        t = coords["t"]
        expected = -torch.sin(x) * torch.exp(-t)
        torch.testing.assert_close(u_xx, expected, rtol=1e-4, atol=1e-6)

    def test_diff2_x_matches_get_derivative(
        self, sin_product_provider: AutogradProvider
    ) -> None:
        u = sin_product_provider.get_field("u")
        diff_result = sin_product_provider.diff(u, "x", 2)
        get_deriv_result = sin_product_provider.get_derivative("u", "x", 2)
        torch.testing.assert_close(diff_result, get_deriv_result, rtol=1e-5, atol=1e-8)


@pytest.mark.unit
class TestDiffProductRule:

    def test_product_rule_sin_product(
        self, sin_product_provider: AutogradProvider, coords: dict[str, Tensor]
    ) -> None:
        u = sin_product_provider.get_field("u")
        u_x = sin_product_provider.diff(u, "x", 1)
        product = u * u_x
        result = sin_product_provider.diff(product, "x", 1)

        x = coords["x"]
        t = coords["t"]
        expected = torch.cos(2 * x) * torch.exp(-2 * t)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)

    def test_product_rule_poly(self, poly_provider: AutogradProvider) -> None:
        u = poly_provider.get_field("u")
        u_x = poly_provider.diff(u, "x", 1)
        product = u * u_x
        result = poly_provider.diff(product, "x", 1)

        x = poly_provider.coords["x"]
        t = poly_provider.coords["t"]
        expected = 15 * x**4 * t**4
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.unit
class TestDiffNested:

    def test_mixed_partial_xt(
        self, sin_product_provider: AutogradProvider, coords: dict[str, Tensor]
    ) -> None:
        u = sin_product_provider.get_field("u")
        u_t = sin_product_provider.diff(u, "t", 1)
        u_tx = sin_product_provider.diff(u_t, "x", 1)

        x = coords["x"]
        t = coords["t"]
        expected = -torch.cos(x) * torch.exp(-t)
        torch.testing.assert_close(u_tx, expected, rtol=1e-4, atol=1e-6)

    def test_mixed_partial_commutativity(
        self, sin_product_provider: AutogradProvider
    ) -> None:
        u = sin_product_provider.get_field("u")

        u_t = sin_product_provider.diff(u, "t", 1)
        u_tx = sin_product_provider.diff(u_t, "x", 1)

        u_x = sin_product_provider.diff(u, "x", 1)
        u_xt = sin_product_provider.diff(u_x, "t", 1)

        torch.testing.assert_close(u_tx, u_xt, rtol=1e-4, atol=1e-6)

    def test_higher_order_via_nesting(
        self, sin_product_provider: AutogradProvider
    ) -> None:
        u = sin_product_provider.get_field("u")


        u_x = sin_product_provider.diff(u, "x", 1)
        u_xx_nested = sin_product_provider.diff(u_x, "x", 1)


        u_xx_direct = sin_product_provider.diff(u, "x", 2)

        torch.testing.assert_close(u_xx_nested, u_xx_direct, rtol=1e-4, atol=1e-6)







@pytest.mark.unit
class TestExecutorDiffEndToEnd:

    def test_diff_x_u(
        self,
        executor: PythonExecutor,
        autograd_context: ExecutionContext,
    ) -> None:
        result = executor.execute("diff_x(u)", autograd_context)
        assert result.used_diff is True
        assert torch.isfinite(result.value).all()

    def test_diff2_x_u(
        self,
        executor: PythonExecutor,
        autograd_context: ExecutionContext,
    ) -> None:
        result = executor.execute("diff2_x(u)", autograd_context)
        assert result.used_diff is True
        assert torch.isfinite(result.value).all()

    def test_diff_t_u(
        self,
        executor: PythonExecutor,
        autograd_context: ExecutionContext,
    ) -> None:
        result = executor.execute("diff_t(u)", autograd_context)
        assert result.used_diff is True
        assert torch.isfinite(result.value).all()

    def test_diff_x_mul_u_ux(
        self,
        executor: PythonExecutor,
        autograd_context: ExecutionContext,
    ) -> None:
        result = executor.execute("diff_x(mul(u, u_x))", autograd_context)
        assert result.used_diff is True
        assert torch.isfinite(result.value).all()

    def test_nested_diff_x_diff_t_u(
        self,
        executor: PythonExecutor,
        autograd_context: ExecutionContext,
    ) -> None:
        result = executor.execute("diff_x(diff_t(u))", autograd_context)
        assert result.used_diff is True
        assert torch.isfinite(result.value).all()

    def test_has_open_form_diff_detection(self) -> None:
        from kd.core.expr.executor import has_open_form_diff

        assert has_open_form_diff("diff_x(u)") is True
        assert has_open_form_diff("diff2_x(u)") is True
        assert has_open_form_diff("diff_t(u)") is True
        assert has_open_form_diff("diff_x(mul(u, u_x))") is True
        assert has_open_form_diff("diff_x(diff_t(u))") is True

        assert has_open_form_diff("u_x") is False
        assert has_open_form_diff("add(u_x, u_xx)") is False

    def test_executor_used_diff_flag(
        self,
        executor: PythonExecutor,
        autograd_context: ExecutionContext,
    ) -> None:
        result = executor.execute("add(u, u)", autograd_context)
        assert result.used_diff is False







@pytest.mark.unit
class TestFiniteDiffProviderDiff:

    def test_fd_diff_returns_finite_tensor(self) -> None:
        dataset = _make_grid_dataset(50, 30)
        provider = FiniteDiffProvider(dataset, max_order=2)

        expr = torch.randn(50, 30, dtype=torch.float64)
        result = provider.diff(expr, "x", 1)
        assert isinstance(result, Tensor)
        assert result.shape == expr.shape
        assert torch.isfinite(result).all()

    def test_fd_diff_works_for_both_axes(self) -> None:
        dataset = _make_grid_dataset(50, 30)
        provider = FiniteDiffProvider(dataset, max_order=2)


        u = dataset.get_field("u")

        result_x = provider.diff(u, "x", 1)
        assert isinstance(result_x, Tensor)
        assert torch.isfinite(result_x).all()

        result_t = provider.diff(u, "t", 1)
        assert isinstance(result_t, Tensor)
        assert torch.isfinite(result_t).all()

    def test_fd_diff_works_for_order_2(self) -> None:
        dataset = _make_grid_dataset(50, 30)
        provider = FiniteDiffProvider(dataset, max_order=2)

        u = dataset.get_field("u")
        result = provider.diff(u, "x", 2)
        assert isinstance(result, Tensor)
        assert torch.isfinite(result).all()

    def test_fd_diff_matches_precomputed(self) -> None:
        dataset = _make_grid_dataset(50, 30)
        provider = FiniteDiffProvider(dataset, max_order=2)

        u = dataset.get_field("u")
        diff_result = provider.diff(u, "x", 1)
        precomputed = provider.get_derivative("u", "x", 1)

        torch.testing.assert_close(diff_result, precomputed, rtol=1e-12, atol=1e-12)

    def test_fd_get_derivative_still_works(self) -> None:
        dataset = _make_grid_dataset(50, 30)
        provider = FiniteDiffProvider(dataset, max_order=2)

        result = provider.get_derivative("u", "x", 1)
        assert isinstance(result, Tensor)
        assert torch.isfinite(result).all()







@pytest.mark.numerical
class TestDiffNumericalStability:

    def test_all_results_finite(self, sin_product_provider: AutogradProvider) -> None:
        u = sin_product_provider.get_field("u")

        for axis in ["x", "t"]:
            for order in [1, 2, 3]:
                result = sin_product_provider.diff(u, axis, order)
                assert torch.isfinite(result).all(), (
                    f"diff(u, '{axis}', {order}) produced NaN/Inf"
                )

    def test_dtype_preserved(self, sin_product_provider: AutogradProvider) -> None:
        u = sin_product_provider.get_field("u")
        u_x = sin_product_provider.diff(u, "x", 1)
        assert u_x.dtype == u.dtype

    def test_third_order_accuracy(
        self, sin_product_provider: AutogradProvider, coords: dict[str, Tensor]
    ) -> None:
        u = sin_product_provider.get_field("u")
        u_xxx = sin_product_provider.diff(u, "x", 3)

        x = coords["x"]
        t = coords["t"]
        expected = -torch.cos(x) * torch.exp(-t)
        torch.testing.assert_close(u_xxx, expected, rtol=1e-3, atol=1e-5)

    def test_result_stays_in_computation_graph(
        self, sin_product_provider: AutogradProvider
    ) -> None:
        u = sin_product_provider.get_field("u")
        u_x = sin_product_provider.diff(u, "x", 1)
        assert u_x.requires_grad, "diff result should remain in computation graph"

    def test_disconnected_tensor_raises(
        self, sin_product_provider: AutogradProvider
    ) -> None:
        disconnected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="not connected"):
            sin_product_provider.diff(disconnected, "x", 1)
