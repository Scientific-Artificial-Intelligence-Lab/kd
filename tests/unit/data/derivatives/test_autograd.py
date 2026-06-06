
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from kd.data.derivatives.autograd import AutogradProvider
from kd.data.derivatives.base import DerivativeProvider
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)






class SinModel(nn.Module):

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return torch.sin(x)


class PolyModel(nn.Module):

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return x**2


class MultiFieldModel(nn.Module):

    def forward(self, x: Tensor, t: Tensor) -> dict[str, Tensor]:
        return {"u": torch.sin(x), "v": torch.cos(x)}


class SinProductModel(nn.Module):

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return torch.sin(x) * torch.exp(-t)







def _make_coords_1d(n: int = 50) -> dict[str, Tensor]:
    x = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64, requires_grad=True)
    t = torch.zeros(n, dtype=torch.float64, requires_grad=True)
    return {"x": x, "t": t}


def _make_coords_2d(n_x: int = 50, n_t: int = 30) -> dict[str, Tensor]:
    x_1d = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
    t_1d = torch.linspace(0, 1, n_t, dtype=torch.float64)
    X, T = torch.meshgrid(x_1d, t_1d, indexing="ij")
    x = X.reshape(-1).clone().detach().requires_grad_(True)
    t = T.reshape(-1).clone().detach().requires_grad_(True)
    return {"x": x, "t": t}


def _make_dataset_1d(n: int = 50) -> PDEDataset:
    x = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64)
    u = torch.sin(x)
    return PDEDataset(
        name="test_autograd_1d",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo(name="x", values=x)},
        axis_order=["x"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="x",
    )


def _make_dataset_2d(n_x: int = 50, n_t: int = 30) -> PDEDataset:
    x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
    t = torch.linspace(0, 1, n_t, dtype=torch.float64)
    X, T = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(X) * torch.exp(-T)
    return PDEDataset(
        name="test_autograd_2d",
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
def sin_provider() -> AutogradProvider:
    model = SinModel().double()
    coords = _make_coords_1d(50)
    dataset = _make_dataset_1d(50)
    return AutogradProvider(model=model, coords=coords, dataset=dataset)


@pytest.fixture
def poly_provider() -> AutogradProvider:
    model = PolyModel().double()
    coords = _make_coords_1d(50)
    dataset = _make_dataset_1d(50)
    return AutogradProvider(model=model, coords=coords, dataset=dataset)


@pytest.fixture
def sin_product_provider() -> AutogradProvider:
    model = SinProductModel().double()
    coords = _make_coords_2d(50, 30)
    dataset = _make_dataset_2d(50, 30)
    return AutogradProvider(model=model, coords=coords, dataset=dataset)







class TestSmoke:

    @pytest.mark.smoke
    def test_autograd_provider_importable(self) -> None:
        from kd.data.derivatives.autograd import AutogradProvider

        assert AutogradProvider is not None

    @pytest.mark.smoke
    def test_provider_init(self, sin_provider: AutogradProvider) -> None:
        assert sin_provider is not None
        assert isinstance(sin_provider, DerivativeProvider)

    @pytest.mark.smoke
    def test_get_field_returns_tensor(self, sin_provider: AutogradProvider) -> None:
        result = sin_provider.get_field("u")
        assert isinstance(result, torch.Tensor)

    @pytest.mark.smoke
    def test_diff_returns_tensor(self, sin_provider: AutogradProvider) -> None:
        field = sin_provider.get_field("u")
        result = sin_provider.diff(field, "x", order=1)
        assert isinstance(result, torch.Tensor)

    @pytest.mark.smoke
    def test_get_derivative_returns_tensor(
        self, sin_provider: AutogradProvider
    ) -> None:
        result = sin_provider.get_derivative("u", "x", order=1)
        assert isinstance(result, torch.Tensor)







class TestInit:

    @pytest.mark.unit
    def test_init_stores_model(self) -> None:
        model = SinModel().double()
        coords = _make_coords_1d()
        dataset = _make_dataset_1d()
        provider = AutogradProvider(model=model, coords=coords, dataset=dataset)
        assert provider.model is model

    @pytest.mark.unit
    def test_init_stores_coords(self) -> None:
        model = SinModel().double()
        coords = _make_coords_1d()
        dataset = _make_dataset_1d()
        provider = AutogradProvider(model=model, coords=coords, dataset=dataset)
        assert provider.coords is coords

    @pytest.mark.unit
    def test_init_stores_dataset(self) -> None:
        model = SinModel().double()
        coords = _make_coords_1d()
        dataset = _make_dataset_1d()
        provider = AutogradProvider(model=model, coords=coords, dataset=dataset)
        assert provider.dataset is dataset

    @pytest.mark.unit
    def test_init_coords_requires_grad(self) -> None:
        model = SinModel().double()

        x = torch.linspace(0, 2 * math.pi, 50, dtype=torch.float64)
        t = torch.zeros(50, dtype=torch.float64)
        coords = {"x": x, "t": t}
        dataset = _make_dataset_1d()



        with pytest.raises(ValueError, match="requires_grad"):
            AutogradProvider(model=model, coords=coords, dataset=dataset)

    @pytest.mark.unit
    def test_init_coords_keys_match_dataset_axes(self) -> None:
        model = SinModel().double()

        coords = {
            "y": torch.linspace(0, 1, 50, dtype=torch.float64, requires_grad=True),
            "z": torch.zeros(50, dtype=torch.float64, requires_grad=True),
        }
        dataset = _make_dataset_1d()

        with pytest.raises((KeyError, ValueError)):
            AutogradProvider(model=model, coords=coords, dataset=dataset)







class TestGetField:

    @pytest.mark.unit
    def test_get_field_sin(self, sin_provider: AutogradProvider) -> None:
        result = sin_provider.get_field("u")
        x = sin_provider.coords["x"]
        expected = torch.sin(x)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-8)

    @pytest.mark.unit
    def test_get_field_poly(self, poly_provider: AutogradProvider) -> None:
        result = poly_provider.get_field("u")
        x = poly_provider.coords["x"]
        expected = x**2
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-8)

    @pytest.mark.unit
    def test_get_field_invalid_name(self, sin_provider: AutogradProvider) -> None:
        with pytest.raises(KeyError):
            sin_provider.get_field("nonexistent")

    @pytest.mark.unit
    def test_get_field_result_in_computation_graph(
        self, sin_provider: AutogradProvider
    ) -> None:
        result = sin_provider.get_field("u")
        assert result.requires_grad, "get_field result should require grad"







class TestDiff:

    @pytest.mark.unit
    def test_diff_sin_order1(self, sin_provider: AutogradProvider) -> None:
        u = sin_provider.get_field("u")
        u_x = sin_provider.diff(u, "x", order=1)
        x = sin_provider.coords["x"]
        expected = torch.cos(x)
        torch.testing.assert_close(u_x, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_diff_sin_order2(self, sin_provider: AutogradProvider) -> None:
        u = sin_provider.get_field("u")
        u_xx = sin_provider.diff(u, "x", order=2)
        x = sin_provider.coords["x"]
        expected = -torch.sin(x)
        torch.testing.assert_close(u_xx, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_diff_sin_order3(self, sin_provider: AutogradProvider) -> None:
        u = sin_provider.get_field("u")
        u_xxx = sin_provider.diff(u, "x", order=3)
        x = sin_provider.coords["x"]
        expected = -torch.cos(x)
        torch.testing.assert_close(u_xxx, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_diff_poly_order1(self, poly_provider: AutogradProvider) -> None:
        u = poly_provider.get_field("u")
        u_x = poly_provider.diff(u, "x", order=1)
        x = poly_provider.coords["x"]
        expected = 2 * x
        torch.testing.assert_close(u_x, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_diff_poly_order2(self, poly_provider: AutogradProvider) -> None:
        u = poly_provider.get_field("u")
        u_xx = poly_provider.diff(u, "x", order=2)
        expected = torch.full_like(u_xx, 2.0)
        torch.testing.assert_close(u_xx, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_diff_preserves_shape(self, sin_provider: AutogradProvider) -> None:
        u = sin_provider.get_field("u")
        u_x = sin_provider.diff(u, "x", order=1)
        assert u_x.shape == u.shape

    @pytest.mark.unit
    def test_diff_create_graph(self, sin_provider: AutogradProvider) -> None:
        u = sin_provider.get_field("u")
        u_x = sin_provider.diff(u, "x", order=1)

        assert u_x.requires_grad, "diff result should remain in computation graph"


        u_xx = sin_provider.diff(u_x, "x", order=1)
        x = sin_provider.coords["x"]
        expected = -torch.sin(x)
        torch.testing.assert_close(u_xx, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_diff_2d_partial_x(self, sin_product_provider: AutogradProvider) -> None:
        u = sin_product_provider.get_field("u")
        u_x = sin_product_provider.diff(u, "x", order=1)
        x = sin_product_provider.coords["x"]
        t = sin_product_provider.coords["t"]
        expected = torch.cos(x) * torch.exp(-t)
        torch.testing.assert_close(u_x, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_diff_2d_partial_t(self, sin_product_provider: AutogradProvider) -> None:
        u = sin_product_provider.get_field("u")
        u_t = sin_product_provider.diff(u, "t", order=1)
        x = sin_product_provider.coords["x"]
        t = sin_product_provider.coords["t"]
        expected = -torch.sin(x) * torch.exp(-t)
        torch.testing.assert_close(u_t, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_diff_invalid_axis(self, sin_provider: AutogradProvider) -> None:
        u = sin_provider.get_field("u")
        with pytest.raises(KeyError):
            sin_provider.diff(u, "nonexistent", order=1)

    @pytest.mark.unit
    def test_diff_invalid_order_zero(self, sin_provider: AutogradProvider) -> None:
        u = sin_provider.get_field("u")
        with pytest.raises(ValueError, match="order"):
            sin_provider.diff(u, "x", order=0)

    @pytest.mark.unit
    def test_diff_invalid_order_negative(self, sin_provider: AutogradProvider) -> None:
        u = sin_provider.get_field("u")
        with pytest.raises(ValueError, match="order"):
            sin_provider.diff(u, "x", order=-1)

    @pytest.mark.unit
    def test_diff_open_form_expression(self, sin_provider: AutogradProvider) -> None:
        u = sin_provider.get_field("u")
        expr = u * u
        d_expr = sin_provider.diff(expr, "x", order=1)
        x = sin_provider.coords["x"]
        expected = torch.sin(2 * x)
        torch.testing.assert_close(d_expr, expected, rtol=1e-4, atol=1e-6)







class TestGetDerivative:

    @pytest.mark.unit
    def test_get_derivative_matches_diff(self, sin_provider: AutogradProvider) -> None:
        result = sin_provider.get_derivative("u", "x", order=1)
        u = sin_provider.get_field("u")
        expected = sin_provider.diff(u, "x", order=1)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-8)

    @pytest.mark.unit
    def test_get_derivative_sin_order1(self, sin_provider: AutogradProvider) -> None:
        result = sin_provider.get_derivative("u", "x", order=1)
        x = sin_provider.coords["x"]
        expected = torch.cos(x)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_get_derivative_sin_order2(self, sin_provider: AutogradProvider) -> None:
        result = sin_provider.get_derivative("u", "x", order=2)
        x = sin_provider.coords["x"]
        expected = -torch.sin(x)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_get_derivative_invalid_field(self, sin_provider: AutogradProvider) -> None:
        with pytest.raises(KeyError):
            sin_provider.get_derivative("nonexistent", "x", order=1)

    @pytest.mark.unit
    def test_get_derivative_invalid_axis(self, sin_provider: AutogradProvider) -> None:
        with pytest.raises(KeyError):
            sin_provider.get_derivative("u", "nonexistent", order=1)

    @pytest.mark.unit
    def test_get_derivative_invalid_order(self, sin_provider: AutogradProvider) -> None:
        with pytest.raises(ValueError):
            sin_provider.get_derivative("u", "x", order=0)
        with pytest.raises(ValueError):
            sin_provider.get_derivative("u", "x", order=-1)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "field,axis,order",
        [
            ("u", "x", 1),
            ("u", "x", 2),
            ("u", "t", 1),
        ],
    )
    def test_get_derivative_2d(
        self,
        sin_product_provider: AutogradProvider,
        field: str,
        axis: str,
        order: int,
    ) -> None:
        result = sin_product_provider.get_derivative(field, axis, order=order)
        assert isinstance(result, torch.Tensor)
        assert torch.isfinite(result).all()







class TestAvailableDerivatives:

    @pytest.mark.unit
    def test_available_derivatives_returns_list(
        self, sin_provider: AutogradProvider
    ) -> None:
        result = sin_provider.available_derivatives()
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 3

    @pytest.mark.unit
    def test_available_derivatives_contains_expected(
        self, sin_provider: AutogradProvider
    ) -> None:
        result = sin_provider.available_derivatives()

        assert ("u", "x", 1) in result

    @pytest.mark.unit
    def test_available_derivatives_2d(
        self, sin_product_provider: AutogradProvider
    ) -> None:
        result = sin_product_provider.available_derivatives()
        assert ("u", "x", 1) in result
        assert ("u", "t", 1) in result

    @pytest.mark.unit
    def test_available_derivatives_format(self, sin_provider: AutogradProvider) -> None:
        result = sin_provider.available_derivatives()
        for field, axis, order in result:
            assert isinstance(field, str)
            assert isinstance(axis, str)
            assert isinstance(order, int)
            assert order >= 1







class TestErrors:

    @pytest.mark.unit
    def test_diff_order_type_check(self, sin_provider: AutogradProvider) -> None:
        u = sin_provider.get_field("u")
        with pytest.raises((TypeError, ValueError)):
            sin_provider.diff(u, "x", order=1.5)

    @pytest.mark.unit
    def test_get_field_lhs_field_valid(self, sin_provider: AutogradProvider) -> None:
        lhs_field = sin_provider.dataset.lhs_field
        result = sin_provider.get_field(lhs_field)
        assert isinstance(result, torch.Tensor)







class TestNumericalStability:

    @pytest.mark.numerical
    def test_diff_result_finite(self, sin_provider: AutogradProvider) -> None:
        u = sin_provider.get_field("u")
        for order in [1, 2, 3]:
            result = sin_provider.diff(u, "x", order=order)
            assert torch.isfinite(result).all(), (
                f"Order {order} derivative contains NaN/Inf"
            )

    @pytest.mark.numerical
    def test_diff_high_order_stability(self) -> None:
        model = SinModel().double()
        coords = _make_coords_1d(50)
        dataset = _make_dataset_1d(50)
        provider = AutogradProvider(
            model=model, coords=coords, dataset=dataset, max_order=4
        )
        u = provider.get_field("u")
        u_4 = provider.diff(u, "x", order=4)
        x = provider.coords["x"]
        expected = torch.sin(x)
        assert torch.isfinite(u_4).all(), "4th order derivative has NaN/Inf"
        torch.testing.assert_close(u_4, expected, rtol=1e-3, atol=1e-5)

    @pytest.mark.numerical
    def test_diff_dtype_preserved(self, sin_provider: AutogradProvider) -> None:
        u = sin_provider.get_field("u")
        u_x = sin_provider.diff(u, "x", order=1)
        assert u_x.dtype == u.dtype

    @pytest.mark.numerical
    def test_get_derivative_all_finite(
        self, sin_product_provider: AutogradProvider
    ) -> None:
        for field, axis, order in sin_product_provider.available_derivatives():
            result = sin_product_provider.get_derivative(field, axis, order=order)
            assert torch.isfinite(result).all(), (
                f"Derivative ({field}, {axis}, {order}) contains NaN/Inf"
            )







class TestAccuracyParametrized:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "order,expected_fn",
        [
            (1, lambda x: torch.cos(x)),
            (2, lambda x: -torch.sin(x)),
            (3, lambda x: -torch.cos(x)),
        ],
        ids=["sin_order1", "sin_order2", "sin_order3"],
    )
    def test_sin_derivatives(
        self,
        sin_provider: AutogradProvider,
        order: int,
        expected_fn: object,
    ) -> None:
        u = sin_provider.get_field("u")
        result = sin_provider.diff(u, "x", order=order)
        x = sin_provider.coords["x"]
        expected = expected_fn(x)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "order,expected_fn",
        [
            (1, lambda x: 2 * x),
            (2, lambda x: torch.full_like(x, 2.0)),
        ],
        ids=["poly_order1", "poly_order2"],
    )
    def test_poly_derivatives(
        self,
        poly_provider: AutogradProvider,
        order: int,
        expected_fn: object,
    ) -> None:
        u = poly_provider.get_field("u")
        result = poly_provider.diff(u, "x", order=order)
        x = poly_provider.coords["x"]
        expected = expected_fn(x)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)







class TestReviewFixes:

    @pytest.mark.unit
    def test_diff_disconnected_graph_raises_value_error(
        self, sin_provider: AutogradProvider
    ) -> None:

        disconnected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="not connected"):
            sin_provider.diff(disconnected, "x", order=1)

    @pytest.mark.unit
    def test_init_max_order_validation(self) -> None:
        model = SinModel().double()
        coords = _make_coords_1d()
        dataset = _make_dataset_1d()
        with pytest.raises(ValueError, match="max_order"):
            AutogradProvider(model=model, coords=coords, dataset=dataset, max_order=0)

    @pytest.mark.unit
    def test_init_max_order_negative(self) -> None:
        model = SinModel().double()
        coords = _make_coords_1d()
        dataset = _make_dataset_1d()
        with pytest.raises(ValueError, match="max_order"):
            AutogradProvider(model=model, coords=coords, dataset=dataset, max_order=-1)

    @pytest.mark.unit
    def test_diff_rejects_order_above_max_order_default(
        self, sin_provider: AutogradProvider
    ) -> None:
        u = sin_provider.get_field("u")
        with pytest.raises(ValueError, match=r"exceeds max_order"):
            sin_provider.diff(u, "x", order=4)

    @pytest.mark.unit
    def test_diff_rejects_order_above_custom_max_order(self) -> None:
        model = SinModel().double()
        coords = _make_coords_1d()
        dataset = _make_dataset_1d()
        provider = AutogradProvider(
            model=model, coords=coords, dataset=dataset, max_order=1
        )
        u = provider.get_field("u")
        with pytest.raises(ValueError, match=r"exceeds max_order 1"):
            provider.diff(u, "x", order=2)

    @pytest.mark.unit
    def test_diff_at_max_order_still_accepted(
        self, sin_provider: AutogradProvider
    ) -> None:
        u = sin_provider.get_field("u")
        result = sin_provider.diff(u, "x", order=3)
        assert result.shape == u.shape
