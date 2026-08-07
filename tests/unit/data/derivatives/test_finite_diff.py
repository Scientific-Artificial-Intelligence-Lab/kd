
from __future__ import annotations

import math

import pytest
import torch

from kd.data.derivatives import (
    DerivativeProvider,
    FiniteDiffProvider,
)
from kd.data.derivatives.finite_diff import central_diff
from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType






class TestSmoke:

    @pytest.mark.smoke
    def test_provider_init(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)
        assert provider is not None
        assert isinstance(provider, DerivativeProvider)

    @pytest.mark.smoke
    def test_get_derivative_basic(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)
        result = provider.get_derivative("u", "x", order=1)
        assert isinstance(result, torch.Tensor)
        assert result.shape == simple_1d_dataset.get_shape()







class TestCentralDiff:

    @pytest.mark.unit
    def test_central_diff_order_1_sin(self) -> None:
        n = 100
        x = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64)
        dx = x[1] - x[0]
        f = torch.sin(x)

        df = central_diff(f, dx.item(), axis=0, order=1)
        expected = torch.cos(x)



        assert torch.allclose(df[2:-2], expected[2:-2], rtol=1e-3, atol=1e-6)

    @pytest.mark.unit
    def test_central_diff_order_2_sin(self) -> None:
        n = 100
        x = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64)
        dx = x[1] - x[0]
        f = torch.sin(x)

        d2f = central_diff(f, dx.item(), axis=0, order=2)
        expected = -torch.sin(x)


        assert torch.allclose(d2f[2:-2], expected[2:-2], rtol=1e-3, atol=1e-6)

    @pytest.mark.unit
    def test_central_diff_order_3_sin(self) -> None:
        n = 200
        x = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64)
        dx = x[1] - x[0]
        f = torch.sin(x)

        d3f = central_diff(f, dx.item(), axis=0, order=3)
        expected = -torch.cos(x)


        assert torch.allclose(d3f[4:-4], expected[4:-4], rtol=1e-2, atol=1e-5)

    @pytest.mark.unit
    def test_central_diff_preserves_shape(self) -> None:
        f = torch.randn(50, dtype=torch.float64)
        dx = 0.1
        df = central_diff(f, dx, axis=0, order=1)
        assert df.shape == f.shape

    @pytest.mark.unit
    def test_central_diff_2d_axis_0(self) -> None:
        n_x, n_t = 64, 32
        x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
        t = torch.linspace(0, 1, n_t, dtype=torch.float64)
        dx = x[1] - x[0]

        X, T = torch.meshgrid(x, t, indexing="ij")
        f = torch.sin(X)

        df = central_diff(f, dx.item(), axis=0, order=1)
        expected = torch.cos(X)


        assert torch.allclose(df[2:-2,:], expected[2:-2,:], rtol=1e-3, atol=1e-6)

    @pytest.mark.unit
    def test_central_diff_2d_axis_1(self) -> None:
        n_x, n_t = 64, 32
        x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
        t = torch.linspace(0, 1, n_t, dtype=torch.float64)
        dt = t[1] - t[0]

        X, T = torch.meshgrid(x, t, indexing="ij")
        f = torch.exp(-T)

        df = central_diff(f, dt.item(), axis=1, order=1)
        expected = -torch.exp(-T)


        assert torch.allclose(df[:, 2:-2], expected[:, 2:-2], rtol=1e-3, atol=1e-6)

    @pytest.mark.unit
    def test_central_diff_invalid_order(self) -> None:
        f = torch.randn(50, dtype=torch.float64)
        dx = 0.1

        with pytest.raises(ValueError):
            central_diff(f, dx, axis=0, order=0)

        with pytest.raises(ValueError):
            central_diff(f, dx, axis=0, order=4)

        with pytest.raises(ValueError):
            central_diff(f, dx, axis=0, order=-1)

    @pytest.mark.unit
    def test_central_diff_invalid_axis(self) -> None:
        f = torch.randn(50, dtype=torch.float64)
        dx = 0.1

        with pytest.raises((ValueError, IndexError)):
            central_diff(f, dx, axis=1, order=1)







class TestProviderInit:

    @pytest.mark.unit
    def test_provider_requires_grid_topology(
        self, scattered_dataset: PDEDataset
    ) -> None:
        with pytest.raises(ValueError, match="[Gg]rid"):
            FiniteDiffProvider(scattered_dataset)

    @pytest.mark.unit
    def test_provider_accepts_grid_topology(
        self, simple_1d_dataset: PDEDataset
    ) -> None:
        assert simple_1d_dataset.topology == DataTopology.GRID
        provider = FiniteDiffProvider(simple_1d_dataset)
        assert provider is not None

    @pytest.mark.unit
    def test_provider_max_order_default(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)

        derivatives = provider.available_derivatives()
        orders = [order for _, _, order in derivatives]
        assert max(orders) == 3

    @pytest.mark.unit
    def test_provider_max_order_custom(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset, max_order=2)
        derivatives = provider.available_derivatives()
        orders = [order for _, _, order in derivatives]
        assert max(orders) == 2


class TestProviderGetDerivative:

    @pytest.mark.unit
    def test_provider_get_derivative_shape(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)
        deriv = provider.get_derivative("u", "x", order=1)
        expected_shape = simple_1d_dataset.get_shape()
        assert deriv.shape == expected_shape

    @pytest.mark.unit
    def test_provider_get_derivative_dtype(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)
        deriv = provider.get_derivative("u", "x", order=1)
        original_dtype = simple_1d_dataset.get_field("u").dtype
        assert deriv.dtype == original_dtype

    @pytest.mark.unit
    def test_provider_get_derivative_2d(self, simple_2d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_2d_dataset)


        u_x = provider.get_derivative("u", "x", order=1)
        assert u_x.shape == simple_2d_dataset.get_shape()


        u_t = provider.get_derivative("u", "t", order=1)
        assert u_t.shape == simple_2d_dataset.get_shape()


class TestProviderAvailableDerivatives:

    @pytest.mark.unit
    def test_available_derivatives_1d(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset, max_order=3)
        derivatives = provider.available_derivatives()


        expected = [("u", "x", 1), ("u", "x", 2), ("u", "x", 3)]
        for deriv in expected:
            assert deriv in derivatives

    @pytest.mark.unit
    def test_available_derivatives_2d(self, simple_2d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_2d_dataset, max_order=2)
        derivatives = provider.available_derivatives()


        expected_contains = [
            ("u", "x", 1),
            ("u", "x", 2),
            ("u", "t", 1),
            ("u", "t", 2),
        ]
        for deriv in expected_contains:
            assert deriv in derivatives


class TestProviderDiff:





    @pytest.mark.smoke
    def test_diff_returns_tensor(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)
        field_values = simple_1d_dataset.get_field("u")
        result = provider.diff(field_values, "x", order=1)
        assert isinstance(result, torch.Tensor)
        assert result.shape == field_values.shape

    @pytest.mark.smoke
    def test_diff_preserves_dtype(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)
        field_values = simple_1d_dataset.get_field("u")
        result = provider.diff(field_values, "x", order=1)
        assert result.dtype == field_values.dtype





    @pytest.mark.unit
    def test_diff_matches_get_derivative_1d_order1(
        self, simple_1d_dataset: PDEDataset
    ) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)
        field_values = simple_1d_dataset.get_field("u")

        diff_result = provider.diff(field_values, "x", order=1)
        precomputed = provider.get_derivative("u", "x", order=1)

        torch.testing.assert_close(diff_result, precomputed, rtol=1e-12, atol=1e-12)

    @pytest.mark.unit
    def test_diff_matches_get_derivative_1d_order2(
        self, simple_1d_dataset: PDEDataset
    ) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)
        field_values = simple_1d_dataset.get_field("u")

        diff_result = provider.diff(field_values, "x", order=2)
        precomputed = provider.get_derivative("u", "x", order=2)

        torch.testing.assert_close(diff_result, precomputed, rtol=1e-12, atol=1e-12)

    @pytest.mark.unit
    def test_diff_matches_get_derivative_1d_order3(
        self, simple_1d_dataset: PDEDataset
    ) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)
        field_values = simple_1d_dataset.get_field("u")

        diff_result = provider.diff(field_values, "x", order=3)
        precomputed = provider.get_derivative("u", "x", order=3)

        torch.testing.assert_close(diff_result, precomputed, rtol=1e-12, atol=1e-12)

    @pytest.mark.unit
    def test_diff_matches_get_derivative_2d_both_axes(
        self, simple_2d_dataset: PDEDataset
    ) -> None:
        provider = FiniteDiffProvider(simple_2d_dataset)
        field_values = simple_2d_dataset.get_field("u")

        for axis_name in ["x", "t"]:
            diff_result = provider.diff(field_values, axis_name, order=1)
            precomputed = provider.get_derivative("u", axis_name, order=1)
            torch.testing.assert_close(diff_result, precomputed, rtol=1e-12, atol=1e-12)





    @pytest.mark.unit
    def test_diff_composite_u_squared(self, simple_2d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_2d_dataset)
        u = simple_2d_dataset.get_field("u")
        u_x = provider.get_derivative("u", "x", order=1)


        u_sq = u**2
        d_u_sq = provider.diff(u_sq, "x", order=1)


        expected = 2 * u * u_x


        torch.testing.assert_close(
            d_u_sq[2:-2,:], expected[2:-2,:], rtol=1e-3, atol=1e-5
        )

    @pytest.mark.unit
    def test_diff_composite_polynomial(self, polynomial_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(polynomial_1d_dataset)
        u = polynomial_1d_dataset.get_field("u")
        x = polynomial_1d_dataset.get_coords("x")

        u_sq = u**2
        d_u_sq = provider.diff(u_sq, "x", order=1)

        expected = 6 * x**5


        torch.testing.assert_close(d_u_sq[4:-4], expected[4:-4], rtol=1e-2, atol=1e-4)





    @pytest.mark.unit
    def test_diff_all_orders_produce_finite_results(
        self, simple_2d_dataset: PDEDataset
    ) -> None:
        provider = FiniteDiffProvider(simple_2d_dataset)
        expr = simple_2d_dataset.get_field("u")

        for order in [1, 2, 3]:
            result = provider.diff(expr, "x", order=order)
            assert torch.isfinite(result).all(), (
                f"diff(expr, 'x', {order}) produced NaN/Inf"
            )
            assert result.shape == expr.shape

    @pytest.mark.unit
    def test_diff_order2_sin(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)
        u = simple_1d_dataset.get_field("u")
        x = simple_1d_dataset.get_coords("x")

        result = provider.diff(u, "x", order=2)
        expected = -torch.sin(x)

        torch.testing.assert_close(result[2:-2], expected[2:-2], rtol=1e-3, atol=1e-6)

    @pytest.mark.unit
    def test_diff_order3_sin(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)
        u = simple_1d_dataset.get_field("u")
        x = simple_1d_dataset.get_coords("x")

        result = provider.diff(u, "x", order=3)
        expected = -torch.cos(x)


        torch.testing.assert_close(result[4:-4], expected[4:-4], rtol=1e-2, atol=1e-4)





    @pytest.mark.unit
    def test_diff_invalid_axis_raises_key_error(
        self, simple_1d_dataset: PDEDataset
    ) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)
        expr = torch.randn(100, dtype=torch.float64)

        with pytest.raises(KeyError):
            provider.diff(expr, "nonexistent_axis", order=1)

    @pytest.mark.unit
    def test_diff_invalid_axis_y_on_xt_dataset(
        self, simple_2d_dataset: PDEDataset
    ) -> None:
        provider = FiniteDiffProvider(simple_2d_dataset)
        expr = simple_2d_dataset.get_field("u")

        with pytest.raises(KeyError):
            provider.diff(expr, "y", order=1)





    @pytest.mark.numerical
    def test_diff_periodic_matches_get_derivative(self) -> None:
        n = 128
        x_full = torch.linspace(0.0, 1.0, n + 1, dtype=torch.float64)
        x = x_full[:-1]
        u = torch.sin(2 * math.pi * x)

        ds = PDEDataset(
            name="periodic_diff_test",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={"x": AxisInfo(name="x", values=x, is_periodic=True)},
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="x",
        )

        provider = FiniteDiffProvider(ds)
        field_values = ds.get_field("u")

        diff_result = provider.diff(field_values, "x", order=1)
        precomputed = provider.get_derivative("u", "x", order=1)


        torch.testing.assert_close(diff_result, precomputed, rtol=1e-12, atol=1e-12)

    @pytest.mark.numerical
    def test_diff_periodic_boundary_accuracy(self) -> None:
        n = 128
        x_full = torch.linspace(0.0, 1.0, n + 1, dtype=torch.float64)
        x = x_full[:-1]
        u = torch.sin(2 * math.pi * x)

        ds = PDEDataset(
            name="periodic_diff_accuracy",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={"x": AxisInfo(name="x", values=x, is_periodic=True)},
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="x",
        )

        provider = FiniteDiffProvider(ds)
        result = provider.diff(u, "x", order=1)

        expected = 2 * math.pi * torch.cos(2 * math.pi * x)
        abs_err = (result - expected).abs()

        interior_max_err = abs_err[4:-4].max().item()
        boundary_max_err = max(
            abs_err[:2].max().item(),
            abs_err[-2:].max().item(),
        )


        assert boundary_max_err < 10 * interior_max_err, (
            f"Periodic diff() boundary error ({boundary_max_err:.2e}) is much "
            f"worse than interior ({interior_max_err:.2e}). "
            f"is_periodic may not be passed to central_diff."
        )





    @pytest.mark.numerical
    def test_diff_nan_input_raises(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)
        expr = torch.randn(100, dtype=torch.float64)
        expr[10] = float("nan")

        with pytest.raises(ValueError, match="NaN"):
            provider.diff(expr, "x", order=1)

    @pytest.mark.numerical
    def test_diff_inf_input_raises(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)
        expr = torch.randn(100, dtype=torch.float64)
        expr[10] = float("inf")

        with pytest.raises(ValueError, match="Inf"):
            provider.diff(expr, "x", order=1)

    @pytest.mark.unit
    def test_diff_invalid_order_raises(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)
        expr = simple_1d_dataset.get_field("u")

        with pytest.raises(ValueError):
            provider.diff(expr, "x", order=0)
        with pytest.raises(ValueError):
            provider.diff(expr, "x", order=4)
        with pytest.raises(ValueError):
            provider.diff(expr, "x", order=-1)

    @pytest.mark.unit
    def test_diff_wrong_shape_tensor(self, simple_2d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_2d_dataset)



        wrong_shape = torch.randn(3, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="doesn't match"):
            provider.diff(wrong_shape, "x", order=1)


        wrong_size = torch.randn(100, 32, dtype=torch.float64)
        with pytest.raises(ValueError, match="doesn't match"):
            provider.diff(wrong_size, "x", order=1)


        wrong_axis2 = torch.randn(64, 100, dtype=torch.float64)
        with pytest.raises(ValueError, match="doesn't match"):
            provider.diff(wrong_axis2, "t", order=1)


class TestProviderErrors:

    @pytest.mark.unit
    def test_provider_invalid_field(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)

        with pytest.raises(KeyError):
            provider.get_derivative("nonexistent", "x", order=1)

    @pytest.mark.unit
    def test_provider_invalid_axis(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)

        with pytest.raises(KeyError):
            provider.get_derivative("u", "nonexistent", order=1)

    @pytest.mark.unit
    def test_provider_invalid_order_zero(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)

        with pytest.raises(ValueError):
            provider.get_derivative("u", "x", order=0)

    @pytest.mark.unit
    def test_provider_invalid_order_negative(
        self, simple_1d_dataset: PDEDataset
    ) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)

        with pytest.raises(ValueError):
            provider.get_derivative("u", "x", order=-1)

    @pytest.mark.unit
    def test_provider_order_exceeds_max(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset, max_order=2)

        with pytest.raises(ValueError):
            provider.get_derivative("u", "x", order=3)







class TestNumericalAccuracy:

    @pytest.mark.numerical
    def test_accuracy_sin(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)
        x = simple_1d_dataset.get_coords("x")


        u_x = provider.get_derivative("u", "x", order=1)
        expected_u_x = torch.cos(x)

        assert torch.allclose(u_x[2:-2], expected_u_x[2:-2], rtol=1e-3, atol=1e-6)


        u_xx = provider.get_derivative("u", "x", order=2)
        expected_u_xx = -torch.sin(x)
        assert torch.allclose(u_xx[2:-2], expected_u_xx[2:-2], rtol=1e-3, atol=1e-6)

    @pytest.mark.numerical
    def test_accuracy_polynomial(self, polynomial_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(polynomial_1d_dataset)
        x = polynomial_1d_dataset.get_coords("x")


        u_x = provider.get_derivative("u", "x", order=1)
        expected_u_x = 3 * x**2
        assert torch.allclose(u_x[2:-2], expected_u_x[2:-2], rtol=1e-3, atol=1e-6)


        u_xx = provider.get_derivative("u", "x", order=2)
        expected_u_xx = 6 * x
        assert torch.allclose(u_xx[2:-2], expected_u_xx[2:-2], rtol=1e-3, atol=1e-6)


        u_xxx = provider.get_derivative("u", "x", order=3)
        expected_u_xxx = torch.full_like(x, 6.0)

        assert torch.allclose(u_xxx[4:-4], expected_u_xxx[4:-4], rtol=1e-2, atol=1e-4)

    @pytest.mark.numerical
    def test_accuracy_exp(self) -> None:

        n_points = 100
        x = torch.linspace(
            0, 1, n_points, dtype=torch.float64
        )
        u = torch.exp(x)

        from kd.data.schema import (
            DataTopology,
            PDEDataset,
            TaskType,
        )

        dataset = PDEDataset(
            name="test_exp",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={"x": AxisInfo(name="x", values=x)},
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="x",
        )

        provider = FiniteDiffProvider(dataset)


        u_x = provider.get_derivative("u", "x", order=1)
        expected = torch.exp(x)
        assert torch.allclose(u_x[2:-2], expected[2:-2], rtol=1e-3, atol=1e-6)


        u_xx = provider.get_derivative("u", "x", order=2)
        assert torch.allclose(u_xx[2:-2], expected[2:-2], rtol=1e-3, atol=1e-6)

    @pytest.mark.numerical
    def test_accuracy_2d_mixed(self, simple_2d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_2d_dataset)
        x = simple_2d_dataset.get_coords("x")
        t = simple_2d_dataset.get_coords("t")

        X, T = torch.meshgrid(x, t, indexing="ij")


        u_x = provider.get_derivative("u", "x", order=1)
        expected_u_x = torch.cos(X) * torch.exp(-T)

        assert torch.allclose(
            u_x[2:-2, 2:-2], expected_u_x[2:-2, 2:-2], rtol=1e-3, atol=1e-6
        )


        u_t = provider.get_derivative("u", "t", order=1)
        expected_u_t = -torch.sin(X) * torch.exp(-T)
        assert torch.allclose(
            u_t[2:-2, 2:-2], expected_u_t[2:-2, 2:-2], rtol=1e-3, atol=1e-6
        )

    @pytest.mark.numerical
    def test_relative_error_bounds(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)
        x = simple_1d_dataset.get_coords("x")

        u_x = provider.get_derivative("u", "x", order=1)
        expected = torch.cos(x)


        mask = torch.abs(expected) > 0.1
        if mask.any():
            relative_error = torch.abs(u_x[mask] - expected[mask]) / torch.abs(
                expected[mask]
            )

            interior_mask = mask.clone()
            interior_mask[:2] = False
            interior_mask[-2:] = False
            if interior_mask.any():
                max_rel_error = relative_error[interior_mask[mask]].max()
                assert max_rel_error < 1e-3, f"Max relative error: {max_rel_error}"







class TestCentralDiffAttack:

    @pytest.mark.numerical
    def test_attack_dx_zero(self) -> None:
        f = torch.randn(50, dtype=torch.float64)
        with pytest.raises((ValueError, ZeroDivisionError)):
            central_diff(f, dx=0.0, axis=0, order=1)

    @pytest.mark.numerical
    def test_attack_dx_tiny(self) -> None:
        f = torch.randn(50, dtype=torch.float64)

        with pytest.raises(ValueError, match="dx"):
            central_diff(f, dx=1e-320, axis=0, order=1)

    @pytest.mark.numerical
    def test_attack_dx_huge(self) -> None:
        f = torch.sin(torch.linspace(0, 2 * math.pi, 50, dtype=torch.float64))
        result = central_diff(f, dx=1e15, axis=0, order=1)

        assert torch.isfinite(result).all(), "dx=1e15 should produce finite results"

        assert result.abs().max() < 1e-10, "Large dx produces near-zero derivatives"

    @pytest.mark.numerical
    def test_attack_dx_inf(self) -> None:
        f = torch.randn(50, dtype=torch.float64)
        with pytest.raises(ValueError, match="dx"):
            central_diff(f, dx=float("inf"), axis=0, order=1)

    @pytest.mark.numerical
    def test_attack_insufficient_points_order1(self) -> None:
        f = torch.tensor([1.0, 2.0], dtype=torch.float64)
        with pytest.raises((ValueError, IndexError)):
            central_diff(f, dx=0.1, axis=0, order=1)

    @pytest.mark.numerical
    def test_attack_insufficient_points_order2(self) -> None:
        f = torch.tensor([1.0, 2.0], dtype=torch.float64)
        with pytest.raises((ValueError, IndexError)):
            central_diff(f, dx=0.1, axis=0, order=2)

    @pytest.mark.numerical
    def test_attack_insufficient_points_order3(self) -> None:
        f = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        with pytest.raises((ValueError, IndexError)):
            central_diff(f, dx=0.1, axis=0, order=3)

    @pytest.mark.numerical
    def test_attack_nan_input(self) -> None:
        f = torch.tensor([1.0, float("nan"), 3.0, 4.0, 5.0], dtype=torch.float64)
        with pytest.raises(ValueError):
            central_diff(f, dx=0.1, axis=0, order=1)

    @pytest.mark.numerical
    def test_attack_inf_input(self) -> None:
        f = torch.tensor([1.0, float("inf"), 3.0, 4.0, 5.0], dtype=torch.float64)
        with pytest.raises(ValueError):
            central_diff(f, dx=0.1, axis=0, order=1)


class TestProviderAttack:

    @pytest.mark.unit
    def test_attack_nonuniform_grid(self) -> None:
        from kd.data.schema import (
            DataTopology,
            PDEDataset,
            TaskType,
        )


        x = torch.tensor(
            [0.0, 0.1, 0.5, 0.6, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
            dtype=torch.float64,
        )
        u = torch.sin(x)
        dataset = PDEDataset(
            name="nonuniform",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={"x": AxisInfo(name="x", values=x)},
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="x",
        )

        with pytest.raises(ValueError, match="[Uu]niform"):
            FiniteDiffProvider(dataset)

    @pytest.mark.unit
    def test_attack_max_order_exceeds_supported(
        self, simple_1d_dataset: PDEDataset
    ) -> None:
        with pytest.raises(ValueError, match="max_order"):
            FiniteDiffProvider(simple_1d_dataset, max_order=4)

    @pytest.mark.unit
    def test_attack_max_order_exceeds_supported_large(
        self, simple_1d_dataset: PDEDataset
    ) -> None:
        with pytest.raises(ValueError, match="max_order"):
            FiniteDiffProvider(simple_1d_dataset, max_order=100)

    @pytest.mark.unit
    def test_attack_max_order_zero(self, simple_1d_dataset: PDEDataset) -> None:
        with pytest.raises(ValueError, match="max_order"):
            FiniteDiffProvider(simple_1d_dataset, max_order=0)

    @pytest.mark.unit
    def test_attack_max_order_negative(self, simple_1d_dataset: PDEDataset) -> None:
        with pytest.raises(ValueError, match="max_order"):
            FiniteDiffProvider(simple_1d_dataset, max_order=-1)







class TestBoundaryAccuracy:

    @pytest.mark.numerical
    def test_second_order_boundary_accuracy(self) -> None:
        from kd.data.schema import (
            DataTopology,
            PDEDataset,
            TaskType,
        )


        n = 100
        x = torch.linspace(0, 1, n, dtype=torch.float64)
        u = x**3
        dx = 1.0 / (n - 1)


        dataset = PDEDataset(
            name="cubic",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={"x": AxisInfo(name="x", values=x)},
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="x",
        )
        provider = FiniteDiffProvider(dataset)
        u_xx = provider.get_derivative("u", "x", order=2)

        expected = 6 * x



        assert torch.isfinite(u_xx[0]), "Left boundary should be finite"
        assert abs(u_xx[0].item() - expected[0].item()) < 0.5, (
            f"Left boundary error too large: {abs(u_xx[0].item() - expected[0].item())}"
        )


        assert torch.isfinite(u_xx[-1]), "Right boundary should be finite"
        assert abs(u_xx[-1].item() - expected[-1].item()) < 0.5, (
            f"Right boundary error too large: {abs(u_xx[-1].item() - expected[-1].item())}"
        )

    @pytest.mark.numerical
    def test_second_order_boundary_convergence(self) -> None:
        from kd.data.schema import (
            DataTopology,
            PDEDataset,
            TaskType,
        )

        def compute_boundary_error(n: int) -> tuple[float, float]:
            x = torch.linspace(0, 1, n, dtype=torch.float64)
            u = x**3
            expected = 6 * x

            dataset = PDEDataset(
                name="cubic",
                task_type=TaskType.PDE,
                topology=DataTopology.GRID,
                axes={"x": AxisInfo(name="x", values=x)},
                axis_order=["x"],
                fields={"u": FieldData(name="u", values=u)},
                lhs_field="u",
                lhs_axis="x",
            )
            provider = FiniteDiffProvider(dataset)
            u_xx = provider.get_derivative("u", "x", order=2)

            left_err = abs(u_xx[0].item() - expected[0].item())
            right_err = abs(u_xx[-1].item() - expected[-1].item())
            return left_err, right_err


        _, err_coarse = compute_boundary_error(50)
        _, err_fine = compute_boundary_error(100)



        if err_coarse > 1e-10:
            ratio = err_coarse / err_fine
            assert ratio > 3.0, (
                f"Boundary convergence rate too low: ratio={ratio:.2f}, "
                f"expected >3 for 2nd-order accuracy. "
                f"Errors: coarse={err_coarse:.2e}, fine={err_fine:.2e}"
            )

    @pytest.mark.numerical
    def test_boundary_not_nan_or_inf(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)

        for order in [1, 2, 3]:
            deriv = provider.get_derivative("u", "x", order=order)
            assert torch.isfinite(deriv).all(), (
                f"Order {order} derivative contains NaN/Inf at boundary"
            )


class TestAccuracyParameter:

    @pytest.mark.unit
    def test_accuracy_default_is_4(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset)

        assert provider._accuracy == 4, (
            f"Default accuracy should be 4, got {provider._accuracy}"
        )

    @pytest.mark.unit
    def test_accuracy_4_accepted(self, simple_1d_dataset: PDEDataset) -> None:
        provider = FiniteDiffProvider(simple_1d_dataset, accuracy=4)
        assert provider._accuracy == 4

    @pytest.mark.unit
    def test_accuracy_2_raises(self, simple_1d_dataset: PDEDataset) -> None:
        with pytest.raises(ValueError, match="accuracy"):
            FiniteDiffProvider(simple_1d_dataset, accuracy=2)

    @pytest.mark.unit
    def test_accuracy_6_raises(self, simple_1d_dataset: PDEDataset) -> None:
        with pytest.raises(ValueError, match="accuracy"):
            FiniteDiffProvider(simple_1d_dataset, accuracy=6)

    @pytest.mark.unit
    def test_accuracy_invalid_raises(self, simple_1d_dataset: PDEDataset) -> None:
        with pytest.raises(ValueError, match="accuracy"):
            FiniteDiffProvider(simple_1d_dataset, accuracy=3)
        with pytest.raises(ValueError, match="accuracy"):
            FiniteDiffProvider(simple_1d_dataset, accuracy=0)
        with pytest.raises(ValueError, match="accuracy"):
            FiniteDiffProvider(simple_1d_dataset, accuracy=-1)







class TestPeriodicBoundary:

    @staticmethod
    def _make_periodic_dataset(
        n: int = 128,
        is_periodic: bool = True,
    ) -> PDEDataset:


        x_full = torch.linspace(0.0, 1.0, n + 1, dtype=torch.float64)
        x = x_full[:-1]
        u = torch.sin(2 * math.pi * x)

        return PDEDataset(
            name="periodic_sin",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={"x": AxisInfo(name="x", values=x, is_periodic=is_periodic)},
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="x",
        )

    @pytest.mark.numerical
    def test_periodic_boundary_matches_interior_accuracy(self) -> None:
        ds = self._make_periodic_dataset(n=128, is_periodic=True)
        provider = FiniteDiffProvider(ds)
        u_x = provider.get_derivative("u", "x", order=1)

        x = ds.get_coords("x")
        expected = 2 * math.pi * torch.cos(2 * math.pi * x)

        abs_err = (u_x - expected).abs()


        interior_max_err = abs_err[4:-4].max().item()


        boundary_max_err = max(
            abs_err[:2].max().item(),
            abs_err[-2:].max().item(),
        )



        assert boundary_max_err < 10 * interior_max_err, (
            f"Periodic boundary error ({boundary_max_err:.2e}) is much worse "
            f"than interior ({interior_max_err:.2e}). "
            f"Ratio = {boundary_max_err / max(interior_max_err, 1e-30):.0f}x. "
            f"is_periodic=True was likely ignored."
        )

    @pytest.mark.numerical
    def test_periodic_second_derivative_boundary(self) -> None:
        ds = self._make_periodic_dataset(n=128, is_periodic=True)
        provider = FiniteDiffProvider(ds)
        u_xx = provider.get_derivative("u", "x", order=2)

        x = ds.get_coords("x")
        expected = -((2 * math.pi) ** 2) * torch.sin(2 * math.pi * x)

        abs_err = (u_xx - expected).abs()
        interior_max_err = abs_err[4:-4].max().item()
        boundary_max_err = max(
            abs_err[:2].max().item(),
            abs_err[-2:].max().item(),
        )

        assert boundary_max_err < 10 * interior_max_err, (
            f"Periodic 2nd derivative boundary error ({boundary_max_err:.2e}) "
            f"vs interior ({interior_max_err:.2e}), "
            f"ratio = {boundary_max_err / max(interior_max_err, 1e-30):.0f}x"
        )

    @pytest.mark.numerical
    def test_nonperiodic_boundary_is_worse(self) -> None:
        ds = self._make_periodic_dataset(n=128, is_periodic=False)
        provider = FiniteDiffProvider(ds)
        u_x = provider.get_derivative("u", "x", order=1)

        x = ds.get_coords("x")
        expected = 2 * math.pi * torch.cos(2 * math.pi * x)

        abs_err = (u_x - expected).abs()
        interior_max_err = abs_err[4:-4].max().item()
        boundary_max_err = max(
            abs_err[:2].max().item(),
            abs_err[-2:].max().item(),
        )



        assert boundary_max_err > 10 * interior_max_err, (
            f"Expected non-periodic boundary to be much worse than interior, "
            f"but ratio is only {boundary_max_err / max(interior_max_err, 1e-30):.1f}x"
        )

    @pytest.mark.numerical
    def test_periodic_vs_nonperiodic_boundary_gap(self) -> None:
        n = 128
        ds_periodic = self._make_periodic_dataset(n=n, is_periodic=True)
        ds_nonperiodic = self._make_periodic_dataset(n=n, is_periodic=False)

        provider_p = FiniteDiffProvider(ds_periodic)
        provider_np = FiniteDiffProvider(ds_nonperiodic)

        u_x_p = provider_p.get_derivative("u", "x", order=1)
        u_x_np = provider_np.get_derivative("u", "x", order=1)

        x = ds_periodic.get_coords("x")
        expected = 2 * math.pi * torch.cos(2 * math.pi * x)


        boundary = list(range(2)) + list(range(n - 2, n))
        err_p = (u_x_p[boundary] - expected[boundary]).abs().max().item()
        err_np = (u_x_np[boundary] - expected[boundary]).abs().max().item()


        assert err_p < err_np / 5, (
            f"Periodic boundary error ({err_p:.2e}) should be <5x better "
            f"than non-periodic ({err_np:.2e}), "
            f"ratio = {err_np / max(err_p, 1e-30):.1f}x"
        )

    @pytest.mark.numerical
    def test_periodic_third_derivative_boundary(self) -> None:
        ds = self._make_periodic_dataset(n=128, is_periodic=True)
        provider = FiniteDiffProvider(ds)
        u_xxx = provider.get_derivative("u", "x", order=3)

        x = ds.get_coords("x")
        expected = -((2 * math.pi) ** 3) * torch.cos(2 * math.pi * x)

        abs_err = (u_xxx - expected).abs()
        interior_max_err = abs_err[4:-4].max().item()
        boundary_max_err = max(
            abs_err[:2].max().item(),
            abs_err[-2:].max().item(),
        )



        assert boundary_max_err < 10 * interior_max_err, (
            f"Periodic 3rd derivative boundary error ({boundary_max_err:.2e}) "
            f"vs interior ({interior_max_err:.2e}), "
            f"ratio = {boundary_max_err / max(interior_max_err, 1e-30):.0f}x"
        )

    @pytest.mark.numerical
    def test_periodic_2d_mixed_axes(self) -> None:
        n_x, n_t = 128, 32

        x_full = torch.linspace(0.0, 1.0, n_x + 1, dtype=torch.float64)
        x = x_full[:-1]

        t = torch.linspace(0.0, 1.0, n_t, dtype=torch.float64)

        X, T = torch.meshgrid(x, t, indexing="ij")
        u = torch.sin(2 * math.pi * X) * torch.exp(-T)

        ds = PDEDataset(
            name="periodic_2d",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x, is_periodic=True),
                "t": AxisInfo(name="t", values=t, is_periodic=False),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )

        provider = FiniteDiffProvider(ds)


        u_x = provider.get_derivative("u", "x", order=1)
        expected_u_x = 2 * math.pi * torch.cos(2 * math.pi * X) * torch.exp(-T)


        t_interior = slice(4, -4)
        x_err = (u_x[:, t_interior] - expected_u_x[:, t_interior]).abs()
        x_interior_err = x_err[4:-4,:].max().item()
        x_boundary_err = max(
            x_err[:2,:].max().item(),
            x_err[-2:,:].max().item(),
        )

        assert x_boundary_err < 10 * x_interior_err, (
            f"x (periodic) boundary error ({x_boundary_err:.2e}) should be "
            f"close to interior ({x_interior_err:.2e}), "
            f"ratio = {x_boundary_err / max(x_interior_err, 1e-30):.0f}x"
        )


        u_t = provider.get_derivative("u", "t", order=1)
        expected_u_t = -torch.sin(2 * math.pi * X) * torch.exp(-T)


        x_interior = slice(4, -4)
        t_err = (u_t[x_interior,:] - expected_u_t[x_interior,:]).abs()
        t_interior_err = t_err[:, 4:-4].max().item()
        t_boundary_err = max(
            t_err[:, :2].max().item(),
            t_err[:, -2:].max().item(),
        )


        assert t_boundary_err > 5 * t_interior_err, (
            f"t (non-periodic) boundary should be worse than interior, "
            f"but ratio is only {t_boundary_err / max(t_interior_err, 1e-30):.1f}x"
        )













class TestUniformGridTolerance:

    @pytest.mark.unit
    @pytest.mark.parametrize("n", [100, 500, 1000])
    def test_accepts_float32_linspace(self, n: int) -> None:
        from kd.data.derivatives.finite_diff import _check_uniform_grid

        x = torch.linspace(0.0, 1.0, n, dtype=torch.float32)
        dx = _check_uniform_grid(x, "x")

        assert dx == pytest.approx(1.0 / (n - 1), rel=1e-3)

    @pytest.mark.unit
    def test_accepts_float64_linspace_large(self) -> None:
        from kd.data.derivatives.finite_diff import _check_uniform_grid

        x = torch.linspace(0.0, 1.0, 100_000, dtype=torch.float64)
        dx = _check_uniform_grid(x, "x")
        assert dx == pytest.approx(1.0 / (100_000 - 1), rel=1e-9)

    @pytest.mark.unit
    def test_rejects_geometric_spacing(self) -> None:
        from kd.data.derivatives.finite_diff import _check_uniform_grid


        x = torch.tensor([2.0**i for i in range(10)], dtype=torch.float64)
        with pytest.raises(ValueError, match="non-uniform"):
            _check_uniform_grid(x, "x")

    @pytest.mark.unit
    def test_rejects_log_spacing(self) -> None:
        from kd.data.derivatives.finite_diff import _check_uniform_grid

        x = torch.logspace(0.0, 2.0, 50, dtype=torch.float64)
        with pytest.raises(ValueError, match="non-uniform"):
            _check_uniform_grid(x, "x")

    @pytest.mark.unit
    def test_rejects_degenerate_dx_zero(self) -> None:
        from kd.data.derivatives.finite_diff import _check_uniform_grid

        x = torch.zeros(10, dtype=torch.float64)
        with pytest.raises(ValueError, match="degenerate spacing"):
            _check_uniform_grid(x, "x")

    @pytest.mark.unit
    def test_rejects_subnormal_dx(self) -> None:
        from kd.data.derivatives.finite_diff import _check_uniform_grid


        x = torch.linspace(0.0, 1e-31, 10, dtype=torch.float64)
        with pytest.raises(ValueError, match="degenerate spacing"):
            _check_uniform_grid(x, "x")


class TestProviderFloat32:

    @pytest.mark.unit
    def test_provider_accepts_float32_dataset(self) -> None:
        nx, nt = 64, 32
        x = torch.linspace(-1.0, 1.0, nx + 1, dtype=torch.float32)[:-1]
        t = torch.linspace(0.0, 1.0, nt, dtype=torch.float32)

        u = torch.sin(torch.pi * x).unsqueeze(1) * torch.exp(-t).unsqueeze(0)

        ds = PDEDataset(
            name="float32_test",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x, is_periodic=True),
                "t": AxisInfo(name="t", values=t),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )
        provider = FiniteDiffProvider(ds, max_order=2)

        u_x = provider.get_derivative("u", "x", order=1)
        u_t = provider.get_derivative("u", "t", order=1)
        assert u_x.shape == ds.get_shape()
        assert u_t.shape == ds.get_shape()


class TestOrderCeilingSingleSource:

    @pytest.mark.unit
    def test_consumer_aliases_track_max_supported_order(self) -> None:


        from kd._evaluate_classify import _FD_MAX_ORDER
        from kd.core.integrator import _MAX_STENCIL_ORDER
        from kd.data.derivatives.finite_diff import MAX_SUPPORTED_ORDER
        from kd.harness.consensus_verify import _DEFAULT_MAX_ATOMIC_ORDER

        assert _MAX_STENCIL_ORDER == MAX_SUPPORTED_ORDER
        assert _FD_MAX_ORDER == MAX_SUPPORTED_ORDER
        assert _DEFAULT_MAX_ATOMIC_ORDER == MAX_SUPPORTED_ORDER
