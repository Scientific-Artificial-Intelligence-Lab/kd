
from __future__ import annotations

import math

import pytest
import torch

from kd.core.expr.naming import parse_compound_derivative
from kd.core.integrator import IntegrationResult, integrate_pde
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)






@pytest.fixture
def simple_grid_dataset() -> PDEDataset:
    nx, nt = 32, 10
    x = torch.linspace(0.0, 2 * torch.pi, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    u = torch.sin(x).unsqueeze(-1).expand(nx, nt).clone()
    return PDEDataset(
        name="test-simple",
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


@pytest.fixture
def scattered_dataset() -> PDEDataset:
    nx, nt = 16, 5
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
    u = torch.randn(nx, nt, dtype=torch.float64)
    return PDEDataset(
        name="test-scattered",
        task_type=TaskType.PDE,
        topology=DataTopology.SCATTERED,
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
def mixed_partial_dataset() -> PDEDataset:
    nx, ny, nt = 16, 16, 9
    x = torch.arange(nx, dtype=torch.float64) * (2 * torch.pi / nx)
    y = torch.arange(ny, dtype=torch.float64) * (2 * torch.pi / ny)
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    x_grid, y_grid, t_grid = torch.meshgrid(x, y, t, indexing="ij")
    u = torch.sin(x_grid + y_grid) * torch.exp(-t_grid)
    return PDEDataset(
        name="test-mixed-partial",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=True),
            "y": AxisInfo(name="y", values=y, is_periodic=True),
            "t": AxisInfo(name="t", values=t, is_periodic=False),
        },
        axis_order=["x", "y", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )







class TestIntegrationResult:

    @pytest.mark.smoke
    def test_success_result_construction(self) -> None:
        field = torch.randn(10, 5)
        result = IntegrationResult(success=True, predicted_field=field)
        assert result.success is True
        assert result.predicted_field is not None
        assert result.warning == ""
        assert result.diverged_at_t is None

    @pytest.mark.smoke
    def test_failure_result_construction(self) -> None:
        result = IntegrationResult(
            success=False,
            predicted_field=None,
            warning="NaN detected",
            diverged_at_t=0.42,
        )
        assert result.success is False
        assert result.predicted_field is None
        assert "NaN" in result.warning
        assert result.diverged_at_t == pytest.approx(0.42)







class TestIntegratePdeSmoke:

    @pytest.mark.smoke
    def test_callable(self) -> None:
        assert callable(integrate_pde)

    @pytest.mark.smoke
    def test_returns_integration_result(self, simple_grid_dataset: PDEDataset) -> None:
        result = integrate_pde("u_x", simple_grid_dataset)
        assert isinstance(result, IntegrationResult)







class TestScatteredRejection:

    def test_scattered_returns_failure(self, scattered_dataset: PDEDataset) -> None:
        result = integrate_pde("u_x", scattered_dataset)
        assert isinstance(result, IntegrationResult)
        assert result.success is False
        assert result.predicted_field is None
        assert result.warning

    def test_scattered_does_not_raise(self, scattered_dataset: PDEDataset) -> None:

        integrate_pde("u_x", scattered_dataset)







class TestOutputTensorProperties:

    def test_output_is_torch_tensor(self, simple_grid_dataset: PDEDataset) -> None:
        result = integrate_pde("u_x", simple_grid_dataset)
        assert result.success, f"Integration must succeed: {result.warning}"
        assert isinstance(result.predicted_field, torch.Tensor)

    def test_output_shape_matches_dataset(
        self, simple_grid_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("u_x", simple_grid_dataset)
        assert result.success, f"Integration must succeed: {result.warning}"
        assert result.predicted_field is not None
        expected_shape = simple_grid_dataset.get_shape()
        assert result.predicted_field.shape == expected_shape

    def test_output_is_finite(self, simple_grid_dataset: PDEDataset) -> None:
        result = integrate_pde("u_x", simple_grid_dataset)
        assert result.success, f"Integration must succeed: {result.warning}"
        assert result.predicted_field is not None
        assert torch.isfinite(result.predicted_field).all()







class TestEdgeCases:

    def test_zero_rhs_preserves_initial_condition(
        self, simple_grid_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("0", simple_grid_dataset)
        assert result.success, f"Integration must succeed: {result.warning}"
        assert result.predicted_field is not None
        ic = simple_grid_dataset.get_field("u")[:, 0]

        for t_idx in range(result.predicted_field.shape[-1]):
            torch.testing.assert_close(
                result.predicted_field[:, t_idx],
                ic,
                rtol=1e-4,
                atol=1e-6,
            )

    def test_constant_rhs_linear_growth(self, simple_grid_dataset: PDEDataset) -> None:
        c_val = 2.0
        result = integrate_pde("2.0", simple_grid_dataset)
        assert result.success, f"Integration must succeed: {result.warning}"
        assert result.predicted_field is not None
        t_vals = simple_grid_dataset.get_coords("t")
        ic = simple_grid_dataset.get_field("u")[:, 0]

        for t_idx in range(result.predicted_field.shape[-1]):
            expected = ic + c_val * t_vals[t_idx]
            torch.testing.assert_close(
                result.predicted_field[:, t_idx],
                expected,
                rtol=1e-4,
                atol=1e-6,
            )

    def test_unsupported_lap_function_rejected_before_solve(
        self, simple_grid_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("lap(u)", simple_grid_dataset)

        assert result.success is False
        assert "Unsupported function calls" in result.warning
        assert "lap" in result.warning
        assert "explicit derivative symbols" in result.warning
        assert "Integration failed" not in result.warning

    def test_unknown_symbol_message_states_facts_not_consumers(
        self, simple_grid_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("u + t", simple_grid_dataset)

        assert result.success is False
        assert "unrecognised symbols" in result.warning
        assert "'t'" in result.warning
        for consumer_text in ("Field-comparison", "pde-residual", "plots"):
            assert consumer_text not in result.warning

        assert len(result.warning) < 300, (
            f"Warning too long ({len(result.warning)} chars), "
            "should stay one terse sentence"
        )

    def test_linear_nested_diff_integrates_natively(
        self, simple_grid_dataset: PDEDataset
    ) -> None:
        result = integrate_pde(
            "diff_x(add(u_x, diff_x(u)))",
            simple_grid_dataset,
            max_step=0.05,
        )

        assert result.success is True, result.warning
        assert result.predicted_field is not None

    def test_explicit_coordinate_symbol_works(
        self, simple_grid_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("u + x", simple_grid_dataset)

        assert result.success is True, (
            f"Expected coord-as-variable to work natively; got: {result.warning}"
        )
        assert result.predicted_field is not None

    def test_method_parameter_accepted(self, simple_grid_dataset: PDEDataset) -> None:
        for method in ["RK45", "Radau", "BDF", "RK23"]:
            result = integrate_pde("0", simple_grid_dataset, method=method)
            assert isinstance(result, IntegrationResult)

    def test_max_step_parameter_accepted(self, simple_grid_dataset: PDEDataset) -> None:
        result = integrate_pde("0", simple_grid_dataset, max_step=0.01)
        assert isinstance(result, IntegrationResult)







class TestTimeOnlyDatasetRejection:

    @staticmethod
    def _time_only_dataset() -> PDEDataset:
        t = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
        return PDEDataset(
            name="test-time-only",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={"t": AxisInfo(name="t", values=t, is_periodic=False)},
            axis_order=["t"],
            fields={"u": FieldData(name="u", values=torch.exp(-t))},
            lhs_field="u",
            lhs_axis="t",
        )

    def test_time_only_dataset_returns_failure_not_raise(self) -> None:
        result = integrate_pde("(-1.0)*(u)", self._time_only_dataset())

        assert isinstance(result, IntegrationResult)
        assert result.success is False
        assert result.predicted_field is None
        assert "no spatial axes" in result.warning
        assert "spatial axis" in result.warning

    def test_same_rhs_on_spatial_dataset_succeeds(
        self, simple_grid_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("(-1.0)*(u)", simple_grid_dataset)

        assert result.success is True, result.warning
        assert result.predicted_field is not None







class TestNonUniformSpatialSpacing:

    def test_rejects_non_uniform_spatial_axis(self) -> None:

        nx, nt = 16, 5
        x_vals = torch.tensor(
            [0.1 * (2.0**i - 1.0) for i in range(nx)], dtype=torch.float64
        )
        t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
        u = torch.sin(x_vals).unsqueeze(-1).expand(nx, nt).clone()
        ds = PDEDataset(
            name="non-uniform",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x_vals, is_periodic=False),
                "t": AxisInfo(name="t", values=t, is_periodic=False),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )
        result = integrate_pde("u_x", ds)
        assert isinstance(result, IntegrationResult)
        assert result.success is False
        assert result.warning
        msg = result.warning.lower()
        assert "non-uniform" in msg or "uniform" in msg
        assert "x" in result.warning

    def test_does_not_raise_on_non_uniform(self) -> None:
        nx, nt = 8, 3
        x_vals = torch.tensor(
            [0.0, 0.1, 0.4, 1.0, 1.2, 1.25, 1.5, 1.51], dtype=torch.float64
        )
        t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
        u = torch.zeros(nx, nt, dtype=torch.float64)
        ds = PDEDataset(
            name="non-uniform",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x_vals, is_periodic=False),
                "t": AxisInfo(name="t", values=t, is_periodic=False),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )

        integrate_pde("u_x", ds)

    def test_accepts_uniform_with_floating_point_drift(
        self, simple_grid_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("u_x", simple_grid_dataset)



        if not result.success:
            assert "uniform" not in (result.warning or "").lower()

    def test_rejects_constant_axis_dx_zero(self) -> None:
        nx, nt = 8, 3
        x_vals = torch.zeros(nx, dtype=torch.float64)
        t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
        u = torch.zeros(nx, nt, dtype=torch.float64)
        ds = PDEDataset(
            name="degenerate",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x_vals, is_periodic=False),
                "t": AxisInfo(name="t", values=t, is_periodic=False),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )
        result = integrate_pde("u_x", ds)
        assert isinstance(result, IntegrationResult)
        assert result.success is False
        msg = (result.warning or "").lower()
        assert "x" in (result.warning or "")
        assert "dx" in msg or "degenerate" in msg or "zero" in msg

    def test_accepts_float32_linspace(self) -> None:
        nx, nt = 100, 3
        x_vals = torch.linspace(0.0, 1.0, nx, dtype=torch.float32)
        t = torch.linspace(0.0, 0.5, nt, dtype=torch.float32)
        u = torch.zeros(nx, nt, dtype=torch.float32)
        ds = PDEDataset(
            name="float32-grid",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x_vals, is_periodic=False),
                "t": AxisInfo(name="t", values=t, is_periodic=False),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )
        result = integrate_pde("u", ds)

        if not result.success:
            assert "non-uniform" not in (result.warning or "").lower()
            assert "uniform" not in (result.warning or "").lower()







class TestDivergenceHandling:

    def test_explosive_rhs_does_not_crash(
        self, simple_grid_dataset: PDEDataset
    ) -> None:

        result = integrate_pde("1e10*u", simple_grid_dataset)
        assert isinstance(result, IntegrationResult)
        assert not result.success, "Explosive RHS should fail"
        assert result.warning, "Failed integration must have a warning"

    def test_diverged_at_t_is_set_on_failure(
        self, simple_grid_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("1e10*u", simple_grid_dataset)
        assert not result.success, "Explosive RHS should fail"


        if result.diverged_at_t is not None:
            assert isinstance(result.diverged_at_t, float)
            assert result.diverged_at_t >= 0.0
        else:
            assert "fail" in result.warning.lower() or "nan" in result.warning.lower()







class TestMultiDimensional:

    def test_2d_dataset_accepted(self) -> None:
        nx, ny, nt = 16, 16, 5
        x = torch.linspace(0.0, 2 * torch.pi, nx, dtype=torch.float64)
        y = torch.linspace(0.0, 2 * torch.pi, ny, dtype=torch.float64)
        t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
        u = torch.randn(nx, ny, nt, dtype=torch.float64)
        dataset = PDEDataset(
            name="test-2d",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x, is_periodic=True),
                "y": AxisInfo(name="y", values=y, is_periodic=True),
                "t": AxisInfo(name="t", values=t, is_periodic=False),
            },
            axis_order=["x", "y", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )
        result = integrate_pde("u_x + u_y", dataset)
        assert isinstance(result, IntegrationResult)

    def test_mixed_partial_open_form_integrates(
        self, mixed_partial_dataset: PDEDataset
    ) -> None:
        result = integrate_pde(
            "diff_y(u_x)",
            mixed_partial_dataset,
            method="RK45",
            max_step=0.1,
        )
        assert result.success, f"Integration must succeed: {result.warning}"
        assert result.predicted_field is not None

        predicted = result.predicted_field.to(torch.float64)
        analytical = mixed_partial_dataset.get_field("u").to(torch.float64)
        initial = analytical[:, :, 0]
        final_predicted = predicted[:, :, -1]
        final_analytical = analytical[:, :, -1]

        corr = torch.corrcoef(
            torch.stack([final_predicted.flatten(), final_analytical.flatten()])
        )[0, 1]
        assert corr.item() > 0.99, f"Final-slice correlation {corr.item():.4f} < 0.99"

        decay_ratio = (final_predicted.norm() / initial.norm()).item()
        assert decay_ratio == pytest.approx(math.exp(-1.0), rel=0.15, abs=0.05)







class TestDirichletBoundary:

    def test_dirichlet_dataset_accepted(self) -> None:
        nx, nt = 32, 10
        x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
        t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
        u = torch.zeros(nx, nt, dtype=torch.float64)

        u[:, 0] = torch.sin(torch.pi * x)
        dataset = PDEDataset(
            name="test-dirichlet",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x, is_periodic=False),
                "t": AxisInfo(name="t", values=t, is_periodic=False),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )
        result = integrate_pde("u_xx", dataset)
        assert isinstance(result, IntegrationResult)

    def test_dirichlet_boundaries_preserved(self) -> None:
        nx, nt = 32, 10
        x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
        t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
        u = torch.zeros(nx, nt, dtype=torch.float64)
        u[:, 0] = torch.sin(torch.pi * x)
        dataset = PDEDataset(
            name="test-dirichlet-bc",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x, is_periodic=False),
                "t": AxisInfo(name="t", values=t, is_periodic=False),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )
        result = integrate_pde("u_xx", dataset)
        assert result.success, f"Integration must succeed: {result.warning}"
        assert result.predicted_field is not None

        left_bc = u[0, 0].item()
        right_bc = u[-1, 0].item()
        torch.testing.assert_close(
            result.predicted_field[0,:],
            torch.full((nt,), left_bc, dtype=torch.float64),
            rtol=1e-4,
            atol=1e-6,
        )
        torch.testing.assert_close(
            result.predicted_field[-1,:],
            torch.full((nt,), right_bc, dtype=torch.float64),
            rtol=1e-4,
            atol=1e-6,
        )







class TestCrossFieldGuard:

    @staticmethod
    def _two_field_dataset() -> PDEDataset:
        nx, nt = 32, 10
        x = torch.linspace(0.0, 2 * torch.pi, nx, dtype=torch.float64)
        t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
        u = torch.sin(x).unsqueeze(-1).expand(nx, nt).clone()
        v = torch.cos(x).unsqueeze(-1).expand(nx, nt).clone()
        return PDEDataset(
            name="test-two-field",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x, is_periodic=True),
                "t": AxisInfo(name="t", values=t, is_periodic=False),
            },
            axis_order=["x", "t"],
            fields={
                "u": FieldData(name="u", values=u),
                "v": FieldData(name="v", values=v),
            },
            lhs_field="u",
            lhs_axis="t",
        )

    def test_cross_field_derivative_rejected(self) -> None:
        ds = self._two_field_dataset()
        result = integrate_pde("v_x", ds)
        assert result.success is False, (
            "Cross-field derivative v_x should be rejected, "
            "but integration returned success=True"
        )

        assert result.warning, "Failed result must include a warning"
        assert "v_x" in result.warning

    def test_cross_field_compound_derivative_rejected(self) -> None:
        ds = self._two_field_dataset()
        result = integrate_pde("v_xx", ds)
        assert result.success is False, (
            "Cross-field derivative v_xx should be rejected, "
            "but integration returned success=True"
        )
        assert result.warning, "Failed result must include a warning"
        assert "v_xx" in result.warning

    def test_same_field_derivative_accepted(self) -> None:
        ds = self._two_field_dataset()
        result = integrate_pde("u_x", ds)
        assert result.success is True, (
            f"Same-field derivative u_x should be accepted, "
            f"but got failure: {result.warning}"
        )
        assert result.predicted_field is not None

    def test_multi_field_state_var_still_rejected(self) -> None:
        ds = self._two_field_dataset()
        result = integrate_pde("v", ds)
        assert result.success is False, (
            "Direct cross-field reference 'v' should be rejected, "
            "but integration returned success=True"
        )









class TestStencilConstraintsSurfaceThroughIntegration:

    @staticmethod
    def _tiny_grid_dataset() -> PDEDataset:
        nx, nt = 4, 5
        x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
        t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
        u = torch.sin(x).unsqueeze(-1).expand(nx, nt).clone()
        return PDEDataset(
            name="tiny-grid",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x, is_periodic=False),
                "t": AxisInfo(name="t", values=t, is_periodic=False),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )

    def test_derivative_rhs_below_stencil_minimum_fails_gracefully(self) -> None:
        result = integrate_pde("u_x", self._tiny_grid_dataset())

        assert isinstance(result, IntegrationResult)
        assert result.success is False
        assert result.predicted_field is None
        assert "points" in result.warning

    def test_order_above_stencil_maximum_rejected_preflight(self) -> None:
        nx, nt = 32, 10
        x = torch.linspace(0.0, 2 * torch.pi, nx, dtype=torch.float64)
        t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
        u = torch.sin(x).unsqueeze(-1).expand(nx, nt).clone()
        ds = PDEDataset(
            name="order-cap",
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

        result = integrate_pde("u_xxxx", ds)
        assert result.success is False
        assert "u_xxxx" in result.warning

        control = integrate_pde("u_xxx", ds, method="RK45", max_step=0.05)
        assert "u_xxx" not in (control.warning or "")







class TestMixedPartialConfirmation:

    def test_mixed_partial_parse_correct(self) -> None:
        result = parse_compound_derivative(
            "u_x_y",
            known_fields={"u"},
            known_axes={"x", "y"},
        )
        assert result is not None, "u_x_y should parse as a compound derivative"
        field, axis_orders = result
        assert field == "u"
        assert len(axis_orders) == 2
        assert ("x", 1) in axis_orders
        assert ("y", 1) in axis_orders

    def test_mixed_partial_correct_values(
        self, mixed_partial_dataset: PDEDataset
    ) -> None:
        result = integrate_pde(
            "diff_y(u_x)",
            mixed_partial_dataset,
            method="RK45",
            max_step=0.05,
        )
        assert result.success, f"Integration must succeed: {result.warning}"
        assert result.predicted_field is not None

        predicted = result.predicted_field.to(torch.float64)
        analytical = mixed_partial_dataset.get_field("u").to(torch.float64)



        initial_norm = analytical[:, :, 0].norm()
        final_predicted_norm = predicted[:, :, -1].norm()
        final_analytical_norm = analytical[:, :, -1].norm()


        predicted_ratio = (final_predicted_norm / initial_norm).item()
        analytical_ratio = (final_analytical_norm / initial_norm).item()


        torch.testing.assert_close(
            torch.tensor(predicted_ratio),
            torch.tensor(analytical_ratio),
            rtol=0.05,
            atol=0.02,
        )


        final_err = (predicted[:, :, -1] - analytical[:, :, -1]).norm()
        assert final_err / final_analytical_norm < 0.1, (
            f"Relative pointwise error {final_err / final_analytical_norm:.4f} "
            f"exceeds 10% threshold"
        )







class TestCheckSpatialUniformitySmallAxis:

    @staticmethod
    def _build_dataset(x_vals: torch.Tensor, nt: int = 4) -> PDEDataset:
        t = torch.linspace(0.0, 1.0, nt, dtype=x_vals.dtype)
        u = torch.zeros(x_vals.shape[0], nt, dtype=x_vals.dtype)
        return PDEDataset(
            name="small-axis-check",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x_vals),
                "t": AxisInfo(name="t", values=t),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )

    @pytest.mark.unit
    def test_size_one_axis_emits_warning(self) -> None:
        from kd.core.integrator import _check_spatial_uniformity

        dataset = self._build_dataset(torch.tensor([0.0], dtype=torch.float64))
        warning = _check_spatial_uniformity(dataset, ["x"])
        assert warning is not None, "size==1 must be reported, not silently skipped"
        assert ">=2" in warning or "must have" in warning, (
            f"Warning should mention the size requirement, got: {warning!r}"
        )

    @pytest.mark.unit
    def test_size_two_axis_accepts_when_uniform(self) -> None:
        from kd.core.integrator import _check_spatial_uniformity

        dataset = self._build_dataset(
            torch.tensor([0.0, 0.1], dtype=torch.float64),
        )
        warning = _check_spatial_uniformity(dataset, ["x"])
        assert warning is None, (
            f"size==2 uniform grid should be accepted by both FD and "
            f"integrator, got warning: {warning!r}"
        )
