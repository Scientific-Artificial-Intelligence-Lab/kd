
from __future__ import annotations

import math

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

matplotlib.use("Agg")

from matplotlib.figure import Figure

from kd.core.evaluator import EvaluationResult
from kd.core.integrator import IntegrationResult
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult
from kd.viz.plots.time_slices import plot_time_slices





_TWO_PI = 2.0 * math.pi


def _make_1d_dataset(nx: int = 20, nt: int = 10) -> PDEDataset:
    x = torch.linspace(0, _TWO_PI, nx)
    t = torch.linspace(0, 1, nt)
    u_field = torch.sin(x).unsqueeze(1) * torch.exp(-t).unsqueeze(0)
    return PDEDataset(
        name="test_1d",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=True),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u_field)},
        lhs_field="u",
        lhs_axis="t",
    )


def _make_2d_dataset(nx: int = 10, ny: int = 10, nt: int = 5) -> PDEDataset:
    x = torch.linspace(0, _TWO_PI, nx)
    y = torch.linspace(0, _TWO_PI, ny)
    t = torch.linspace(0, 1, nt)
    u_field = (
        torch.sin(x).reshape(nx, 1, 1)
        * torch.cos(y).reshape(1, ny, 1)
        * torch.exp(-t).reshape(1, 1, nt)
    )
    return PDEDataset(
        name="test_2d",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=True),
            "y": AxisInfo(name="y", values=y, is_periodic=True),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "y", "t"],
        fields={"u": FieldData(name="u", values=u_field)},
        lhs_field="u",
        lhs_axis="t",
    )


def _make_integration_result(
    dataset: PDEDataset,
    *,
    success: bool = True,
    noise_std: float = 0.01,
) -> IntegrationResult:
    if success:
        true_field = dataset.get_field("u")
        return IntegrationResult(
            success=True,
            predicted_field=true_field + torch.randn_like(true_field) * noise_std,
        )
    return IntegrationResult(success=False, warning="Integration failed")


def _make_experiment_result() -> ExperimentResult:
    n = 50
    actual = torch.randn(n)
    predicted = actual + torch.randn(n) * 0.1
    recorder = VizRecorder()
    recorder.log("_best_score", 1.0)
    recorder.log("_best_expr", "u")
    recorder.log("_n_candidates", 10)
    return ExperimentResult(
        best_expression="u",
        best_score=1.0,
        iterations=1,
        early_stopped=False,
        final_eval=EvaluationResult(
            mse=0.01,
            nmse=0.005,
            r2=0.95,
            aic=-100.0,
            complexity=1,
            coefficients=torch.tensor([1.0]),
            is_valid=True,
            error_message="",
            selected_indices=[0],
            residuals=predicted - actual,
            terms=["u"],
            expression="u",
        ),
        actual=actual,
        predicted=predicted,
        dataset_name="test",
        algorithm_name="SGA",
        config={},
        recorder=recorder,
    )







class TestTimeSlicesSmoke:

    def test_callable_returns_tuple(self) -> None:
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, warnings = plot_time_slices(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
            assert isinstance(warnings, list)
        finally:
            plt.close(fig)

    def test_accepts_style_kwarg(self) -> None:
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir, style={"font.size": 12})
        plt.close(fig)

    def test_accepts_n_slices_kwarg(self) -> None:
        ds = _make_1d_dataset(nt=10)
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir, n_slices=5)
        plt.close(fig)







class TestTimeSlicesHappyPath:

    def test_1d_produces_panels(self) -> None:
        ds = _make_1d_dataset(nt=10)
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir, n_slices=3)
        try:

            assert len(fig.get_axes()) >= 3
        finally:
            plt.close(fig)

    def test_2d_produces_panels(self) -> None:
        ds = _make_2d_dataset(nt=5)
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir, n_slices=3)
        try:
            assert len(fig.get_axes()) >= 3
        finally:
            plt.close(fig)

    def test_n_slices_affects_panel_count(self) -> None:
        ds = _make_1d_dataset(nt=20)
        ir = _make_integration_result(ds)
        result = _make_experiment_result()

        fig2, _ = plot_time_slices(result, ds, ir, n_slices=2)
        fig5, _ = plot_time_slices(result, ds, ir, n_slices=5)
        try:
            assert len(fig5.get_axes()) > len(fig2.get_axes())
        finally:
            plt.close(fig2)
            plt.close(fig5)

    def test_panels_have_titles(self) -> None:
        ds = _make_1d_dataset(nt=10)
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir, n_slices=3)
        try:
            titles = [ax.get_title() for ax in fig.get_axes()]
            non_empty = [t for t in titles if t]
            assert len(non_empty) >= 3
        finally:
            plt.close(fig)







class TestTimeSlicesEdgeCases:

    def test_failed_integration_no_crash(self) -> None:
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds, success=False)
        result = _make_experiment_result()
        fig, warnings = plot_time_slices(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
            assert len(warnings) > 0
        finally:
            plt.close(fig)

    def test_single_time_step(self) -> None:
        ds = _make_1d_dataset(nx=10, nt=1)
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir, n_slices=3)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_n_slices_larger_than_nt(self) -> None:
        ds = _make_1d_dataset(nt=2)
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir, n_slices=10)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_diverged_integration(self) -> None:
        ds = _make_1d_dataset()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(
            success=False,
            predicted_field=pred,
            warning="Diverged at t=0.5",
            diverged_at_t=0.5,
        )
        result = _make_experiment_result()
        fig, warnings = plot_time_slices(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_no_figure_leak(self) -> None:
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        figs_before = len(plt.get_fignums())
        fig, _ = plot_time_slices(result, ds, ir)
        plt.close(fig)
        figs_after = len(plt.get_fignums())
        assert figs_after <= figs_before







def _make_ode_dataset_no_axis_order() -> PDEDataset:
    nt = 10
    t = torch.linspace(0, 1, nt)
    u_field = torch.sin(t)
    return PDEDataset(
        name="test_ode",
        task_type=TaskType.ODE,
        axes=None,
        axis_order=None,
        fields={"u": FieldData(name="u", values=u_field)},
        lhs_field="u",
        lhs_axis="t",
    )


def _make_ode_dataset_n_spatial_0() -> PDEDataset:
    nt = 10
    t = torch.linspace(0, 1, nt)
    u_field = torch.sin(t)
    return PDEDataset(
        name="test_ode_t_only",
        task_type=TaskType.ODE,
        axes={"t": AxisInfo(name="t", values=t)},
        axis_order=["t"],
        fields={"u": FieldData(name="u", values=u_field)},
        lhs_field="u",
        lhs_axis="t",
    )


class TestTimeSlicesAxisOrderNone:

    def test_axis_order_none_no_crash(self) -> None:
        ds = _make_ode_dataset_no_axis_order()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(success=True, predicted_field=pred)
        result = _make_experiment_result()
        fig, warnings = plot_time_slices(result, ds, ir)
        try:
            assert isinstance(fig, Figure)

            assert len(warnings) > 0
        finally:
            plt.close(fig)

    def test_axis_order_none_returns_valid_figure(self) -> None:
        ds = _make_ode_dataset_no_axis_order()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(success=True, predicted_field=pred)
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir)
        try:
            assert len(fig.get_axes()) >= 1
        finally:
            plt.close(fig)

    def test_axis_order_none_failed_integration(self) -> None:
        ds = _make_ode_dataset_no_axis_order()
        ir = IntegrationResult(success=False, warning="Integration failed")
        result = _make_experiment_result()
        fig, warnings = plot_time_slices(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)


class TestTimeSlicesNSpatialZero:

    def test_n_spatial_zero_no_crash(self) -> None:
        ds = _make_ode_dataset_n_spatial_0()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(success=True, predicted_field=pred)
        result = _make_experiment_result()
        fig, warnings = plot_time_slices(result, ds, ir)
        try:
            assert isinstance(fig, Figure)

            assert len(warnings) > 0
        finally:
            plt.close(fig)

    def test_n_spatial_zero_returns_valid_figure(self) -> None:
        ds = _make_ode_dataset_n_spatial_0()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(success=True, predicted_field=pred)
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir)
        try:
            assert len(fig.get_axes()) >= 1
        finally:
            plt.close(fig)

    def test_n_spatial_zero_failed_integration(self) -> None:
        ds = _make_ode_dataset_n_spatial_0()
        ir = IntegrationResult(success=False, warning="Integration failed")
        result = _make_experiment_result()
        fig, warnings = plot_time_slices(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)







class TestTimeSlicesDivergedTitle:

    def test_diverged_1d_title_contains_diverged(self) -> None:
        ds = _make_1d_dataset()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(
            success=False,
            predicted_field=pred,
            warning="Diverged at t=0.5",
            diverged_at_t=0.5,
        )
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir)
        try:
            titles = [ax.get_title() for ax in fig.get_axes()]
            all_titles = " ".join(titles)
            assert "DIVERGED" in all_titles, (
                f"Expected 'DIVERGED' in panel titles, got: {titles}"
            )
        finally:
            plt.close(fig)

    def test_diverged_2d_title_contains_diverged(self) -> None:
        ds = _make_2d_dataset()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(
            success=False,
            predicted_field=pred,
            warning="Diverged at t=0.3",
            diverged_at_t=0.3,
        )
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir)
        try:
            titles = [ax.get_title() for ax in fig.get_axes()]
            all_titles = " ".join(titles)
            assert "DIVERGED" in all_titles, (
                f"Expected 'DIVERGED' in panel titles, got: {titles}"
            )
        finally:
            plt.close(fig)

    def test_diverged_without_at_t_still_has_tag(self) -> None:
        ds = _make_1d_dataset()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(
            success=False,
            predicted_field=pred,
            warning="Integration did not succeed",
            diverged_at_t=None,
        )
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir)
        try:
            titles = [ax.get_title() for ax in fig.get_axes()]
            all_titles = " ".join(titles)
            assert "DIVERGED" in all_titles, (
                f"Expected 'DIVERGED' in panel titles, got: {titles}"
            )
        finally:
            plt.close(fig)

    def test_non_diverged_no_diverged_tag(self) -> None:
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds, success=True)
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir)
        try:
            titles = [ax.get_title() for ax in fig.get_axes()]
            all_titles = " ".join(titles)
            assert "DIVERGED" not in all_titles, (
                f"Non-diverged should not have DIVERGED tag, got: {titles}"
            )
        finally:
            plt.close(fig)







class TestTimeSlicesNormalRegression:

    def test_1d_normal_dataset_unchanged(self) -> None:
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, warnings = plot_time_slices(result, ds, ir, n_slices=3)
        try:
            assert isinstance(fig, Figure)
            assert len(fig.get_axes()) >= 3

            guard_warnings = [
                w for w in warnings if "spatial" in w.lower() or "axis" in w.lower()
            ]
            assert len(guard_warnings) == 0, (
                f"Normal dataset should not trigger guards: {guard_warnings}"
            )
        finally:
            plt.close(fig)

    def test_2d_normal_dataset_unchanged(self) -> None:
        ds = _make_2d_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, warnings = plot_time_slices(result, ds, ir, n_slices=3)
        try:
            assert isinstance(fig, Figure)
            assert len(fig.get_axes()) >= 3
        finally:
            plt.close(fig)
