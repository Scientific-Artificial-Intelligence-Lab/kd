
from __future__ import annotations

import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from matplotlib.figure import Figure

matplotlib.use("Agg")

from kd.core.evaluator import EvaluationResult
from kd.core.integrator import IntegrationResult
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult
from kd.viz.plots._dim_utils import _pick_time_steps, _slice_nd_to_2d
from kd.viz.plots.field import plot_field_comparison
from kd.viz.plots.pde_residual import plot_pde_residual_field





_TWO_PI = 2.0 * math.pi


def _make_1d_pde_dataset(nx: int = 20, nt: int = 10) -> PDEDataset:
    x = torch.linspace(0, _TWO_PI, nx)
    t = torch.linspace(0, 1, nt)

    u_field = torch.sin(x).unsqueeze(1) * torch.exp(-t).unsqueeze(0)
    return PDEDataset(
        name="test_burgers_1d",
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


def _make_2d_pde_dataset(nx: int = 10, ny: int = 10, nt: int = 5) -> PDEDataset:
    x = torch.linspace(0, _TWO_PI, nx)
    y = torch.linspace(0, _TWO_PI, ny)
    t = torch.linspace(0, 1, nt)

    u_field = (
        torch.sin(x).reshape(nx, 1, 1)
        * torch.cos(y).reshape(1, ny, 1)
        * torch.exp(-t).reshape(1, 1, nt)
    )
    return PDEDataset(
        name="test_diffusion_2d",
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
    add_noise_std: float = 0.01,
) -> IntegrationResult:
    if success:
        true_field = dataset.get_field("u")
        noise = torch.randn_like(true_field) * add_noise_std
        return IntegrationResult(
            success=True,
            predicted_field=true_field + noise,
        )
    return IntegrationResult(
        success=False,
        warning="Test: integration diverged",
    )


def _make_diverged_integration_result(
    dataset: PDEDataset,
) -> IntegrationResult:
    true_field = dataset.get_field("u")

    pred = true_field.clone()

    return IntegrationResult(
        success=False,
        predicted_field=pred,
        warning="Integration diverged at t=0.5",
        diverged_at_t=0.5,
    )


def _make_experiment_result(n_samples: int = 50) -> ExperimentResult:
    actual = torch.sin(torch.linspace(0, _TWO_PI, n_samples))
    predicted = actual + torch.randn(n_samples) * 0.1
    residuals = predicted - actual
    recorder = VizRecorder()
    for s in [10.0, 5.0, 2.0]:
        recorder.log("_best_score", s)
        recorder.log("_best_expr", "u + u_x")
        recorder.log("_n_candidates", 20)
    return ExperimentResult(
        best_expression="add(u, u_x)",
        best_score=2.0,
        iterations=3,
        early_stopped=False,
        final_eval=EvaluationResult(
            mse=0.01,
            nmse=0.005,
            r2=0.95,
            aic=-100.0,
            complexity=2,
            coefficients=torch.tensor([1.0, -0.5]),
            is_valid=True,
            error_message="",
            selected_indices=[0, 1],
            residuals=residuals,
            terms=["u", "u_x"],
            expression="add(u, u_x)",
        ),
        actual=actual,
        predicted=predicted,
        dataset_name="test",
        algorithm_name="SGA",
        config={"max_iter": 3},
        recorder=recorder,
    )







class TestPickTimeSteps:

    def test_three_from_ten(self) -> None:
        result = _pick_time_steps(10, 3)
        assert isinstance(result, list)
        assert len(result) == 3

        assert result[0] == 0
        assert result[-1] == 9

    def test_single_timestep(self) -> None:
        result = _pick_time_steps(1, 3)
        assert result == [0]

    def test_two_timesteps(self) -> None:
        result = _pick_time_steps(2, 3)
        assert len(result) <= 2
        assert 0 in result
        assert 1 in result

    def test_n_equals_n_t(self) -> None:
        result = _pick_time_steps(4, 4)
        assert result == [0, 1, 2, 3]

    def test_n_larger_than_n_t(self) -> None:
        result = _pick_time_steps(3, 10)
        assert len(result) == 3
        assert result == [0, 1, 2]

    def test_all_indices_valid_range(self) -> None:
        result = _pick_time_steps(20, 5)
        for idx in result:
            assert 0 <= idx < 20

    def test_indices_are_sorted(self) -> None:
        result = _pick_time_steps(100, 7)
        assert result == sorted(result)

    def test_no_duplicates(self) -> None:
        result = _pick_time_steps(50, 5)
        assert len(result) == len(set(result))

    @given(
        n_t=st.integers(min_value=1, max_value=500),
        n=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=50)
    def test_property_length_and_bounds(self, n_t: int, n: int) -> None:
        result = _pick_time_steps(n_t, n)
        assert len(result) == min(n, n_t)
        assert result == sorted(set(result))
        for idx in result:
            assert 0 <= idx < n_t

        if n >= 2 and n_t >= 2:
            assert result[0] == 0
            assert result[-1] == n_t - 1







class TestSliceNdTo2d:

    def test_2d_identity(self) -> None:
        arr = np.random.default_rng(42).standard_normal((5, 7))
        result = _slice_nd_to_2d(arr, axes=(0, 1))
        np.testing.assert_array_equal(result, arr)

    def test_3d_to_2d_keeps_first_two(self) -> None:
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((4, 6, 10))
        result = _slice_nd_to_2d(arr, axes=(0, 1))
        assert result.shape == (4, 6)

        expected = arr[:, :, 5]
        np.testing.assert_array_equal(result, expected)

    def test_3d_to_2d_keeps_last_two(self) -> None:
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((8, 4, 6))
        result = _slice_nd_to_2d(arr, axes=(1, 2))
        assert result.shape == (4, 6)
        expected = arr[4, :,:]
        np.testing.assert_array_equal(result, expected)

    def test_4d_to_2d(self) -> None:
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((6, 8, 10, 12))
        result = _slice_nd_to_2d(arr, axes=(1, 3))
        assert result.shape == (8, 12)

        expected = arr[3, :, 5,:]
        np.testing.assert_array_equal(result, expected)

    def test_output_shape_always_2d(self) -> None:
        rng = np.random.default_rng(42)
        for ndim in range(2, 6):
            shape = tuple(range(3, 3 + ndim))
            arr = rng.standard_normal(shape)
            result = _slice_nd_to_2d(arr, axes=(0, ndim - 1))
            assert result.ndim == 2

    def test_preserves_selected_axes_sizes(self) -> None:
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((5, 7, 9, 11))
        result = _slice_nd_to_2d(arr, axes=(1, 3))
        assert result.shape == (7, 11)

    @given(
        extra_dims=st.integers(min_value=0, max_value=3),
    )
    @settings(max_examples=20)
    def test_property_output_is_2d(self, extra_dims: int) -> None:
        ndim = 2 + extra_dims
        shape = tuple([4] * ndim)
        arr = np.ones(shape)
        result = _slice_nd_to_2d(arr, axes=(0, ndim - 1))
        assert result.ndim == 2







class TestPlotFieldComparison:

    def test_1d_custom_axis_labels_use_dataset_names(
        self,
        custom_axis_dataset: PDEDataset,
    ) -> None:
        ir = _make_integration_result(custom_axis_dataset)
        result = _make_experiment_result()

        fig, _ = plot_field_comparison(result, custom_axis_dataset, ir)
        try:
            plot_axes = fig.get_axes()[:3]
            assert {ax.get_xlabel() for ax in plot_axes} == {"tau"}
            assert {ax.get_ylabel() for ax in plot_axes} == {"xi"}
        finally:
            plt.close(fig)

    def test_1d_spatial_returns_figure_and_warnings(self) -> None:
        ds = _make_1d_pde_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()

        fig, warnings = plot_field_comparison(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
            assert isinstance(warnings, list)
        finally:
            plt.close(fig)

    def test_1d_spatial_has_three_panels(self) -> None:
        ds = _make_1d_pde_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()

        fig, _ = plot_field_comparison(result, ds, ir)
        try:
            axes = fig.get_axes()

            assert len(axes) >= 3
        finally:
            plt.close(fig)

    def test_1d_spatial_panel_titles_contain_true_predicted_residual(self) -> None:
        ds = _make_1d_pde_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()

        fig, _ = plot_field_comparison(result, ds, ir)
        try:
            titles = [ax.get_title().lower() for ax in fig.get_axes()]
            title_text = " ".join(titles)
            assert "true" in title_text or "truth" in title_text
            assert "predict" in title_text
            assert "resid" in title_text
        finally:
            plt.close(fig)

    def test_2d_spatial_returns_figure(self) -> None:
        ds = _make_2d_pde_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()

        fig, warnings = plot_field_comparison(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
            assert isinstance(warnings, list)
        finally:
            plt.close(fig)

    def test_2d_spatial_has_multiple_panels(self) -> None:
        ds = _make_2d_pde_dataset(nt=5)
        ir = _make_integration_result(ds)
        result = _make_experiment_result()

        fig, _ = plot_field_comparison(result, ds, ir)
        try:


            axes = fig.get_axes()
            assert len(axes) >= 3
        finally:
            plt.close(fig)

    def test_2d_spatial_uses_physical_extent_and_axis_labels(
        self,
        rectangular_2d_dataset: PDEDataset,
    ) -> None:
        ir = _make_integration_result(rectangular_2d_dataset)
        result = _make_experiment_result()

        fig, _ = plot_field_comparison(result, rectangular_2d_dataset, ir)
        try:
            data_axes = [ax for ax in fig.get_axes() if ax.images]
            assert data_axes
            for ax in data_axes:
                assert tuple(ax.images[0].get_extent()) == pytest.approx(
                    (10.0, 14.0, -2.0, 3.0)
                )
                assert ax.get_xlabel() == "eta"
                assert ax.get_ylabel() == "xi"
        finally:
            plt.close(fig)

    def test_integration_failure_no_crash(self) -> None:
        ds = _make_1d_pde_dataset()
        ir = _make_integration_result(ds, success=False)
        result = _make_experiment_result()

        fig, warnings = plot_field_comparison(result, ds, ir)
        try:
            assert isinstance(fig, Figure)

            assert (
                len(warnings) > 0
                or any(
                    "fail" in ax.get_title().lower()
                    or "warn" in ax.get_title().lower()
                    or "n/a" in ax.get_title().lower()
                    for ax in fig.get_axes()
                )
                or _figure_contains_warning_text(fig)
            )
        finally:
            plt.close(fig)

    def test_integration_failure_predicted_panel_shows_message(self) -> None:
        ds = _make_1d_pde_dataset()
        ir = _make_integration_result(ds, success=False)
        result = _make_experiment_result()

        fig, _ = plot_field_comparison(result, ds, ir)
        try:

            all_texts = []
            for ax in fig.get_axes():
                all_texts.extend(t.get_text().lower() for t in ax.texts)
            all_text_joined = " ".join(all_texts)
            has_warning = (
                "fail" in all_text_joined
                or "diverge" in all_text_joined
                or "n/a" in all_text_joined
                or "warning" in all_text_joined
                or "no prediction" in all_text_joined
            )
            assert has_warning, (
                f"Expected warning text in Predicted panel, got: {all_texts}"
            )
        finally:
            plt.close(fig)

    def test_diverged_integration_handled_gracefully(self) -> None:
        ds = _make_1d_pde_dataset()
        ir = _make_diverged_integration_result(ds)
        result = _make_experiment_result()

        fig, warnings = plot_field_comparison(result, ds, ir)
        try:
            assert isinstance(fig, Figure)

        finally:
            plt.close(fig)

    def test_style_parameter_accepted(self) -> None:
        ds = _make_1d_pde_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()

        fig, _ = plot_field_comparison(result, ds, ir, style={"font.size": 12})
        plt.close(fig)

    def test_no_figures_leaked(self) -> None:
        ds = _make_1d_pde_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()

        figs_before = len(plt.get_fignums())
        fig, _ = plot_field_comparison(result, ds, ir)
        plt.close(fig)
        figs_after = len(plt.get_fignums())
        assert figs_after <= figs_before

    def test_return_type_is_tuple(self) -> None:
        ds = _make_1d_pde_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()

        out = plot_field_comparison(result, ds, ir)
        try:
            assert isinstance(out, tuple)
            assert len(out) == 2
            assert isinstance(out[0], Figure)
            assert isinstance(out[1], list)
        finally:
            plt.close(out[0])







class TestPlotPdeResidualField:

    def test_returns_figure_and_warnings(self) -> None:
        result = _make_experiment_result()
        fig, warnings = plot_pde_residual_field(result, field_shape=(10, 5))
        try:
            assert isinstance(fig, Figure)
            assert isinstance(warnings, list)
        finally:
            plt.close(fig)

    def test_produces_panels(self) -> None:
        n_samples = 50
        result = _make_experiment_result(n_samples=n_samples)
        fig, _ = plot_pde_residual_field(result, field_shape=(10, 5))
        try:
            assert len(fig.get_axes()) >= 3
        finally:
            plt.close(fig)

    def test_shape_mismatch_produces_warning(self) -> None:
        result = _make_experiment_result(n_samples=50)
        fig, warnings = plot_pde_residual_field(result, field_shape=(999, 999))
        try:
            assert len(warnings) > 0
        finally:
            plt.close(fig)

    def test_none_field_shape_produces_warning_or_fallback(self) -> None:
        result = _make_experiment_result()
        fig, warnings = plot_pde_residual_field(result, field_shape=None)
        try:
            assert isinstance(fig, Figure)

            assert isinstance(warnings, list)
        finally:
            plt.close(fig)

    def test_style_parameter_accepted(self) -> None:
        result = _make_experiment_result(n_samples=50)
        fig, _ = plot_pde_residual_field(
            result, style={"font.size": 12}, field_shape=(10, 5)
        )
        plt.close(fig)

    def test_return_type(self) -> None:
        result = _make_experiment_result(n_samples=50)
        out = plot_pde_residual_field(result, field_shape=(10, 5))
        try:
            assert isinstance(out, tuple)
            assert len(out) == 2
            assert isinstance(out[0], Figure)
            assert isinstance(out[1], list)
        finally:
            plt.close(out[0])

    def test_no_figures_leaked(self) -> None:
        result = _make_experiment_result(n_samples=50)
        figs_before = len(plt.get_fignums())
        fig, _ = plot_pde_residual_field(result, field_shape=(10, 5))
        plt.close(fig)
        figs_after = len(plt.get_fignums())
        assert figs_after <= figs_before







class TestFieldComparisonEdgeCases:

    def test_very_small_dataset(self) -> None:
        ds = _make_1d_pde_dataset(nx=2, nt=2)
        ir = _make_integration_result(ds)
        result = _make_experiment_result()

        fig, warnings = plot_field_comparison(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_large_noise_in_prediction(self) -> None:
        ds = _make_1d_pde_dataset()
        ir = _make_integration_result(ds, add_noise_std=100.0)
        result = _make_experiment_result()

        fig, warnings = plot_field_comparison(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_integration_result_none_predicted_field(self) -> None:
        ds = _make_1d_pde_dataset()
        ir = IntegrationResult(
            success=False,
            predicted_field=None,
            warning="Total failure",
        )
        result = _make_experiment_result()

        fig, warnings = plot_field_comparison(result, ds, ir)
        try:
            assert isinstance(fig, Figure)

        finally:
            plt.close(fig)

    def test_single_spatial_point(self) -> None:
        ds = _make_1d_pde_dataset(nx=1, nt=5)
        ir = _make_integration_result(ds)
        result = _make_experiment_result()

        fig, warnings = plot_field_comparison(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)


class TestPdeResidualEdgeCases:

    def test_nan_in_actual(self) -> None:
        result = _make_experiment_result(n_samples=50)
        result.actual[0] = float("nan")
        fig, warnings = plot_pde_residual_field(result, field_shape=(10, 5))
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_inf_in_predicted(self) -> None:
        result = _make_experiment_result(n_samples=50)
        result.predicted[0] = float("inf")
        fig, warnings = plot_pde_residual_field(result, field_shape=(10, 5))
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_empty_data(self) -> None:
        result = _make_experiment_result(n_samples=3)
        fig, warnings = plot_pde_residual_field(result, field_shape=(10, 5))
        try:
            assert len(warnings) > 0
        finally:
            plt.close(fig)


class TestDimUtilsEdgeCases:

    def test_pick_time_steps_zero_raises(self) -> None:
        with pytest.raises((ValueError, ZeroDivisionError)):
            _pick_time_steps(0, 3)

    def test_slice_nd_to_2d_same_axis_twice(self) -> None:
        arr = np.ones((5, 7))
        with pytest.raises((ValueError, IndexError)):
            _slice_nd_to_2d(arr, axes=(0, 0))

    def test_slice_nd_to_2d_axis_out_of_range(self) -> None:
        arr = np.ones((5, 7))
        with pytest.raises((ValueError, IndexError)):
            _slice_nd_to_2d(arr, axes=(0, 5))







def _figure_contains_warning_text(fig: Figure) -> bool:
    for ax in fig.get_axes():
        for text in ax.texts:
            t = text.get_text().lower()
            if any(kw in t for kw in ("fail", "warn", "n/a", "diverge", "no ")):
                return True
    return False







class TestFieldComparisonDiverged:

    def test_1d_diverged_title_contains_diverged(self) -> None:
        ds = _make_1d_pde_dataset()
        ir = _make_diverged_integration_result(ds)
        result = _make_experiment_result()

        fig, warnings = plot_field_comparison(result, ds, ir)
        try:
            all_titles = [ax.get_title().upper() for ax in fig.get_axes()]
            all_title_text = " ".join(all_titles)
            assert "DIVERGED" in all_title_text or "DIVERGE" in all_title_text, (
                f"Expected DIVERGED in titles, got: {all_titles}"
            )
        finally:
            plt.close(fig)

    def test_2d_diverged_title_contains_diverged(self) -> None:
        ds = _make_2d_pde_dataset()
        ir = _make_diverged_integration_result(ds)
        result = _make_experiment_result()

        fig, warnings = plot_field_comparison(result, ds, ir)
        try:
            all_titles = [ax.get_title().upper() for ax in fig.get_axes()]
            all_texts = []
            for ax in fig.get_axes():
                all_texts.extend(t.get_text().upper() for t in ax.texts)
            combined = " ".join(all_titles + all_texts)
            assert "DIVERGED" in combined or "DIVERGE" in combined, (
                f"Expected DIVERGED marking in 2D plot, got titles={all_titles}"
            )
        finally:
            plt.close(fig)

    def test_diverged_still_shows_prediction_data(self) -> None:
        ds = _make_1d_pde_dataset()
        ir = _make_diverged_integration_result(ds)
        result = _make_experiment_result()

        fig, _ = plot_field_comparison(result, ds, ir)
        try:


            axes = fig.get_axes()
            assert len(axes) >= 3
        finally:
            plt.close(fig)

    def test_diverged_at_t_in_warning(self) -> None:
        ds = _make_1d_pde_dataset()
        ir = _make_diverged_integration_result(ds)
        result = _make_experiment_result()

        fig, warnings = plot_field_comparison(result, ds, ir)
        try:
            combined = " ".join(warnings).lower()
            all_texts = []
            for ax in fig.get_axes():
                all_texts.extend(t.get_text().lower() for t in ax.texts)
                all_texts.append(ax.get_title().lower())
            combined += " " + " ".join(all_texts)

            has_diverge_info = (
                "diverge" in combined or "0.5" in combined
            )
            assert has_diverge_info, (
                f"Expected divergence info in warnings/text, got: {combined}"
            )
        finally:
            plt.close(fig)







class TestFieldComparison2dPredNone:

    def test_2d_no_pred_has_3_rows(self) -> None:
        ds = _make_2d_pde_dataset(nt=5)
        ir = IntegrationResult(
            success=False,
            predicted_field=None,
            warning="Integration failed completely",
        )
        result = _make_experiment_result()

        fig, warnings = plot_field_comparison(result, ds, ir)
        try:
            n_axes = len(fig.get_axes())


            assert n_axes >= 9, (
                f"Expected 3 rows (>=9 panels) for 2D pred=None, got {n_axes}"
            )
        finally:
            plt.close(fig)

    def test_2d_no_pred_warning_panels_present(self) -> None:
        ds = _make_2d_pde_dataset(nt=5)
        ir = IntegrationResult(
            success=False,
            predicted_field=None,
            warning="Total failure",
        )
        result = _make_experiment_result()

        fig, _ = plot_field_comparison(result, ds, ir)
        try:
            all_texts = []
            for ax in fig.get_axes():
                all_texts.extend(t.get_text().lower() for t in ax.texts)
            text_joined = " ".join(all_texts)
            has_warning = (
                "fail" in text_joined
                or "warning" in text_joined
                or "no prediction" in text_joined
                or "n/a" in text_joined
            )
            assert has_warning, (
                f"Expected warning text in Predicted/Residual rows: {all_texts}"
            )
        finally:
            plt.close(fig)

    def test_2d_no_pred_true_row_still_renders(self) -> None:
        ds = _make_2d_pde_dataset(nt=5)
        ir = IntegrationResult(
            success=False,
            predicted_field=None,
            warning="Integration failed",
        )
        result = _make_experiment_result()

        fig, _ = plot_field_comparison(result, ds, ir)
        try:

            first_row_axes = fig.get_axes()[:3]
            has_images = any(len(ax.images) > 0 for ax in first_row_axes)
            assert has_images, "True row should still render heatmaps"
        finally:
            plt.close(fig)
