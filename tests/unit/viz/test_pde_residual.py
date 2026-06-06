
from __future__ import annotations

import math

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

matplotlib.use("Agg")

from matplotlib.figure import Figure

from kd.core.evaluator import EvaluationResult
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult
from kd.viz.plots.pde_residual import plot_pde_residual_field





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


def _make_result(n_samples: int = 50) -> ExperimentResult:
    actual = torch.randn(n_samples)
    predicted = actual + torch.randn(n_samples) * 0.1
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







class TestPdeResidualDatasetSmoke:

    def test_accepts_dataset_kwarg(self) -> None:
        ds = _make_1d_dataset(nx=10, nt=5)

        result = _make_result(n_samples=10 * 5)
        fig, warnings = plot_pde_residual_field(result, field_shape=(10, 5), dataset=ds)
        try:
            assert isinstance(fig, Figure)
            assert isinstance(warnings, list)
        finally:
            plt.close(fig)

    def test_dataset_none_backward_compat(self) -> None:
        result = _make_result(n_samples=100)
        fig, warnings = plot_pde_residual_field(result, field_shape=(10, 10))
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)







class TestPdeResidualAxisAware1D:

    def test_1d_with_dataset_uses_pcolormesh(self) -> None:
        ds = _make_1d_dataset(nx=10, nt=5)
        result = _make_result(n_samples=10 * 5)
        fig, _ = plot_pde_residual_field(result, field_shape=(10, 5), dataset=ds)
        try:

            has_collections = any(len(ax.collections) > 0 for ax in fig.get_axes())
            assert has_collections, "1D dataset should trigger pcolormesh (collections)"
        finally:
            plt.close(fig)

    def test_1d_without_dataset_uses_generic(self) -> None:
        result = _make_result(n_samples=100)
        fig, _ = plot_pde_residual_field(result, field_shape=(10, 10))
        try:

            has_images_or_lines = any(
                len(ax.images) > 0 or len(ax.get_lines()) > 0 for ax in fig.get_axes()
            )
            assert has_images_or_lines
        finally:
            plt.close(fig)

    def test_1d_rendering_differs_from_no_dataset(self) -> None:
        ds = _make_1d_dataset(nx=10, nt=5)
        result = _make_result(n_samples=10 * 5)

        fig_with, _ = plot_pde_residual_field(result, field_shape=(10, 5), dataset=ds)
        fig_without, _ = plot_pde_residual_field(result, field_shape=(10, 5))
        try:

            cols_with = sum(len(ax.collections) for ax in fig_with.get_axes())
            cols_without = sum(len(ax.collections) for ax in fig_without.get_axes())

            assert cols_with > cols_without, (
                f"Axis-aware should use pcolormesh: "
                f"with={cols_with}, without={cols_without}"
            )
        finally:
            plt.close(fig_with)
            plt.close(fig_without)







class TestPdeResidualAxisAware2D:

    def test_2d_with_dataset_produces_heatmaps(self) -> None:
        ds = _make_2d_dataset(nx=5, ny=5, nt=4)
        result = _make_result(n_samples=5 * 5 * 4)
        fig, _ = plot_pde_residual_field(result, field_shape=(5, 5, 4), dataset=ds)
        try:
            has_images = any(len(ax.images) > 0 for ax in fig.get_axes())
            assert has_images, "2D dataset should trigger heatmap (imshow)"
        finally:
            plt.close(fig)

    def test_2d_axes_have_titles(self) -> None:
        ds = _make_2d_dataset(nx=5, ny=5, nt=4)
        result = _make_result(n_samples=5 * 5 * 4)
        fig, _ = plot_pde_residual_field(result, field_shape=(5, 5, 4), dataset=ds)
        try:
            titles = [ax.get_title() for ax in fig.get_axes() if ax.get_title()]
            assert len(titles) >= 1, "Expected at least one panel title"
        finally:
            plt.close(fig)







class TestPdeResidualDatasetEdgeCases:

    def test_dataset_shape_mismatch_with_field_shape(self) -> None:
        ds = _make_1d_dataset(nx=20, nt=10)
        result = _make_result(n_samples=50)
        fig, warnings = plot_pde_residual_field(result, field_shape=(10, 5), dataset=ds)
        try:
            assert isinstance(fig, Figure)

            assert isinstance(warnings, list)
        finally:
            plt.close(fig)

    def test_nan_data_with_dataset(self) -> None:
        ds = _make_1d_dataset(nx=10, nt=5)
        result = _make_result(n_samples=10 * 5)
        result.actual[0] = float("nan")
        fig, warnings = plot_pde_residual_field(result, field_shape=(10, 5), dataset=ds)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_inf_data_with_dataset(self) -> None:
        ds = _make_1d_dataset(nx=10, nt=5)
        result = _make_result(n_samples=10 * 5)
        result.predicted[0] = float("inf")
        fig, warnings = plot_pde_residual_field(result, field_shape=(10, 5), dataset=ds)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_no_figure_leak(self) -> None:
        ds = _make_1d_dataset(nx=10, nt=5)
        result = _make_result(n_samples=10 * 5)
        figs_before = len(plt.get_fignums())
        fig, _ = plot_pde_residual_field(result, field_shape=(10, 5), dataset=ds)
        plt.close(fig)
        figs_after = len(plt.get_fignums())
        assert figs_after <= figs_before







class TestPdeResidualSilentFallbackWarning:

    def test_no_dataset_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        result = _make_result(n_samples=50)
        with caplog.at_level(logging.WARNING, logger="kd.viz.plots.pde_residual"):
            fig, _ = plot_pde_residual_field(result, field_shape=None)
            plt.close(fig)


        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert len(warning_messages) > 0, (
            "Expected a warning when dataset is None and falling back to 1D plot"
        )

    def test_no_dataset_square_shape_emits_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        result = _make_result(n_samples=100)
        with caplog.at_level(logging.WARNING, logger="kd.viz.plots.pde_residual"):
            fig, _ = plot_pde_residual_field(result, field_shape=(10, 10))
            plt.close(fig)



        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert len(warning_messages) > 0, (
            "Expected a warning about fallback even when field_shape is valid"
        )

    def test_with_dataset_no_fallback_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        ds = _make_1d_dataset(nx=10, nt=5)
        result = _make_result(n_samples=10 * 5)
        with caplog.at_level(logging.WARNING, logger="kd.viz.plots.pde_residual"):
            fig, _ = plot_pde_residual_field(result, field_shape=(10, 5), dataset=ds)
            plt.close(fig)


        fallback_warnings = [
            r.message
            for r in caplog.records
            if r.levelno >= logging.WARNING and "fallback" in r.message.lower()
        ]
        assert len(fallback_warnings) == 0, (
            f"Should not warn about fallback with dataset: {fallback_warnings}"
        )
