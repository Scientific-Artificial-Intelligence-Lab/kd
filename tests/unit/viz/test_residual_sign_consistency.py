
from __future__ import annotations

import math

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

from kd.core.evaluator import EvaluationResult
from kd.core.integrator import IntegrationResult
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult
from kd.viz.plots.error_heatmap import plot_error_heatmap
from kd.viz.plots.field import plot_field_comparison
from kd.viz.plots.pde_residual import plot_pde_residual_field
from kd.viz.plots.residual import plot_residual

_TWO_PI = 2.0 * math.pi



_OVERSHOOT = 0.25





_EXPECTED_SIGN = "Predicted - True"


def _dataset_1d(nx: int = 12, nt: int = 8) -> PDEDataset:
    x = torch.linspace(0, _TWO_PI, nx)
    t = torch.linspace(0, 1, nt)
    u = torch.sin(x).unsqueeze(1) * torch.exp(-t).unsqueeze(0)
    return PDEDataset(
        name="sign_1d",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=True),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def _dataset_2d(nx: int = 6, ny: int = 6, nt: int = 4) -> PDEDataset:
    x = torch.linspace(0, _TWO_PI, nx)
    y = torch.linspace(0, _TWO_PI, ny)
    t = torch.linspace(0, 1, nt)
    u = (
        torch.sin(x).reshape(nx, 1, 1)
        * torch.cos(y).reshape(1, ny, 1)
        * torch.exp(-t).reshape(1, 1, nt)
    )
    return PDEDataset(
        name="sign_2d",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=True),
            "y": AxisInfo(name="y", values=y, is_periodic=True),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "y", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def _overshooting_integration(dataset: PDEDataset) -> IntegrationResult:
    return IntegrationResult(
        success=True,
        predicted_field=dataset.get_field("u") + _OVERSHOOT,
    )


def _overshooting_result(n_samples: int = 24) -> ExperimentResult:
    actual = torch.sin(torch.linspace(0, _TWO_PI, n_samples))
    predicted = actual + _OVERSHOOT
    return ExperimentResult(
        best_expression="u_x",
        best_score=0.1,
        iterations=1,
        early_stopped=False,
        final_eval=EvaluationResult(
            score=0.1,
            complexity=1,
            r2=0.9,
            mse=0.01,
            nmse=0.01,
            coefficients=torch.tensor([1.0]),
            is_valid=True,
            residuals=predicted - actual,
            terms=["u_x"],
        ),
        actual=actual,
        predicted=predicted,
        dataset_name="sign",
        algorithm_name="Test",
        config={},
        recorder=VizRecorder(),
    )


def _panel_values(ax: matplotlib.axes.Axes) -> np.ndarray:
    for artist in (*ax.collections, *ax.images):
        array = np.asarray(artist.get_array(), dtype=np.float64)
        if array.size:
            return array[np.isfinite(array)]
    for line in ax.lines:
        array = np.asarray(line.get_ydata(), dtype=np.float64)
        if array.size:
            return array[np.isfinite(array)]
    pytest.fail("panel drew no data")


def _colorbar_labels(fig: matplotlib.figure.Figure) -> list[str]:
    return [ax.get_ylabel() + ax.get_xlabel() for ax in fig.axes]


@pytest.mark.unit
def test_field_1d_residual_panel_is_predicted_minus_true() -> None:
    dataset = _dataset_1d()
    fig, _ = plot_field_comparison(
        _overshooting_result(), dataset, _overshooting_integration(dataset)
    )
    residual_axes = [ax for ax in fig.axes if ax.get_title().startswith("Residual")]
    assert residual_axes, "1D field comparison drew no residual panel"
    for ax in residual_axes:
        assert np.all(_panel_values(ax) > 0)
        assert _EXPECTED_SIGN in ax.get_title()


@pytest.mark.unit
def test_field_2d_residual_panels_are_predicted_minus_true() -> None:
    dataset = _dataset_2d()
    fig, _ = plot_field_comparison(
        _overshooting_result(), dataset, _overshooting_integration(dataset)
    )


    residual_axes = [ax for ax in fig.axes if ax.get_title().startswith("Residual")]
    assert residual_axes, "2D field comparison drew no residual panel"
    for ax in residual_axes:
        assert np.all(_panel_values(ax) > 0)
        assert _EXPECTED_SIGN in ax.get_title()


@pytest.mark.unit
@pytest.mark.parametrize("dataset_factory", [_dataset_1d, _dataset_2d])
def test_error_heatmap_is_predicted_minus_true(dataset_factory) -> None:
    dataset = dataset_factory()
    fig, _ = plot_error_heatmap(
        _overshooting_result(), dataset, _overshooting_integration(dataset)
    )
    heatmap_axes = [ax for ax in fig.axes if ax.images]
    assert heatmap_axes, "error heatmap drew no image"
    for ax in heatmap_axes:
        assert np.all(_panel_values(ax) > 0)
    assert any(_EXPECTED_SIGN in label for label in _colorbar_labels(fig))


@pytest.mark.unit
def test_residual_figure_names_its_sign_on_both_panels() -> None:
    fig, _ = plot_residual(_overshooting_result(), field_shape=(4, 6))
    labelled = [
        ax
        for ax in fig.axes
        if _EXPECTED_SIGN in (ax.get_title() + ax.get_xlabel() + ax.get_ylabel())
    ]
    assert len(labelled) == 2, (
        "both the histogram and the spatial residual panel must name the sign; "
        f"labelled: {[ax.get_title() for ax in labelled]}"
    )


@pytest.mark.unit
def test_pde_residual_panel_names_its_sign() -> None:
    fig, _ = plot_pde_residual_field(_overshooting_result(), field_shape=(4, 6))
    residual_axes = [ax for ax in fig.axes if ax.get_title().startswith("Residual")]
    assert residual_axes, "pde residual figure drew no residual panel"
    for ax in residual_axes:
        assert np.all(_panel_values(ax) > 0)
        assert _EXPECTED_SIGN in ax.get_title()
