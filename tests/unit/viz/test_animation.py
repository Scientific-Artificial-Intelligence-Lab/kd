
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

matplotlib.use("Agg")

from matplotlib.animation import FuncAnimation

from kd.core.evaluator import EvaluationResult
from kd.core.integrator import IntegrationResult
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult
from kd.viz.plots._dim_utils import _pick_animation_frames
from kd.viz.plots.animation import plot_field_animation


def _make_1d_dataset(nx: int = 8, nt: int = 5) -> PDEDataset:
    x = torch.linspace(0.0, 1.0, nx)
    t = torch.linspace(0.0, 1.0, nt)
    u = torch.sin(x).reshape(nx, 1) * torch.exp(-t).reshape(1, nt)
    return PDEDataset(
        name="one_dimensional",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def _make_experiment_result() -> ExperimentResult:
    actual = torch.linspace(0.0, 1.0, 8)
    predicted = actual.clone()
    recorder = VizRecorder()
    recorder.log("_best_score", 1.0)
    recorder.log("_best_expr", "u")
    recorder.log("_n_candidates", 1)
    return ExperimentResult(
        best_expression="u",
        best_score=1.0,
        iterations=1,
        early_stopped=False,
        final_eval=EvaluationResult(
            mse=0.0,
            nmse=0.0,
            r2=1.0,
            score=0.0,
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


def _close_animation(anim: FuncAnimation | None) -> None:
    if anim is not None:
        anim._draw_was_started = True
        plt.close(anim._fig)


def test_pick_animation_frames_evenly_spaced_with_endpoints() -> None:
    assert _pick_animation_frames(10, 4) == [0, 3, 6, 9]
    assert _pick_animation_frames(3, 24) == [0, 1, 2]
    with pytest.raises(ValueError, match="n_t"):
        _pick_animation_frames(0, 4)
    with pytest.raises(ValueError, match="max_frames"):
        _pick_animation_frames(4, 0)


def test_plot_field_animation_frame_count_extent_and_fixed_norm(
    rectangular_2d_dataset: PDEDataset,
) -> None:
    true_field = rectangular_2d_dataset.get_field("u")
    ir = IntegrationResult(success=True, predicted_field=true_field * 2.0)
    result = _make_experiment_result()

    anim, warnings = plot_field_animation(
        result,
        rectangular_2d_dataset,
        ir,
        max_frames=4,
        fps=5,
    )
    try:
        assert warnings == []
        assert isinstance(anim, FuncAnimation)
        assert list(anim.new_frame_seq()) == _pick_animation_frames(6, 4)
        data_axes = [ax for ax in anim._fig.axes if ax.images]
        assert len(data_axes) == 2
        clims = {ax.images[0].get_clim() for ax in data_axes}
        assert len(clims) == 1
        for ax in data_axes:
            assert tuple(ax.images[0].get_extent()) == pytest.approx(
                (10.0, 14.0, -2.0, 3.0)
            )
            assert ax.get_xlabel() == "eta"
            assert ax.get_ylabel() == "xi"
    finally:
        _close_animation(anim)


def test_plot_field_animation_skips_non_2d_dataset() -> None:
    ds = _make_1d_dataset()
    ir = IntegrationResult(success=True, predicted_field=ds.get_field("u"))
    result = _make_experiment_result()

    anim, warnings = plot_field_animation(result, ds, ir)

    assert anim is None
    assert any("2D" in warning or "2 spatial" in warning for warning in warnings)


def test_plot_field_animation_success_false_renders_true_only(
    rectangular_2d_dataset: PDEDataset,
) -> None:
    ir = IntegrationResult(
        success=False,
        predicted_field=rectangular_2d_dataset.get_field("u"),
        warning="Integration failed",
    )
    result = _make_experiment_result()

    anim, warnings = plot_field_animation(result, rectangular_2d_dataset, ir)
    try:
        assert isinstance(anim, FuncAnimation)
        assert any("Integration failed" in warning for warning in warnings)
        data_axes = [ax for ax in anim._fig.axes if ax.images]
        assert len(data_axes) == 1
        assert "Predicted" not in " ".join(ax.get_title() for ax in data_axes)
    finally:
        _close_animation(anim)
