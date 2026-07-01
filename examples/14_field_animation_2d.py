
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sympy as sp
import torch
from matplotlib.animation import PillowWriter

import kd
from kd.core.evaluator import EvaluationResult
from kd.core.integrator import integrate_pde
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult
from kd.viz.plots.animation import plot_field_animation

OUT_DIR = Path(__file__).parent / "out"
OUT_PATH = OUT_DIR / "field_animation_burgers2d.gif"
_DOWNSAMPLE_STEP = 4


def _load_downsampled_burgers_2d() -> PDEDataset:
    full = kd.load_burgers_2d()
    x = full.get_coords("x")[::_DOWNSAMPLE_STEP].clone()
    y = full.get_coords("y")[::_DOWNSAMPLE_STEP].clone()
    t = full.get_coords("t")[::_DOWNSAMPLE_STEP].clone()
    u = full.get_field("u")[
        ::_DOWNSAMPLE_STEP,
        ::_DOWNSAMPLE_STEP,
        ::_DOWNSAMPLE_STEP,
    ].clone()

    return PDEDataset(
        name="burgers-2d-demo-downsampled",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=False),
            "y": AxisInfo(name="y", values=y, is_periodic=False),
            "t": AxisInfo(name="t", values=t, is_periodic=False),
        },
        axis_order=["x", "y", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
        lhs_order=1,
        ground_truth=full.ground_truth,
    )


def _make_demo_result(dataset: PDEDataset) -> ExperimentResult:
    actual = torch.linspace(0.0, 1.0, 8)
    recorder = VizRecorder()
    recorder.log("_best_score", 0.0)
    recorder.log("_best_expr", dataset.ground_truth or "u")
    recorder.log("_n_candidates", 1)

    return ExperimentResult(
        best_expression=dataset.ground_truth or "u",
        best_score=0.0,
        iterations=1,
        early_stopped=False,
        final_eval=EvaluationResult(
            mse=0.0,
            nmse=0.0,
            r2=1.0,
            aic=0.0,
            complexity=1,
            coefficients=torch.tensor([1.0]),
            is_valid=True,
            error_message="",
            selected_indices=[0],
            residuals=torch.zeros_like(actual),
            terms=["u"],
            expression="u",
        ),
        actual=actual,
        predicted=actual,
        dataset_name=dataset.name,
        algorithm_name="demo",
        config={},
        recorder=recorder,
    )


def main() -> int:
    if not PillowWriter.isAvailable():
        print("PillowWriter is unavailable; skipping GIF generation.")
        return 0

    dataset = _load_downsampled_burgers_2d()
    rhs = sp.sympify("-u*u_x - u*u_y + 0.01*u_xx + 0.01*u_yy")
    integration_result = integrate_pde(rhs, dataset, method="RK45")
    if not integration_result.success:
        print(f"Integration failed: {integration_result.warning}")
        return 1

    result = _make_demo_result(dataset)
    animation, warnings = plot_field_animation(result, dataset, integration_result)
    if animation is None:
        print(f"Animation skipped: {'; '.join(warnings)}")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    animation.save(OUT_PATH, writer=PillowWriter(fps=8))
    plt.close("all")

    print(f"Saved GIF: {OUT_PATH}")
    print(f"Open this GIF: {OUT_PATH}")
    for warning in warnings:
        print(f"[viz warning] {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
