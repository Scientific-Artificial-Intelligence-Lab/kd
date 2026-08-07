"""Example 14 - Generate a 2D Burgers field animation.

Loads the bundled EqGPT Burgers_2D benchmark, integrates the known
ground-truth RHS on a uniform downsample of the grid, then saves a
True | Predicted GIF animation.

Output goes to ``examples/out/field_animation_burgers2d.gif``.

Run: python examples/14_field_animation_2d.py
      open examples/out/field_animation_burgers2d.gif
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

import kd
from kd import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd.core import integrate_pde
from kd.viz.plots import plot_field_animation

OUT_DIR = Path(__file__).parent / "out"
OUT_PATH = OUT_DIR / "field_animation_burgers2d.gif"
_DOWNSAMPLE_STEP = 4


def _load_downsampled_burgers_2d() -> PDEDataset:
    """Load bundled Burgers_2D and keep a smaller uniform grid for the demo."""
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


def main() -> int:
    """Generate the Burgers_2D field animation GIF."""
    if not PillowWriter.isAvailable():
        print("PillowWriter is unavailable; skipping GIF generation.")
        return 0

    dataset = _load_downsampled_burgers_2d()
    rhs = "-u*u_x - u*u_y + 0.01*u_xx + 0.01*u_yy"
    integration_result = integrate_pde(rhs, dataset, method="RK45")
    if not integration_result.success:
        print(f"Integration failed: {integration_result.warning}")
        return 1

    animation, warnings = plot_field_animation(dataset, integration_result)
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
