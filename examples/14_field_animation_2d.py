"""Example 14 - Generate a 2D Burgers field animation.

Loads the bundled EqGPT Burgers_2D benchmark, integrates the known
ground-truth RHS on a uniform downsample of the grid, then saves a
True | Predicted GIF animation.

Output goes to ``examples/out/field_animation_burgers2d.gif``.

Run: python examples/14_field_animation_2d.py
      open examples/out/field_animation_burgers2d.gif
"""

from pathlib import Path

import kd
from kd.core import integrate_pde
from kd.data import stride_subsample
from kd.viz.plots import save_field_animation

OUT_PATH = Path(__file__).parent / "out" / "field_animation_burgers2d.gif"

# 1. Load and uniformly subsample every axis to keep the animation small.
dataset = stride_subsample(kd.load_burgers_2d(), stride=4)

# 2. Integrate the benchmark's known RHS on that same grid.
rhs = "-u*u_x - u*u_y + 0.01*u_xx + 0.01*u_yy"
prediction = integrate_pde(rhs, dataset, method="RK45")

# 3. Save the True | Predicted movie. Rendering and file errors propagate.
notes = save_field_animation(dataset, prediction, OUT_PATH)
print(f"Saved GIF: {OUT_PATH}")
for note in notes:
    print(f"[viz warning] {note}")
