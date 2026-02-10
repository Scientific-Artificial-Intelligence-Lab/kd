"""KD_DSCV N-D example: 2D spatial data with Regular/FD mode.

This script demonstrates how to use KD_DSCV with N-D (multi-spatial-dimension)
data via the PDEDataset API:

- Construct a 2D spatial dataset u(x, y, t) = sin(x)cos(y)exp(-t)
- Run PDE discovery with KD_DSCV (Regular/FD mode)
- Visualize with both legacy dscv_viz and unified kd.viz APIs
"""

import numpy as np

from _bootstrap import ensure_project_root_on_syspath

PROJECT_ROOT = ensure_project_root_on_syspath()

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module='numpy.*')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow.*')

from kd.dataset import PDEDataset
from kd.model import KD_DSCV
from kd.viz.dscv_viz import (
    plot_expression_tree,
    plot_density,
    plot_evolution,
    plot_pde_residual_analysis,
    plot_field_comparison as dscv_plot_field_comparison,
    plot_actual_vs_predicted,
)
from kd.viz.discover_eq2latex import discover_program_to_latex
from kd.viz.equation_renderer import render_latex_to_image

np.random.seed(42)


# 1. Generate synthetic 2D spatial data (x, y, t)
# 生成合成 2D 空间数据

nx, ny, nt = 20, 20, 30
x = np.linspace(0, 2 * np.pi, nx)
y = np.linspace(0, 2 * np.pi, ny)
t = np.linspace(0, 1, nt)

X, Y, T = np.meshgrid(x, y, t, indexing="ij")

# u(x, y, t) = sin(x) * cos(y) * exp(-t)  →  u_t = -u
u = np.sin(X) * np.cos(Y) * np.exp(-T)

print(f"[DSCV N-D] Generated 2D spatial data with shape: {u.shape}")
print(f"  x: {nx} points, y: {ny} points, t: {nt} points")


# 2. Construct PDEDataset using N-D API
# 使用 N-D API 构建 PDEDataset

dataset = PDEDataset(
    equation_name="2d_decay",
    fields_data={"u": u},
    coords_1d={"x": x, "y": y, "t": t},
    axis_order=["x", "y", "t"],
    target_field="u",
    lhs_axis="t",
)

print(f"[DSCV N-D] Created PDEDataset:")
print(f"  equation_name: {dataset.equation_name}")
print(f"  axis_order: {dataset.axis_order}")


# 3. Create model with 2D N-D tokens and train
# 使用 2D N-D 算子创建模型并训练

model = KD_DSCV(
    binary_operators=["add", "mul", "Diff"],
    unary_operators=['n2'],
    n_samples_per_batch=500,
)

model.import_dataset(dataset)
step_output = model.train(n_epochs=11)

print(f"\n[DSCV N-D] Best expression: {step_output['expression']}")
print(f"[DSCV N-D] Best reward: {step_output['r']}")


# 4. Legacy DSCV visualizations
# 旧版 DSCV 可视化

print("\n[DSCV N-D Viz] Legacy API calls:")

print("  render_latex_to_image...")
render_latex_to_image(discover_program_to_latex(step_output['program']))

print("  plot_expression_tree...")
plot_expression_tree(model)

print("  plot_density...")
plot_density(model, epoches=[2, 5, 10])

print("  plot_evolution...")
plot_evolution(model)

# NOTE: Legacy plot_pde_residual_analysis / dscv_plot_field_comparison do not
# support N-D data (they assume 2D (x, t) layout). Use the unified API
# (plot_field_comparison, plot_residuals from kd.viz) as N-D alternatives.
# print("  plot_pde_residual_analysis...")
# plot_pde_residual_analysis(model, step_output['program'])
# print("  plot_field_comparison (legacy DSCV)...")
# dscv_plot_field_comparison(model, step_output['program'])

print("  plot_actual_vs_predicted...")
plot_actual_vs_predicted(model, step_output['program'])


# 5. Unified kd.viz API
# 统一 kd.viz API

from kd.viz import (
    configure,
    render_equation,
    plot_parity,
    plot_field_comparison,
    plot_residuals,
)

SAVE_DIR = PROJECT_ROOT / "artifacts" / "dscv_nd_viz"
configure(save_dir=SAVE_DIR)

print("\n[DSCV N-D Viz] Unified API calls:")

print("  render_equation...")
render_equation(model)

print("  plot_parity...")
plot_parity(model, title="KD_DSCV N-D Parity")

print("  plot_field_comparison...")
plot_field_comparison(model, x_coords=None, t_coords=None, true_field=None, predicted_field=None)

print("  plot_residuals...")
plot_residuals(model, actual=None, predicted=None, bins=40)

print("\n[DSCV N-D] Done!")
