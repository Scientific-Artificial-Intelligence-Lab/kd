"""KD_SGA N-D example: 3D structured grid data.

This script demonstrates how to use the N-D PDEDataset API with KD_SGA:

- Construct a 3D dataset using fields_data/coords_1d/axis_order
- Run PDE discovery on higher-dimensional data
- The N-D API supports arbitrary spatial dimensions (x, y, z, ...) plus time
"""

from _bootstrap import ensure_project_root_on_syspath

# 1. Project root & imports
PROJECT_ROOT = ensure_project_root_on_syspath()

import numpy as np
from kd.dataset import PDEDataset
from kd.model.kd_sga import KD_SGA


# 2. Generate synthetic 3D data (x, y, t)
# 生成合成 3D 数据

# Define 1D coordinates for each axis
# 定义每个轴的一维坐标
nx, ny, nt = 20, 20, 30
x = np.linspace(0, 2 * np.pi, nx)
y = np.linspace(0, 2 * np.pi, ny)
t = np.linspace(0, 1, nt)

# Create 3D meshgrid
# 创建 3D 网格
X, Y, T = np.meshgrid(x, y, t, indexing="ij")

# Synthetic solution: u(x, y, t) = sin(x) * cos(y) * exp(-t)
# This satisfies: u_t = -u (simple decay equation)
# 合成解：满足 u_t = -u 的简单衰减方程
u = np.sin(X) * np.cos(Y) * np.exp(-T)

print(f"[N-D Example] Generated 3D data with shape: {u.shape}")
print(f"  x: {nx} points, y: {ny} points, t: {nt} points")


# 3. Construct PDEDataset using N-D API
# 使用 N-D API 构造 PDEDataset

dataset = PDEDataset(
    equation_name="3d_decay",
    # N-D mode parameters
    fields_data={"u": u},           # Dict of field arrays
    coords_1d={"x": x, "y": y, "t": t},  # Dict of 1D coordinate arrays
    axis_order=["x", "y", "t"],     # Defines array dimension order
    target_field="u",               # Which field to discover equation for
    lhs_axis="t",                   # Time axis for u_t = ...
)

print(f"[N-D Example] Created PDEDataset:")
print(f"  equation_name: {dataset.equation_name}")
print(f"  axis_order: {dataset.axis_order}")
print(f"  target_field: {dataset.target_field}")
print(f"  lhs_axis: {dataset.lhs_axis}")


# 4. Verify compatibility layer still works
# 验证兼容层仍然可用

print(f"\n[N-D Example] Compatibility layer access:")
print(f"  dataset.x shape: {dataset.x.shape if dataset.x is not None else 'None'}")
print(f"  dataset.t shape: {dataset.t.shape if dataset.t is not None else 'None'}")
print(f"  dataset.usol shape: {dataset.usol.shape}")


# 5. Run KD_SGA discovery
# 运行 KD_SGA 发现

print("\n[N-D Example] Starting PDE discovery...")

model = KD_SGA(
    sga_run=50,      # Reduced for demo; increase for better results
    num=15,          # Population size
    depth=3,         # Expression tree depth
    width=4,         # Expression tree width
    p_var=0.5,
    p_mute=0.3,
    p_cro=0.5,
    seed=42,
)

model.fit_dataset(dataset)


# 6. Print results
# 打印结果

print("\n" + "=" * 60)
print("[Result] Discovered equation:")
print(f"  {model.best_pde_}")
print(f"  Score: {model.best_score_:.6f}")

latex = model.equation_latex()
print(f"\n[Result] LaTeX form:")
print(f"  {latex}")

print("\n[N-D Example] Expected: u_t = -u (decay equation)")
print("=" * 60)


# 7. Visualization via kd.viz API / 使用 kd.viz 统一 API 可视化
# Test all unified viz calls with N-D data to verify compatibility.

from kd.viz import (
    configure,
    render_equation,
    plot_field_comparison,
    plot_time_slices,
    plot_parity,
    plot_residuals,
)

SAVE_DIR = PROJECT_ROOT / "artifacts" / "sga_nd_viz"
configure(save_dir=SAVE_DIR)

print("\n[N-D Viz] Rendering equation...")
render_equation(model)

print("[N-D Viz] Plotting field comparison...")
plot_field_comparison(model, x_coords=None, t_coords=None, true_field=None, predicted_field=None)

# TODO: [viz-nd] plot_time_slices fails with N-D data:
#   IndexError: index 29 is out of bounds for axis 1 with size 20
#   SGA adapter _time_slices assumes usol shape (nx, nt) but N-D gives (nx, ny, nt)
# print("[N-D Viz] Plotting time slices...")
# plot_time_slices(
#     model,
#     x_coords=None,
#     t_coords=None,
#     true_field=None,
#     predicted_field=None,
#     slice_times=[0.0, 0.5, 1.0],
# )

print("[N-D Viz] Plotting parity...")
plot_parity(model, title="KD_SGA N-D Parity")

# TODO: [viz-nd] plot_residuals produces empty output with N-D data:
#   warning: Failed to access residual fields: 'ProblemContext' object has no
#   attribute 'right_side_full_origin'
#   SGA adapter residual extraction assumes 1D context attributes
print("[N-D Viz] Plotting residuals...")
plot_residuals(model, actual=None, predicted=None, bins=40)

print("\n[N-D Viz] All visualizations completed!")
print(f"  Output directory: {SAVE_DIR}")
