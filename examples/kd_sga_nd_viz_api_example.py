"""KD_SGA N-D example with unified viz API.

This script demonstrates how to:

- Construct a 3D dataset using the N-D PDEDataset API
- Run PDE discovery with KD_SGA on higher-dimensional data
- Visualize results exclusively through the unified ``kd.viz`` facade
  (render_equation, plot_field_comparison, plot_time_slices, plot_parity,
  plot_residuals)
"""

from _bootstrap import ensure_project_root_on_syspath

# 1. Project root & imports / 工程根目录与依赖导入
PROJECT_ROOT = ensure_project_root_on_syspath()

import numpy as np
from kd.dataset import PDEDataset
from kd.model.kd_sga import KD_SGA
from kd.viz import (
    configure,
    render_equation,
    plot_field_comparison,
    plot_time_slices,
    plot_parity,
    plot_residuals,
)


# 2. Generate synthetic 3D data (x, y, t)
# 生成合成 3D 数据

nx, ny, nt = 20, 20, 30
x = np.linspace(0, 2 * np.pi, nx)
y = np.linspace(0, 2 * np.pi, ny)
t = np.linspace(0, 1, nt)

X, Y, T = np.meshgrid(x, y, t, indexing="ij")

# Synthetic solution: u(x, y, t) = sin(x) * cos(y) * exp(-t)
# This satisfies: u_t = -u (simple decay equation)
# 合成解：满足 u_t = -u 的简单衰减方程
u = np.sin(X) * np.cos(Y) * np.exp(-T)

print(f"[SGA N-D Viz API] Generated 3D data with shape: {u.shape}")
print(f"  x: {nx} points, y: {ny} points, t: {nt} points")


# 3. Construct PDEDataset using N-D API
# 使用 N-D API 构造 PDEDataset

dataset = PDEDataset(
    equation_name="3d_decay",
    fields_data={"u": u},
    coords_1d={"x": x, "y": y, "t": t},
    axis_order=["x", "y", "t"],
    target_field="u",
    lhs_axis="t",
)

print(f"[SGA N-D Viz API] Created PDEDataset:")
print(f"  equation_name: {dataset.equation_name}")
print(f"  axis_order: {dataset.axis_order}")
print(f"  target_field: {dataset.target_field}")
print(f"  lhs_axis: {dataset.lhs_axis}")


# 4. Create and train KD_SGA / 创建并训练 KD_SGA 模型

model = KD_SGA(
    sga_run=50,
    num=15,
    depth=3,
    width=4,
    p_var=0.5,
    p_mute=0.3,
    p_cro=0.5,
    seed=42,
)

print("\n[SGA N-D Viz API] Starting PDE discovery with fit_dataset(PDEDataset)...")
model.fit_dataset(dataset)


# 5. Print results / 打印结果

print("\n" + "=" * 60)
print("[Result] Discovered equation:")
print(f"  {model.best_pde_}")
print(f"  Score: {model.best_score_:.6f}")

latex_full = model.equation_latex()
latex_structure = model.equation_latex(include_coefficients=False)

print(f"\n[Result] LaTeX (with coefficients):")
print(f"  {latex_full}")
print(f"[Result] LaTeX (structure only):")
print(f"  {latex_structure}")
print("=" * 60)


# 6. Unified viz API / 使用 kd.viz 统一 API 可视化

SAVE_DIR = PROJECT_ROOT / "artifacts" / "sga_nd_viz_api"
configure(save_dir=SAVE_DIR)

print("\n[N-D Viz API] Rendering equation...")
render_equation(model)

print("[N-D Viz API] Plotting field comparison...")
plot_field_comparison(model, x_coords=None, t_coords=None, true_field=None, predicted_field=None)

print("[N-D Viz API] Plotting time slices...")
plot_time_slices(
    model,
    x_coords=None,
    t_coords=None,
    true_field=None,
    predicted_field=None,
    slice_times=[0.0, 0.5, 1.0],
)

print("[N-D Viz API] Plotting parity...")
plot_parity(model, title="KD_SGA N-D Viz API Parity")

print("[N-D Viz API] Plotting residuals...")
plot_residuals(model, actual=None, predicted=None, bins=40)

print("\n[N-D Viz API] All visualizations completed!")
print(f"  Output directory: {SAVE_DIR}")
