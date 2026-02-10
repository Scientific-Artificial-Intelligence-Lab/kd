"""KD_DSCV_SPR N-D example: 2D spatial data with Sparse/PINN mode.

This script demonstrates how to use KD_DSCV_SPR with N-D (multi-spatial-dimension)
data via the PDEDataset API:

- Construct a 2D spatial dataset u(x, y, t) = sin(x)cos(y)exp(-t)
- Run PDE discovery with KD_DSCV_SPR (Sparse/PINN mode)
- Visualize with both legacy dscv_viz and unified kd.viz APIs

NOTE: Custom N-D dataset names are handled gracefully by the PINN init.
Training and viz sections are guarded with try/except for robustness.
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
from kd.model import KD_DSCV_SPR
from kd.viz.dscv_viz import (
    plot_expression_tree,
    plot_density,
    plot_evolution,
    plot_spr_residual_analysis,
    plot_spr_field_comparison,
    plot_spr_actual_vs_predicted,
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

print(f"[DSCV_SPR N-D] Generated 2D spatial data with shape: {u.shape}")
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

print(f"[DSCV_SPR N-D] Created PDEDataset:")
print(f"  equation_name: {dataset.equation_name}")
print(f"  axis_order: {dataset.axis_order}")


# 3. Create model with PINN-compatible operators and attempt training
# 使用 PINN 兼容算子创建模型并尝试训练

model = KD_DSCV_SPR(
    n_samples_per_batch=300,
    binary_operators=["add_t", "mul_t", "div_t", "diff_t", "diff2_t"],
    unary_operators=['n2_t'],
)

# NOTE: load_inner_data now gracefully handles unknown dataset names for N-D
# PDEDatasets. Training may still fail for other reasons, so we keep try/except.
step_output = None
try:
    model.import_dataset(dataset, sample_ratio=0.1, colloc_num=20000, random_state=0)
    step_output = model.train(n_epochs=50, verbose=True)
    print(f"\n[DSCV_SPR N-D] Best expression: {step_output['expression']}")
    print(f"[DSCV_SPR N-D] Best reward: {step_output['r']}")
except Exception as e:
    print(f"\n[DSCV_SPR N-D] Training failed (expected for custom N-D datasets):")
    print(f"  {type(e).__name__}: {e}")
    print("  Skipping all visualization calls that depend on training output.")


# 4. Legacy SPR visualizations (require successful training)
# 旧版 SPR 可视化（需要训练成功）

if step_output is not None:
    print("\n[DSCV_SPR N-D Viz] Legacy API calls:")

    print("  render_latex_to_image...")
    render_latex_to_image(discover_program_to_latex(step_output['program']))

    print("  plot_expression_tree...")
    plot_expression_tree(model)

    print("  plot_density...")
    plot_density(model)

    print("  plot_evolution...")
    plot_evolution(model)

    try:
        print("  plot_spr_residual_analysis...")
        plot_spr_residual_analysis(model, step_output['program'])
    except Exception as e:
        print(f"  plot_spr_residual_analysis skipped (N-D not supported): {e}")

    try:
        print("  plot_spr_field_comparison...")
        plot_spr_field_comparison(model, step_output['program'])
    except Exception as e:
        print(f"  plot_spr_field_comparison skipped (N-D not supported): {e}")

    try:
        print("  plot_spr_actual_vs_predicted...")
        plot_spr_actual_vs_predicted(model, step_output['program'])
    except Exception as e:
        print(f"  plot_spr_actual_vs_predicted skipped (N-D not supported): {e}")
else:
    print("\n[DSCV_SPR N-D Viz] Skipped legacy API calls (no training output).")


# 5. Unified kd.viz API (render_equation needs trained model)
# 统一 kd.viz API（render_equation 需要已训练的模型）

if step_output is not None:
    from kd.viz import (
        configure,
        render_equation,
        plot_parity,
        plot_field_comparison,
        plot_residuals,
    )

    SAVE_DIR = PROJECT_ROOT / "artifacts" / "dscvspr_nd_viz"
    configure(save_dir=SAVE_DIR)

    print("\n[DSCV_SPR N-D Viz] Unified API calls:")

    print("  render_equation...")
    render_equation(model)

    print("  plot_parity...")
    plot_parity(model, title="KD_DSCV_SPR N-D Parity")

    print("  plot_field_comparison...")
    plot_field_comparison(model, x_coords=None, t_coords=None, true_field=None, predicted_field=None)

    print("  plot_residuals...")
    plot_residuals(model, actual=None, predicted=None, bins=40)
else:
    print("[DSCV_SPR N-D Viz] Skipped unified API calls (no training output).")

print("\n[DSCV_SPR N-D] Done!")
