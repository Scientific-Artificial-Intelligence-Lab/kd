"""KD_DSCV complete example: Regular/FD mode with 1D + N-D data.

Demonstrates the DISCOVER algorithm for PDE discovery using finite-difference
mode with the unified kd.viz visualization API.

Usage:
    python examples/dscv_example.py
"""

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy.*")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow.*")

import numpy as np

from kd.dataset import PDEDataset, load_pde
from kd.model import KD_DSCV
from kd.viz import VizRequest, configure, render


# ============================================================
# 1. Data Loading & Model / 数据加载与模型
# ============================================================

dataset = load_pde("burgers")

# Supported binary operators / 支持的二元算子:
#   'add', 'sub', 'mul', 'div', 'diff', 'diff2', 'diff3', 'diff4'
# Supported unary operators / 支持的一元算子:
#   'n2', 'n3', 'n4', 'n5', 'sin', 'cos', 'tan', 'sigmoid', 'logabs', 'expneg'
model = KD_DSCV(
    binary_operators=["add", "mul", "diff"],  # Binary ops (add required) / 二元算子
    unary_operators=["n2"],                   # Unary ops / 一元算子
    n_samples_per_batch=500,                  # Traversals per batch / 每批采样数
    # n_iterations=100,    # [default] Max training iterations / 最大训练迭代数
    # out_path='./log/',   # [default] Log output path / 日志输出路径
    # core_num=1,          # [default] CPU cores for parallel eval / CPU 核数
    # seed=0,              # [default] Random seed / 随机种子
)

np.random.seed(42)

# import_dataset + train is the two-step API / 两步式 API
# Alternative: model.fit_dataset(dataset, n_epochs=11)
model.import_dataset(dataset)
step_output = model.train(n_epochs=11)
print(f"Best: {step_output['expression']}  (reward: {step_output['r']})")


# ============================================================
# 2. Visualization / 可视化
# ============================================================

configure(save_dir="artifacts/dscv_viz")

render(VizRequest("equation", model))
# Legacy: render_latex_to_image(discover_program_to_latex(step_output["program"]))
render(VizRequest("search_evolution", model))
# Legacy: plot_evolution(model, output_dir=...)
render(VizRequest("density", model, options={"epoches": [2, 5, 10]}))
# Legacy: plot_density(model, epoches=[2, 5, 10], output_dir=...)
render(VizRequest("tree", model))
# Legacy: plot_expression_tree(model, output_dir=...)
render(VizRequest("residual", model))
# Legacy: plot_pde_residual_analysis(model, step_output["program"], output_dir=...)
render(VizRequest("field_comparison", model))
# Legacy: plot_field_comparison(model, step_output["program"], output_dir=...)
render(VizRequest("parity", model))
# Legacy: plot_actual_vs_predicted(model, step_output["program"], output_dir=...)


# ============================================================
# 3. N-D Data (3D spatial grid) / N-D 数据
# ============================================================
# Note: N-D mode uses uppercase "Diff" operator / 注意：N-D 模式使用大写 "Diff"

from _nd_data import make_diffusion_2d

x, y, t, u = make_diffusion_2d()

nd_dataset = PDEDataset(
    equation_name="2d_diffusion",
    fields_data={"u": u},
    coords_1d={"x": x, "y": y, "t": t},
    axis_order=["x", "y", "t"],
    target_field="u",
    lhs_axis="t",
)

nd_model = KD_DSCV(
    binary_operators=["add", "mul", "Diff", "Diff2"],  # Uppercase for N-D / N-D 用大写
    unary_operators=["n2"],
    n_samples_per_batch=500,
)

np.random.seed(42)
nd_model.import_dataset(nd_dataset)
nd_result = nd_model.train(n_epochs=11)
print(f"N-D best: {nd_result['expression']}  (reward: {nd_result['r']})")

configure(save_dir="artifacts/dscv_nd_viz")
render(VizRequest("equation", nd_model))
render(VizRequest("search_evolution", nd_model))
render(VizRequest("density", nd_model))
render(VizRequest("tree", nd_model))
render(VizRequest("residual", nd_model))
render(VizRequest("field_comparison", nd_model))
render(VizRequest("parity", nd_model))
