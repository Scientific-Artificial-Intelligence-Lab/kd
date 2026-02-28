"""KD_DSCV_SPR complete example: Sparse/PINN mode with 1D + N-D data.

Demonstrates the DISCOVER + PINN algorithm for PDE discovery from sparse
observations with the unified kd.viz visualization API.

Usage:
    python examples/dscv_spr_example.py
"""

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy.*")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow.*")

import numpy as np

from kd.dataset import PDEDataset, load_pde
from kd.model import KD_DSCV_SPR
from kd.viz import VizRequest, configure, render


# ============================================================
# 1. Data Loading & Model / 数据加载与模型
# ============================================================

dataset = load_pde("burgers")

# SPR operators must be PyTorch-compatible (ending with '_t')
# SPR 算子必须兼容 PyTorch（以 '_t' 结尾）
model = KD_DSCV_SPR(
    n_samples_per_batch=300,                                    # Traversals per batch / 每批采样数
    binary_operators=["add_t", "mul_t", "div_t", "diff_t", "diff2_t"],  # Binary ops / 二元算子
    unary_operators=["n2_t"],                                   # Unary ops / 一元算子
    # n_iterations=100,    # [default] Max training iterations / 最大训练迭代数
    # out_path='./log/',   # [default] Log output path / 日志输出路径
    # seed=0,              # [default] Random seed / 随机种子
)

np.random.seed(42)

# import_dataset with PINN-specific parameters / 含 PINN 特有参数
model.import_dataset(
    dataset,
    sample_ratio=0.1,      # Fraction of grid points as observations / 观测点比例
    colloc_num=20000,       # Number of collocation points for PINN / PINN 配置点数量
    random_state=0,         # Random seed for sampling / 采样随机种子
    # noise_level=None,    # [default] Noise to add to observations / 观测噪声水平
    # spline_sample=False, # [default] Use spline-based sampling / 样条采样
)

step_output = model.train(n_epochs=200, verbose=True)
print(f"Best: {step_output['expression']}  (reward: {step_output['r']})")


# ============================================================
# 2. Visualization / 可视化
# ============================================================

configure(save_dir="artifacts/dscvspr_viz")

render(VizRequest("equation", model))
# Legacy: render_latex_to_image(discover_program_to_latex(step_output["program"]))
render(VizRequest("search_evolution", model))
# Legacy: plot_evolution(model, output_dir=...)
render(VizRequest("density", model))
# Legacy: plot_density(model, output_dir=...)
render(VizRequest("tree", model))
# Legacy: plot_expression_tree(model, output_dir=...)
# Note: Generic residual/field_comparison/parity use FD data path,
# which is incompatible with SPR models. Use spr_* variants instead.
render(VizRequest("spr_residual", model))
# Legacy: plot_spr_residual_analysis(model, step_output["program"], output_dir=...)
render(VizRequest("spr_field_comparison", model))
# Legacy: plot_spr_field_comparison(model, step_output["program"], output_dir=...)


# ============================================================
# 3. N-D Data (3D spatial grid) / N-D 数据
# ============================================================
# Note: N-D SPR may fail for custom datasets — guarded with try/except
# 注意：N-D SPR 对自定义数据集可能失败，使用 try/except 保护

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

nd_model = KD_DSCV_SPR(
    n_samples_per_batch=300,
    binary_operators=["add_t", "mul_t", "div_t", "diff_t", "diff2_t"],
    unary_operators=["n2_t"],
)

np.random.seed(42)
nd_output = None
try:
    nd_model.import_dataset(
        nd_dataset, sample_ratio=0.1, colloc_num=20000, random_state=0,
    )
    nd_output = nd_model.train(n_epochs=50, verbose=True)
    print(f"N-D best: {nd_output['expression']}  (reward: {nd_output['r']})")
except Exception as e:
    print(f"N-D SPR training failed (expected for custom datasets): {e}")

if nd_output is not None:
    configure(save_dir="artifacts/dscvspr_nd_viz")
    render(VizRequest("equation", nd_model))
    render(VizRequest("search_evolution", nd_model))
    render(VizRequest("density", nd_model))
    render(VizRequest("tree", nd_model))
    # Note: Generic parity/residual/field_comparison use FD data path,
    # incompatible with SPR. spr_residual/spr_field_comparison may also
    # fail on custom N-D datasets due to PINN data structure differences.
