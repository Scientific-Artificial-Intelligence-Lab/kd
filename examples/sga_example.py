"""KD_SGA complete example: built-in data, N-D data, and custom data.

Demonstrates the SGA (Symbolic Genetic Algorithm) workflow for PDE discovery
with three data sources and unified kd.viz visualization.

Usage:
    python examples/sga_example.py
"""

import numpy as np

from kd.dataset import PDEDataset, load_pde
from kd.model.kd_sga import KD_SGA
from kd.viz import (
    configure,
    render_equation,
    plot_field_comparison,
    plot_time_slices,
    plot_parity,
    plot_residuals,
)

# ============================================================
# 1. Built-in Dataset / 内置数据集
# ============================================================

# Load the Burgers PDE dataset / 加载 Burgers PDE 数据集
dataset = load_pde("burgers")

# Create KD_SGA model with full parameter documentation
# 创建 KD_SGA 模型（含全参数说明）
model = KD_SGA(
    sga_run=30,               # Number of SGA runs / SGA 运行轮数
    num=20,                   # Population size / 种群大小
    depth=4,                  # Expression tree depth / 表达式树深度
    # width=5,                # [default] Tree width (branch factor) / 树宽度
    # p_var=0.5,              # [default] Variable mutation prob / 变量变异概率
    # p_mute=0.3,             # [default] Gene mutation prob / 基因突变概率
    # p_cro=0.5,              # [default] Crossover prob / 交叉概率
    seed=0,                   # Random seed / 随机种子
    # use_autograd=False,     # [default] Use autograd for derivatives / 自动求导
    # max_epoch=100000,       # [default] Max NN training epochs / NN 最大训练轮数
    # use_metadata=False,     # [default] Metadata augmentation / 元数据增强
    # delete_edges=False,     # [default] Prune zero-weight edges / 修剪零权重边
)

model.fit_dataset(dataset)
print(model.best_pde_)
print(model.equation_latex())


# ============================================================
# 2. Visualization via kd.viz / 可视化
# ============================================================

configure(save_dir="artifacts/sga_viz")

# Render discovered equation as LaTeX image / 渲染方程为 LaTeX 图片
render_equation(model)

# Field comparison heatmap / 场对比热力图
plot_field_comparison(
    model, x_coords=None, t_coords=None,
    true_field=None, predicted_field=None,
)

# Time slices at t=0.0, 0.5, 1.0 / 时间切片
plot_time_slices(
    model, x_coords=None, t_coords=None,
    true_field=None, predicted_field=None,
    slice_times=[0.0, 0.5, 1.0],
)

# Parity plot: predicted RHS vs true u_t / 奇偶图
plot_parity(model, title="KD_SGA Parity Plot")

# Residual diagnostics / 残差诊断
plot_residuals(model, actual=None, predicted=None, bins=40)


# ============================================================
# 3. N-D Data (3D spatial grid) / N-D 数据（3D 空间网格）
# ============================================================

from _nd_data import make_diffusion_2d

x, y, t, u = make_diffusion_2d()
print(f"N-D data shape: {u.shape}")

# Construct N-D PDEDataset / 构造 N-D 数据集
nd_dataset = PDEDataset(
    equation_name="2d_diffusion",
    fields_data={"u": u},               # Field arrays / 场数据字典
    coords_1d={"x": x, "y": y, "t": t}, # 1D coordinate arrays / 坐标数组
    axis_order=["x", "y", "t"],          # Dimension order / 维度顺序
    target_field="u",                    # Field to discover equation for / 目标场
    lhs_axis="t",                        # Time axis for u_t = ... / 时间轴
)

nd_model = KD_SGA(sga_run=50, num=15, depth=3, seed=42)
nd_model.fit_dataset(nd_dataset)
print(nd_model.best_pde_)

configure(save_dir="artifacts/sga_nd_viz")
render_equation(nd_model)
plot_field_comparison(nd_model)
plot_parity(nd_model, title="KD_SGA N-D Parity")
plot_residuals(nd_model)


# ============================================================
# 4. Custom Data / 自定义数据
# ============================================================

# Construct synthetic field u(x,t) = sin(pi*x) * cos(pi*t) / 构造合成场
x_custom = np.linspace(0.0, 1.0, 32)
t_custom = np.linspace(0.0, 1.0, 33)
xx, tt = np.meshgrid(x_custom, t_custom, indexing="ij")
u_custom = np.sin(np.pi * xx) * np.cos(np.pi * tt)

# PDEDataset with pde_data=None for fully custom data
# pde_data=None 表示完全自定义数据（不使用内置数据集）
custom_dataset = PDEDataset(
    equation_name="custom_test",
    pde_data=None,       # No built-in .mat file / 不使用内置数据文件
    domain=None,         # No domain info needed / 不需要域信息
    epi=0.0,             # No noise added / 不添加噪声
    x=x_custom,          # Spatial coordinates / 空间坐标
    t=t_custom,          # Temporal coordinates / 时间坐标
    usol=u_custom,       # Solution field / 解场
)

custom_model = KD_SGA(
    sga_run=100, num=20, depth=4,
    seed=0, use_autograd=False, use_metadata=False,
)
custom_model.fit_dataset(custom_dataset, problem_name="custom_test")
print(custom_model.best_pde_)

configure(save_dir="artifacts/sga_custom")
render_equation(custom_model)
plot_parity(custom_model, title="KD_SGA Custom Data Parity")
