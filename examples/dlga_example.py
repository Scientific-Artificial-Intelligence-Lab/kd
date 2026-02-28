"""KD_DLGA complete example: PDE discovery with full visualization.

Demonstrates the DLGA (Deep Learning Genetic Algorithm) workflow including
training, prediction, and all available kd.viz diagnostic plots.

Usage:
    python examples/dlga_example.py
"""

from kd.dataset import load_pde
from kd.model.kd_dlga import KD_DLGA
from kd.viz import (
    configure,
    render_equation,
    plot_training_curve,
    plot_validation_curve,
    plot_search_evolution,
    plot_optimization,
    plot_field_comparison,
    plot_residuals,
    plot_time_slices,
    plot_parity,
    plot_derivative_relationships,
)


# ============================================================
# 1. Data Loading / 数据加载
# ============================================================

# Load the KdV PDE dataset / 加载 KdV PDE 数据集
dataset = load_pde("kdv")


# ============================================================
# 2. Model Initialization / 模型初始化
# ============================================================

# Create KD_DLGA model with full parameter documentation
# 创建 KD_DLGA 模型（含全参数说明）
model = KD_DLGA(
    operators=["u", "u_x", "u_xx", "u_xxx"],  # Candidate operator library / 候选算子库
    epi=0.1,                                   # Sparsity penalty / 方程简洁度惩罚项
    input_dim=2,                               # Input dimension (x, t) / 输入维度
    verbose=False,                             # Suppress training logs / 静默训练日志
    max_iter=9000,                             # Max NN training iterations / 最大训练迭代次数
    # --- GA kwargs (passed to underlying GA) / 遗传算法参数 ---
    # population_size=100,   # [default] GA population / GA 种群大小
    # n_generations=50,      # [default] GA generations / GA 代数
)


# ============================================================
# 3. Training / 训练
# ============================================================

configure(save_dir="artifacts/dlga_viz")

model.fit_dataset(
    dataset,
    sample=1000,               # Sample 1000 random grid points / 随机采样 1000 点
    sample_method="random",    # Sampling strategy / 采样策略
    Xy_from="sample",          # Use sampled points (not full grid) / 使用采样点
)


# ============================================================
# 4. Prediction on Full Grid / 全网格预测
# ============================================================

# Generate predictions on the complete (x, t) mesh for visualization
# 在完整网格上生成预测，用于可视化对比
X_full = dataset.mesh()
u_pred = model.predict(X_full).reshape(dataset.get_size())


# ============================================================
# 5. Visualization / 可视化
# ============================================================

# Equation rendering / 方程渲染
render_equation(model)

# Training & validation loss curves / 训练和验证损失曲线
plot_training_curve(model)
plot_validation_curve(model)

# GA search evolution & optimization diagnostics / GA 搜索过程与优化诊断
plot_search_evolution(model)
plot_optimization(model)

# Field comparison: true vs predicted / 场对比
# Note: DLGA requires explicit coords and fields / 注意：DLGA 需要显式传入坐标和场
plot_field_comparison(
    model,
    x_coords=dataset.x,
    t_coords=dataset.t,
    true_field=dataset.usol,
    predicted_field=u_pred,
)

# Residual diagnostics / 残差诊断
plot_residuals(
    model,
    actual=dataset.usol.reshape(-1),
    predicted=u_pred.reshape(-1),
    coordinates=X_full,
    bins=40,
)

# Time slices / 时间切片
plot_time_slices(
    model,
    x_coords=dataset.x,
    t_coords=dataset.t,
    true_field=dataset.usol,
    predicted_field=u_pred,
    slice_times=[0.25, 0.5, 0.75],
)

# Derivative-term relationships (top N RHS terms) / 导数项关系
plot_derivative_relationships(model, top_n_terms=4)

# Parity plot / 奇偶图
plot_parity(model, title="KD_DLGA Parity Plot")
