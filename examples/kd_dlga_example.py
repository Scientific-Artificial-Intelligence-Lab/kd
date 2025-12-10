from _bootstrap import ensure_project_root_on_syspath

PROJECT_ROOT = ensure_project_root_on_syspath()

# --- 依赖导入 / Dependency Imports ---
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


# --- 数据加载 / Data Loading ---
# 使用统一入口加载 KdV 方程数据集。
# Load the KdV PDE dataset via the unified loader.
kdv_data = load_pde("kdv")


# --- 模型初始化 / Model Initialization ---
# 初始化 KD_DLGA 模型：
#   - operators: 候选算子库（这里使用常见的 u, u_x, u_xx, u_xxx）
#   - epi: 控制方程稀疏度/简洁度的惩罚系数
#   - input_dim: 输入维度，这里是 (x, t) → 2
#   - max_iter: 内部神经网络训练迭代次数
model = KD_DLGA(
    operators=["u", "u_x", "u_xx", "u_xxx"],  # 候选算子库 / Candidate operator library.
    epi=0.1,  # 方程简洁度惩罚项 / Sparsity penalty for equation.
    input_dim=2,  # 输入维度，与 (x,t) 一致 / Input dimension, matches (x, t).
    verbose=False,
    max_iter=9000,
)


# --- 可视化配置 / Visualization configuration ---
# 设置可视化输出目录；所有由 kd.viz 生成的图像都会写入此目录。
# Configure the viz façade so that all generated figures are written under this directory.
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "dlga_example"
configure(save_dir=OUTPUT_ROOT)


# --- 模型训练 / Model Training ---
# 使用统一的 fit_dataset 入口训练 DLGA：
#   - sample=1000: 从网格上随机抽取 1000 个样本点进行训练；
#   - sample_method='random': 使用随机采样策略；
#   - Xy_from='sample': 明确表示使用样本点而不是完整网格。
print("\n[KD_DLGA Example] Training DLGA model with fit_dataset(PDEDataset)...")
model.fit_dataset(
    kdv_data,
    sample=1000,
    sample_method="random",
    Xy_from="sample",
)


# --- 预测生成 / Prediction on full grid ---
# 在完整 (x, t) 网格上生成预测解 u_pred，用于后续场对比与残差分析。
print("\n[KD_DLGA Example] Generating predictions on full (x, t) grid...")
X_full = kdv_data.mesh()  # 创建用于可视化的完整网格 / Full grid for field visualisation.
u_pred_field = model.predict(X_full).reshape(kdv_data.get_size())


# --- 结果可视化（统一 kd.viz façade） / Results visualisation via kd.viz ---

# 方程渲染（LaTeX → PNG）
# Render the discovered PDE as a LaTeX equation image.
render_equation(model)

# 训练/验证损失曲线
# Training and validation loss curves.
plot_training_curve(model)
plot_validation_curve(model)

# 遗传算法搜索过程与优化诊断
# GA search evolution and optimization diagnostics.
plot_search_evolution(model)
plot_optimization(model)

# PDE 场对比（真实场 vs 预测场）
# PDE field comparison: true solution vs DLGA-predicted solution.
plot_field_comparison(
    model,
    x_coords=kdv_data.x,
    t_coords=kdv_data.t,
    true_field=kdv_data.usol,
    predicted_field=u_pred_field,
)

# 残差诊断（u_t - RHS）——这里使用完整网格上的点。
# Residual diagnostics (u_t - RHS) on the full grid.
plot_residuals(
    model,
    actual=kdv_data.usol.reshape(-1),
    predicted=u_pred_field.reshape(-1),
    coordinates=X_full,
    bins=40,
)

# 时间切片对比
# Time-slice comparison of u(x, t) at several representative times.
plot_time_slices(
    model,
    x_coords=kdv_data.x,
    t_coords=kdv_data.t,
    true_field=kdv_data.usol,
    predicted_field=u_pred_field,
    slice_times=[0.25, 0.5, 0.75],
)

# 导数项关系（自动解析最重要的 RHS 项）
# Derivative-term relationships: automatically focuses on the most significant RHS terms.
plot_derivative_relationships(model, top_n_terms=4)

# 方程奇偶图
# Parity plot comparing predicted PDE RHS vs true LHS.
plot_parity(model, title="Final Validation of Discovered PDE")
