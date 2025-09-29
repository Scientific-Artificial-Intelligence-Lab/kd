import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)


# --- 依赖导入 / Dependency Imports ---
from kd.dataset import load_pde, load_pde_dataset
from kd.model.kd_dlga import KD_DLGA
from kd.viz.dlga_viz import *
from kd.viz.equation_renderer import render_latex_to_image


# --- 数据加载 / Data Loading ---
kdv_data = load_pde('kdv')

# 从总数据集中随机抽取1000个样本点作为训练数据 / Randomly sample 1000 points from the dataset for training.
X_train, y_train = kdv_data.sample(n_samples=1000)


# --- 模型初始化 / Model Initialization ---
model = KD_DLGA(
    operators=['u', 'u_x', 'u_xx', 'u_xxx'],        # 定义候选算子库 / Define the candidate operator library.
    epi=0.1,                                        # 方程的简洁度惩罚项 / Simplicity penalty for the equation.
    input_dim=2,                                    # 输入数据的维度，须与X_train列数匹配 / Input dimension, must match X_train's columns.
    verbose=False,                                  # 是否在进化中打印每代的最优解 / Whether to print the best solution in each generation.
    max_iter=9000                                   # 内部神经网络的最大训练迭代次数 / Max training iterations for the internal NN.
)


# --- 模型训练与预测 / Model Training & Prediction ---
print("\nTraining DLGA model...")
model.fit(X_train, y_train)

print("\nGenerating predictions...")
X_full = kdv_data.mesh()                            # 创建用于可视化的完整网格 / Create a full grid for visualization.
u_pred = model.predict(X_full)
u_pred = u_pred.reshape(kdv_data.get_size())


# --- 结果可视化 / Results Visualization ---
render_latex_to_image(model.eq_latex)               # 将发现的最佳方程渲染成图片 / Render the best discovered equation to an image.
plot_training_loss(model)                           # 绘制训练损失曲线 / Plot the training loss curve.
plot_validation_loss(model)                         # 绘制验证损失曲线 / Plot the validation loss curve.

# 绘制最优解的适应度与复杂度随代数进化的曲线
# Plot the evolution of fitness and complexity of the best solution over generations.
plot_optimization_analysis(model)

# 将真实解 u 与预测解 u_pred 进行热图比较 
# Compare true (u) and predicted (u_pred) solutions via heatmaps.
plot_pde_comparison(kdv_data.x, kdv_data.t, kdv_data.usol, u_pred)

# 绘制训练点上的残差分布图，以及在整个求解域上的残差统计直方图
# Plot the residual distribution on training points and a histogram of residuals over the entire domain.
plot_residual_analysis(model, X_train, y_train, kdv_data.usol, u_pred)

# 在特定时间点，绘制真实解与预测解的横截面对比图。
# Plot cross-section comparisons of true and predicted solutions at specific times.
plot_time_slices(kdv_data.x, kdv_data.t, kdv_data.usol, u_pred, slice_times=[0.25, 0.5, 0.75])

# 此函数会自动解析模型中的最优解，并为最重要的项生成关系图, 核心思想是：关键的RHS项与LHS之间应该存在简单的（通常是线性的）关系。
# This function automatically parses the best solution and plots the relationships between key terms.
# The core idea is that a key RHS term should have a simple (usually linear) relationship with the LHS.
plot_derivative_relationships(model)

# 生成最终发现方程的奇偶图 / Generate a parity plot for the final discovered equation.
plot_pde_parity(model, title="Final Validation of Discovered Equation")