import numpy as np

from _bootstrap import ensure_project_root_on_syspath

PROJECT_ROOT = ensure_project_root_on_syspath()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='numpy.*')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow.*')


from kd.model import KD_DSCV_SPR
from kd.viz.discover_eq2latex import discover_program_to_latex 
from kd.viz.equation_renderer import render_latex_to_image
from kd.viz.dscv_viz import *
from kd.dataset import load_pde

np.random.seed(42)

# 1. 通过统一入口加载 Burgers 数据集
pde_dataset = load_pde('burgers')

# 实例化模型。此处定义的算子必须与PyTorch兼容（通常以'_t'结尾）
# Instantiate the model. The operators defined here must be compatible with PyTorch (usually ending with '_t').
model = KD_DSCV_SPR(
    n_samples_per_batch = 300, # Number of generated traversals by agent per batch
    binary_operators = ["add_t", "mul_t", "div_t", "diff_t", "diff2_t"],
    unary_operators = ['n2_t'],
)

# 2. 使用统一入口构建稀疏数据并运行训练
model.import_dataset(
    pde_dataset,
    sample_ratio=0.1,
    colloc_num=20000,
    random_state=0,
)

step_output = model.train(n_epochs=200, verbose=True)

print(f"Current best expression is {step_output['expression']} and its reward is {step_output['r']}")

# 将模型发现的最佳符号程序（program）转换为LaTeX字符串并渲染成图片
# Converts the best program discovered by the model into a LaTeX string and renders it as an image.
render_latex_to_image(discover_program_to_latex(step_output['program']))

# 将最终发现方程的结构可视化为表达式树
# Visualizes the structure of the finally discovered equation as an expression tree.
plot_expression_tree(model)

# 绘制奖励值的概率密度分布图，以观察已发现解的质量分布
# Plots the probability density distribution of the reward values to observe the quality distribution of the discovered solutions.
plot_density(model)

# 绘制最优奖励值随训练轮次的进化曲线
# Plots the evolution curve of the optimal reward value over training epochs.
plot_evolution(model)

# 分析由最终发现的PDE的残差
# Analyzes the residuals of the PDE discovered by the final model.
plot_spr_residual_analysis(model, step_output['program'])

# 将PINN预测的解场与真实的解场进行热图对比
# Compares the predicted solution field by the PINN with the true solution field via heatmaps.
plot_spr_field_comparison(model, step_output['program'])

# 生成“奇偶图”，逐点对比PINN的预测值与真实值，以量化最终方程的预测精度
# Generates a "scatter plot" to compare the PINN's predictions with the true values point by point, quantifying the prediction accuracy of the final equation.
plot_spr_actual_vs_predicted(model, step_output['program'])
