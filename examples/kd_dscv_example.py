import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

import scipy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='numpy.*')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow.*')
from kd.model import KD_DSCV
from kd.viz.dscv_viz import *
from kd.viz.discover_eq2latex import discover_program_to_latex 
from kd.viz.equation_renderer import render_latex_to_image


# 支持的算子包括：/ Supported operators include:
#   - 二元算子（binary_operators）: 
#       'add', 'sub', 'mul', 'div', 'diff', 'diff2', 'diff3', 'diff4', 
#   - 一元算子（unary_operators）: 
#       'n2', 'n3', 'n4', 'n5', 'sin', 'cos', 'tan', 'sigmoid', 'logabs', 'expneg'
model = KD_DSCV(
    binary_operators=["add", "mul", "diff"],                    # 定义可用的二元算子 (add 必须存在) / Defines available binary operators (add must be present)
    unary_operators=['n2'],                                     # 定义可用的一元算子 / Defines available unary operators
    n_samples_per_batch=500,                                    # 影响训练效率与探索范围 / Affects training efficiency and exploration range
)

np.random.seed(42)

model.import_inner_data(dataset='Burgers', data_type='regular')

step_output = model.train(n_epochs=51)
print(f"Current best expression is {step_output['expression']} and its reward is {step_output['r']}")


# 将模型发现的最佳程序（program）转换为美观的LaTeX数学公式并渲染成图片
# Converts the best program discovered by the model into a beautiful LaTeX mathematical formula and renders it as an image.
render_latex_to_image(discover_program_to_latex(step_output['program']))

# 绘制最终发现方程的表达式树（Expression Tree），直观地展示其结构
# Draws the Expression Tree of the finally discovered equation, visually showing its structure.
plot_expression_tree(model)

# 绘制在训练过程中，奖励（Reward）的概率密度分布图 用于观察Agent发现的解的质量分布情况。通过指定epoches参数，可以观察分布随训练的演化
# Plots the probability density distribution of the Reward during the training process.
plot_density(model)
plot_density(model, epoches = [10,30,50])

# 绘制最优奖励值随训练轮次（Epoch）的进化曲线 用于判断Agent的学习过程是否在稳定地收敛
# Plots the evolution curve of the optimal reward value over training epochs
plot_evolution(model)

# 绘制最终发现的PDE的残差分析图
# Plots the residual analysis graph of the finally discovered PDE.
plot_pde_residual_analysis(model, step_output['program'])

# 将真实解的热图与由发现的PDE积分得到的解进行并排比较
# Plots a side-by-side comparison of the heatmap of the true solution and the solution
plot_field_comparison(model, step_output['program'])

# 将真实解与由发现的PDE积分得到的解在每个点上进行比较。
# Compares the true solution with the solution obtained by integrating the discovered PDE as a scatter plot at each point
plot_actual_vs_predicted(model, step_output['program'])