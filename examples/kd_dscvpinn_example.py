import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='numpy.*')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow.*')


from kd.model import KD_DSCV_Pinn
from kd.viz.discover_eq2latex import discover_program_to_latex 
from kd.viz.equation_renderer import render_latex_to_image
from kd.viz.dscv_viz import *
from kd.dataset import load_burgers_equation, load_pde_dataset

np.random.seed(42)

burgers_data = load_burgers_equation()

# burgers_data = load_pde_dataset(filename="KdV_equation.mat", x_key='x', t_key='tt', u_key='uu')  

x, y = burgers_data.sample(n_samples=0.1)
lb, ub = burgers_data.mesh_bounds()

# 实例化模型。此处定义的算子必须与PyTorch兼容（通常以'_t'结尾）
model = KD_DSCV_Pinn(
    n_samples_per_batch = 100, # Number of generated traversals by agent per batch
    binary_operators = ["add_t", "mul_t", "div_t", "diff_t", "diff2_t"],
    unary_operators = ['n2_t'],
)


step_output = model.fit(x, y, [lb, ub], n_epochs=1)

print(f"Current best expression is {step_output['expression']} and its reward is {step_output['r']}")

# 将模型发现的最佳符号程序（program）转换为LaTeX字符串并渲染成图片。
render_latex_to_image(discover_program_to_latex(step_output['program']))

# 将最终发现方程的结构可视化为表达式树。
plot_expression_tree(model)

# 绘制奖励值的概率密度分布图，以观察已发现解的质量分布。
plot_density(model)

# 绘制最优奖励值随训练轮次的进化曲线，以评估Agent学习过程的收敛性。
plot_evolution(model)

# 分析由PINN强制执行的、最终发现的PDE的残差。
plot_pinn_residual_analysis(model, step_output['program'])

# 将PINN预测的解场与真实的解场进行热图对比。
plot_pinn_field_comparison(model, step_output['program'])

# 生成“奇偶图”，逐点对比PINN的预测值与真实值，以量化最终方程的预测精度。
plot_pinn_actual_vs_predicted(model, step_output['program'])