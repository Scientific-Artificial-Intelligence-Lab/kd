"""
Custom SGA example: 构造一个简单双正弦场 `u(x,t)=sin(pi*x)*cos(pi*t)`，验证 KD_SGA 在
无解析模板的 custom 模式下可以跑通。

重点：
- 通过 PDEDataset 直接传入 x/t/usol 需显式传 pde_data=None 占位
- problem_name 使用自定义标签，不在内置三基准集合中。
- 默认关闭 metadata/autograd。
"""

import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

from kd.dataset import PDEDataset
from kd.model.kd_sga import KD_SGA
from kd.viz import VizRequest, configure, render, render_equation

# --- 构造自定义数据 ---
x = np.linspace(0.0, 1.0, 32)
t = np.linspace(0.0, 1.0, 33)
xx, tt = np.meshgrid(x, t, indexing="ij")
u = np.sin(np.pi * xx) * np.cos(np.pi * tt)

# PDEDataset 需要 pde_data 位置参数，可传 None 再使用 x/t/usol
dataset = PDEDataset(
    equation_name="custom_sga_test",
    pde_data=None,
    domain=None,
    epi=0.0,
    x=x,
    t=t,
    usol=u,
)

# --- 初始化与训练 ---
model = KD_SGA(
    sga_run=100,      # 略小的迭代方便快速 smoke
    num=20,
    depth=4,
    width=5,
    seed=0,
    use_autograd=False,
    use_metadata=False,
)

print("Running KD_SGA on custom dataset (no ground truth templates)...")
model.fit_dataset(dataset, problem_name="custom_sga_test")

print("\nBest PDE (raw string):")
print(model.best_pde_)

try:
    print("\nEquation in LaTeX:")
    print(model.equation_latex())
except Exception as exc:
    print(f"无法生成 LaTeX 表达式（{exc}），可忽略。")

# 不调用 legacy plot_results（仅支持内置三基准）。使用 kd.viz 适配器生成基础诊断图。
configure(save_dir=os.path.join(kd_main_dir, "artifacts", "sga_custom"))
render_equation(model)
render(VizRequest(kind="field_comparison", target=model))
render(VizRequest(kind="time_slices", target=model, options={"slice_times": [0.0, 0.5, 1.0]}))
render(VizRequest(kind="parity", target=model))
render(VizRequest(kind="residual", target=model))  # custom 场景缺少解析真解，可能提示 warning
