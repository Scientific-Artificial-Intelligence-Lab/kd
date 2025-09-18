"""Dataset-driven usage of KD_DSCV_SPR (Mode 2 / sparse PINN pipeline)."""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

from kd.dataset import load_pde
from kd.model.kd_dscv import KD_DSCV_SPR

# 1. 加载数据集（示例使用 Burgers 方程）
pde_dataset = load_pde('burgers')

# 2. 初始化 R-DISCOVER (Mode2) 模型
model = KD_DSCV_SPR(
    n_iterations=5,
    n_samples_per_batch=50,
    binary_operators=["add_t", "mul_t", "div_t", "diff_t", "diff2_t"],
    unary_operators=['n2_t'],
)

# 3. 使用新入口导入 PDEDataset。显式设置 random_state 以保证可重复
model.import_dataset(
    pde_dataset,
    sample_ratio=0.05,
    colloc_num=256,
    random_state=0,
)

# 4. 演示性地运行一次迭代（真实任务可调用 model.fit 或更长训练）
step_result = model.train(n_epochs=1, verbose=False)
print(f"Current reward snapshot: {step_result['r']}")
