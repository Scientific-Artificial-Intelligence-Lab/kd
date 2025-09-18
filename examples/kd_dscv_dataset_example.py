"""Dataset-driven usage of KD_DSCV (Mode 1 / regular grids)."""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

from kd.dataset import load_pde
from kd.model.kd_dscv import KD_DSCV

# 1. 通过统一入口加载 PDE 数据集
pde_dataset = load_pde('chafee-infante')

# 2. 初始化 DISCOVER (Mode1) 模型参数
model = KD_DSCV(
    n_iterations=20,
    n_samples_per_batch=200,
    binary_operators=["add", "mul", "diff", "diff2"],
    unary_operators=['n2'],
)

# 3. 新接口：直接使用 PDEDataset
model.import_dataset(pde_dataset)

# 4. 运行少量迭代演示（真实任务建议调高 n_iterations）
result = model.train(n_epochs=10, verbose=False)
print(f"Current best expression: {result['expression']}")
