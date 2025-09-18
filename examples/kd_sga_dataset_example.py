"""Dataset-driven usage of KD_SGA."""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

from kd.dataset import load_pde
from kd.model.kd_sga import KD_SGA

# 1. 通过统一入口加载数据
pde_dataset = load_pde('chafee-infante')

# 2. 初始化模型，与旧接口保持一致
model = KD_SGA(sga_run=10, depth=3)

# 3. 使用新增接口直接提供数据集
model.fit_dataset(pde_dataset)

print(f"The discovered equation is: {model.best_pde_}")
model.plot_results()
