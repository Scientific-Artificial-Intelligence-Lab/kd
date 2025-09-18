import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)


from kd.dataset import load_pde
from kd.model.kd_sga import KD_SGA

# 1. 通过统一入口加载数据集
pde_dataset = load_pde('chafee-infante')

# 2. 创建并配置参数 (所有在 SGA config 中定义的参数都已被兼容)
model = KD_SGA(sga_run=10, depth=3)

# 3. 使用新接口直接传入 PDEDataset
model.fit_dataset(pde_dataset)

# 4. 查看结果
print(f"The discovered equation is: {model.best_pde_}")

# 5. 可视化
model.plot_results()

