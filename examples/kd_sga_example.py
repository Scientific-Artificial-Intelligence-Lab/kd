import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)


from kd.model.kd_sga import KD_SGA

# 1. 创建并配置参数
# 所有在 __init__ 中定义的参数都可以在这里设置
model = KD_SGA(sga_run=100, depth=3, seed=42) 

# 2. 加入数据（当前通过 problem_name）并训练
model.fit(problem_name='chafee-infante')

# 3. 查看结果
print(f"The discovered equation is: {model.best_pde_}")

# 4. 可视化
model.plot_results()