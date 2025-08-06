"""
Minimal example: Using KD_SGA to discover PDEs from data (SGA-PDE, no KD data pipeline).
"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

from kd.model.kd_sga import KD_SGA
from kd.model.sga.codes import Data_generator

if __name__ == "__main__":
    # 加载 SGA 自带的测试数据
    u = Data_generator.u
    t = Data_generator.t
    x = Data_generator.x

    # 实例化 SGA-PDE 求解器
    model = KD_SGA(num=5, depth=2, width=2)  # 参数可根据需要调整

    # 拟合模型
    best_eq, best_aic = model.fit(u=u, t=t, x=x, max_gen=2, verbose=True)

    print("\nBest discovered equation (concise):")
    print(best_eq.concise_visualize())
    print("Best AIC score:", best_aic)
