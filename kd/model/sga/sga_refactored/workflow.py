# main_refactored.py

import numpy as np
import sys

# -------------------------------------------------------------------
# 关键：确保 `codes` 和 `sga_refactored` 都在Python的搜索路径中
# (当我们用 `python -m ...` 运行时，这通常是自动处理的)
# -------------------------------------------------------------------

# 从我们重构的模块中导入工具
from .data_utils import prepare_workspace
from .operators import get_default_library

# 从原始的、现在被当作“包”的模块中导入
from ..codes.sga import SGA
from ..codes import pde, tree, setup, configure, Data_generator

print("✅  Successfully imported all necessary modules!")
print("-" * 40)

def inject_dependencies(workspace, symbol_library):
    """
    这是一个核心的“适配器”函数。
    它负责将我们准备好的数据和符号库，传给原始模块所期望的全局变量。
    """
    print("   - Injecting dependencies into original modules...")

    # 1. 注入数据依赖
    #    pde.py 依赖 ut, default_terms 等
    pde.ut = workspace['ut'].reshape(-1, 1) # pde模块期望的是一维数组
    pde.n, pde.m = workspace['u'].shape
    
    #    构建 pde.py 依赖的 default_terms
    u_flat = workspace['u'].reshape(-1, 1)
    pde.default_terms = np.hstack((u_flat,)).reshape(-1, 1)  # 暂时只注入 u，与原始 setup.py 一致
    pde.default_names = ['u']
    pde.num_default = pde.default_terms.shape[1]

    # 2. 注入符号库依赖（保持原 shape，和 setup.py 一致）
    tree.ROOT = symbol_library['ROOT']
    tree.OPS = symbol_library['OPS']
    tree.VARS = symbol_library['VARS']
    tree.OP1 = symbol_library['OP1']
    tree.OP2 = symbol_library['OP2']
    tree.den = symbol_library['den']
    
    # 3. 注入其他可能存在的依赖
    #    例如，pde.py 中还用到了 dx, dt
    pde.dx = workspace['dx']
    pde.dt = workspace['dt']
    
    print("   - Dependencies injected successfully.")

def run_sga_process():
    """
    重构后的主流程。
    """
    print("🚀  Starting the refactored SGA process...")

    # --- 步骤 1: 加载数据 ---
    u, t, x = Data_generator.u, Data_generator.t, Data_generator.x
    print(f"   - Data loaded for problem: '{configure.problem}', shape: {u.shape}")

    # --- 步骤 2: 调用我们重构的、干净的函数 ---
    print("   - Preparing workspace using refactored modules...")
    workspace = prepare_workspace(u, t, x)
    symbol_library = get_default_library(workspace)

    # --- 步骤 3: 依赖注入 ---
    inject_dependencies(workspace, symbol_library)

    # --- 步骤 4: 实例化并运行SGA引擎 ---
    print("   - Instantiating and running the SGA Engine...")
    
    # 直接使用原始的SGA类
    sga_instance = SGA(
        num=20,     # population_size
        depth=4,    # max_tree_depth
        width=5,    # max_tree_width
        p_var=0.5,
        p_rep=1.0,
        p_mute=0.3,
        p_cro=0.5
    )
    # 只运行几代来快速测试
    sga_instance.run(gen=5) 
    
    # --- 步骤 5: 捕获并展示结果 ---
    best_eq_obj, best_aic = sga_instance.the_best()
    
    print("-" * 40)
    print("✅  Refactored process finished successfully!")
    print(f"   - Best AIC score: {best_aic:.4f}")
    print(f"   - Discovered Equation: {best_eq_obj.concise_visualize()}")


if __name__ == "__main__":
    run_sga_process()
