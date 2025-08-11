# kd/model/kd_sga.py

import numpy as np
from ..base import BaseEstimator

from .sga.sgapde.config import SolverConfig
from .sga.sgapde.context import ProblemContext
from .sga.sgapde.solver import SGAPDE_Solver
from .sga.sgapde import visualizer as sga_visualizer


class KD_SGA(BaseEstimator):
    """
    一个使用符号遗传算法（SGA）发现偏微分方程（PDE）的模型。
    这是对 sgapde 库的一个封装，以适应 kd 框架。

    遵循 scikit-learn API 风格，通过 __init__ 设置参数，通过 fit 执行计算。
    """
    
    def __init__(self, sga_run=100, num=20, depth=4, width=5, 
                 p_var=0.5, p_mute=0.3, p_cro=0.5, seed=0, 
                 use_autograd=False, max_epoch=100000, 
                 use_metadata=False, delete_edges=False):
        """
        初始化 KD_SGA 模型。

        所有参数都直接对应 sgapde.config.SolverConfig 中的配置项。
        """
        # BaseEstimator 的 __init__ 会自动帮我们处理参数赋值
        # 但为了清晰，我们在这里也显式声明
        self.sga_run = sga_run
        self.num = num
        self.depth = depth
        self.width = width
        self.p_var = p_var
        self.p_mute = p_mute
        self.p_cro = p_cro
        self.seed = seed
        self.use_autograd = use_autograd
        self.max_epoch = max_epoch
        self.use_metadata = use_metadata
        self.delete_edges = delete_edges
        
    def fit(self, problem_name: str):
        """
        根据预设的问题名称加载数据，并执行PDE发现算法。

        这是我们当前阶段的临时数据接口，未来可以扩展。

        参数:
            problem_name (str): 预设的数据集名称, 
                                例如 'chafee-infante', 'burgers', 'kdv'。
        """
        print(f"--- Starting SGA PDE Discovery for problem: {problem_name} ---")

        # 1. 根据 self 的属性和传入的 problem_name 创建 SolverConfig
        # 我们没有传入 u_data, x_data, t_data，所以它会自动从文件加载
        config = SolverConfig(
            problem_name=problem_name,
            sga_run=self.sga_run,
            num=self.num,
            depth=self.depth,
            width=self.width,
            p_var=self.p_var,
            p_mute=self.p_mute,
            p_cro=self.p_cro,
            seed=self.seed,
            use_autograd=self.use_autograd,
            max_epoch=self.max_epoch,
            use_metadata=self.use_metadata,
            delete_edges=self.delete_edges
        )
        
        # 2. 创建数据上下文，完成所有数据预处理
        context = ProblemContext(config)
        
        # 3. 创建并运行求解器
        solver = SGAPDE_Solver(config)
        best_pde, best_score = solver.run(context)
        
        # 4. 存储结果到模型的属性中（以后缀 _ 结尾）
        self.best_pde_ = best_pde
        self.best_score_ = best_score
        self.context_ = context # 保存完整的上下文，以备可视化使用
        self.config_ = config   # 保存此次运行的配置
        
        print("\n--- SGA PDE Discovery Finished ---")
        print(f"Best PDE Found: {self.best_pde_}")
        print(f"AIC Score: {self.best_score_}")

        return self

    def plot_results(self):
        """
        调用 sgapde 自带的可视化工具来绘制结果和诊断图。
        """
        if not hasattr(self, 'context_'):
            raise RuntimeError("You must call fit() before plotting results.")
        
        print("INFO: Generating visualization plots...")
        sga_visualizer.plot_figures(self.context_, self.config_)