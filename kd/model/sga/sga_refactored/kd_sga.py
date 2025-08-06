import numpy as np
from ..codes.sga import SGA
from ..codes import pde, tree, Data_generator, configure
from .data_utils import prepare_workspace
from .operators import get_default_library
from .main import inject_dependencies

class KD_SGA:
    """
    KD_SGA: 面向 kd 框架的 SGA-PDE 求解器适配器类。
    继承自 BaseGa（此处为原型，后续可替换为实际基类）。
    支持参数注册、fit 主流程、依赖注入。
    """
    def __init__(self, 
                 num=20, 
                 depth=4, 
                 width=5, 
                 p_var=0.5, 
                 p_rep=1.0, 
                 p_mute=0.3, 
                 p_cro=0.5,
                 data_mode='finite_difference'):
        """
        参数:
            num: SGA 种群规模
            depth: 符号树最大深度
            width: PDE 最大项数
            p_var: 生成树时变量概率
            p_rep: 替换概率
            p_mute: 突变概率
            p_cro: 交叉概率
            data_mode: 数据导数计算模式（'finite_difference' 或 'autograd'）
        """
        self.num = num
        self.depth = depth
        self.width = width
        self.p_var = p_var
        self.p_rep = p_rep
        self.p_mute = p_mute
        self.p_cro = p_cro
        self.data_mode = data_mode
        # 可扩展更多参数

    def get_params(self, deep=True):
        return {
            "num": self.num,
            "depth": self.depth,
            "width": self.width,
            "p_var": self.p_var,
            "p_rep": self.p_rep,
            "p_mute": self.p_mute,
            "p_cro": self.p_cro,
            "data_mode": self.data_mode
        }

    def set_params(self, **params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self

    def fit(self, u=None, t=None, x=None, max_gen=5, verbose=True):
        """
        执行 SGA-PDE 主流程。
        参数:
            u, t, x: 输入数据（如不指定则用 Data_generator 默认数据）
            max_gen: SGA 迭代代数
            verbose: 是否打印详细日志
        返回:
            best_eq_obj, best_aic
        """
        if u is None or t is None or x is None:
            u, t, x = Data_generator.u, Data_generator.t, Data_generator.x
        if verbose:
            print(f"[KD_SGA] Data loaded: u.shape={u.shape}")

        # 1. 数据预处理与导数计算
        workspace = prepare_workspace(u, t, x, mode=self.data_mode)
        symbol_library = get_default_library(workspace)

        # 2. 依赖注入
        inject_dependencies(workspace, symbol_library)
        if verbose:
            print("[KD_SGA] Dependencies injected.")

        # 3. 实例化并运行 SGA
        sga_instance = SGA(
            num=self.num,
            depth=self.depth,
            width=self.width,
            p_var=self.p_var,
            p_rep=self.p_rep,
            p_mute=self.p_mute,
            p_cro=self.p_cro
        )
        sga_instance.run(gen=max_gen)
        best_eq_obj, best_aic = sga_instance.the_best()
        if verbose:
            print("[KD_SGA] Best AIC score:", best_aic)
            print("[KD_SGA] Discovered Equation:", best_eq_obj.concise_visualize())
        return best_eq_obj, best_aic
