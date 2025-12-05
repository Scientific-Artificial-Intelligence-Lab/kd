# kd/model/kd_sga.py

from typing import Any, Dict, Optional, Type

from ..base import BaseEstimator

from .sga.sgapde.config import SolverConfig
from .sga.sgapde.context import ProblemContext
from .sga.sgapde.solver import SGAPDE_Solver
from .sga.sgapde.equation import sga_equation_to_latex
from .sga.sgapde import visualizer as sga_visualizer


class KD_SGA(BaseEstimator):
    """
    一个使用符号遗传算法 SGA 发现偏微分方程 PDE 的模型。
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
        
    # 旧版 fit 方法，保留以兼容现有代码
    def fit(self, problem_name: str):
        """
        旧接口已废弃：请改用 fit_dataset(dataset)。

        这样可以统一从 kd.dataset.load_pde 入口加载数据，避免 legacy 文件路径依赖。
        """
        raise RuntimeError(
            "KD_SGA.fit(problem_name) 已废弃，请改用 KD_SGA.fit_dataset(PDEDataset)。"
            " 如需兼容旧脚本，请先通过 kd.dataset.load_pde(problem_name) 获取数据集。"
        )

    def equation_latex(
        self,
        *,
        include_coefficients: bool = True,
    ) -> str:
        """Return the discovered equation formatted as LaTeX."""

        details = getattr(self, 'best_equation_details_', None)
        if details is None:
            raise RuntimeError('Equation details are not available. Call fit_dataset() first.')
        return sga_equation_to_latex(details, include_coefficients=include_coefficients)

    def equation_structure_latex(self) -> str:
        """Return the equation structure without coefficients."""

        return self.equation_latex(include_coefficients=False)

    def fit_dataset(
        self,
        dataset: Any,
        *,
        problem_name: Optional[str] = None,
        context_cls: Optional[Type] = None,
        solver_cls: Optional[Type] = None,
    ):
        """
        新增接口：直接使用 :class:`~kd.dataset.PDEDataset` 执行 SGA.

        Args:
            dataset: 由 ``kd.dataset.load_pde`` 返回的 PDEDataset 对象。
            problem_name: 可选，覆盖用于 SolverConfig 的问题名称。
            context_cls: 可选，注入自定义 ProblemContext 子类（测试用）。
            solver_cls: 可选，注入自定义 SGAPDE_Solver 子类（测试用）。
        """
        from kd.dataset import PDEDataset  # 避免模块级循环依赖
        from .sga.adapter import SGADataAdapter

        if not isinstance(dataset, PDEDataset):
            raise TypeError("dataset 必须是 PDEDataset 实例")

        adapter = SGADataAdapter(dataset)
        solver_kwargs: Dict[str, Any] = adapter.to_solver_kwargs()

        inferred_name = solver_kwargs.pop("problem_name", None)
        aliases = getattr(dataset, 'aliases', {}) if hasattr(dataset, 'aliases') else {}
        sga_problem = aliases.get('sga_problem') if isinstance(aliases, dict) else None
        legacy_problem = aliases.get('legacy') if isinstance(aliases, dict) else None
        problem_label = problem_name or sga_problem or legacy_problem or inferred_name or "custom_dataset"

        actual_context_cls = context_cls or ProblemContext
        actual_solver_cls = solver_cls or SGAPDE_Solver

        print(
            f"--- Starting SGA PDE Discovery for problem: {problem_label} (dataset mode) ---"
        )

        config = SolverConfig(
            problem_name=problem_label,
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
            delete_edges=self.delete_edges,
            **solver_kwargs,
        )

        context = actual_context_cls(config)
        solver = actual_solver_cls(config)
        best_pde, best_score = solver.run(context)

        self.best_pde_ = best_pde
        self.best_score_ = best_score
        self.context_ = context
        self.config_ = config
        self.dataset_ = dataset
        self.best_equation_details_ = getattr(solver, 'best_equation_details_', None)

        print("\n--- SGA PDE Discovery Finished ---")
        print(f"Best PDE Found: {self.best_pde_}")
        print(f"AIC Score: {self.best_score_}")

        return self

    def plot_results(self):
        """
        调用 sgapde 自带的可视化工具来绘制结果和诊断图。
        """
        if not hasattr(self, 'context_'):
            raise RuntimeError("You must call fit_dataset() before plotting results.")

        # legacy 可视化仅支持三种内置 benchmark，custom 数据会缺少解析模板
        supported = {"chafee-infante", "burgers", "kdv"}
        problem = getattr(self, "config_", None)
        normalized_name = None
        if problem is not None:
            normalized_name = SolverConfig._normalize_problem_name(problem.problem_name)
        has_ground_truth = getattr(problem, "has_ground_truth", False)
        if normalized_name not in supported or not has_ground_truth:
            print(
                "INFO: legacy SGA visualizer 仅针对内置基准（chafee-infante/burgers/kdv），"
                "已跳过 plot_figures。可使用 kd.viz 的 SGA 适配器进行可视化。"
            )
            return

        print("INFO: Generating legacy visualization plots...")
        sga_visualizer.plot_figures(self.context_, self.config_)
