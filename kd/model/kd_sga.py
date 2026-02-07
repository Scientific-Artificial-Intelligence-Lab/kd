# kd/model/kd_sga.py

from typing import Any, Dict, Optional, Type

from ..base import BaseEstimator

from .sga.sgapde.config import SolverConfig
from .sga.sgapde.context import ProblemContext
from .sga.sgapde.solver import SGAPDE_Solver
from .sga.sgapde.equation import sga_equation_to_latex
from .sga.sgapde import visualizer as sga_visualizer


class KD_SGA(BaseEstimator):
    """KD wrapper around the SGA-PDE solver.

    This class exposes a scikit-learn–style interface around the upstream
    ``sgapde`` implementation for discovering PDEs of the form
    ``u_t = N(u, u_x, ...)``.

    Supports N-D datasets via optional parameters. Default behavior infers
    ``target_field`` and ``lhs_axis`` from the dataset when not specified.
    """

    def __init__(
        self,
        sga_run: int = 100,
        num: int = 20,
        depth: int = 4,
        width: int = 5,
        p_var: float = 0.5,
        p_mute: float = 0.3,
        p_cro: float = 0.5,
        seed: int = 0,
        use_autograd: bool = False,
        max_epoch: int = 100000,
        use_metadata: bool = False,
        delete_edges: bool = False,
        # N-D 参数（可选，推断为主）
        target_field: Optional[str] = None,
        lhs_axis: Optional[str] = None,
        primary_spatial_axis: Optional[str] = None,
    ):
        """Initialise a KD_SGA solver.

        All parameters correspond directly to fields in
        :class:`sgapde.config.SolverConfig`.

        Args:
            target_field: Field name for the PDE target (e.g., "u", "rho").
                If None, inferred from dataset or defaults to "u".
            lhs_axis: Axis for the LHS derivative (e.g., "t").
                If None, inferred from dataset or defaults to "t".
            primary_spatial_axis: Override for the primary spatial axis used
                in legacy ux/uxx naming when no "x" axis exists.
        """
        # BaseEstimator.__init__ will help manage parameters,
        # but we keep explicit attributes for clarity.
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
        # N-D 参数
        self.target_field = target_field
        self.lhs_axis = lhs_axis
        self.primary_spatial_axis = primary_spatial_axis
        
    # 旧版 fit 方法，保留以兼容现有代码
    def fit(self, problem_name: str):
        """Deprecated legacy entry point, kept for backwards compatibility.

        Use :meth:`fit_dataset` together with :func:`kd.dataset.load_pde`
        instead of ``fit(problem_name)``.
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
        """Run SGA directly on a :class:`kd.dataset.PDEDataset` instance.

        Args:
            dataset: Dataset returned by :func:`kd.dataset.load_pde`.
            problem_name: Optional override for the problem name used inside
                :class:`SolverConfig`.
            context_cls: Optional custom :class:`ProblemContext` subclass,
                typically used for testing.
            solver_cls: Optional custom :class:`SGAPDE_Solver` subclass,
                typically used for testing.

        Returns:
            KD_SGA: The fitted estimator instance.
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

        # 推断优先级：用户显式设置 > adapter 推断 > 默认值
        final_target_field = (
            self.target_field
            or solver_kwargs.pop("target_field", None)
            or "u"
        )
        final_lhs_axis = (
            self.lhs_axis
            or solver_kwargs.pop("lhs_axis", None)
            or "t"
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
            target_field=final_target_field,
            lhs_axis=final_lhs_axis,
            primary_spatial_axis=self.primary_spatial_axis,
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
        """Invoke the legacy ``sgapde`` visualiser for built-in benchmarks.

        This is kept for backwards compatibility and only works for the three
        supported benchmarks (Chafee–Infante, Burgers, KdV). For new and
        custom datasets, prefer the unified :mod:`kd.viz` façade.
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
