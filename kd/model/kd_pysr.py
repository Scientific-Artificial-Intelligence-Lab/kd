"""PySR 封装：KD 中的通用符号回归后端。

本模块为 PySR 提供一个轻量级、可选依赖的 KD 封装：

- 不强绑定 PDE，只作为通用 `fit(X, y)` 风格的符号回归器；
- 未安装 `pysr` 时，KD 其它功能完全不受影响；
- 已安装 `pysr` 时，可以通过 :class:`KD_PySR` 进行回归并导出方程与 LaTeX。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..base import BaseEstimator

try:  # 惰性导入，避免在未安装 pysr 时影响整个 KD
    from pysr import PySRRegressor  # type: ignore
except Exception:  # pragma: no cover - 在未安装 pysr 的环境下走这里
    PySRRegressor = None  # type: ignore


class KD_PySR(BaseEstimator):
    """KD 中对 PySR 的最小封装，用于通用符号回归。

    设计要点：

    - 接口保持 scikit-learn 风格：``fit`` / ``predict``；
    - 提供 ``best_equation_`` 与 ``equation_latex``，方便与 KD 其它后端对齐；
    - 不在 KD 1.x 中强行加入 PDE 相关接口（如 ``fit_dataset``），只聚焦通用 SR。
    """

    def __init__(
        self,
        *,
        niterations: int = 1000,
        binary_operators: Optional[List[str]] = None,
        unary_operators: Optional[List[str]] = None,
        extra_model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """构造 KD_PySR。

        参数基本直接透传给 :class:`pysr.PySRRegressor`：

        Args:
            niterations: PySR 迭代次数，对应 PySRRegressor 的 ``niterations``。
            binary_operators: 二元算子列表，传给 ``binary_operators``。
            unary_operators: 一元算子列表，传给 ``unary_operators``。
            extra_model_kwargs: 其余需要传给 PySRRegressor 的关键字参数。
        """
        # 不调用 BaseEstimator.__init__，避免其对未初始化属性做 get_params。
        self.niterations = int(niterations)
        self.binary_operators = list(binary_operators) if binary_operators is not None else None
        self.unary_operators = list(unary_operators) if unary_operators is not None else None
        self.extra_model_kwargs = dict(extra_model_kwargs) if extra_model_kwargs is not None else {}

        # 训练后填充的属性
        self._model: Any = None
        self.best_equation_: Optional[str] = None
        self._sympy_expr_: Any = None

        # Cache training data for downstream visualization in the current process.
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_backend_available(self) -> None:
        """Ensure that PySR backend is available."""
        if PySRRegressor is None:
            raise RuntimeError(
                "KD_PySR requires the 'pysr' package and its dependencies "
                "(including a working Julia environment). "
                "No pysr installation was detected in this environment. "
                "Please install it via `pip install kd[pysr]` or "
                "`pip install pysr`."
            )

    def _build_backend(self) -> Any:
        """Construct the underlying PySRRegressor instance."""
        self._ensure_backend_available()

        kwargs: Dict[str, Any] = {"niterations": self.niterations}
        if self.binary_operators is not None:
            kwargs["binary_operators"] = self.binary_operators
        if self.unary_operators is not None:
            kwargs["unary_operators"] = self.unary_operators
        kwargs.update(self.extra_model_kwargs)

        return PySRRegressor(**kwargs)  # type: ignore[call-arg]

    def _update_equation_info(self) -> None:
        """Extract the best equation and SymPy expression from the PySR model."""
        model = self._model
        if model is None:
            return

        best_equation: Optional[str] = None
        sympy_expr: Any = None

        # 优先使用 PySR 的 get_best() 与 sympy() 接口（0.13+）
        try:
            get_best = getattr(model, "get_best", None)
            if callable(get_best):
                best = get_best()
                try:
                    # pandas.Series 或 dict 都支持这种访问方式
                    best_equation = str(best["equation"])
                except Exception:
                    # 兜底：直接字符串化整行
                    best_equation = str(best)
        except Exception:
            best_equation = None

        try:
            sympy_fn = getattr(model, "sympy", None)
            if callable(sympy_fn):
                sympy_expr = sympy_fn()
        except Exception:
            sympy_expr = None

        # 再兜底：直接用模型的 __str__
        if best_equation is None:
            try:
                best_equation = str(model)
            except Exception:
                best_equation = None

        self.best_equation_ = best_equation
        self._sympy_expr_ = sympy_expr

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """在给定的 ``(X, y)`` 上拟合符号回归模型。

        Args:
            X: 形状为 ``(n_samples, n_features)`` 的特征矩阵。
            y: 形状为 ``(n_samples,)`` 或 ``(n_samples, 1)`` 的目标值。

        Returns:
            KD_PySR: 自身实例，便于链式调用。
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).reshape(-1)

        if X_arr.ndim != 2:
            raise ValueError(f"KD_PySR.fit 期望 X 为二维数组，但收到 X.shape={X_arr.shape}")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"KD_PySR.fit 收到不匹配的样本数：X.shape[0]={X_arr.shape[0]}, y.shape[0]={y_arr.shape[0]}"
            )

        self.X_train_ = X_arr
        self.y_train_ = y_arr

        self._model = self._build_backend()
        self._model.fit(X_arr, y_arr)
        self._update_equation_info()
        return self

    def predict(self, X):
        """使用已训练的 PySR 模型进行预测。"""
        if self._model is None:
            raise RuntimeError("KD_PySR.predict 在模型尚未 fit 时被调用。请先调用 fit(X, y)。")
        X_arr = np.asarray(X)
        return self._model.predict(X_arr)

    def equation_latex(self, lhs: str = "y") -> str:
        """返回当前模型学到的最佳方程的 LaTeX 表达形式。

        优先使用 PySR 自带的 ``latex()`` 接口；若不可用，则使用 SymPy 表达式
        转换；若仍失败，则退化为 ``best_equation_`` 的字符串形式。

        注意：返回值**不包含** ``$`` 包裹，渲染时由上层 viz 统一处理。
        """
        if self._model is None:
            raise RuntimeError("KD_PySR.equation_latex 在模型尚未 fit 时被调用。请先调用 fit(X, y)。")

        # 1. 尝试使用 PySR 自带的 latex()（如果有的话）
        latex_str: Optional[str] = None
        try:
            latex_fn = getattr(self._model, "latex", None)
            if callable(latex_fn):
                candidate = latex_fn()
                if isinstance(candidate, str) and candidate.strip():
                    latex_str = candidate
        except Exception:
            latex_str = None

        # 2. 使用 SymPy 表达式兜底
        if latex_str is None and self._sympy_expr_ is not None:
            try:
                import sympy as sp

                rhs = sp.latex(self._sympy_expr_)
                latex_str = f"{lhs} = {rhs}"
            except Exception:
                latex_str = None

        # 3. 最后退化为纯字符串形式
        if latex_str is None:
            if self.best_equation_:
                latex_str = str(self.best_equation_)
            else:
                latex_str = str(self._model)

        return latex_str


__all__ = ["KD_PySR"]
