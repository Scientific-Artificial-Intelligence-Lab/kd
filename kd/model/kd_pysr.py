"""PySR integration: generic symbolic regression backend for KD.

This module provides a lightweight, optional wrapper around PySR:

* It does not assume any PDE structure and only exposes a generic
  ``fit(X, y)`` style symbolic regressor.
* If ``pysr`` is not installed, all other KD features remain fully usable.
* When ``pysr`` is available, :class:`KD_PySR` can be used to fit models and
  export equations and LaTeX strings.
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
    """Minimal KD wrapper around PySR for generic symbolic regression.

    Key design points:

    * Exposes a scikit-learn–style interface (``fit`` / ``predict``).
    * Provides ``best_equation_`` and :meth:`equation_latex` for alignment
      with other KD backends.
    * Does not introduce PDE-specific interfaces such as ``fit_dataset`` in
      KD 1.x, focusing on generic regression instead.
    """

    def __init__(
        self,
        *,
        niterations: int = 1000,
        binary_operators: Optional[List[str]] = None,
        unary_operators: Optional[List[str]] = None,
        extra_model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a KD_PySR estimator.

        Parameters are forwarded almost directly to
        :class:`pysr.PySRRegressor`.

        Args:
            niterations: Number of PySR optimisation iterations (``niterations``).
            binary_operators: List of binary operators passed to
                ``binary_operators``.
            unary_operators: List of unary operators passed to
                ``unary_operators``.
            extra_model_kwargs: Additional keyword arguments forwarded to the
                underlying :class:`pysr.PySRRegressor`.
        """
        # Do not call BaseEstimator.__init__ to avoid accessing attributes
        # before they are initialised.
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
        """Fit a symbolic regression model on the given ``(X, y)`` pairs.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target array of shape ``(n_samples,)`` or ``(n_samples, 1)``.

        Returns:
            KD_PySR: The fitted estimator instance.
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
        """Predict with the fitted PySR model."""
        if self._model is None:
            raise RuntimeError(
                "KD_PySR.predict was called before the model was fitted. "
                "Call fit(X, y) first."
            )
        X_arr = np.asarray(X)
        return self._model.predict(X_arr)

    def equation_latex(self, lhs: str = "y") -> str:
        """Return the best learned equation in LaTeX form.

        The method tries, in order:

        * PySR's own ``latex()`` method (if available),
        * conversion from the cached SymPy expression,
        * a plain string representation of ``best_equation_`` or the model.

        Note:
            The returned string does *not* include surrounding ``$`` markers;
            rendering layers (such as :mod:`kd.viz`) are responsible for
            wrapping it appropriately.
        """
        if self._model is None:
            raise RuntimeError(
                "KD_PySR.equation_latex was called before the model was fitted. "
                "Call fit(X, y) first."
            )

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
