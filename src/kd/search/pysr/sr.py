
from __future__ import annotations

import keyword
import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import sympy

from kd.core.expr.registry import FunctionRegistry
from kd.core.expr.sympy_bridge import from_sympy
from kd.search.pysr.backend import PySRBackend, default_backend_factory
from kd.search.pysr.config import PySRConfig



BackendFactory = Callable[[PySRConfig], PySRBackend]

_NMSE_EPS = 1e-12





_KD_IR_RESERVED_NAMES: frozenset[str] = frozenset(
    FunctionRegistry.create_default().list_names()
)
_DIFF_RESERVED_PATTERN = re.compile(r"^diff[0-9]*_[a-z]+$")


class PySRSymbolicRegressor:

    def __init__(
        self,
        *,
        config: PySRConfig | None = None,
        backend_factory: BackendFactory = default_backend_factory,
    ) -> None:
        self._config = config if config is not None else PySRConfig()
        self._backend_factory = backend_factory


        self.best_sympy_: Any = None
        self.best_expr_: str | None = None
        self.best_score_: float = float("inf")
        self.var_names_: list[str] | None = None
        self.n_features_in_: int | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        var_names: Sequence[str] | None = None,
    ) -> PySRSymbolicRegressor:
        x_arr, y_arr, names = self._validate_fit_inputs(X, y, var_names)
        internal_names = _generic_feature_names(x_arr.shape[1])

        backend = self._backend_factory(self._config)
        backend.fit(x_arr, y_arr, internal_names)

        internal_expr = sympy.sympify(backend.best_sympy())
        rendered_expr = _render_expression(internal_expr, internal_names, names)
        best_expr = _convert_to_kd_ir(rendered_expr)
        y_pred = _evaluate_expression(rendered_expr, names, x_arr)
        best_score = _nmse(y_arr, y_pred)

        self.best_sympy_ = rendered_expr
        self.best_expr_ = best_expr
        self.var_names_ = names
        self.n_features_in_ = x_arr.shape[1]
        self.best_score_ = best_score
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = self._validate_predict_inputs(X)
        return self._evaluate(x_arr)

    @staticmethod
    def _validate_fit_inputs(
        X: np.ndarray,
        y: np.ndarray,
        var_names: Sequence[str] | None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        x_arr = np.asarray(X, dtype=float)
        y_raw = np.asarray(y, dtype=float)
        if y_raw.ndim >= 2 and (y_raw.ndim != 2 or y_raw.shape[1] != 1):
            raise ValueError(
                "PySRSymbolicRegressor.fit expects y with shape (n,) or (n, 1), "
                f"got {y_raw.shape}"
            )
        y_arr = y_raw.reshape(-1)
        if x_arr.ndim != 2:
            raise ValueError(
                f"PySRSymbolicRegressor.fit expects 2-D X, got {x_arr.shape}"
            )
        if x_arr.shape[0] == 0 or x_arr.shape[1] == 0:
            raise ValueError(
                f"PySRSymbolicRegressor.fit expects non-empty X, got {x_arr.shape}"
            )
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                "PySRSymbolicRegressor.fit received mismatched sample counts: "
                f"X.shape[0]={x_arr.shape[0]}, y.shape[0]={y_arr.shape[0]}"
            )
        _raise_if_not_finite(x_arr, name="X")
        _raise_if_not_finite(y_arr, name="y")
        if isinstance(var_names, str):
            raise ValueError(
                "var_names must be a sequence of names, not a single string"
            )
        names = (
            _generic_feature_names(x_arr.shape[1])
            if var_names is None
            else list(var_names)
        )
        if len(names) != x_arr.shape[1]:
            raise ValueError(
                "var_names length must equal the number of features: "
                f"len(var_names)={len(names)}, n_features={x_arr.shape[1]}"
            )
        _validate_var_names(names)
        return x_arr, y_arr, names

    def _validate_predict_inputs(self, X: np.ndarray) -> np.ndarray:
        if self.best_sympy_ is None or self.var_names_ is None:
            raise RuntimeError(
                "PySRSymbolicRegressor.predict was called before fit(X, y)."
            )
        x_arr = np.asarray(X, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError(
                f"PySRSymbolicRegressor.predict expects 2-D X, got {x_arr.shape}"
            )
        if self.n_features_in_ is not None and x_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                "PySRSymbolicRegressor.predict received mismatched feature count: "
                f"X.shape[1]={x_arr.shape[1]}, expected={self.n_features_in_}"
            )
        return x_arr

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        if self.best_sympy_ is None or self.var_names_ is None:
            raise RuntimeError("PySRSymbolicRegressor must be fit before evaluation.")
        return _evaluate_expression(self.best_sympy_, self.var_names_, X)


def _generic_feature_names(n_features: int) -> list[str]:
    return [f"x{index + 1}" for index in range(n_features)]


def _render_expression(
    expr: sympy.Expr,
    internal_names: Sequence[str],
    rendered_names: Sequence[str],
) -> sympy.Expr:
    substitutions = {
        sympy.Symbol(source): sympy.Symbol(target)
        for source, target in zip(internal_names, rendered_names, strict=True)
    }
    return sympy.sympify(expr.xreplace(substitutions))


def _validate_var_names(names: Sequence[str]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(
                f"var_names must be valid Python identifiers, got {name!r}"
            )
        if keyword.iskeyword(name):
            raise ValueError(f"var_name {name!r} is a Python keyword")
        if name in _KD_IR_RESERVED_NAMES or _is_diff_reserved_name(name):
            raise ValueError(f"var_name {name!r} collides with a reserved kd-IR token")
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        joined = ", ".join(repr(name) for name in duplicates)
        raise ValueError(f"var_names must be unique; duplicate names: {joined}")


def _is_diff_reserved_name(name: str) -> bool:
    return _DIFF_RESERVED_PATTERN.match(name) is not None


def _raise_if_not_finite(values: np.ndarray, *, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")


def _convert_to_kd_ir(expr: sympy.Expr) -> str:
    try:
        return from_sympy(expr)
    except ValueError as exc:
        raise RuntimeError(
            "PySR returned an expression unconvertible to kd IR "
            f"({expr!s}; {exc}). Restrict PySRConfig operators to the kd-IR "
            "subset (+,-,*,/,sin,cos,exp,log)."
        ) from exc


def _evaluate_expression(
    expr: sympy.Expr,
    names: Sequence[str],
    X: np.ndarray,
) -> np.ndarray:
    symbols = [sympy.Symbol(name) for name in names]
    func = sympy.lambdify(symbols, expr, modules=["numpy"])
    columns = [X[:, index] for index in range(X.shape[1])]
    values = np.asarray(func(*columns), dtype=float)
    if values.ndim == 0:
        return np.full(X.shape[0], float(values), dtype=float)
    return values.reshape(-1)


def _nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if (
        y_true.shape != y_pred.shape
        or not np.all(np.isfinite(y_true))
        or not np.all(np.isfinite(y_pred))
    ):
        return float("inf")
    mse = float(np.mean(np.square(y_true - y_pred)))
    variance = float(np.var(y_true))
    return mse / max(variance, _NMSE_EPS)


__all__ = ["PySRSymbolicRegressor"]
