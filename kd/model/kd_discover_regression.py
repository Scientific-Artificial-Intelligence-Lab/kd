from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .kd_discover import KD_Discover
from .discover.functions import PlaceholderConstant


@dataclass
class _RegressionArrayData:
    X: np.ndarray
    y: np.ndarray
    var_names: list[str]

    def get_data(self) -> dict:
        return {
            "X": [self.X[:, i : i + 1] for i in range(self.X.shape[1])],
            "y": self.y.reshape(-1, 1),
            "n_input_var": self.X.shape[1],
            "var_names": self.var_names,
        }


class KD_Discover_Regression(KD_Discover):
    """Discover wrapper for generic symbolic regression on tabular data."""

    def __init__(self, *args, config_out=None, **kwargs):
        super().__init__(*args, config_out=None, **kwargs)
        self.base_config_file = "./discover/config/config_regression.json"
        self.set_config(config_out)
        self.best_program_ = None
        self.result_ = None

    def _validate_inputs(self, X, y, var_names: Sequence[str] | None):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)

        if X_arr.ndim != 2:
            raise ValueError(f"KD_Discover_Regression.fit 期望 X 为二维数组，但收到 X.shape={X_arr.shape}")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"KD_Discover_Regression.fit 收到不匹配的样本数：X.shape[0]={X_arr.shape[0]}, y.shape[0]={y_arr.shape[0]}"
            )

        if var_names is None:
            names = [f"x{i + 1}" for i in range(X_arr.shape[1])]
        else:
            names = list(var_names)
            if len(names) != X_arr.shape[1]:
                raise ValueError(
                    f"var_names 长度必须等于特征数：len(var_names)={len(names)}, n_features={X_arr.shape[1]}"
                )
        return X_arr, y_arr, names

    @staticmethod
    def _render_named_expression(expression: str, var_names: Sequence[str]) -> str:
        rendered = expression
        for index, name in enumerate(var_names, start=1):
            rendered = rendered.replace(f"x{index}", name)
        return rendered

    @staticmethod
    def _format_constant(value: float) -> str:
        scalar = float(np.asarray(value).reshape(-1)[0])
        rounded = round(scalar, 6)
        text = f"{rounded:.6f}".rstrip("0").rstrip(".")
        if "." not in text:
            text += ".0"
        return text

    def _serialize_program(self, program) -> str:
        parts = []
        for token in program.traversal:
            if isinstance(token, PlaceholderConstant) and token.value is not None:
                parts.append(self._format_constant(token.value))
            else:
                parts.append(repr(token))
        return ",".join(parts)

    def fit(self, X, y, var_names: Sequence[str] | None = None, *, verbose: bool = False):
        X_arr, y_arr, names = self._validate_inputs(X, y, var_names)

        self.data_class = _RegressionArrayData(X_arr, y_arr, names)
        self.dataset = "custom_regression"
        self.dataset_ = None
        self.setup()

        raw_result = self.train(n_epochs=self.n_iterations, verbose=verbose)
        program = raw_result["program"]
        self.best_program_ = program

        expression = self._serialize_program(program)
        result = {
            "expression": expression,
            "expression_named": self._render_named_expression(expression, names),
            "constants": [float(c) for c in program.get_constants()],
            "reward": float(raw_result["r"]),
            "mse": float(program.evaluate["nmse_test"]),
            "program": program,
            "var_names": names,
        }
        self.result_ = result
        return result

    def set_task(self):
        if self.operator and "const" not in self.operator:
            self.operator = list(self.operator) + ["const"]
        super().set_task()

    def predict(self, X):
        if self.best_program_ is None:
            raise RuntimeError("KD_Discover_Regression.predict was called before fit(X, y).")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError(f"KD_Discover_Regression.predict 期望 X 为二维数组，但收到 X.shape={X_arr.shape}")
        y_hat, _, _ = self.best_program_.execute_direct(X_arr)
        return y_hat.reshape(-1)
