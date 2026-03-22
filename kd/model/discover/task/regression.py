from __future__ import annotations

from typing import Sequence

import numpy as np

from .task import HierarchicalTask
from ..functions import create_tokens
from ..library import Library


def make_regression_metric(name: str, parsimony_coeff: float = 0.0):
    """Create a regression reward metric.

    Args:
        name: Metric name. ``"inv_nrmse"`` supported.
        parsimony_coeff: Per-token penalty on expression length.
            ``reward *= max(0, 1 - parsimony_coeff * n_tokens)``.
            Set to 0.0 (default) to disable.
    """
    if name != "inv_nrmse":
        raise ValueError(f"Unrecognized regression reward function: {name}")

    def metric(y: np.ndarray, y_hat: np.ndarray, n_tokens: int = 0) -> float:
        variance = max(float(np.var(y)), 1e-12)
        mse = float(np.mean((y - y_hat) ** 2))
        accuracy = 1.0 / (1.0 + np.sqrt(mse / variance))
        if parsimony_coeff > 0.0 and n_tokens > 0:
            penalty = max(0.0, 1.0 - parsimony_coeff * n_tokens)
            return accuracy * penalty
        return accuracy

    return metric, 0.0, 1.0


class RegressionTask(HierarchicalTask):
    """Direct symbolic regression over tabular ``(X, y)`` data."""

    task_type = "regression"

    def __init__(
        self,
        function_set: Sequence[str],
        dataset: str,
        metric: str = "inv_nrmse",
        metric_params: Sequence[float] = (),
        threshold: float = 1e-12,
        protected: bool = False,
        reward_noise: float = 0.0,
        use_torch: bool = False,
        decision_tree_threshold_set=None,
        parsimony_coeff: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        if metric_params:
            raise ValueError("RegressionTask does not use metric_params.")

        self.name = dataset
        self.function_set = list(function_set)
        self.threshold = threshold
        self.protected = protected
        self.use_torch = use_torch
        self.decision_tree_threshold_set = decision_tree_threshold_set
        self.parsimony_coeff = float(parsimony_coeff)
        self.metric, self.invalid_reward, self.max_reward = make_regression_metric(
            metric, parsimony_coeff=self.parsimony_coeff
        )
        self.stochastic = reward_noise > 0.0

    def load_data(self, dataset: dict) -> None:
        self.x = [np.asarray(col, dtype=float).reshape(-1, 1) for col in dataset["X"]]
        self.y = np.asarray(dataset["y"], dtype=float).reshape(-1, 1)
        self.n_input_var = int(dataset.get("n_input_var", len(self.x)))
        self.var_names = list(dataset.get("var_names", [f"x{i + 1}" for i in range(self.n_input_var)]))

        tokens = create_tokens(
            n_input_var=self.n_input_var,
            function_set=self.function_set,
            protected=self.protected,
            task_type="regression",
            decision_tree_threshold_set=self.decision_tree_threshold_set,
        )
        self.library = Library(tokens)

    def reward_function(self, p):
        y_hat, _, weights = p.execute_direct(self.x)
        if p.invalid or y_hat is None:
            return self.invalid_reward, [0.0], None, None
        reward = self.metric(self.y, y_hat, n_tokens=len(p.traversal))
        return reward, weights, y_hat, None

    def mse_function(self, p) -> float:
        y_hat, _, _ = p.execute_direct(self.x)
        if p.invalid or y_hat is None:
            return float("inf")
        return float(np.mean((self.y - y_hat) ** 2))

    def evaluate(self, p) -> dict:
        y_hat, _, _ = p.execute_direct(self.x)
        if p.invalid or y_hat is None:
            nmse_test = None
            success = False
            mse = None
        else:
            mse = float(np.mean((self.y - y_hat) ** 2))
            nmse_test = mse
            success = mse < self.threshold
        return {
            "nmse_test": nmse_test,
            "mse": mse,
            "success": success,
        }
