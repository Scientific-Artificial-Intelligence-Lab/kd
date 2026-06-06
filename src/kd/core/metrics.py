
from __future__ import annotations

import math
from collections.abc import Callable





ScorerFn = Callable[[float, int], float]
"""Unified scorer signature: (mse, k) -> score.

``mse`` is mean squared error; ``k`` is model complexity (active term count).
Lower scores indicate better models.
"""





_NMSE_EPS: float = 1e-15
"""Default epsilon for NMSE target-variance guard."""

_MSE_FLOOR: float = 1e-15
"""MSE values at or below this threshold are treated as perfect fit."""







def aic(mse: float, k: int, n: int) -> float:
    if n <= 0:
        return float("inf")
    if not math.isfinite(mse) or mse < 0.0:
        return float("inf")
    if mse <= _MSE_FLOOR:
        return -float("inf")
    return n * math.log(mse) + 2.0 * k


def aic_no_n(mse: float, k: int, ratio: float = 1.0) -> float:
    if not math.isfinite(mse) or mse <= 0.0:
        return float("inf")
    return 2.0 * k * ratio + 2.0 * math.log(mse)


def aicc(mse: float, k: int, n: int) -> float:
    if n <= k + 1:
        return float("inf")
    base = aic(mse, k, n)
    if not math.isfinite(base):
        return base
    correction = 2.0 * k * (k + 1) / (n - k - 1)
    return base + correction


def bic(mse: float, k: int, n: int) -> float:
    if not math.isfinite(mse) or mse < 0.0:
        return float("inf")
    if mse <= _MSE_FLOOR:
        return -float("inf")
    if n <= 0:
        return float("inf")
    return n * math.log(mse) + k * math.log(n)


def nmse(mse: float, target_var: float, eps: float = _NMSE_EPS) -> float:
    if not math.isfinite(mse):
        return mse
    if target_var > eps:
        return mse / target_var
    return mse







def make_aic_scorer(n: int) -> ScorerFn:

    def _scorer(mse: float, k: int) -> float:
        return aic(mse, k, n)

    return _scorer


def make_sga_scorer(ratio: float = 1.0) -> ScorerFn:

    def _scorer(mse: float, k: int) -> float:
        return aic_no_n(mse, k, ratio)

    return _scorer


def make_bic_scorer(n: int) -> ScorerFn:

    def _scorer(mse: float, k: int) -> float:
        return bic(mse, k, n)

    return _scorer
