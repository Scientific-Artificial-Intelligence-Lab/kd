
from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


DEFAULT_COMPLEXITY_WEIGHT = 0.01


def sparse_reward(
    y: FloatArray,
    y_hat: FloatArray,
    n_terms: int,
    *,
    complexity_weight: float = DEFAULT_COMPLEXITY_WEIGHT,
) -> float:
    y_arr = np.asarray(y)
    y_hat_arr = np.asarray(y_hat)
    if y_arr.shape != y_hat_arr.shape:
        raise ValueError(
            f"y and y_hat must have identical shapes, got {y_arr.shape} "
            f"vs {y_hat_arr.shape} (a mismatch would broadcast silently)"
        )

    return float(
        (1 - complexity_weight * n_terms)
        / (1 + np.sqrt(np.mean((y_arr - y_hat_arr) ** 2) / np.var(y_arr)))
    )


def rounded_sparse_reward(
    y: FloatArray,
    y_hat: FloatArray,
    n_terms: int,
    *,
    complexity_weight: float = DEFAULT_COMPLEXITY_WEIGHT,
) -> float:
    return round(
        sparse_reward(y, y_hat, n_terms, complexity_weight=complexity_weight), 4
    )
