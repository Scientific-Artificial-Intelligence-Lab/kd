
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np


INVALID_REWARD: Final[float] = 0.0

_REWARD_OVERFLOW_CLAMP: Final[float] = 1e4

WAVE_SPARSITY_ALPHA: Final[float] = 0.02


@dataclass(frozen=True)
class RewardResult:

    reward: float
    r2: float
    coefficients: np.ndarray
    n_terms: int


def _invalid(n_terms: int) -> RewardResult:
    return RewardResult(
        reward=INVALID_REWARD,
        r2=float("nan"),
        coefficients=np.empty(0, dtype=np.float64),
        n_terms=n_terms,
    )


def _dedup_columns(matrix: np.ndarray) -> np.ndarray:
    kept: list[np.ndarray] = []
    for col_idx in range(matrix.shape[1]):
        col = matrix[:, col_idx]
        if not any(np.array_equal(col, existing) for existing in kept):
            kept.append(col)
    if not kept:
        return np.empty((matrix.shape[0], 0), dtype=matrix.dtype)
    return np.column_stack(kept)


def _drop_inf_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix[~np.isposinf(matrix).any(axis=1)]
    return matrix[~np.isneginf(matrix).any(axis=1)]


def compute_reward(A: np.ndarray, *, sparsity_alpha: float) -> RewardResult:
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[1] == 0:
        return _invalid(0)

    deduped = _dedup_columns(A)
    n_terms = deduped.shape[1]

    filtered = _drop_inf_rows(deduped)
    if filtered.shape[0] == 0:
        return _invalid(n_terms)

    if np.isnan(filtered).any():
        return _invalid(n_terms)

    lhs = -filtered[:, 0]
    centered_denominator = float(np.sum((lhs - lhs.mean()) ** 2))
    if centered_denominator == 0.0:
        return _invalid(n_terms)

    try:
        coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(
            filtered[:, 1:], lhs, rcond=None
        )
    except np.linalg.LinAlgError:
        return _invalid(n_terms)

    rhs = filtered[:, 1:] @ coefficients
    r2 = 1.0 - float(np.sum((lhs - rhs) ** 2)) / centered_denominator
    reward = float((1.0 - sparsity_alpha * np.log10(n_terms)) * r2)






    if np.isnan(reward):
        return _invalid(n_terms)

    if reward > _REWARD_OVERFLOW_CLAMP:
        return RewardResult(
            reward=INVALID_REWARD, r2=r2, coefficients=coefficients, n_terms=n_terms
        )
    return RewardResult(
        reward=reward, r2=r2, coefficients=coefficients, n_terms=n_terms
    )
