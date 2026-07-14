


from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from kd.search.eqgpt.reward import (
    INVALID_REWARD,
    WAVE_SPARSITY_ALPHA,
    RewardResult,
    compute_reward,
)

ALPHA = 0.02


def _linear_system(seed: int = 0, n: int = 40) -> tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(seed)
    c1 = rng.standard_normal(n)
    c2 = rng.standard_normal(n)
    b = 2.0 * c1 + 3.0 * c2
    return np.column_stack([-b, c1, c2]), 2.0, 3.0







def test_reward_perfect_fit_matches_formula() -> None:
    A, k1, k2 = _linear_system()
    result = compute_reward(A, sparsity_alpha=ALPHA)
    assert isinstance(result, RewardResult)
    expected = (1.0 - ALPHA * np.log10(3)) * 1.0
    assert result.r2 == pytest.approx(1.0, abs=1e-9)
    assert result.reward == pytest.approx(expected, abs=1e-9)
    assert result.n_terms == 3
    np.testing.assert_allclose(result.coefficients, [k1, k2], atol=1e-9)


def test_reward_alpha_is_per_problem_not_constant() -> None:
    A, _, _ = _linear_system()
    wave = compute_reward(A, sparsity_alpha=WAVE_SPARSITY_ALPHA).reward
    membrane = compute_reward(A, sparsity_alpha=1.0).reward
    assert membrane == pytest.approx((1.0 - 1.0 * np.log10(3)) * 1.0, abs=1e-9)
    assert membrane < wave


def test_reward_deduplicates_columns_before_penalty() -> None:
    A, _, _ = _linear_system()
    dup = np.column_stack([A[:, 0], A[:, 1], A[:, 1], A[:, 2]])
    result = compute_reward(dup, sparsity_alpha=ALPHA)
    assert result.n_terms == 3
    assert result.reward == pytest.approx((1.0 - ALPHA * np.log10(3)) * 1.0, abs=1e-9)


def test_reward_sparsity_penalty_is_monotone_in_terms() -> None:
    A, _, _ = _linear_system()
    with_zero = np.column_stack([A, np.zeros(A.shape[0])])
    r_small = compute_reward(A, sparsity_alpha=ALPHA)
    r_big = compute_reward(with_zero, sparsity_alpha=ALPHA)
    assert r_big.n_terms == 4 and r_small.n_terms == 3
    assert r_big.reward < r_small.reward


def test_reward_removes_inf_rows() -> None:
    A, _, _ = _linear_system()
    a_pinf = A.copy()
    a_pinf[2, 1] = np.inf
    a_ninf = A.copy()
    a_ninf[4, 2] = -np.inf
    expected = (1.0 - ALPHA * np.log10(3)) * 1.0
    assert compute_reward(a_pinf, sparsity_alpha=ALPHA).reward == pytest.approx(
        expected, abs=1e-9
    )
    assert compute_reward(a_ninf, sparsity_alpha=ALPHA).reward == pytest.approx(
        expected, abs=1e-9
    )


def test_reward_single_column_is_not_zero() -> None:
    A, _, _ = _linear_system()
    b = -A[:, 0]
    expected_r2 = 1.0 - (b**2).sum() / ((b - b.mean()) ** 2).sum()
    result = compute_reward(A[:, :1], sparsity_alpha=ALPHA)
    assert result.coefficients.size == 0
    assert result.n_terms == 1
    assert result.r2 == pytest.approx(expected_r2, abs=1e-9)
    assert result.reward == pytest.approx(expected_r2, abs=1e-9)
    assert result.reward != INVALID_REWARD


def test_reward_can_be_negative_for_poor_fit() -> None:
    rng = np.random.default_rng(1)
    n = 40
    lhs = 5.0 + rng.standard_normal(n)
    unrelated = rng.standard_normal(n)
    unrelated = unrelated - (unrelated @ lhs / (lhs @ lhs)) * lhs
    A = np.column_stack([-lhs, unrelated])
    result = compute_reward(A, sparsity_alpha=ALPHA)
    assert result.reward < 0.0







def test_reward_nan_matrix_is_invalid_zero() -> None:
    A, _, _ = _linear_system()
    A[3, 1] = np.nan
    result = compute_reward(A, sparsity_alpha=ALPHA)
    assert result.reward == INVALID_REWARD
    assert np.isnan(result.r2)


def test_reward_empty_matrix_is_invalid_zero() -> None:

    result = compute_reward(np.empty((0, 0)), sparsity_alpha=ALPHA)
    assert result.reward == INVALID_REWARD
    assert np.isnan(result.r2)
    assert result.coefficients.size == 0


def test_reward_zero_column_matrix_is_invalid_zero() -> None:

    result = compute_reward(np.empty((5, 0)), sparsity_alpha=ALPHA)
    assert result.reward == INVALID_REWARD
    assert np.isnan(result.r2)


def test_reward_all_rows_dropped_is_invalid_zero() -> None:
    A, _, _ = _linear_system(n=5)
    A[:, 1] = np.inf
    result = compute_reward(A, sparsity_alpha=ALPHA)
    assert result.reward == INVALID_REWARD
    assert np.isnan(result.r2)


def test_compute_reward_upcasts_float32_to_float64() -> None:
    A32 = _linear_system()[0].astype(np.float32)
    r32 = compute_reward(A32, sparsity_alpha=ALPHA)
    r64 = compute_reward(A32.astype(np.float64), sparsity_alpha=ALPHA)
    assert r32.reward == r64.reward
    assert r32.r2 == r64.r2
    assert r32.n_terms == r64.n_terms
    np.testing.assert_array_equal(r32.coefficients, r64.coefficients)


def test_compute_reward_guards_overflow_nan() -> None:
    rng = np.random.default_rng(0)
    n = 40
    huge = 1e160
    A = np.column_stack(
        [
            -huge * rng.standard_normal(n),
            huge * rng.standard_normal(n),
            huge * rng.standard_normal(n),
        ]
    )


    with np.errstate(over="ignore", invalid="ignore"):
        result = compute_reward(A, sparsity_alpha=ALPHA)
    assert result.reward == INVALID_REWARD
    assert np.isnan(result.r2)
    assert result.coefficients.size == 0


def test_reward_constant_lhs_is_invalid_zero() -> None:
    rng = np.random.default_rng(3)
    n = 20
    A = np.column_stack(
        [np.full(n, 2.0), rng.standard_normal(n), rng.standard_normal(n)]
    )
    result = compute_reward(A, sparsity_alpha=ALPHA)
    assert result.reward == INVALID_REWARD
    assert np.isnan(result.r2)







@settings(max_examples=25, deadline=None)
@given(
    data=st.lists(
        st.lists(
            st.floats(
                min_value=-10, max_value=10, allow_nan=False, allow_infinity=False
            ),
            min_size=2,
            max_size=4,
        ),
        min_size=6,
        max_size=20,
    ),
    alpha=st.floats(min_value=0.0, max_value=1.0),
)
def test_reward_never_exceeds_one(data: list[list[float]], alpha: float) -> None:
    rows = min(len(r) for r in data)
    A = np.array([r[:rows] for r in data], dtype=np.float64)


    assume(np.ptp(A[:, 0]) > 1e-6)
    result = compute_reward(A, sparsity_alpha=alpha)
    assert result.reward <= 1.0 + 1e-9
