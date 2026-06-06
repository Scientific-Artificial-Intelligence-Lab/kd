
from __future__ import annotations

import math

import pytest

from kd.core.evaluator import EvaluationResult
from kd.search.discover.evaluation.reward import compute_reward

DEFAULT_ALPHA = 0.01
FIXED_COMPLEXITY = 4
FIXED_NMSE = 1.0


def _make_result(
    *,
    nmse: float = FIXED_NMSE,
    complexity: int = FIXED_COMPLEXITY,
    is_valid: bool = True,
) -> EvaluationResult:
    return EvaluationResult(
        mse=nmse,
        nmse=nmse,
        r2=1.0 - nmse,
        complexity=complexity,
        is_valid=is_valid,
    )


class TestComputeRewardUnit:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_good_fit_has_higher_reward_than_poor_fit(self) -> None:
        good = _make_result(nmse=0.25)
        poor = _make_result(nmse=9.0)
        assert compute_reward(good) > compute_reward(poor)

    @pytest.mark.unit
    def test_higher_complexity_has_lower_reward(self) -> None:
        simple = _make_result(complexity=2)
        complex_ = _make_result(complexity=8)
        assert compute_reward(simple) > compute_reward(complex_)

    @pytest.mark.unit
    def test_alpha_zero_ignores_complexity(self) -> None:
        result = _make_result(nmse=4.0, complexity=99)
        expected = 1.0 / (1.0 + math.sqrt(4.0))
        assert compute_reward(result, alpha=0.0) == pytest.approx(expected)

    @pytest.mark.unit
    def test_invalid_result_returns_zero(self) -> None:
        result = _make_result(is_valid=False)
        assert compute_reward(result) == 0.0

    @pytest.mark.unit
    def test_infinite_nmse_returns_zero(self) -> None:
        result = _make_result(nmse=float("inf"))
        assert compute_reward(result) == 0.0

    @pytest.mark.unit
    def test_nan_nmse_returns_zero(self) -> None:
        result = _make_result(nmse=float("nan"))
        assert compute_reward(result) == 0.0

    @pytest.mark.unit
    def test_perfect_fit_returns_complexity_penalty(self) -> None:
        result = _make_result(nmse=0.0, complexity=3)
        expected = 1.0 - DEFAULT_ALPHA * 3.0
        assert compute_reward(result) == pytest.approx(expected)

    @pytest.mark.unit
    def test_zero_complexity_removes_complexity_penalty(self) -> None:
        result = _make_result(nmse=9.0, complexity=0)
        expected = 1.0 / (1.0 + math.sqrt(9.0))
        assert compute_reward(result) == pytest.approx(expected)

    @pytest.mark.unit
    def test_large_complexity_is_clamped_to_zero(self) -> None:
        result = _make_result(nmse=0.0, complexity=101)
        assert compute_reward(result) == 0.0


class TestComputeRewardInvariants:

    @pytest.mark.unit
    @pytest.mark.parametrize("nmse", [0.0, 0.25, 1.0, 4.0, 25.0])
    @pytest.mark.parametrize("complexity", [0, 1, 5, 25, 150])
    def test_reward_is_bounded_between_zero_and_one(
        self, nmse: float, complexity: int
    ) -> None:
        result = _make_result(nmse=nmse, complexity=complexity)
        reward = compute_reward(result)
        assert 0.0 <= reward <= 1.0

    @pytest.mark.unit
    def test_reward_monotonically_decreases_with_nmse(self) -> None:
        results = [
            _make_result(nmse=nmse, complexity=3)
            for nmse in [0.0, 0.25, 1.0, 4.0]
        ]
        rewards = [compute_reward(result) for result in results]
        assert rewards == pytest.approx(sorted(rewards, reverse=True))

    @pytest.mark.unit
    def test_reward_monotonically_decreases_with_complexity(self) -> None:
        results = [
            _make_result(nmse=0.5, complexity=complexity)
            for complexity in [0, 1, 5, 10, 150]
        ]
        rewards = [compute_reward(result) for result in results]
        assert rewards == pytest.approx(sorted(rewards, reverse=True))
