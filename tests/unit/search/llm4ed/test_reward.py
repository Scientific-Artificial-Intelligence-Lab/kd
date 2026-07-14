
from __future__ import annotations

import numpy as np
import pytest

from kd.search.llm4ed.reward import rounded_sparse_reward, sparse_reward


class TestSparseReward:
    def test_perfect_fit_is_complexity_only(self) -> None:
        y = np.array([0.0, 1.0, 2.0, 3.0])
        assert sparse_reward(y, y.copy(), 2) == pytest.approx(0.98)
        assert sparse_reward(y, y.copy(), 5) == pytest.approx(0.95)

    def test_hand_computed_value(self) -> None:


        y = np.array([0.0, 2.0])
        y_hat = np.array([1.0, 1.0])
        assert sparse_reward(y, y_hat, 3) == pytest.approx(0.97 / 2)

    def test_population_variance_not_sample(self) -> None:
        y = np.array([0.0, 2.0])
        y_hat = np.array([1.0, 1.0])
        wrong = 0.97 / (1.0 + np.sqrt(0.5))
        assert sparse_reward(y, y_hat, 3) != pytest.approx(wrong)

    def test_reward_decreases_with_term_count(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        rewards = [sparse_reward(y, y.copy(), n) for n in range(1, 6)]
        assert rewards == sorted(rewards, reverse=True)

    def test_column_shapes_accepted(self) -> None:
        y = np.array([[0.0], [2.0]])
        y_hat = np.array([[1.0], [1.0]])
        assert sparse_reward(y, y_hat, 3) == pytest.approx(0.97 / 2)

    def test_shape_mismatch_raises(self) -> None:
        y = np.array([[0.0], [2.0]])
        y_hat = np.array([1.0, 1.0])
        with pytest.raises(ValueError, match="shape"):
            sparse_reward(y, y_hat, 3)

    def test_returns_python_float(self) -> None:
        y = np.array([0.0, 1.0])
        assert type(sparse_reward(y, y.copy(), 1)) is float


class TestRoundedSparseReward:
    def test_rounds_to_four_decimals(self) -> None:
        y = np.array([0.0, 2.0])
        y_hat = np.array([1.0, 0.9])
        raw = sparse_reward(y, y_hat, 2)
        assert rounded_sparse_reward(y, y_hat, 2) == round(raw, 4)

        assert raw != round(raw, 4)

    def test_matches_raw_when_exact(self) -> None:
        y = np.array([0.0, 1.0, 2.0, 3.0])
        assert rounded_sparse_reward(y, y.copy(), 2) == 0.98
