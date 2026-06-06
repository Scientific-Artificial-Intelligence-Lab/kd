
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
import torch

from kd.search.discover.engine import CandidateSnapshot
from kd.search.discover.stability import stability_select


class _FakeEvaluator:

    def __init__(
        self,
        theta_map: dict[tuple[str, ...], np.ndarray],
        lhs: Sequence[float] | np.ndarray,
    ) -> None:
        self._theta_map = {
            key: torch.tensor(value, dtype=torch.float64)
            for key, value in theta_map.items()
        }
        self._lhs = torch.tensor(lhs, dtype=torch.float64)
        self.calls: list[tuple[tuple[str, ...], bool]] = []

    @property
    def lhs_target(self) -> torch.Tensor:
        return self._lhs

    def build_theta_matrix(
        self,
        terms: list[str],
        *,
        skip_invalid: bool = False,
    ) -> tuple[torch.Tensor, list[str]]:
        key = tuple(terms)
        self.calls.append((key, skip_invalid))
        return self._theta_map[key], list(terms)


def _candidate(
    expression: str,
    reward: float,
    nmse: float,
    n_nodes: int,
    terms: list[str],
) -> CandidateSnapshot:
    return CandidateSnapshot(
        expression=expression,
        reward=reward,
        nmse=nmse,
        n_nodes=n_nodes,
        terms=terms,
    )


class TestStabilitySelection:

    def test_duplicate_removal_by_nmse_keeps_shorter_candidate(self) -> None:
        lhs = np.linspace(1.0, 2.0, 20)
        theta = lhs[:, None]
        evaluator = _FakeEvaluator(
            {
                ("long_term",): theta,
                ("short_term",): theta,
                ("unique_term",): theta,
            },
            lhs,
        )
        candidates = [
            _candidate("long_expr", 0.9, 0.1, 9, ["long_term"]),
            _candidate("short_expr", 0.8, 0.100000001, 3, ["short_term"]),
            _candidate("unique_expr", 0.7, 0.2, 4, ["unique_term"]),
        ]

        stability_select(
            candidates,
            evaluator,
            top_k=2,
            n_bootstrap=8,
            n_inner_bootstrap=4,
            rng=np.random.default_rng(0),
        )

        assert [call[0] for call in evaluator.calls] == [
            ("short_term",),
            ("unique_term",),
        ]

    def test_build_theta_matrix_called_once_per_tested_candidate(self) -> None:
        lhs = np.linspace(1.0, 2.0, 24)
        theta = lhs[:, None]
        evaluator = _FakeEvaluator(
            {
                ("term_a",): theta,
                ("term_b",): theta,
            },
            lhs,
        )
        candidates = [
            _candidate("expr_a", 0.9, 0.1, 2, ["term_a"]),
            _candidate("expr_b", 0.8, 0.2, 2, ["term_b"]),
        ]

        stability_select(
            candidates,
            evaluator,
            top_k=2,
            n_bootstrap=20,
            n_inner_bootstrap=5,
            rng=np.random.default_rng(1),
        )

        assert len(evaluator.calls) == 2

    def test_more_stable_candidate_can_beat_higher_reward_candidate(self) -> None:
        x = np.linspace(1.0, 2.0, 80)
        noise = np.random.default_rng(7).normal(scale=0.1, size=x.shape[0])
        lhs = x
        evaluator = _FakeEvaluator(
            {
                ("unstable_x", "unstable_noise"): np.column_stack([noise, noise**2]),
                ("stable_x",): x[:, None],
            },
            lhs,
        )
        candidates = [
            _candidate(
                "higher_reward_but_unstable",
                0.95,
                0.1,
                4,
                ["unstable_x", "unstable_noise"],
            ),
            _candidate("lower_reward_but_stable", 0.50, 0.2, 2, ["stable_x"]),
        ]

        result = stability_select(
            candidates,
            evaluator,
            top_k=2,
            n_bootstrap=100,
            n_inner_bootstrap=10,
            rng=np.random.default_rng(2),
        )

        assert result.selected.expression == "lower_reward_but_stable"
        assert result.vote_counts[1] > result.vote_counts[0]
        for stats in result.candidates:
            assert stats.mse.shape == (100,)
            assert stats.cv.shape == (100,)
            assert stats.score.shape == (100,)
            assert np.isfinite(stats.mse).all()
            assert np.isfinite(stats.cv).all()
            assert np.isfinite(stats.score).all()

    def test_vote_counts_uses_minlength_for_all_candidates(self) -> None:
        lhs = np.linspace(1.0, 2.0, 24)
        noise = np.random.default_rng(4).normal(scale=0.1, size=lhs.shape[0])
        evaluator = _FakeEvaluator(
            {
                ("stable",): lhs[:, None],
                ("unstable_a", "unstable_b"): np.column_stack([noise, noise**2]),
            },
            lhs,
        )
        candidates = [
            _candidate("stable_expr", 0.7, 0.1, 2, ["stable"]),
            _candidate("unstable_expr", 0.6, 0.2, 4, ["unstable_a", "unstable_b"]),
        ]

        result = stability_select(
            candidates,
            evaluator,
            top_k=2,
            n_bootstrap=16,
            n_inner_bootstrap=6,
            rng=np.random.default_rng(3),
        )

        assert len(result.vote_counts) == 2
        assert sum(result.vote_counts) == 16
        assert result.selected.expression == "stable_expr"

    def test_top_k_zero_rejected(self) -> None:
        lhs = np.linspace(1.0, 2.0, 10)
        evaluator = _FakeEvaluator({("u",): lhs[:, None]}, lhs)

        with pytest.raises(ValueError, match="top_k"):
            stability_select(
                [_candidate("expr", 0.5, 0.1, 1, ["u"])],
                evaluator,
                top_k=0,
            )

    def test_rank_deficient_theta_produces_finite_statistics(self) -> None:
        lhs = np.linspace(1.0, 2.0, 12)
        theta = np.column_stack(
            [lhs, lhs**2, lhs**3, lhs**4, lhs**5, lhs**6, lhs**7],
        )
        evaluator = _FakeEvaluator({("a", "b", "c", "d", "e", "f", "g"): theta}, lhs)

        result = stability_select(
            [
                _candidate(
                    "rank_deficient",
                    0.4,
                    0.2,
                    7,
                    ["a", "b", "c", "d", "e", "f", "g"],
                ),
            ],
            evaluator,
            top_k=1,
            n_bootstrap=20,
            n_inner_bootstrap=5,
            rng=np.random.default_rng(5),
        )

        stats = result.candidates[0]
        assert np.isfinite(stats.mse).all()
        assert np.isfinite(stats.cv).all()
        assert np.isfinite(stats.score).all()
