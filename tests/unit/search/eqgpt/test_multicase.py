
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch

from kd.core.term_cache import TermColumnCache
from kd.data.loaders.wave_breaking import WaveBreakingCase
from kd.data.loaders.wave_breaking_eval import wave_breaking_star_grid
from kd.search.eqgpt._multicase import (
    WAVE_LHS_ORDER,
    WAVE_MAX_ATOMIC_ORDER,
    MultiCaseEvaluator,
    WaveCaseBundle,
    assemble_pinned_matrix,
)
from kd.search.eqgpt.reward import compute_reward

pytestmark = pytest.mark.unit

_ALPHA = 0.02







class _ExecOut:

    def __init__(self, value: torch.Tensor) -> None:
        self.value = value


class FakeExecutor:

    def __init__(self, columns: dict[str, torch.Tensor]) -> None:
        self._columns = columns

    def execute(self, term: str, context: Any = None) -> _ExecOut:


        return _ExecOut(self._columns[term])


class FakeContext:
    pass


def _bundle(
    case_name: str,
    columns: dict[str, torch.Tensor],
    pinned: torch.Tensor,
    generation: int,
) -> WaveCaseBundle:
    return WaveCaseBundle(
        case_name=case_name,
        executor=FakeExecutor(columns),
        context=FakeContext(),
        pinned_lhs=pinned,
        n_points=pinned.numel(),
        cache_generation=generation,
    )


def _reward_of(pinned: torch.Tensor, col: torch.Tensor) -> float:
    matrix = np.column_stack([pinned.numpy(), col.numpy()])
    return compute_reward(matrix, sparsity_alpha=_ALPHA).reward


def _r2_of(pinned: torch.Tensor, col: torch.Tensor) -> float:
    matrix = np.column_stack([pinned.numpy(), col.numpy()])
    return compute_reward(matrix, sparsity_alpha=_ALPHA).r2







class TestConstantsAndBundle:
    def test_wave_order_constants(self) -> None:
        assert WAVE_MAX_ATOMIC_ORDER == 3
        assert WAVE_LHS_ORDER == 1

    def test_wave_case_bundle_is_constructible(self) -> None:
        b = _bundle("c", {}, torch.zeros(4, dtype=torch.float64), generation=3)
        assert b.case_name == "c"
        assert b.n_points == 4
        assert b.cache_generation == 3







class TestAssemblePinnedMatrix:
    def test_column_order_and_float64(self) -> None:
        pinned = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        cols = {
            "a": torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64),
            "b": torch.tensor([9.0, 10.0, 11.0, 12.0], dtype=torch.float64),
        }
        matrix = assemble_pinned_matrix(
            ["a", "b"],
            executor=FakeExecutor(cols),
            context=FakeContext(),
            pinned_lhs=pinned,
            cache=TermColumnCache(),
            cache_generation=0,
        )
        assert matrix.dtype == np.float64
        assert matrix.shape == (4, 3)
        np.testing.assert_allclose(matrix[:, 0], pinned.numpy())
        np.testing.assert_allclose(matrix[:, 1], cols["a"].numpy())
        np.testing.assert_allclose(matrix[:, 2], cols["b"].numpy())

    def test_length_mismatch_raises_value_error(self) -> None:
        pinned = torch.zeros(5, dtype=torch.float64)
        ex = FakeExecutor({"a": torch.zeros(4, dtype=torch.float64)})
        with pytest.raises(ValueError):
            assemble_pinned_matrix(
                ["a"],
                executor=ex,
                context=FakeContext(),
                pinned_lhs=pinned,
                cache=TermColumnCache(),
                cache_generation=0,
            )

    def test_cache_isolated_per_generation(self) -> None:
        term = "u_x"
        col0 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        col1 = torch.tensor([-9.0, -8.0, -7.0, -6.0], dtype=torch.float64)
        shared = TermColumnCache()

        m0 = assemble_pinned_matrix(
            [term],
            executor=FakeExecutor({term: col0}),
            context=FakeContext(),
            pinned_lhs=torch.ones(4, dtype=torch.float64),
            cache=shared,
            cache_generation=0,
        )
        m1 = assemble_pinned_matrix(
            [term],
            executor=FakeExecutor({term: col1}),
            context=FakeContext(),
            pinned_lhs=2.0 * torch.ones(4, dtype=torch.float64),
            cache=shared,
            cache_generation=1,
        )
        np.testing.assert_allclose(m0[:, 1], col0.numpy())
        np.testing.assert_allclose(m1[:, 1], col1.numpy())


        assert not np.allclose(m1[:, 1], col0.numpy())


class TestDualGridProvenance:

    @staticmethod
    def _synthetic_case() -> WaveBreakingCase:

        t = torch.linspace(0.0, 1.0, 50, dtype=torch.float64)
        z = torch.zeros(50, dtype=torch.float64)
        return WaveBreakingCase(
            name="N_synthetic",
            t=t,
            x=z,
            eta=z,
            g=2,
            tp_seconds=1.2,
            a=80,
            lamda=2.25,
            prefix="N",
        )

    def test_real_grid_row_counts_differ(self) -> None:
        case = self._synthetic_case()
        x_r, t_r = wave_breaking_star_grid(case, points_per_window=50)
        x_c, t_c = wave_breaking_star_grid(case, points_per_window=100)
        assert x_r.numel() == 3 * 50
        assert x_c.numel() == 3 * 100
        assert t_r.numel() == t_c.numel()
        nt = t_r.numel()
        assert 3 * 50 * nt != 3 * 100 * nt

    def test_assembled_row_counts_track_the_grid(self) -> None:
        case = self._synthetic_case()
        x_r, t = wave_breaking_star_grid(case, points_per_window=50)
        x_c, _ = wave_breaking_star_grid(case, points_per_window=100)
        n_reward = x_r.numel() * t.numel()
        n_coeff = x_c.numel() * t.numel()

        m_r = assemble_pinned_matrix(
            ["u_x"],
            executor=FakeExecutor({"u_x": torch.zeros(n_reward, dtype=torch.float64)}),
            context=FakeContext(),
            pinned_lhs=torch.zeros(n_reward, dtype=torch.float64),
            cache=TermColumnCache(),
            cache_generation=0,
        )
        m_c = assemble_pinned_matrix(
            ["u_x"],
            executor=FakeExecutor({"u_x": torch.zeros(n_coeff, dtype=torch.float64)}),
            context=FakeContext(),
            pinned_lhs=torch.zeros(n_coeff, dtype=torch.float64),
            cache=TermColumnCache(),
            cache_generation=1,
        )
        assert m_r.shape[0] == n_reward
        assert m_c.shape[0] == n_coeff
        assert m_r.shape[0] != m_c.shape[0]

    def test_reward_column_against_coeff_target_raises(self) -> None:
        case = self._synthetic_case()
        x_r, t = wave_breaking_star_grid(case, points_per_window=50)
        x_c, _ = wave_breaking_star_grid(case, points_per_window=100)
        n_reward = x_r.numel() * t.numel()
        n_coeff = x_c.numel() * t.numel()
        with pytest.raises(ValueError):
            assemble_pinned_matrix(
                ["u_x"],
                executor=FakeExecutor(
                    {"u_x": torch.zeros(n_reward, dtype=torch.float64)}
                ),
                context=FakeContext(),
                pinned_lhs=torch.zeros(n_coeff, dtype=torch.float64),
                cache=TermColumnCache(),
                cache_generation=0,
            )







class TestMeanOverSurvivors:

    PINNED = torch.tensor([-1.0, 1.0, -2.0, 2.0], dtype=torch.float64)
    COL_R2_ONE = torch.tensor([1.0, -1.0, 2.0, -2.0], dtype=torch.float64)
    COL_R2_PARTIAL = torch.tensor([1.0, -1.0, 0.0, 0.0], dtype=torch.float64)
    COL_NAN = torch.full((4,), float("nan"), dtype=torch.float64)
    COL_CONST = torch.tensor(
        [1.0, 1.0, 1.0, 1.0], dtype=torch.float64
    )
    TERM = "u_x"

    def _evaluator(
        self, reward_specs: list[tuple[str, torch.Tensor]]
    ) -> MultiCaseEvaluator:
        reward_bundles = [
            _bundle(name, {self.TERM: col}, self.PINNED, gen)
            for gen, (name, col) in enumerate(reward_specs)
        ]


        coeff_bundles = [
            _bundle(name, {self.TERM: col}, self.PINNED, 100 + gen)
            for gen, (name, col) in enumerate(reward_specs)
        ]
        return MultiCaseEvaluator(
            reward_bundles=reward_bundles,
            coeff_bundles=coeff_bundles,
            primary_case=reward_specs[0][0],
            sparsity_alpha=_ALPHA,
        )

    def test_mean_excludes_degenerate_case(self) -> None:

        assert math.isfinite(_r2_of(self.PINNED, self.COL_R2_ONE))
        assert math.isfinite(_r2_of(self.PINNED, self.COL_R2_PARTIAL))
        assert math.isnan(_r2_of(self.PINNED, self.COL_NAN))

        r_a = _reward_of(self.PINNED, self.COL_R2_ONE)
        r_b = _reward_of(self.PINNED, self.COL_R2_PARTIAL)
        expected_mean = (r_a + r_b) / 2.0
        count_zero_wrong = (r_a + r_b + 0.0) / 3.0

        mc = self._evaluator(
            [
                ("caseA", self.COL_R2_ONE),
                ("caseB", self.COL_R2_PARTIAL),
                ("caseC", self.COL_NAN),
            ]
        )
        result = mc.score_candidate(candidate=self.TERM, terms=[self.TERM])
        assert result.is_valid is True
        assert result.score == pytest.approx(expected_mean, rel=1e-9, abs=1e-12)

        assert result.score != pytest.approx(count_zero_wrong, rel=1e-9)

    def test_zero_reward_finite_r2_is_a_survivor(self) -> None:
        r_zero = _reward_of(self.PINNED, self.COL_CONST)
        r_good = _reward_of(self.PINNED, self.COL_R2_ONE)
        assert r_zero == pytest.approx(0.0, abs=1e-12)
        assert math.isfinite(_r2_of(self.PINNED, self.COL_CONST))
        assert math.isnan(_r2_of(self.PINNED, self.COL_NAN))

        expected_mean = (r_zero + r_good) / 2.0

        mc = self._evaluator(
            [
                ("finiteZero", self.COL_CONST),
                ("good", self.COL_R2_ONE),
                ("nanZero", self.COL_NAN),
            ]
        )
        result = mc.score_candidate(candidate=self.TERM, terms=[self.TERM])
        assert result.is_valid is True
        assert result.score == pytest.approx(expected_mean, rel=1e-9, abs=1e-12)


        assert result.score != pytest.approx(r_good, rel=1e-9)

    def test_all_cases_failing_is_invalid_not_zero(self) -> None:
        mc = self._evaluator(
            [
                ("c1", self.COL_NAN),
                ("c2", self.COL_NAN),
            ]
        )
        result = mc.score_candidate(candidate=self.TERM, terms=[self.TERM])
        assert result.is_valid is False

    def test_per_case_rewards_map_nan_for_nonsurvivors(self) -> None:
        mc = self._evaluator(
            [
                ("caseA", self.COL_R2_ONE),
                ("caseB", self.COL_R2_PARTIAL),
                ("caseC", self.COL_NAN),
            ]
        )
        rewards = mc.per_case_rewards([self.TERM])
        assert rewards["caseA"] == pytest.approx(
            _reward_of(self.PINNED, self.COL_R2_ONE), rel=1e-9, abs=1e-12
        )
        assert rewards["caseB"] == pytest.approx(
            _reward_of(self.PINNED, self.COL_R2_PARTIAL), rel=1e-9, abs=1e-12
        )
        assert math.isnan(rewards["caseC"])

    def test_case_names_in_bundle_order(self) -> None:
        mc = self._evaluator(
            [
                ("caseA", self.COL_R2_ONE),
                ("caseB", self.COL_R2_PARTIAL),
            ]
        )
        assert mc.case_names == ["caseA", "caseB"]


class TestBuildFinalResult:
    def test_final_result_stamps_pool_mean_and_shares_target_rowspace(self) -> None:

        pinned = torch.tensor([-1.0, 1.0, -2.0, 2.0, -3.0, 3.0], dtype=torch.float64)
        col = torch.tensor([1.0, -1.0, 2.0, -2.0, 3.0, -3.0], dtype=torch.float64)
        term = "u_x"
        mc = MultiCaseEvaluator(
            reward_bundles=[_bundle("caseA", {term: col}, pinned, 0)],
            coeff_bundles=[_bundle("caseA", {term: col}, pinned, 100)],
            primary_case="caseA",
            sparsity_alpha=_ALPHA,
        )

        best_reward = 0.77
        result = mc.build_final_result([term], best_reward=best_reward)

        assert result.score == pytest.approx(best_reward)
        assert result.coefficients is not None
        assert result.is_valid is True
        assert result.residuals is not None

        target = mc.result_target()
        assert target.numel() == pinned.numel()

        torch.testing.assert_close(
            target.detach().cpu().double(),
            (-pinned).double(),
            rtol=1e-6,
            atol=1e-9,
        )

        assert result.residuals.numel() == target.numel()
