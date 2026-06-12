
from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.nn import Parameter

from kd.core.evaluator import EvaluationResult
from kd.search.discover.core.batch import Batch
from kd.search.discover.engine import DiscoverEngine
from kd.search.discover.engine_types import EngineState
from kd.search.discover.tokens.library import Library, LibraryConfig
from kd.search.discover.tokens.validator import CandidateValidator
from kd.search.discover.training.strategy import BaselineState, RSPGStrategy





LIB_CONFIG = LibraryConfig(
    operators=["add", "mul", "diff_x", "diff2_x"],
    state_vars=["u"],
    coord_vars=["x", "t"],
)

MAX_LENGTH = 15
BATCH_SIZE = 4



X_TOKENS: tuple[str, ...] = ("diff_x", "u")
X_IR = "diff_x(u)"
Y_TOKENS: tuple[str, ...] = ("diff2_x", "u")
Y_IR = "diff2_x(u)"



REWARD_X_ON_A = 0.9
NMSE_X_ON_A = 0.05
COEFF_X_ON_A = 2.5

REWARD_X_ON_B = 0.4
NMSE_X_ON_B = 0.62
COEFF_X_ON_B = -1.25

REWARD_Y_ON_B = 0.5
NMSE_Y_ON_B = 0.43







class ScriptedGenerator:

    def __init__(
        self,
        library: Library,
        scripts: Sequence[Sequence[str]],
    ) -> None:
        if not scripts:
            raise ValueError("scripts must contain at least one token row.")
        self._library = library
        self._scripts: list[list[int]] = [
            [library.names.index(name) for name in row] for row in scripts
        ]
        self._call_idx = 0
        self._param = Parameter(torch.zeros(1))
        self._state: dict[str, Any] = {"scripted": True}

    @property
    def library(self) -> Library:
        return self._library

    def sample(self, batch_size: int) -> Batch:
        row = self._scripts[min(self._call_idx, len(self._scripts) - 1)]
        self._call_idx += 1
        length = len(row)
        n_tokens = len(self._library.tokens)
        actions = np.tile(
            np.asarray(row, dtype=np.int32),
            (batch_size, 1),
        )
        obs = np.zeros((batch_size, 4, length), dtype=np.float32)
        priors = np.ones((batch_size, length, n_tokens), dtype=np.float32)
        lengths = np.full(batch_size, length, dtype=np.int32)
        return Batch(actions=actions, obs=obs, priors=priors, lengths=lengths)

    def make_neglogp_and_entropy(
        self,
        batch: Batch,
        entropy_gamma: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        batch_len = batch.actions.shape[0]
        anchor = self._param.sum()
        return torch.zeros(batch_len) + anchor, torch.zeros(batch_len) + anchor

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def parameters(self) -> Iterator[Parameter]:
        yield self._param

    def state_dict(self) -> dict[str, Any]:
        return dict(self._state)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._state = dict(state_dict)

    def train(self, mode: bool = True) -> None:
        pass


@dataclass(frozen=True)
class EvalSpec:

    reward: float = 0.0
    nmse: float = 0.5
    coefficient: float = 1.0
    valid: bool = True
    error: str = ""


class TableEvaluator:

    def __init__(self, table: dict[str, EvalSpec]) -> None:
        self._table = dict(table)
        self.calls: list[str] = []

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        self.calls.append(expr)
        spec = self._table.get(expr)
        if spec is None:
            raise AssertionError(
                f"TableEvaluator got unexpected expression {expr!r}; "
                f"known: {sorted(self._table)}"
            )
        if not spec.valid:
            return EvaluationResult(
                mse=0.0,
                nmse=0.0,
                r2=0.0,
                complexity=0,
                is_valid=False,
                error_message=spec.error,
                expression=expr,
            )
        return EvaluationResult(
            mse=spec.nmse,
            nmse=spec.nmse,
            r2=spec.reward,
            complexity=1,
            is_valid=True,
            expression=expr,
            terms=[expr],
            coefficients=torch.tensor([spec.coefficient]),
            selected_indices=None,
        )


def passthrough_reward(result: EvaluationResult) -> float:
    if not result.is_valid:
        return 0.0
    return float(result.r2)


class TogglableRewardAdapter:

    def __init__(self) -> None:
        self._nonfinite = False

    def make_nonfinite(self) -> None:
        self._nonfinite = True

    def __call__(self, result: EvaluationResult) -> float:
        if not result.is_valid:
            return 0.0
        if self._nonfinite:
            return math.inf
        return float(result.r2)







@pytest.fixture
def lib() -> Library:
    return Library.from_config(LIB_CONFIG)


def _make_engine(
    lib: Library,
    scripts: Sequence[Sequence[str]],
    reward_adapter: TogglableRewardAdapter | None = None,
) -> tuple[DiscoverEngine, ScriptedGenerator]:
    generator = ScriptedGenerator(lib, scripts)
    engine = DiscoverEngine(
        generator=generator,
        strategy=RSPGStrategy(epsilon=0.5, baseline="R_e", entropy_weight=0.005),
        reward_adapter=(
            reward_adapter if reward_adapter is not None else passthrough_reward
        ),
        validator=CandidateValidator(lib, max_length=MAX_LENGTH),
        deduplicator=None,
        batch_size=BATCH_SIZE,
    )
    return engine, generator


def _seed_champion_on_scale_a(
    engine: DiscoverEngine,
) -> TableEvaluator:
    eval_a = TableEvaluator(
        {
            X_IR: EvalSpec(
                reward=REWARD_X_ON_A,
                nmse=NMSE_X_ON_A,
                coefficient=COEFF_X_ON_A,
            ),
        }
    )
    engine.run_cycle(eval_a, n_iterations=1)
    assert engine.best_expression == X_IR, "precondition: X must win cycle A"
    assert engine.best_reward == pytest.approx(REWARD_X_ON_A), (
        "precondition: champion reward must sit at the inflated scale-A "
        "value (distinguishable from both 0.0 and any scale-B value)"
    )
    best = engine.best_result
    assert best is not None and best.is_valid, (
        "precondition: champion result must be a valid scale-A evaluation"
    )
    assert best.nmse == pytest.approx(NMSE_X_ON_A), (
        "precondition: champion result fields must come from scale A"
    )
    return eval_a







@pytest.mark.unit
class TestRebaseScaleDrift:

    def test_a_stale_champion_repriced_lets_new_candidate_win(
        self, lib: Library
    ) -> None:
        engine, _ = _make_engine(lib, scripts=[X_TOKENS, Y_TOKENS])
        _seed_champion_on_scale_a(engine)

        eval_b = TableEvaluator(
            {
                X_IR: EvalSpec(
                    reward=REWARD_X_ON_B,
                    nmse=NMSE_X_ON_B,
                    coefficient=COEFF_X_ON_B,
                ),
                Y_IR: EvalSpec(reward=REWARD_Y_ON_B, nmse=NMSE_Y_ON_B),
            }
        )
        engine.run_cycle(eval_b, n_iterations=1)

        assert engine.best_expression == Y_IR, (
            f"on scale B the champion X reprices to {REWARD_X_ON_B} and the "
            f"new candidate Y at {REWARD_Y_ON_B} must take over; a stale "
            f"scale-A reward must not suppress it"
        )
        assert engine.best_reward == pytest.approx(REWARD_Y_ON_B)

    def test_d_surviving_champion_refreshes_reward_and_result(
        self, lib: Library
    ) -> None:
        engine, _ = _make_engine(lib, scripts=[X_TOKENS, Y_TOKENS])
        _seed_champion_on_scale_a(engine)

        eval_b = TableEvaluator(
            {
                X_IR: EvalSpec(
                    reward=REWARD_X_ON_B,
                    nmse=NMSE_X_ON_B,
                    coefficient=COEFF_X_ON_B,
                ),
                Y_IR: EvalSpec(reward=0.2, nmse=1.8, coefficient=0.1),
            }
        )
        engine.run_cycle(eval_b, n_iterations=1)

        assert engine.best_expression == X_IR
        assert engine.best_reward == pytest.approx(REWARD_X_ON_B), (
            "surviving champion must be repriced to the current scale"
        )
        best = engine.best_result
        assert best is not None
        assert best.nmse == pytest.approx(NMSE_X_ON_B), (
            "best_result must be the fresh scale-B evaluation, not the "
            "stale scale-A one"
        )
        assert best.coefficients is not None
        torch.testing.assert_close(
            best.coefficients,
            torch.tensor([COEFF_X_ON_B]),
            rtol=1e-6,
            atol=1e-8,
        )


@pytest.mark.unit
class TestRebaseInvalidChampion:

    def test_b_invalid_champion_releases_gate_for_valid_candidate(
        self, lib: Library
    ) -> None:
        engine, _ = _make_engine(lib, scripts=[X_TOKENS, Y_TOKENS])
        _seed_champion_on_scale_a(engine)

        eval_b = TableEvaluator(
            {
                X_IR: EvalSpec(valid=False, error="X invalid on scale B"),
                Y_IR: EvalSpec(reward=0.3, nmse=NMSE_Y_ON_B),
            }
        )
        engine.run_cycle(eval_b, n_iterations=1)

        assert engine.best_expression == Y_IR, (
            "invalid champion must release the gate so a valid candidate can take over"
        )
        assert engine.best_reward == pytest.approx(0.3)

    def test_c_invalid_champion_without_valid_candidates(self, lib: Library) -> None:
        engine, _ = _make_engine(lib, scripts=[X_TOKENS, Y_TOKENS])
        _seed_champion_on_scale_a(engine)

        eval_b = TableEvaluator(
            {
                X_IR: EvalSpec(valid=False, error="X invalid on scale B"),
                Y_IR: EvalSpec(valid=False, error="Y invalid on scale B"),
            }
        )
        engine.run_cycle(eval_b, n_iterations=1)

        assert engine.best_expression == X_IR, (
            "champion expression must be kept even when invalid on the new scale"
        )
        assert engine.best_reward == 0.0, (
            "invalid champion must release the gate to INITIAL_BEST_REWARD"
        )
        best = engine.best_result
        assert best is not None
        assert best.is_valid is False, (
            "best_result must honestly record the invalid re-evaluation"
        )
        assert best.error_message == "X invalid on scale B", (
            "the recorded invalid result must come from scale B's "
            "evaluation of the champion"
        )


@pytest.mark.unit
class TestRebaseFinitenessContract:

    def test_e_nonfinite_rebased_reward_raises_value_error(self, lib: Library) -> None:
        adapter = TogglableRewardAdapter()
        engine, _ = _make_engine(
            lib,
            scripts=[X_TOKENS],
            reward_adapter=adapter,
        )
        _seed_champion_on_scale_a(engine)

        adapter.make_nonfinite()
        eval_b = TableEvaluator(
            {X_IR: EvalSpec(reward=REWARD_X_ON_B, nmse=NMSE_X_ON_B)}
        )
        with pytest.raises(ValueError):
            engine.run_cycle(eval_b, n_iterations=0)


@pytest.mark.unit
class TestRebaseAfterRestore:

    def test_f_restored_placeholder_repriced_on_next_cycle(self, lib: Library) -> None:
        engine, _ = _make_engine(lib, scripts=[Y_TOKENS])
        restored = EngineState(
            controller_state_dict={"scripted": True},
            baseline_state=BaselineState(),
            best_reward=REWARD_X_ON_A,
            best_expression=X_IR,
            best_result_terms=[X_IR],
            best_result_coefficients=[COEFF_X_ON_A],
        )
        engine.state = restored

        placeholder = engine.best_result
        assert placeholder is not None, "precondition: placeholder rebuilt"
        assert placeholder.is_valid is False, (
            "precondition: restored best_result is the synthetic invalid placeholder"
        )
        assert math.isinf(placeholder.nmse), (
            "precondition: placeholder carries sentinel metrics"
        )
        assert engine.best_reward == pytest.approx(REWARD_X_ON_A), (
            "precondition: restored reward sits at the saved high value"
        )

        live = TableEvaluator(
            {
                X_IR: EvalSpec(reward=0.45, nmse=0.33, coefficient=1.5),
            }
        )
        engine.run_cycle(live, n_iterations=0)

        best = engine.best_result
        assert best is not None
        assert best.is_valid is True, (
            "after run_cycle the placeholder must be replaced by a real "
            "evaluation on the live evaluator"
        )
        assert best.nmse == pytest.approx(0.33)
        assert engine.best_reward == pytest.approx(0.45), (
            "restored reward must be repriced on the live scale"
        )
        assert engine.best_expression == X_IR







@pytest.mark.unit
class TestRebaseRegressionLocks:

    def test_g_fresh_engine_zero_iterations_never_touches_evaluator(
        self, lib: Library
    ) -> None:
        engine, _ = _make_engine(lib, scripts=[Y_TOKENS])
        evaluator = TableEvaluator({Y_IR: EvalSpec(reward=0.5)})

        engine.run_cycle(evaluator, n_iterations=0)

        assert evaluator.calls == [], (
            "a fresh engine (best_expression == '') must not send any "
            "evaluation request during run_cycle entry"
        )

    def test_g_fresh_engine_never_evaluates_empty_expression(
        self, lib: Library
    ) -> None:
        engine, _ = _make_engine(lib, scripts=[Y_TOKENS])
        evaluator = TableEvaluator({Y_IR: EvalSpec(reward=0.5)})

        engine.run_cycle(evaluator, n_iterations=1)

        assert "" not in evaluator.calls
        assert evaluator.calls == [Y_IR] * BATCH_SIZE, (
            "first cycle must evaluate exactly the proposed batch "
            "(dedup disabled -> one call per row), nothing else"
        )

    def test_h_same_evaluator_two_cycles_is_idempotent(self, lib: Library) -> None:
        engine, _ = _make_engine(lib, scripts=[X_TOKENS])
        evaluator = TableEvaluator(
            {X_IR: EvalSpec(reward=0.5, nmse=0.4, coefficient=1.0)}
        )

        engine.run_cycle(evaluator, n_iterations=1)
        assert engine.best_expression == X_IR, "precondition: X won cycle 1"
        assert engine.best_reward == pytest.approx(0.5)
        first_best = engine.best_result
        assert first_best is not None

        engine.run_cycle(evaluator, n_iterations=1)

        assert engine.best_expression == X_IR
        assert engine.best_reward == pytest.approx(0.5), (
            "same scale -> repricing is idempotent"
        )
        second_best = engine.best_result
        assert second_best is not None
        assert second_best.nmse == pytest.approx(first_best.nmse)

    def test_i_within_cycle_ratchet_unchanged(self, lib: Library) -> None:
        engine, _ = _make_engine(lib, scripts=[X_TOKENS, Y_TOKENS])
        evaluator = TableEvaluator(
            {
                X_IR: EvalSpec(reward=0.8, nmse=0.1),
                Y_IR: EvalSpec(reward=0.3, nmse=0.9),
            }
        )

        engine.run_cycle(evaluator, n_iterations=2)

        assert engine.best_expression == X_IR, (
            "same-scale ratchet semantics must be preserved inside a cycle"
        )
        assert engine.best_reward == pytest.approx(0.8)







@pytest.mark.unit
class TestRebaseTieAndGridSemantics:

    def test_tie_keeps_incumbent_champion(self, lib: Library) -> None:
        engine, _ = _make_engine(lib, scripts=[X_TOKENS, Y_TOKENS])
        _seed_champion_on_scale_a(engine)

        tie = 0.5
        eval_b = TableEvaluator(
            {
                X_IR: EvalSpec(reward=tie, nmse=NMSE_X_ON_B),
                Y_IR: EvalSpec(reward=tie, nmse=NMSE_Y_ON_B),
            }
        )
        engine.run_cycle(eval_b, n_iterations=1)

        assert engine.best_reward == tie
        assert engine.best_expression == X_IR, (
            "exact tie must keep the incumbent champion (kd contract; "
            "documented delta vs DSO round-end tie-breaking)"
        )

    def test_rebase_reward_lands_on_batch_float32_grid(self, lib: Library) -> None:
        off_grid = 0.7
        engine, _ = _make_engine(lib, scripts=[X_TOKENS])
        evaluator = TableEvaluator({X_IR: EvalSpec(reward=off_grid, nmse=0.4)})

        engine.run_cycle(evaluator, n_iterations=1)
        first_reward = engine.best_reward
        assert first_reward == float(np.float32(off_grid)), (
            "precondition: the batch path stores the float32-grid value"
        )

        engine.run_cycle(evaluator, n_iterations=0)

        assert engine.best_reward == first_reward, (
            "rebase on the same scale must reproduce the batch-path value "
            "bit-for-bit (float32 grid), not a float64 ULP-shifted neighbor"
        )







@pytest.mark.unit
class TestRebasePublicName:

    def test_j_public_rebase_best_reprices_champion(self, lib: Library) -> None:
        engine, _ = _make_engine(lib, scripts=[X_TOKENS])
        _seed_champion_on_scale_a(engine)

        eval_b = TableEvaluator(
            {
                X_IR: EvalSpec(
                    reward=REWARD_X_ON_B,
                    nmse=NMSE_X_ON_B,
                    coefficient=COEFF_X_ON_B,
                ),
            }
        )
        engine.rebase_best(eval_b)

        assert eval_b.calls == [X_IR]
        assert engine.best_reward == pytest.approx(REWARD_X_ON_B)
        assert engine.best_expression == X_IR
        best = engine.best_result
        assert best is not None
        assert best.nmse == pytest.approx(NMSE_X_ON_B)

    def test_j_private_alias_is_preserved(self) -> None:
        assert DiscoverEngine._rebase_best is DiscoverEngine.rebase_best
