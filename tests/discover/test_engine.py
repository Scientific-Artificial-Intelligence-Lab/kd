
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from kd.core.evaluator import EvaluationResult
from kd.search.discover.controller.lstm import LSTMController
from kd.search.discover.engine import DiscoverEngine, EngineState
from kd.search.discover.evaluation.dedup import Deduplicator
from kd.search.discover.evaluation.reward import compute_reward
from kd.search.discover.tokens.library import Library, LibraryConfig
from kd.search.discover.tokens.prior import (
    DiffChildConstraint,
    LengthConstraint,
    PriorSystem,
)
from kd.search.discover.tokens.validator import CandidateValidator
from kd.search.discover.training.strategy import BaselineState, RSPGStrategy

if TYPE_CHECKING:
    from kd.search.discover.core.batch import Batch
    from kd.search.discover.engine_types import BoolArray, Generator





BURGERS_CONFIG = LibraryConfig(
    operators=["add", "mul", "sub", "div", "sin", "cos", "diff_x", "diff2_x"],
    state_vars=["u"],
    coord_vars=["x", "t"],
)

MAX_LENGTH = 15
MIN_LENGTH = 4
BATCH_SIZE = 16
SEED = 42






class MockEvaluator:

    def __init__(self, default_nmse: float = 0.5) -> None:
        self.default_nmse = default_nmse
        self.evaluated: list[str] = []

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        self.evaluated.append(expr)
        return EvaluationResult(
            mse=self.default_nmse,
            nmse=self.default_nmse,
            r2=max(0.0, 1.0 - self.default_nmse),
            complexity=3,
            is_valid=True,
            expression=expr,
        )


class RankedEvaluator:

    def __init__(self) -> None:
        self.evaluated: list[str] = []
        self._call_count = 0

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        self.evaluated.append(expr)
        nmse = 0.01 if self._call_count == 0 else 5.0 + self._call_count
        self._call_count += 1
        return EvaluationResult(
            mse=nmse,
            nmse=nmse,
            r2=max(0.0, 1.0 - nmse),
            complexity=3,
            is_valid=True,
            expression=expr,
        )


class InvalidEvaluator:

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        return EvaluationResult(
            mse=0.0,
            nmse=0.0,
            r2=1.0,
            complexity=0,
            is_valid=False,
            expression=expr,
        )







@pytest.fixture
def lib() -> Library:
    return Library.from_config(BURGERS_CONFIG)


@pytest.fixture
def prior_system(lib: Library) -> PriorSystem:
    return PriorSystem(
        lib,
        [
            LengthConstraint(lib, min_=MIN_LENGTH, max_=MAX_LENGTH),
            DiffChildConstraint(lib),
        ],
    )


@pytest.fixture
def controller(
    lib: Library,
    prior_system: PriorSystem,
) -> LSTMController:
    torch.manual_seed(SEED)
    return LSTMController(
        library=lib,
        prior_system=prior_system,
        num_units=16,
        num_layers=1,
        embedding_dim=4,
    )


@pytest.fixture
def strategy() -> RSPGStrategy:
    return RSPGStrategy(epsilon=0.5, baseline="R_e", entropy_weight=0.005)


@pytest.fixture
def validator(lib: Library) -> CandidateValidator:
    return CandidateValidator(lib, max_length=MAX_LENGTH)


@pytest.fixture
def deduplicator(lib: Library) -> Deduplicator:
    return Deduplicator(lib)


@pytest.fixture
def engine(
    controller: LSTMController,
    strategy: RSPGStrategy,
    validator: CandidateValidator,
    deduplicator: Deduplicator,
) -> DiscoverEngine:
    torch.manual_seed(SEED)
    return DiscoverEngine(
        generator=controller,
        strategy=strategy,
        reward_adapter=compute_reward,
        validator=validator,
        deduplicator=deduplicator,
        batch_size=BATCH_SIZE,
    )


@pytest.fixture
def engine_no_dedup(
    controller: LSTMController,
    strategy: RSPGStrategy,
    validator: CandidateValidator,
) -> DiscoverEngine:
    torch.manual_seed(SEED)
    return DiscoverEngine(
        generator=controller,
        strategy=strategy,
        reward_adapter=compute_reward,
        validator=validator,
        deduplicator=None,
        batch_size=BATCH_SIZE,
    )







class TestStepWiseCycle:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_full_cycle_completes(self, engine: DiscoverEngine) -> None:
        torch.manual_seed(SEED)
        ir_strings = engine.propose()
        assert isinstance(ir_strings, list)
        assert len(ir_strings) > 0

        evaluator = MockEvaluator(default_nmse=0.5)
        results = [evaluator.evaluate_expression(ir) for ir in ir_strings]
        engine.receive_results(results)
        engine.update()

    @pytest.mark.unit
    def test_propose_twice_without_update_raises(
        self,
        engine: DiscoverEngine,
    ) -> None:
        torch.manual_seed(SEED)
        engine.propose()
        with pytest.raises(RuntimeError):
            engine.propose()

    @pytest.mark.unit
    def test_receive_before_propose_raises(
        self,
        engine: DiscoverEngine,
    ) -> None:
        with pytest.raises(RuntimeError):
            engine.receive_results([])

    @pytest.mark.unit
    def test_update_before_receive_raises(
        self,
        engine: DiscoverEngine,
    ) -> None:
        torch.manual_seed(SEED)
        engine.propose()
        with pytest.raises(RuntimeError):
            engine.update()

    @pytest.mark.unit
    def test_receive_wrong_count_raises(
        self,
        engine: DiscoverEngine,
    ) -> None:
        torch.manual_seed(SEED)
        ir_strings = engine.propose()
        evaluator = MockEvaluator()

        if len(ir_strings) > 1:
            wrong = [evaluator.evaluate_expression(ir) for ir in ir_strings[:-1]]
        else:
            wrong = [evaluator.evaluate_expression("x")] * 2
        with pytest.raises(ValueError):
            engine.receive_results(wrong)

    @pytest.mark.unit
    def test_cycle_resets_pending_state(
        self,
        engine: DiscoverEngine,
    ) -> None:
        torch.manual_seed(SEED)
        evaluator = MockEvaluator()


        irs = engine.propose()
        engine.receive_results(
            [evaluator.evaluate_expression(ir) for ir in irs],
        )
        engine.update()


        irs2 = engine.propose()
        assert isinstance(irs2, list)
        assert len(irs2) > 0







class TestScatterRewards:

    @pytest.mark.unit
    def test_all_invalid_results_produce_zero_pg_loss(
        self,
        engine: DiscoverEngine,
    ) -> None:
        torch.manual_seed(SEED)
        ir_strings = engine.propose()
        results = [
            EvaluationResult(
                mse=0.0,
                nmse=0.0,
                r2=1.0,
                complexity=0,
                is_valid=False,
                expression=ir,
            )
            for ir in ir_strings
        ]
        engine.receive_results(results)
        engine.update()
        metrics = engine.last_metrics
        assert metrics["pg_loss"] == 0.0
        assert metrics["reward_max"] == 0.0
        assert metrics["n_eval_valid"] == 0.0

    @pytest.mark.unit
    def test_all_valid_results_produce_nonzero_pg_loss(
        self,
        engine: DiscoverEngine,
    ) -> None:
        torch.manual_seed(SEED)
        metrics = engine.run_iteration(RankedEvaluator())
        assert metrics["pg_loss"] != 0.0
        assert metrics["n_eval_valid"] > 0.0

    @pytest.mark.unit
    def test_best_expression_matches_best_evaluation(
        self,
        engine: DiscoverEngine,
    ) -> None:
        evaluator = RankedEvaluator()
        torch.manual_seed(SEED)
        engine.run_iteration(evaluator)

        assert engine.best_expression == evaluator.evaluated[0]

    @pytest.mark.unit
    def test_evaluator_receives_only_unique_irs(
        self,
        engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluator()
        torch.manual_seed(SEED)
        engine.run_iteration(evaluator)
        assert len(evaluator.evaluated) == len(set(evaluator.evaluated))







class TestRunIteration:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_returns_metrics_dict(self, engine: DiscoverEngine) -> None:
        evaluator = MockEvaluator(default_nmse=0.5)
        torch.manual_seed(SEED)
        metrics = engine.run_iteration(evaluator)
        assert isinstance(metrics, dict)
        for key, value in metrics.items():
            assert isinstance(key, str), f"key {key!r} not str"
            assert isinstance(value, (int, float)), f"{key}: {type(value)}"

    @pytest.mark.unit
    def test_metrics_contain_expected_keys(
        self,
        engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluator(default_nmse=0.5)
        torch.manual_seed(SEED)
        metrics = engine.run_iteration(evaluator)
        for key in ("reward_max", "best_reward", "pg_loss", "total_loss"):
            assert key in metrics, f"Missing key: {key}"

    @pytest.mark.unit
    def test_evaluator_called_for_each_unique_ir(
        self,
        engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluator()
        torch.manual_seed(SEED)
        engine.run_iteration(evaluator)
        assert len(evaluator.evaluated) > 0
        assert len(evaluator.evaluated) <= BATCH_SIZE

    @pytest.mark.unit
    def test_all_invalid_evaluations_still_completes(
        self,
        engine: DiscoverEngine,
    ) -> None:
        torch.manual_seed(SEED)
        metrics = engine.run_iteration(InvalidEvaluator())
        assert isinstance(metrics, dict)

        assert metrics["best_reward"] == 0.0

    @pytest.mark.unit
    def test_evaluator_exception_clears_pending(
        self,
        engine: DiscoverEngine,
    ) -> None:

        class BrokenEvaluator:
            def evaluate_expression(self, expr: str) -> EvaluationResult:
                raise RuntimeError("eval failed")

        torch.manual_seed(SEED)
        with pytest.raises(RuntimeError, match="eval failed"):
            engine.run_iteration(BrokenEvaluator())


        torch.manual_seed(SEED + 10)
        ir_strings = engine.propose()
        assert len(ir_strings) > 0

    @pytest.mark.unit
    def test_run_multiple_iterations(
        self,
        engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluator(default_nmse=0.5)
        torch.manual_seed(SEED)
        final_state = engine.run(evaluator, n_iterations=3)
        assert isinstance(final_state, EngineState)
        assert final_state.best_reward > 0.0
        assert len(final_state.best_expression) > 0

        assert len(evaluator.evaluated) >= 3







class TestNullDeduplicator:

    @pytest.mark.unit
    def test_cycle_without_dedup(
        self,
        engine_no_dedup: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluator()
        torch.manual_seed(SEED)
        metrics = engine_no_dedup.run_iteration(evaluator)
        assert isinstance(metrics, dict)
        assert "best_reward" in metrics

    @pytest.mark.unit
    def test_no_dedup_evaluates_all_valid(
        self,
        engine_no_dedup: DiscoverEngine,
    ) -> None:
        torch.manual_seed(SEED)
        ir_strings = engine_no_dedup.propose()

        evaluator = MockEvaluator()
        results = [evaluator.evaluate_expression(ir) for ir in ir_strings]
        engine_no_dedup.receive_results(results)
        engine_no_dedup.update()

        assert len(evaluator.evaluated) == len(ir_strings)

        assert len(ir_strings) >= 1







class TestStateManagement:

    @pytest.mark.unit
    def test_state_type(self, engine: DiscoverEngine) -> None:
        state = engine.state
        assert isinstance(state, EngineState)
        assert isinstance(state.controller_state_dict, dict)
        assert isinstance(state.baseline_state, BaselineState)
        assert isinstance(state.best_reward, float)
        assert isinstance(state.best_expression, str)

    @pytest.mark.unit
    def test_state_roundtrip(self, engine: DiscoverEngine) -> None:
        evaluator = MockEvaluator(default_nmse=0.1)
        torch.manual_seed(SEED)
        engine.run_iteration(evaluator)

        state_after_iter1 = engine.state
        saved_weights = copy.deepcopy(state_after_iter1.controller_state_dict)
        saved_best = state_after_iter1.best_reward


        engine.run_iteration(evaluator)


        engine.state = state_after_iter1
        restored = engine.state

        assert restored.best_reward == saved_best
        for key in saved_weights:
            assert torch.equal(
                restored.controller_state_dict[key],
                saved_weights[key],
            ), f"Weight mismatch for {key}"

    @pytest.mark.unit
    def test_best_reward_tracks_maximum(
        self,
        engine: DiscoverEngine,
    ) -> None:
        evaluator = RankedEvaluator()
        torch.manual_seed(SEED)
        engine.run_iteration(evaluator)
        assert engine.best_reward > 0.0
        assert isinstance(engine.best_expression, str)
        assert len(engine.best_expression) > 0

    @pytest.mark.unit
    def test_best_expression_is_ir_string(
        self,
        engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluator(default_nmse=0.1)
        torch.manual_seed(SEED)
        engine.run_iteration(evaluator)
        expr = engine.best_expression

        assert isinstance(expr, str)
        assert len(expr) > 0







class TestInvariants:

    @pytest.mark.unit
    def test_propose_length_le_batch_size(
        self,
        engine: DiscoverEngine,
    ) -> None:
        torch.manual_seed(SEED)
        ir_strings = engine.propose()
        assert len(ir_strings) <= BATCH_SIZE

    @pytest.mark.unit
    def test_propose_returns_nonempty_strings(
        self,
        engine: DiscoverEngine,
    ) -> None:
        torch.manual_seed(SEED)
        ir_strings = engine.propose()
        for ir in ir_strings:
            assert isinstance(ir, str)
            assert len(ir) > 0

    @pytest.mark.unit
    def test_params_change_after_cycle(
        self,
        engine: DiscoverEngine,
    ) -> None:
        state_before = copy.deepcopy(engine.state.controller_state_dict)

        evaluator = MockEvaluator(default_nmse=0.1)
        torch.manual_seed(SEED)
        engine.run_iteration(evaluator)

        state_after = engine.state.controller_state_dict
        any_changed = any(
            not torch.equal(state_before[k], state_after[k]) for k in state_before
        )
        assert any_changed, "No parameters changed after run_iteration"

    @pytest.mark.unit
    def test_validator_filters_before_evaluation(
        self,
        engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluator()
        torch.manual_seed(SEED)
        engine.run_iteration(evaluator)


        assert 0 < len(evaluator.evaluated) <= BATCH_SIZE

    @pytest.mark.unit
    def test_run_produces_finite_metrics(
        self,
        engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluator()
        torch.manual_seed(SEED)
        metrics = engine.run_iteration(evaluator)
        for key, value in metrics.items():
            assert np.isfinite(value), f"{key} is not finite: {value}"







class MixedValidityEvaluator:

    def __init__(self, n_valid: int = 1, valid_nmse: float = 0.1) -> None:
        self.n_valid = n_valid
        self.valid_nmse = valid_nmse
        self.evaluated: list[str] = []
        self._call_count = 0

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        self.evaluated.append(expr)
        idx = self._call_count
        self._call_count += 1
        if idx < self.n_valid:
            return EvaluationResult(
                mse=self.valid_nmse,
                nmse=self.valid_nmse,
                r2=max(0.0, 1.0 - self.valid_nmse),
                complexity=3,
                is_valid=True,
                expression=expr,
            )
        return EvaluationResult(
            mse=0.0,
            nmse=0.0,
            r2=1.0,
            complexity=0,
            is_valid=False,
            expression=expr,
        )


class TestInvalidFilter:

    @pytest.mark.unit
    def test_update_passes_valid_mask_to_strategy(
        self,
        controller: LSTMController,
        strategy: RSPGStrategy,
        validator: CandidateValidator,
        deduplicator: Deduplicator,
    ) -> None:
        torch.manual_seed(SEED)
        engine = DiscoverEngine(
            generator=controller,
            strategy=strategy,
            reward_adapter=compute_reward,
            validator=validator,
            deduplicator=deduplicator,
            batch_size=BATCH_SIZE,
        )
        evaluator = MockEvaluator(default_nmse=0.5)


        ir_strings = engine.propose()
        results = [evaluator.evaluate_expression(ir) for ir in ir_strings]
        engine.receive_results(results)


        original_train_step = strategy.train_step
        captured_kwargs: dict[str, object] = {}

        def spy_train_step(
            controller: Generator,
            batch: Batch,
            rewards: torch.Tensor,
            baseline_state: BaselineState,
            **kwargs: BoolArray | None,
        ) -> tuple[dict[str, float], BaselineState]:
            captured_kwargs.update(kwargs)
            return original_train_step(
                controller,
                batch,
                rewards,
                baseline_state,
                **kwargs,
            )

        strategy.train_step = spy_train_step
        engine.update()


        assert "valid_mask" in captured_kwargs
        mask = captured_kwargs["valid_mask"]
        assert mask is not None
        assert hasattr(mask, "dtype")

    @pytest.mark.unit
    def test_all_invalid_batch_does_not_crash(
        self,
        engine: DiscoverEngine,
    ) -> None:
        torch.manual_seed(SEED)
        metrics = engine.run_iteration(InvalidEvaluator())
        assert isinstance(metrics, dict)

    @pytest.mark.unit
    def test_all_invalid_batch_metrics_report_zero_eval_valid(
        self,
        engine: DiscoverEngine,
    ) -> None:
        torch.manual_seed(SEED)
        metrics = engine.run_iteration(InvalidEvaluator())
        assert "n_eval_valid" in metrics
        assert metrics["n_eval_valid"] == 0.0

    @pytest.mark.unit
    def test_mixed_batch_trains_only_on_valid(
        self,
        engine: DiscoverEngine,
    ) -> None:
        evaluator = MixedValidityEvaluator(n_valid=1, valid_nmse=0.1)
        torch.manual_seed(SEED)
        engine.run_iteration(evaluator)


        assert engine.best_reward > 0.0

        assert engine.best_expression == evaluator.evaluated[0]

    @pytest.mark.unit
    def test_metrics_include_n_invalid_in_topk(
        self,
        engine: DiscoverEngine,
    ) -> None:
        evaluator = MixedValidityEvaluator(n_valid=1, valid_nmse=0.5)
        torch.manual_seed(SEED)
        metrics = engine.run_iteration(evaluator)
        assert "n_invalid_in_topk" in metrics
        assert isinstance(metrics["n_invalid_in_topk"], (int, float))

    @pytest.mark.unit
    def test_best_reward_only_from_valid_expressions(
        self,
        engine: DiscoverEngine,
    ) -> None:
        torch.manual_seed(SEED)
        engine.run_iteration(InvalidEvaluator())

        assert engine.best_reward == 0.0
        assert engine.best_expression == ""

    @pytest.mark.unit
    def test_integration_propose_receive_update_with_invalid(
        self,
        controller: LSTMController,
        strategy: RSPGStrategy,
        validator: CandidateValidator,
        deduplicator: Deduplicator,
    ) -> None:
        torch.manual_seed(SEED)
        engine = DiscoverEngine(
            generator=controller,
            strategy=strategy,
            reward_adapter=compute_reward,
            validator=validator,
            deduplicator=deduplicator,
            batch_size=BATCH_SIZE,
        )


        evaluator = MixedValidityEvaluator(n_valid=2, valid_nmse=0.2)
        ir_strings = engine.propose()
        results = [evaluator.evaluate_expression(ir) for ir in ir_strings]
        engine.receive_results(results)
        engine.update()


        assert engine.best_reward > 0.0


        evaluator2 = MockEvaluator(default_nmse=0.3)
        ir_strings2 = engine.propose()
        results2 = [evaluator2.evaluate_expression(ir) for ir in ir_strings2]
        engine.receive_results(results2)
        engine.update()

    @pytest.mark.unit
    def test_consecutive_all_invalid_iterations(
        self,
        engine: DiscoverEngine,
    ) -> None:
        torch.manual_seed(SEED)
        for _ in range(3):
            metrics = engine.run_iteration(InvalidEvaluator())
            assert isinstance(metrics, dict)

        assert engine.best_reward == 0.0
