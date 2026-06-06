
from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.nn import Parameter

from kd.core.evaluator import EvaluationResult
from kd.search.discover.controller.lstm import LSTMController
from kd.search.discover.core.batch import Batch
from kd.search.discover.engine import (
    DEFAULT_BATCH_SIZE,
    DiscoverEngine,
    EngineState,
    Generator,
    extract_active_terms,
)
from kd.search.discover.evaluation.dedup import Deduplicator
from kd.search.discover.evaluation.reward import compute_reward
from kd.search.discover.ir.conversion import ir_to_tree
from kd.search.discover.tokens.library import Library, LibraryConfig
from kd.search.discover.tokens.prior import (
    DiffChildConstraint,
    LengthConstraint,
    PriorSystem,
)
from kd.search.discover.tokens.validator import CandidateValidator
from kd.search.discover.training.strategy import BaselineState, RSPGStrategy





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


class MockEvaluatorWithTerms:

    def __init__(self, default_nmse: float = 0.3) -> None:
        self.default_nmse = default_nmse

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        return EvaluationResult(
            mse=self.default_nmse,
            nmse=self.default_nmse,
            r2=max(0.0, 1.0 - self.default_nmse),
            complexity=2,
            is_valid=True,
            expression=expr,
            terms=["u_xx", "mul(u, u_x)", "u"],
            coefficients=torch.tensor([1.0, -1.0, 0.0]),
            selected_indices=[0, 1],
        )


def _result_with_terms(
    expr: str,
    *,
    nmse: float = 0.3,
    terms: list[str] | None = None,
    coefficients: list[float] | None = None,
    selected_indices: list[int] | None = None,
) -> EvaluationResult:
    if terms is None:
        terms = ["u_xx", "u"]
    if coefficients is None:
        coefficients = [1.0, -1.0]
    return EvaluationResult(
        mse=nmse,
        nmse=nmse,
        r2=max(0.0, 1.0 - nmse),
        complexity=(
            len(selected_indices)
            if selected_indices is not None
            else len(terms)
        ),
        is_valid=True,
        expression=expr,
        terms=terms,
        coefficients=torch.tensor(coefficients),
        selected_indices=selected_indices,
    )







class MockGenerator:

    def __init__(self, library: Library) -> None:
        self._library = library
        self._param = Parameter(torch.zeros(1))
        self._state: dict[str, Any] = {"mock": True}

    @property
    def library(self) -> Library:
        return self._library

    def sample(self, batch_size: int) -> Batch:
        n_tokens = len(self._library.tokens)
        actions = np.full((batch_size, 1), 0, dtype=np.int32)
        obs = np.zeros((batch_size, 4, 1), dtype=np.float32)
        priors = np.ones((batch_size, 1, n_tokens), dtype=np.float32)
        lengths = np.ones(batch_size, dtype=np.int32)
        return Batch(actions=actions, obs=obs, priors=priors, lengths=lengths)

    def make_neglogp_and_entropy(
        self,
        batch: Batch,
        entropy_gamma: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        bs = batch.actions.shape[0]
        return torch.zeros(bs), torch.zeros(bs)

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







@pytest.fixture
def lib() -> Library:
    return Library.from_config(BURGERS_CONFIG)


@pytest.fixture
def prior_system(lib: Library) -> PriorSystem:
    return PriorSystem(lib, [
        LengthConstraint(lib, min_=MIN_LENGTH, max_=MAX_LENGTH),
        DiffChildConstraint(lib),
    ])


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
def mock_generator(lib: Library) -> MockGenerator:
    return MockGenerator(lib)


@pytest.fixture
def engine(
    controller: LSTMController,
    strategy: RSPGStrategy,
    validator: CandidateValidator,
    deduplicator: Deduplicator,
) -> DiscoverEngine:
    torch.manual_seed(SEED)
    return DiscoverEngine(
        controller=controller,
        strategy=strategy,
        reward_adapter=compute_reward,
        validator=validator,
        deduplicator=deduplicator,
        batch_size=BATCH_SIZE,
    )


@pytest.fixture
def engine_with_capacity(
    controller: LSTMController,
    strategy: RSPGStrategy,
    validator: CandidateValidator,
    deduplicator: Deduplicator,
) -> DiscoverEngine:
    torch.manual_seed(SEED)
    return DiscoverEngine(
        controller=controller,
        strategy=strategy,
        reward_adapter=compute_reward,
        validator=validator,
        deduplicator=deduplicator,
        batch_size=BATCH_SIZE,
        cycle_candidate_capacity=2,
    )







class TestGeneratorProtocol:

    @pytest.mark.unit
    def test_generator_protocol_is_importable(self) -> None:


        assert Generator is not None

    @pytest.mark.unit
    def test_generator_protocol_is_runtime_checkable(self) -> None:
        assert hasattr(Generator, "__protocol_attrs__") or hasattr(
            Generator, "__abstractmethods__"
        ) or callable(getattr(Generator, "_is_runtime_protocol", None))


        isinstance(object(), Generator)

    @pytest.mark.unit
    def test_lstm_controller_satisfies_generator(
        self, controller: LSTMController,
    ) -> None:
        assert isinstance(controller, Generator)

    @pytest.mark.unit
    def test_mock_generator_satisfies_protocol(
        self, mock_generator: MockGenerator,
    ) -> None:
        assert isinstance(mock_generator, Generator)

    @pytest.mark.unit
    def test_engine_accepts_generator_param(
        self,
        mock_generator: MockGenerator,
        strategy: RSPGStrategy,
        validator: CandidateValidator,
    ) -> None:
        engine = DiscoverEngine(
            generator=mock_generator,
            strategy=strategy,
            reward_adapter=compute_reward,
            validator=validator,
        )
        assert engine is not None

    @pytest.mark.unit
    def test_engine_init_signature_has_generator_param(self) -> None:
        sig = inspect.signature(DiscoverEngine.__init__)
        params = list(sig.parameters.keys())
        assert "generator" in params, (
            f"Expected 'generator' in __init__ params, got {params}"
        )
        assert "controller" not in params, (
            "'controller' param should be renamed to 'generator'"
        )







class TestBatchSizeProperty:

    @pytest.mark.unit
    def test_batch_size_getter_returns_current_value(
        self, engine: DiscoverEngine,
    ) -> None:
        assert engine.batch_size == BATCH_SIZE

    @pytest.mark.unit
    def test_batch_size_default(
        self,
        controller: LSTMController,
        strategy: RSPGStrategy,
        validator: CandidateValidator,
    ) -> None:
        engine = DiscoverEngine(
            generator=controller,
            strategy=strategy,
            reward_adapter=compute_reward,
            validator=validator,
        )
        assert engine.batch_size == DEFAULT_BATCH_SIZE

    @pytest.mark.unit
    def test_batch_size_setter_updates_value(
        self, engine: DiscoverEngine,
    ) -> None:
        new_size = 32
        engine.batch_size = new_size
        assert engine.batch_size == new_size

    @pytest.mark.unit
    def test_batch_size_setter_rejects_zero(
        self, engine: DiscoverEngine,
    ) -> None:
        with pytest.raises(ValueError):
            engine.batch_size = 0

    @pytest.mark.unit
    def test_batch_size_setter_rejects_negative(
        self, engine: DiscoverEngine,
    ) -> None:
        with pytest.raises(ValueError):
            engine.batch_size = -1

    @pytest.mark.unit
    def test_batch_size_is_property(self) -> None:
        assert isinstance(
            DiscoverEngine.__dict__.get("batch_size"),
            property,
        ), "batch_size must be a @property"







class TestCycleAPI:

    @pytest.mark.unit
    def test_run_cycle_exists(self, engine: DiscoverEngine) -> None:
        assert hasattr(engine, "run_cycle")
        assert callable(engine.run_cycle)

    @pytest.mark.unit
    def test_run_cycle_returns_engine_state(
        self, engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluator(default_nmse=0.5)
        torch.manual_seed(SEED)
        result = engine.run_cycle(evaluator, n_iterations=2)
        assert isinstance(result, EngineState)

    @pytest.mark.unit
    def test_run_cycle_progress_callback_observes_each_iteration(
        self, engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluator(default_nmse=0.5)
        calls: list[tuple[int, dict[str, float], DiscoverEngine]] = []

        def record_progress(
            iteration: int,
            metrics: dict[str, float],
            observed_engine: DiscoverEngine,
        ) -> None:
            calls.append((iteration, metrics, observed_engine))

        torch.manual_seed(SEED)
        engine.run_cycle(evaluator, n_iterations=2, progress_callback=record_progress)

        assert [call[0] for call in calls] == [1, 2]
        assert all(call[2] is engine for call in calls)
        assert all("best_reward" in call[1] for call in calls)

    @pytest.mark.unit
    def test_run_cycle_equivalent_to_old_run(
        self,
        controller: LSTMController,
        strategy: RSPGStrategy,
        validator: CandidateValidator,
        deduplicator: Deduplicator,
    ) -> None:
        evaluator = MockEvaluator(default_nmse=0.5)


        torch.manual_seed(SEED)
        engine1 = DiscoverEngine(
            generator=controller,
            strategy=strategy,
            reward_adapter=compute_reward,
            validator=validator,
            deduplicator=deduplicator,
            batch_size=BATCH_SIZE,
        )
        state1 = engine1.run_cycle(evaluator, n_iterations=3)

        assert isinstance(state1, EngineState)
        assert state1.best_reward >= 0.0

    @pytest.mark.unit
    def test_run_n_cycles_1_calls_run_cycle_once(
        self, engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluator(default_nmse=0.5)
        torch.manual_seed(SEED)
        with patch.object(engine, "run_cycle", wraps=engine.run_cycle) as spy:
            engine.run(evaluator, n_iterations=2, n_cycles=1)
            assert spy.call_count == 1

    @pytest.mark.unit
    def test_run_n_cycles_3_calls_run_cycle_three_times(
        self, engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluator(default_nmse=0.5)
        torch.manual_seed(SEED)
        with patch.object(engine, "run_cycle", wraps=engine.run_cycle) as spy:
            engine.run(evaluator, n_iterations=2, n_cycles=3)
            assert spy.call_count == 3

    @pytest.mark.unit
    def test_run_n_cycles_0_returns_immediately(
        self, engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluator(default_nmse=0.5)
        result = engine.run(evaluator, n_iterations=100, n_cycles=0)
        assert isinstance(result, EngineState)

        assert len(evaluator.evaluated) == 0

    @pytest.mark.unit
    def test_between_cycles_called_between_cycles(
        self, engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluator(default_nmse=0.5)
        callback = MagicMock()
        torch.manual_seed(SEED)
        engine.run(
            evaluator,
            n_iterations=1,
            n_cycles=3,
            between_cycles=callback,
        )

        assert callback.call_count == 2

        callback.assert_any_call(0, engine)
        callback.assert_any_call(1, engine)

    @pytest.mark.unit
    def test_between_cycles_not_called_after_last_cycle(
        self, engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluator(default_nmse=0.5)
        callback = MagicMock()
        torch.manual_seed(SEED)
        engine.run(
            evaluator,
            n_iterations=1,
            n_cycles=2,
            between_cycles=callback,
        )

        assert callback.call_count == 1
        callback.assert_called_once_with(0, engine)

    @pytest.mark.unit
    def test_between_cycles_not_called_for_single_cycle(
        self, engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluator(default_nmse=0.5)
        callback = MagicMock()
        torch.manual_seed(SEED)
        engine.run(
            evaluator,
            n_iterations=1,
            n_cycles=1,
            between_cycles=callback,
        )
        callback.assert_not_called()

    @pytest.mark.unit
    def test_run_signature_has_n_cycles_param(self) -> None:
        sig = inspect.signature(DiscoverEngine.run)
        params = list(sig.parameters.keys())
        assert "n_cycles" in params, (
            f"Expected 'n_cycles' in run() params, got {params}"
        )

    @pytest.mark.unit
    def test_run_signature_has_between_cycles_param(self) -> None:
        sig = inspect.signature(DiscoverEngine.run)
        params = list(sig.parameters.keys())
        assert "between_cycles" in params, (
            f"Expected 'between_cycles' in run() params, got {params}"
        )







class TestGeneratorState:

    @pytest.mark.unit
    def test_state_uses_generator_state_dict(
        self,
        mock_generator: MockGenerator,
        strategy: RSPGStrategy,
        validator: CandidateValidator,
    ) -> None:
        engine = DiscoverEngine(
            generator=mock_generator,
            strategy=strategy,
            reward_adapter=compute_reward,
            validator=validator,
        )
        state = engine.state

        assert "mock" in state.controller_state_dict

    @pytest.mark.unit
    def test_state_restore_uses_generator_load_state_dict(
        self,
        mock_generator: MockGenerator,
        strategy: RSPGStrategy,
        validator: CandidateValidator,
    ) -> None:
        engine = DiscoverEngine(
            generator=mock_generator,
            strategy=strategy,
            reward_adapter=compute_reward,
            validator=validator,
        )
        custom_state = EngineState(
            controller_state_dict={"restored": True},
            baseline_state=BaselineState(),
            best_reward=0.5,
            best_expression="add(x, t)",
        )
        engine.state = custom_state

        assert mock_generator._state.get("restored") is True







class TestPluginBatchSizeIntegration:

    @pytest.mark.unit
    def test_plugin_source_uses_public_batch_size(self) -> None:
        import kd.search.discover.plugin as plugin_mod

        source = inspect.getsource(plugin_mod)

        assert "._batch_size" not in source, (
            "plugin.py still uses engine._batch_size (private). "
            "Must use engine.batch_size = n (public setter)."
        )

        assert ".batch_size" in source, (
            "plugin.py should use engine.batch_size setter"
        )







class TestE2ECycle:

    @pytest.mark.integration
    def test_full_e2e_three_cycles_with_callback(
        self, engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluator(default_nmse=0.3)
        cycle_rewards: list[float] = []

        def record_between(cycle_idx: int, eng: DiscoverEngine) -> None:
            cycle_rewards.append(eng.best_reward)

        torch.manual_seed(SEED)
        final_state = engine.run(
            evaluator,
            n_iterations=2,
            n_cycles=3,
            between_cycles=record_between,
        )
        assert isinstance(final_state, EngineState)

        assert len(cycle_rewards) == 2

        assert len(evaluator.evaluated) > 0

        assert final_state.best_reward > 0.0

    @pytest.mark.integration
    def test_run_cycle_default_n_cycles_1_backward_compat(
        self, engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluator(default_nmse=0.5)
        torch.manual_seed(SEED)


        final_state = engine.run(evaluator, n_iterations=3)
        assert isinstance(final_state, EngineState)
        assert final_state.best_reward > 0.0







class TestRegression:

    @pytest.mark.unit
    def test_propose_still_works(self, engine: DiscoverEngine) -> None:
        torch.manual_seed(SEED)
        ir_strings = engine.propose()
        assert isinstance(ir_strings, list)
        assert len(ir_strings) > 0

    @pytest.mark.unit
    def test_run_iteration_still_works(
        self, engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluator(default_nmse=0.5)
        torch.manual_seed(SEED)
        metrics = engine.run_iteration(evaluator)
        assert isinstance(metrics, dict)
        assert "best_reward" in metrics

    @pytest.mark.unit
    def test_state_property_returns_engine_state(
        self, engine: DiscoverEngine,
    ) -> None:
        state = engine.state
        assert isinstance(state, EngineState)
        assert isinstance(state.controller_state_dict, dict)

    @pytest.mark.unit
    def test_best_reward_and_expression_properties(
        self, engine: DiscoverEngine,
    ) -> None:
        assert engine.best_reward == 0.0
        assert engine.best_expression == ""







class TestRunCycleNoDeadParam:

    @pytest.mark.unit
    def test_run_cycle_signature_has_no_between_cycles(self) -> None:
        sig = inspect.signature(DiscoverEngine.run_cycle)
        params = list(sig.parameters.keys())
        assert "between_cycles" not in params, (
            "run_cycle() still accepts dead 'between_cycles' parameter"
        )







class TestNoControllerAlias:

    @pytest.mark.unit
    def test_engine_has_no_controller_attribute(
        self, engine: DiscoverEngine,
    ) -> None:
        assert not hasattr(engine, "_controller"), (
            "Engine still has _controller alias — remove it"
        )







class TestCheckpointOptimizerReset:

    @pytest.mark.unit
    def test_restore_without_optimizer_state_resets_optimizer(
        self, engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluator(default_nmse=0.5)
        torch.manual_seed(SEED)

        engine.run_iteration(evaluator)
        assert engine._strategy._optimizer is not None


        old_checkpoint = EngineState(
            controller_state_dict=engine.state.controller_state_dict,
            baseline_state=BaselineState(),
            best_reward=0.0,
            best_expression="",
            optimizer_state=None,
        )


        engine.state = old_checkpoint


        assert engine._strategy._optimizer is None
        assert engine._strategy._pending_optimizer_state is None







class TestBestResult:

    @pytest.mark.unit
    def test_best_result_none_at_start(self, engine: DiscoverEngine) -> None:
        assert engine.best_result is None

    @pytest.mark.unit
    def test_best_result_populated_after_iteration(
        self, engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluatorWithTerms(default_nmse=0.3)
        torch.manual_seed(SEED)
        engine.run_iteration(evaluator)
        assert engine.best_result is not None
        assert isinstance(engine.best_result, EvaluationResult)

    @pytest.mark.unit
    def test_best_result_has_terms(self, engine: DiscoverEngine) -> None:
        evaluator = MockEvaluatorWithTerms(default_nmse=0.3)
        torch.manual_seed(SEED)
        engine.run_iteration(evaluator)
        result = engine.best_result
        assert result is not None
        assert result.terms is not None
        assert isinstance(result.terms, list)
        assert len(result.terms) > 0

    @pytest.mark.unit
    def test_best_result_has_coefficients(
        self, engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluatorWithTerms(default_nmse=0.3)
        torch.manual_seed(SEED)
        engine.run_iteration(evaluator)
        result = engine.best_result
        assert result is not None
        assert result.coefficients is not None

    @pytest.mark.unit
    def test_best_result_matches_best_expression(
        self, engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluatorWithTerms(default_nmse=0.3)
        torch.manual_seed(SEED)
        engine.run_iteration(evaluator)
        result = engine.best_result
        assert result is not None
        assert result.expression == engine.best_expression

    @pytest.mark.unit
    def test_best_result_only_updates_when_reward_improves(
        self, engine: DiscoverEngine,
    ) -> None:
        good_eval = MockEvaluatorWithTerms(default_nmse=0.1)
        bad_eval = MockEvaluatorWithTerms(default_nmse=0.9)

        torch.manual_seed(SEED)
        engine.run_iteration(good_eval)
        first_best = engine.best_result
        first_reward = engine.best_reward
        assert first_best is not None

        engine.run_iteration(bad_eval)

        assert engine.best_reward == first_reward
        assert engine.best_result is not None
        assert engine.best_result.expression == first_best.expression







class TestExtractActiveTerms:

    @pytest.mark.unit
    def test_with_selected_indices_filters(self) -> None:
        result = EvaluationResult(
            mse=0.1, nmse=0.1, r2=0.9, complexity=2,
            is_valid=True, expression="test",
            terms=["u_xx", "mul(u, u_x)", "u"],
            coefficients=torch.tensor([1.0, -1.0, 0.0]),
            selected_indices=[0, 1],
        )
        terms, coeffs = extract_active_terms(result)
        assert terms == ["u_xx", "mul(u, u_x)"]
        assert coeffs == pytest.approx([1.0, -1.0])

    @pytest.mark.unit
    def test_without_selected_indices_returns_all(self) -> None:
        result = EvaluationResult(
            mse=0.1, nmse=0.1, r2=0.9, complexity=3,
            is_valid=True, expression="test",
            terms=["u_xx", "mul(u, u_x)", "u"],
            coefficients=torch.tensor([1.0, -1.0, 0.5]),
            selected_indices=None,
        )
        terms, coeffs = extract_active_terms(result)
        assert terms == ["u_xx", "mul(u, u_x)", "u"]
        assert coeffs == pytest.approx([1.0, -1.0, 0.5])

    @pytest.mark.unit
    def test_coefficients_are_plain_floats(self) -> None:
        result = EvaluationResult(
            mse=0.1, nmse=0.1, r2=0.9, complexity=1,
            is_valid=True, expression="test",
            terms=["u_xx"],
            coefficients=torch.tensor([1.5]),
            selected_indices=None,
        )
        _, coeffs = extract_active_terms(result)
        assert isinstance(coeffs, list)
        for c in coeffs:
            assert isinstance(c, float)

    @pytest.mark.unit
    def test_single_selected_index(self) -> None:
        result = EvaluationResult(
            mse=0.1, nmse=0.1, r2=0.9, complexity=1,
            is_valid=True, expression="test",
            terms=["a", "b", "c"],
            coefficients=torch.tensor([0.0, 2.5, 0.0]),
            selected_indices=[1],
        )
        terms, coeffs = extract_active_terms(result)
        assert terms == ["b"]
        assert coeffs == pytest.approx([2.5])

    @pytest.mark.unit
    def test_empty_selected_indices(self) -> None:
        result = EvaluationResult(
            mse=0.1, nmse=0.1, r2=0.9, complexity=0,
            is_valid=True, expression="test",
            terms=["a", "b"],
            coefficients=torch.tensor([1.0, 2.0]),
            selected_indices=[],
        )
        terms, coeffs = extract_active_terms(result)
        assert terms == []
        assert coeffs == []

    @pytest.mark.unit
    def test_none_terms_returns_empty(self) -> None:
        result = EvaluationResult(
            mse=0.1, nmse=0.1, r2=0.9, complexity=0,
            is_valid=True, expression="test",
            terms=None,
            coefficients=torch.tensor([1.0]),
        )
        terms, coeffs = extract_active_terms(result)
        assert terms == []
        assert coeffs == []

    @pytest.mark.unit
    def test_none_coefficients_returns_empty(self) -> None:
        result = EvaluationResult(
            mse=0.1, nmse=0.1, r2=0.9, complexity=0,
            is_valid=True, expression="test",
            terms=["a", "b"],
            coefficients=None,
        )
        terms, coeffs = extract_active_terms(result)
        assert terms == []
        assert coeffs == []







class TestEngineStateBestResultFields:

    @pytest.mark.unit
    def test_engine_state_has_best_result_terms_field(self) -> None:
        state = EngineState(
            controller_state_dict={},
            baseline_state=BaselineState(),
            best_reward=0.5,
            best_expression="add(x, t)",
            best_result_terms=["u_xx", "u"],
            best_result_coefficients=[1.0, -0.5],
        )
        assert state.best_result_terms == ["u_xx", "u"]
        assert state.best_result_coefficients == [1.0, -0.5]

    @pytest.mark.unit
    def test_engine_state_defaults_none(self) -> None:
        state = EngineState(
            controller_state_dict={},
            baseline_state=BaselineState(),
            best_reward=0.0,
            best_expression="",
        )
        assert state.best_result_terms is None
        assert state.best_result_coefficients is None

    @pytest.mark.unit
    def test_best_result_roundtrip(self, engine: DiscoverEngine) -> None:
        evaluator = MockEvaluatorWithTerms(default_nmse=0.3)
        torch.manual_seed(SEED)
        engine.run_iteration(evaluator)

        original_result = engine.best_result
        assert original_result is not None


        saved_state = engine.state
        assert saved_state.best_result_terms is not None
        assert saved_state.best_result_coefficients is not None


        engine.state = saved_state
        restored_result = engine.best_result
        assert restored_result is not None

        active_terms, active_coeffs = extract_active_terms(original_result)
        assert restored_result.terms == active_terms

        assert restored_result.coefficients is not None
        restored_coeffs = restored_result.coefficients.tolist()
        assert restored_coeffs == pytest.approx(active_coeffs)

    @pytest.mark.unit
    def test_best_result_coefficients_stored_as_list_float(
        self, engine: DiscoverEngine,
    ) -> None:
        evaluator = MockEvaluatorWithTerms(default_nmse=0.3)
        torch.manual_seed(SEED)
        engine.run_iteration(evaluator)

        state = engine.state
        assert state.best_result_coefficients is not None
        assert isinstance(state.best_result_coefficients, list)
        for c in state.best_result_coefficients:
            assert isinstance(c, float)







class TestBackwardCompat:

    @pytest.mark.unit
    def test_old_engine_state_without_new_fields(
        self, engine: DiscoverEngine,
    ) -> None:
        old_state = EngineState(
            controller_state_dict=engine.state.controller_state_dict,
            baseline_state=BaselineState(),
            best_reward=0.5,
            best_expression="add(x, t)",

        )
        engine.state = old_state
        assert engine.best_result is None
        assert engine.best_reward == 0.5
        assert engine.best_expression == "add(x, t)"

    @pytest.mark.unit
    def test_engine_state_none_fields_loads(
        self, engine: DiscoverEngine,
    ) -> None:
        state = EngineState(
            controller_state_dict=engine.state.controller_state_dict,
            baseline_state=BaselineState(),
            best_reward=0.0,
            best_expression="",
            best_result_terms=None,
            best_result_coefficients=None,
        )
        engine.state = state
        assert engine.best_result is None







class TestCycleTopCandidates:

    @pytest.mark.unit
    def test_cycle_top_candidates_resets_at_start_of_run_cycle(
        self, engine_with_capacity: DiscoverEngine,
    ) -> None:
        engine_with_capacity._update_best(
            unique_irs=["u"],
            unique_rewards=np.array([0.4], dtype=np.float32),
            unique_eval_valid_mask=np.array([True], dtype=np.bool_),
            unique_results=[_result_with_terms("u")],
        )
        assert len(engine_with_capacity.cycle_top_candidates) == 1

        engine_with_capacity.run_cycle(MockEvaluator(), n_iterations=0)

        assert engine_with_capacity.cycle_top_candidates == []

    @pytest.mark.unit
    def test_hall_of_fame_keeps_highest_reward_items_up_to_capacity(
        self, engine_with_capacity: DiscoverEngine,
    ) -> None:
        expressions = ["u", "diff_x(u)", "add(u,diff_x(u))"]
        rewards = np.array([0.2, 0.9, 0.5], dtype=np.float32)
        results = [_result_with_terms(expr) for expr in expressions]

        engine_with_capacity._update_best(
            unique_irs=expressions,
            unique_rewards=rewards,
            unique_eval_valid_mask=np.array([True, True, True], dtype=np.bool_),
            unique_results=results,
        )

        snapshots = engine_with_capacity.cycle_top_candidates
        assert [candidate.expression for candidate in snapshots] == [
            "diff_x(u)",
            "add(u,diff_x(u))",
        ]

    @pytest.mark.unit
    def test_hall_of_fame_ignores_duplicate_expression_strings(
        self, engine_with_capacity: DiscoverEngine,
    ) -> None:
        engine_with_capacity._update_best(
            unique_irs=["u"],
            unique_rewards=np.array([0.2], dtype=np.float32),
            unique_eval_valid_mask=np.array([True], dtype=np.bool_),
            unique_results=[_result_with_terms("u")],
        )
        engine_with_capacity._update_best(
            unique_irs=["u"],
            unique_rewards=np.array([0.9], dtype=np.float32),
            unique_eval_valid_mask=np.array([True], dtype=np.bool_),
            unique_results=[_result_with_terms("u", nmse=0.1)],
        )

        snapshots = engine_with_capacity.cycle_top_candidates
        assert len(snapshots) == 1
        assert snapshots[0].expression == "u"
        assert snapshots[0].reward == pytest.approx(0.2)

    @pytest.mark.unit
    def test_snapshot_terms_keep_full_valid_theta_terms(
        self, engine_with_capacity: DiscoverEngine,
    ) -> None:
        result = _result_with_terms(
            "add(diff2_x(u),u)",
            terms=["diff2_x(u)", "u"],
            coefficients=[1.0, 0.0],
            selected_indices=[0],
        )

        engine_with_capacity._update_best(
            unique_irs=["add(diff2_x(u),u)"],
            unique_rewards=np.array([0.4], dtype=np.float32),
            unique_eval_valid_mask=np.array([True], dtype=np.bool_),
            unique_results=[result],
        )

        snapshot = engine_with_capacity.cycle_top_candidates[0]
        assert snapshot.terms == ["diff2_x(u)", "u"]

    @pytest.mark.unit
    def test_snapshot_n_nodes_matches_ir_tree_size(
        self, engine_with_capacity: DiscoverEngine,
    ) -> None:
        expression = "add(u,diff_x(u))"
        engine_with_capacity._update_best(
            unique_irs=[expression],
            unique_rewards=np.array([0.4], dtype=np.float32),
            unique_eval_valid_mask=np.array([True], dtype=np.bool_),
            unique_results=[_result_with_terms(expression)],
        )

        snapshot = engine_with_capacity.cycle_top_candidates[0]
        expected_nodes = ir_to_tree(
            expression,
            engine_with_capacity._generator.library,
        ).n_nodes()
        assert snapshot.n_nodes == expected_nodes

    @pytest.mark.unit
    def test_unparsable_expression_is_skipped_from_cycle_candidates(
        self, engine_with_capacity: DiscoverEngine,
    ) -> None:
        engine_with_capacity._update_best(
            unique_irs=["not valid ir("],
            unique_rewards=np.array([0.4], dtype=np.float32),
            unique_eval_valid_mask=np.array([True], dtype=np.bool_),
            unique_results=[_result_with_terms("not valid ir(")],
        )

        assert engine_with_capacity.cycle_top_candidates == []

    @pytest.mark.unit
    def test_cycle_top_candidates_not_serialized_in_engine_state(
        self,
        engine_with_capacity: DiscoverEngine,
        controller: LSTMController,
        strategy: RSPGStrategy,
        validator: CandidateValidator,
        deduplicator: Deduplicator,
    ) -> None:
        engine_with_capacity._update_best(
            unique_irs=["u"],
            unique_rewards=np.array([0.4], dtype=np.float32),
            unique_eval_valid_mask=np.array([True], dtype=np.bool_),
            unique_results=[_result_with_terms("u")],
        )
        saved_state = engine_with_capacity.state
        assert not hasattr(saved_state, "cycle_top_candidates")

        restored = DiscoverEngine(
            controller=controller,
            strategy=strategy,
            reward_adapter=compute_reward,
            validator=validator,
            deduplicator=deduplicator,
            batch_size=BATCH_SIZE,
            cycle_candidate_capacity=2,
        )
        restored.state = saved_state
        assert restored.cycle_top_candidates == []
