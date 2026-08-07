
from __future__ import annotations

import copy
import pickle
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from kd.core.evaluator import EvaluationResult
from kd.search.discover.controller.lstm import LSTMController
from kd.search.discover.engine import DiscoverEngine, EngineState
from kd.search.discover.evaluation.dedup import Deduplicator
from kd.search.discover.evaluation.reward import compute_reward
from kd.search.discover.plugin import DISCOVERPlugin
from kd.search.discover.tokens.library import Library, LibraryConfig
from kd.search.discover.tokens.prior import (
    DiffChildConstraint,
    LengthConstraint,
    PriorSystem,
)
from kd.search.discover.tokens.validator import CandidateValidator
from kd.search.discover.training.strategy import BaselineState, RSPGStrategy
from kd.search.protocol import PlatformComponents





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
def mock_evaluator() -> MockEvaluator:
    return MockEvaluator(default_nmse=0.5)


@pytest.fixture
def mock_components(mock_evaluator: MockEvaluator) -> PlatformComponents:
    return PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=mock_evaluator,
        context=MagicMock(training_result=None),
        registry=MagicMock(),
        recorder=None,
    )


@pytest.fixture
def plugin(mock_components: PlatformComponents) -> DISCOVERPlugin:
    torch.manual_seed(SEED)
    p = DISCOVERPlugin()
    p.prepare(mock_components)
    return p


def _run_one_iteration(engine: DiscoverEngine) -> None:
    evaluator = MockEvaluator(default_nmse=0.5)
    engine.run_iteration(evaluator)


def _adam_step_as_int(step: Any) -> int:
    if isinstance(step, torch.Tensor):
        return int(step.item())
    return int(step)







class TestStrategyOptimizerState:

    @pytest.mark.unit
    def test_optimizer_state_none_before_train_step(
        self,
        strategy: RSPGStrategy,
    ) -> None:
        assert strategy.optimizer_state is None

    @pytest.mark.unit
    def test_optimizer_state_dict_after_train_step(
        self,
        controller: LSTMController,
        strategy: RSPGStrategy,
    ) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        rewards = torch.rand(BATCH_SIZE)
        strategy.train_step(controller, batch, rewards, BaselineState())

        opt_state = strategy.optimizer_state
        assert isinstance(opt_state, dict)

        assert "state" in opt_state
        assert "param_groups" in opt_state

    @pytest.mark.unit
    def test_set_optimizer_state_restores_adam_moments(
        self,
        controller: LSTMController,
        strategy: RSPGStrategy,
    ) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        rewards = torch.rand(BATCH_SIZE)


        strategy.train_step(controller, batch, rewards, BaselineState())
        state_after_step1 = strategy.optimizer_state
        assert state_after_step1 is not None
        params_after_step1 = {
            n: p.clone().detach() for n, p in controller.named_parameters()
        }


        strategy.train_step(controller, batch, rewards, BaselineState())
        params_after_step2 = {
            n: p.clone().detach() for n, p in controller.named_parameters()
        }


        controller.load_state_dict(
            {n: p.clone() for n, p in params_after_step1.items()},
        )
        strategy.optimizer_state = copy.deepcopy(state_after_step1)


        strategy.train_step(controller, batch, rewards, BaselineState())
        params_replayed = {
            n: p.clone().detach() for n, p in controller.named_parameters()
        }


        for name in params_after_step2:
            torch.testing.assert_close(
                params_replayed[name],
                params_after_step2[name],
                atol=1e-6,
                rtol=1e-5,
                msg=f"Replayed param {name} diverged from original step 2",
            )

    @pytest.mark.unit
    def test_set_optimizer_state_before_init_deferred(
        self,
        controller: LSTMController,
    ) -> None:
        strategy = RSPGStrategy(epsilon=0.5, baseline="R_e")


        torch.manual_seed(SEED)
        ref_strategy = RSPGStrategy(epsilon=0.5, baseline="R_e")
        batch = controller.sample(BATCH_SIZE)
        rewards = torch.rand(BATCH_SIZE)
        ref_strategy.train_step(controller, batch, rewards, BaselineState())
        ref_opt_state = ref_strategy.optimizer_state
        assert ref_opt_state is not None


        assert ref_opt_state["state"], "reference optimizer state is empty"


        torch.manual_seed(SEED)
        controller_fresh = LSTMController(
            library=controller.library,
            prior_system=controller.prior_system,
            num_units=16,
            num_layers=1,
            embedding_dim=4,
        )


        strategy.optimizer_state = copy.deepcopy(ref_opt_state)


        batch2 = controller_fresh.sample(BATCH_SIZE)
        rewards2 = torch.rand(BATCH_SIZE)
        strategy.train_step(controller_fresh, batch2, rewards2, BaselineState())


        restored = strategy.optimizer_state
        assert restored is not None
        assert restored["state"], "optimizer state empty after deferred restore"

        assert set(restored["state"].keys()) == set(ref_opt_state["state"].keys())


        for key, ref_entry in ref_opt_state["state"].items():
            ref_step = _adam_step_as_int(ref_entry["step"])
            got_step = _adam_step_as_int(restored["state"][key]["step"])
            assert got_step == ref_step + 1, (
                f"param {key}: step {got_step} != ref {ref_step} + 1 "
                "(deferred optimizer_state restore was dropped)"
            )







class TestEngineStateCheckpoint:

    @pytest.mark.unit
    def test_engine_state_includes_optimizer_state_after_iteration(
        self,
        engine: DiscoverEngine,
    ) -> None:
        torch.manual_seed(SEED)
        _run_one_iteration(engine)

        state = engine.state
        assert hasattr(state, "optimizer_state")
        assert state.optimizer_state is not None
        assert isinstance(state.optimizer_state, dict)
        assert "state" in state.optimizer_state
        assert "param_groups" in state.optimizer_state

    @pytest.mark.unit
    def test_engine_state_optimizer_state_none_before_training(
        self,
        engine: DiscoverEngine,
    ) -> None:
        state = engine.state
        assert state.optimizer_state is None

    @pytest.mark.unit
    def test_engine_state_backward_compat_defaults(self) -> None:
        state = EngineState(
            controller_state_dict={"w": torch.zeros(3)},
            baseline_state=BaselineState(),
            best_reward=0.5,
            best_expression="sin(x)",
        )
        assert state.optimizer_state is None
        assert state.extras is None

    @pytest.mark.unit
    def test_engine_state_with_extras(self) -> None:
        extras = {"pinn_weights": [1.0, 2.0], "term_bank": {"x": 0.5}}
        state = EngineState(
            controller_state_dict={"w": torch.zeros(3)},
            baseline_state=BaselineState(),
            best_reward=0.5,
            best_expression="sin(x)",
            extras=extras,
        )
        assert state.extras is not None
        assert state.extras["pinn_weights"] == [1.0, 2.0]
        assert state.extras["term_bank"]["x"] == 0.5

    @pytest.mark.unit
    def test_state_roundtrip_preserves_optimizer_state(
        self,
        engine: DiscoverEngine,
    ) -> None:
        torch.manual_seed(SEED)
        _run_one_iteration(engine)


        saved_state = engine.state
        saved_params = copy.deepcopy(saved_state.controller_state_dict)
        assert saved_state.optimizer_state is not None


        _run_one_iteration(engine)
        _run_one_iteration(engine)


        engine.state = saved_state

        restored_state = engine.state
        assert restored_state.optimizer_state is not None

        for key in saved_params:
            torch.testing.assert_close(
                restored_state.controller_state_dict[key],
                saved_params[key],
            )

    @pytest.mark.unit
    def test_state_roundtrip_extras_preserved(
        self,
        engine: DiscoverEngine,
    ) -> None:
        torch.manual_seed(SEED)
        _run_one_iteration(engine)

        state = engine.state

        state_with_extras = EngineState(
            controller_state_dict=state.controller_state_dict,
            baseline_state=state.baseline_state,
            best_reward=state.best_reward,
            best_expression=state.best_expression,
            optimizer_state=state.optimizer_state,
            extras={"custom_key": "custom_value", "epoch": 42},
        )
        engine.state = state_with_extras

        restored = engine.state
        assert restored.extras is not None
        assert restored.extras["custom_key"] == "custom_value"
        assert restored.extras["epoch"] == 42

    @pytest.mark.unit
    def test_restore_state_without_optimizer_state(
        self,
        engine: DiscoverEngine,
    ) -> None:
        torch.manual_seed(SEED)
        _run_one_iteration(engine)


        old_state = EngineState(
            controller_state_dict=engine.state.controller_state_dict,
            baseline_state=engine.state.baseline_state,
            best_reward=engine.state.best_reward,
            best_expression=engine.state.best_expression,
        )

        engine.state = old_state


        _run_one_iteration(engine)







class TestPluginCheckpointSerialization:

    @pytest.mark.unit
    def test_plugin_state_includes_optimizer_state(
        self,
        plugin: DISCOVERPlugin,
    ) -> None:
        torch.manual_seed(SEED)
        candidates = plugin.propose(BATCH_SIZE)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        state = plugin.state
        engine_state = state["engine_state"]
        assert "optimizer_state" in engine_state

    @pytest.mark.unit
    def test_plugin_state_restores_optimizer_state(
        self,
        plugin: DISCOVERPlugin,
    ) -> None:
        torch.manual_seed(SEED)
        candidates = plugin.propose(BATCH_SIZE)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        saved_state = plugin.state
        saved_score = plugin.best_score


        candidates2 = plugin.propose(BATCH_SIZE)
        results2 = plugin.evaluate(candidates2)
        plugin.update(results2)


        plugin.state = saved_state
        assert plugin.best_score == saved_score

    @pytest.mark.unit
    def test_plugin_state_without_optimizer_state_backward_compat(
        self,
        plugin: DISCOVERPlugin,
    ) -> None:
        torch.manual_seed(SEED)
        candidates = plugin.propose(BATCH_SIZE)
        results = plugin.evaluate(candidates)
        plugin.update(results)


        current_state = plugin.state
        engine_state = current_state["engine_state"]
        old_format: dict[str, Any] = {
            "algorithm": "discover",
            "engine_state": {
                "controller_state_dict": engine_state["controller_state_dict"],
                "baseline_state": engine_state["baseline_state"],
                "best_reward": current_state["engine_state"]["best_reward"],
                "best_expression": current_state["engine_state"]["best_expression"],
            },
        }

        plugin.state = old_format


        candidates3 = plugin.propose(BATCH_SIZE)
        assert len(candidates3) > 0

    @pytest.mark.unit
    def test_plugin_state_extras_roundtrip(
        self,
        plugin: DISCOVERPlugin,
    ) -> None:
        torch.manual_seed(SEED)
        candidates = plugin.propose(BATCH_SIZE)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        state = plugin.state

        state["engine_state"]["extras"] = {"phase": 2, "pinn": True}
        plugin.state = state

        restored = plugin.state
        assert restored["engine_state"].get("extras") is not None
        assert restored["engine_state"]["extras"]["phase"] == 2
        assert restored["engine_state"]["extras"]["pinn"] is True

    @pytest.mark.unit
    def test_plugin_state_pickle_serializable_with_optimizer(
        self,
        plugin: DISCOVERPlugin,
    ) -> None:
        torch.manual_seed(SEED)
        candidates = plugin.propose(BATCH_SIZE)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        state = plugin.state

        data = pickle.dumps(state)
        restored = pickle.loads(data)
        assert isinstance(restored, dict)
        assert "engine_state" in restored
        assert "optimizer_state" in restored["engine_state"]
