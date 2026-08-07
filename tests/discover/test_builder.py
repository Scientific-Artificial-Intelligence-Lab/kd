
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from kd.core.evaluator import EvaluationResult
from kd.search.discover.builder import (
    build_controller,
    build_engine,
    build_library,
    build_prior_system,
    build_strategy,
)
from kd.search.discover.config import DiscoverConfig, PINNConfig
from kd.search.discover.controller.lstm import LSTMController
from kd.search.discover.engine import DiscoverEngine
from kd.search.discover.evaluation.reward import compute_reward
from kd.search.discover.tokens.library import Library, LibraryConfig
from kd.search.discover.tokens.prior import (
    DiffChildConstraint,
    DiffDescendantConstraint,
    InverseUnaryConstraint,
    LengthConstraint,
    PriorSystem,
    RepeatConstraint,
    TrigConstraint,
)
from kd.search.discover.training.strategy import RSPGStrategy
from kd.search.protocol import PlatformComponents





SEED = 42
CUSTOM_OPERATORS = ["add", "mul", "sin", "diff_x"]
CUSTOM_STATE_VARS = ["u"]
CUSTOM_COORD_VARS = ["x", "t"]






class MockEvaluator:

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        return EvaluationResult(
            mse=0.5,
            nmse=0.5,
            r2=0.5,
            complexity=3,
            is_valid=True,
            expression=expr,
        )







@pytest.fixture
def default_config() -> DiscoverConfig:
    return DiscoverConfig()


@pytest.fixture
def custom_config() -> DiscoverConfig:
    return DiscoverConfig(
        library=LibraryConfig(
            operators=list(CUSTOM_OPERATORS),
            state_vars=list(CUSTOM_STATE_VARS),
            coord_vars=list(CUSTOM_COORD_VARS),
        ),
        min_length=3,
        max_length=20,
        batch_size=32,
        num_units=64,
        num_layers=2,
        embedding_dim=16,
        epsilon=0.10,
        baseline="ewma_R",
        entropy_weight=0.01,
        gamma=0.95,
        reward_alpha=0.02,
    )







class TestBuildLibrary:

    def test_returns_library(self, default_config: DiscoverConfig) -> None:
        library = build_library(default_config)
        assert isinstance(library, Library)

    def test_default_token_count(self, default_config: DiscoverConfig) -> None:
        library = build_library(default_config)
        expected_ops = len(default_config.library.operators)
        expected_vars = len(default_config.library.state_vars)
        expected_coords = len(default_config.library.coord_vars)
        assert len(library.tokens) == expected_ops + expected_vars + expected_coords

    def test_custom_operators(self, custom_config: DiscoverConfig) -> None:
        library = build_library(custom_config)
        expected = (
            len(CUSTOM_OPERATORS) + len(CUSTOM_STATE_VARS) + len(CUSTOM_COORD_VARS)
        )
        assert len(library.tokens) == expected







class TestBuildPriorSystem:

    def test_returns_prior_system(self, default_config: DiscoverConfig) -> None:
        library = build_library(default_config)
        ps = build_prior_system(library, default_config)
        assert isinstance(ps, PriorSystem)

    def test_includes_length_constraint(self, default_config: DiscoverConfig) -> None:
        library = build_library(default_config)
        ps = build_prior_system(library, default_config)
        length_priors = [p for p in ps.priors if isinstance(p, LengthConstraint)]
        assert len(length_priors) == 1

    def test_length_constraint_uses_config_min_length(self) -> None:
        config = DiscoverConfig(min_length=6, max_length=25)
        library = build_library(config)
        ps = build_prior_system(library, config)
        length_priors = [p for p in ps.priors if isinstance(p, LengthConstraint)]
        assert length_priors[0].min_ == 6

    def test_length_constraint_uses_config_max_length(self) -> None:
        config = DiscoverConfig(min_length=3, max_length=30)
        library = build_library(config)
        ps = build_prior_system(library, config)
        length_priors = [p for p in ps.priors if isinstance(p, LengthConstraint)]
        assert length_priors[0].max_ == 30

    def test_includes_diff_child_constraint(
        self,
        default_config: DiscoverConfig,
    ) -> None:
        library = build_library(default_config)
        ps = build_prior_system(library, default_config)
        diff_priors = [p for p in ps.priors if isinstance(p, DiffChildConstraint)]
        assert len(diff_priors) == 1



    def test_includes_repeat_constraint(
        self,
        default_config: DiscoverConfig,
    ) -> None:
        library = build_library(default_config)
        ps = build_prior_system(library, default_config)
        repeat_priors = [p for p in ps.priors if isinstance(p, RepeatConstraint)]
        assert len(repeat_priors) == 1

    def test_includes_trig_constraint(
        self,
        default_config: DiscoverConfig,
    ) -> None:
        library = build_library(default_config)
        ps = build_prior_system(library, default_config)
        trig_priors = [p for p in ps.priors if isinstance(p, TrigConstraint)]
        assert len(trig_priors) == 1

    def test_includes_inverse_constraint(
        self,
        default_config: DiscoverConfig,
    ) -> None:
        library = build_library(default_config)
        ps = build_prior_system(library, default_config)
        inv_priors = [p for p in ps.priors if isinstance(p, InverseUnaryConstraint)]
        assert len(inv_priors) == 1

    def test_includes_diff_descendant_constraint(
        self,
        default_config: DiscoverConfig,
    ) -> None:
        library = build_library(default_config)
        ps = build_prior_system(library, default_config)
        dd_priors = [p for p in ps.priors if isinstance(p, DiffDescendantConstraint)]
        assert len(dd_priors) == 1

    def test_disable_repeat_prior(self) -> None:
        config = DiscoverConfig(use_repeat_prior=False)
        library = build_library(config)
        ps = build_prior_system(library, config)
        assert not any(isinstance(p, RepeatConstraint) for p in ps.priors)

    def test_disable_trig_prior(self) -> None:
        config = DiscoverConfig(use_trig_prior=False)
        library = build_library(config)
        ps = build_prior_system(library, config)
        assert not any(isinstance(p, TrigConstraint) for p in ps.priors)

    def test_disable_inverse_prior(self) -> None:
        config = DiscoverConfig(use_inverse_prior=False)
        library = build_library(config)
        ps = build_prior_system(library, config)
        assert not any(isinstance(p, InverseUnaryConstraint) for p in ps.priors)

    def test_disable_diff_descendant_prior(self) -> None:
        config = DiscoverConfig(use_diff_descendant_prior=False)
        library = build_library(config)
        ps = build_prior_system(library, config)
        assert not any(isinstance(p, DiffDescendantConstraint) for p in ps.priors)

    def test_disable_diff_child_prior(self) -> None:
        config = DiscoverConfig(use_diff_child_prior=False)
        library = build_library(config)
        ps = build_prior_system(library, config)
        assert not any(isinstance(p, DiffChildConstraint) for p in ps.priors)

    def test_default_includes_diff_child_prior(self) -> None:
        config = DiscoverConfig()
        library = build_library(config)
        ps = build_prior_system(library, config)
        assert any(isinstance(p, DiffChildConstraint) for p in ps.priors)

    def test_repeat_config_passthrough(self) -> None:
        config = DiscoverConfig(
            repeat_tokens=["mul"],
            repeat_max=3,
        )
        library = build_library(config)
        ps = build_prior_system(library, config)
        repeat_priors = [p for p in ps.priors if isinstance(p, RepeatConstraint)]
        assert len(repeat_priors) == 1
        rc = repeat_priors[0]
        assert rc.max_ == 3
        mul_idx = library.name_to_index("mul")
        assert mul_idx in rc.target_tokens

    def test_all_priors_disabled_only_length_and_diffchild(self) -> None:
        config = DiscoverConfig(
            use_repeat_prior=False,
            use_trig_prior=False,
            use_inverse_prior=False,
            use_diff_descendant_prior=False,
        )
        library = build_library(config)
        ps = build_prior_system(library, config)
        assert len(ps.priors) == 2
        assert isinstance(ps.priors[0], LengthConstraint)
        assert isinstance(ps.priors[1], DiffChildConstraint)

    def test_default_config_prior_count(
        self,
        default_config: DiscoverConfig,
    ) -> None:
        library = build_library(default_config)
        ps = build_prior_system(library, default_config)
        assert len(ps.priors) == 6







class TestBuildController:

    def test_returns_lstm_controller(self, default_config: DiscoverConfig) -> None:
        torch.manual_seed(SEED)
        library = build_library(default_config)
        ps = build_prior_system(library, default_config)
        controller = build_controller(library, ps, default_config)
        assert isinstance(controller, LSTMController)

    def test_num_units(self, custom_config: DiscoverConfig) -> None:
        torch.manual_seed(SEED)
        library = build_library(custom_config)
        ps = build_prior_system(library, custom_config)
        controller = build_controller(library, ps, custom_config)
        assert controller.rnn.hidden_size == custom_config.num_units

    def test_num_layers(self, custom_config: DiscoverConfig) -> None:
        torch.manual_seed(SEED)
        library = build_library(custom_config)
        ps = build_prior_system(library, custom_config)
        controller = build_controller(library, ps, custom_config)
        assert len(controller.rnn.cells) == custom_config.num_layers

    def test_embedding_dim(self, default_config: DiscoverConfig) -> None:
        torch.manual_seed(SEED)
        library = build_library(default_config)
        ps = build_prior_system(library, default_config)
        controller = build_controller(library, ps, default_config)
        assert controller.action_embedding.embedding_dim == default_config.embedding_dim

    def test_library_attached(self, default_config: DiscoverConfig) -> None:
        torch.manual_seed(SEED)
        library = build_library(default_config)
        ps = build_prior_system(library, default_config)
        controller = build_controller(library, ps, default_config)
        assert controller.library is library

    def test_default_initializer_is_xavier(
        self,
        default_config: DiscoverConfig,
    ) -> None:
        torch.manual_seed(SEED)
        library = build_library(default_config)
        ps = build_prior_system(library, default_config)
        controller = build_controller(library, ps, default_config)

        assert any(
            p.dim() >= 2 and not torch.all(p == 0.0) for p in controller.parameters()
        )

    def test_zeros_initializer_threaded_through(self) -> None:
        config = DiscoverConfig(initializer="zeros")
        library = build_library(config)
        ps = build_prior_system(library, config)
        controller = build_controller(library, ps, config)
        assert controller.initializer == "zeros"
        for name, param in controller.rnn.named_parameters():
            assert torch.all(param == 0.0), (
                f"rnn parameter '{name}' must be zero under initializer='zeros'"
            )
        assert not torch.all(controller.output_layer.weight == 0.0), (
            "output_layer.weight must remain xavier (non-zero) to keep "
            "gradient flowing back into the LSTM"
        )
        assert controller.output_layer.bias is not None
        assert torch.all(controller.output_layer.bias == 0.0), (
            "output_layer.bias must be zero under initializer='zeros'"
        )







class TestBuildStrategy:

    def test_returns_rspg_strategy(self, default_config: DiscoverConfig) -> None:
        strategy = build_strategy(default_config)
        assert isinstance(strategy, RSPGStrategy)

    def test_epsilon(self, custom_config: DiscoverConfig) -> None:
        strategy = build_strategy(custom_config)
        assert strategy.epsilon == custom_config.epsilon

    def test_baseline(self, custom_config: DiscoverConfig) -> None:
        strategy = build_strategy(custom_config)
        assert strategy.baseline == custom_config.baseline

    def test_entropy_weight(self, custom_config: DiscoverConfig) -> None:
        strategy = build_strategy(custom_config)
        assert strategy.entropy_weight == custom_config.entropy_weight

    def test_gamma(self, custom_config: DiscoverConfig) -> None:
        strategy = build_strategy(custom_config)
        assert strategy.gamma == custom_config.gamma







class TestBuildEngine:

    def test_returns_discover_engine(self, default_config: DiscoverConfig) -> None:
        torch.manual_seed(SEED)
        engine = build_engine(default_config)
        assert isinstance(engine, DiscoverEngine)

    def test_can_propose(self, default_config: DiscoverConfig) -> None:
        torch.manual_seed(SEED)
        engine = build_engine(default_config)
        candidates = engine.propose()
        assert len(candidates) > 0
        assert all(isinstance(c, str) for c in candidates)

    def test_can_run_iteration(self, default_config: DiscoverConfig) -> None:
        torch.manual_seed(SEED)
        engine = build_engine(default_config)
        metrics = engine.run_iteration(MockEvaluator())
        assert "reward_max" in metrics
        assert "best_reward" in metrics

    def test_custom_batch_size(self) -> None:
        config = DiscoverConfig(batch_size=8)
        torch.manual_seed(SEED)
        engine = build_engine(config)
        assert engine.batch_size == 8

    def test_stability_selection_enables_cycle_candidate_capacity(self) -> None:
        config = DiscoverConfig(
            stability_selection=3,
            stability_queue_capacity=7,
        )
        torch.manual_seed(SEED)
        engine = build_engine(config)
        assert engine._cycle_candidate_capacity == 7

    def test_stability_selection_disabled_zeroes_cycle_capacity(self) -> None:
        config = DiscoverConfig(
            stability_selection=0,
            stability_queue_capacity=7,
        )
        torch.manual_seed(SEED)
        engine = build_engine(config)
        assert engine._cycle_candidate_capacity == 0

    def test_invalid_stability_queue_capacity_rejected(self) -> None:
        with pytest.raises(ValueError, match="stability_queue_capacity"):
            DiscoverConfig(stability_selection=3, stability_queue_capacity=2)

    def test_reward_adapter_uses_config_alpha(self) -> None:
        config = DiscoverConfig(reward_alpha=0.05)
        torch.manual_seed(SEED)
        engine = build_engine(config)
        result = EvaluationResult(
            mse=0.5,
            nmse=0.5,
            r2=0.5,
            complexity=3,
            is_valid=True,
            expression="x",
        )
        expected = compute_reward(result, alpha=0.05)
        actual = engine._reward_adapter(result)
        assert actual == pytest.approx(expected)

        default_reward = compute_reward(result, alpha=0.01)
        assert actual != pytest.approx(default_reward)

    def test_engine_respects_min_length(self) -> None:
        config = DiscoverConfig(min_length=7, max_length=25)
        torch.manual_seed(SEED)
        engine = build_engine(config)
        assert isinstance(engine._generator, LSTMController)
        priors = engine._generator.prior_system.priors
        length_priors = [p for p in priors if isinstance(p, LengthConstraint)]
        assert length_priors[0].min_ == 7

    def test_engine_respects_max_length(self) -> None:
        config = DiscoverConfig(min_length=3, max_length=25)
        torch.manual_seed(SEED)
        engine = build_engine(config)
        assert isinstance(engine._generator, LSTMController)
        priors = engine._generator.prior_system.priors
        length_priors = [p for p in priors if isinstance(p, LengthConstraint)]
        assert length_priors[0].max_ == 25

    def test_engine_validator_receives_min_length(self) -> None:
        config = DiscoverConfig(min_length=6, max_length=25)
        torch.manual_seed(SEED)
        engine = build_engine(config)
        assert engine._validator.min_length == 6

    def test_custom_config_passthrough(self, custom_config: DiscoverConfig) -> None:
        torch.manual_seed(SEED)
        engine = build_engine(custom_config)

        assert isinstance(engine._generator, LSTMController)
        assert engine._generator.rnn.hidden_size == custom_config.num_units
        assert len(engine._generator.rnn.cells) == custom_config.num_layers
        assert (
            engine._generator.action_embedding.embedding_dim
            == custom_config.embedding_dim
        )

        assert engine._strategy.epsilon == custom_config.epsilon
        assert engine._strategy.baseline == custom_config.baseline
        assert engine._strategy.entropy_weight == custom_config.entropy_weight
        assert engine._strategy.gamma == custom_config.gamma

        assert engine.batch_size == custom_config.batch_size

        result = EvaluationResult(
            mse=0.5,
            nmse=0.5,
            r2=0.5,
            complexity=3,
            is_valid=True,
            expression="x",
        )
        assert engine._reward_adapter(result) == pytest.approx(
            compute_reward(result, alpha=custom_config.reward_alpha),
        )












class TestLearningRateAndEntropyGammaFlow:

    def test_config_has_learning_rate_with_default(self) -> None:
        config = DiscoverConfig()
        assert hasattr(config, "learning_rate")
        assert config.learning_rate == pytest.approx(0.001)

    def test_config_has_entropy_gamma_with_default(self) -> None:
        config = DiscoverConfig()
        assert hasattr(config, "entropy_gamma")
        assert config.entropy_gamma == pytest.approx(1.0)

    def test_custom_learning_rate_flows_to_strategy(self) -> None:
        config = DiscoverConfig(learning_rate=0.01)
        strategy = build_strategy(config)
        assert strategy._learning_rate == pytest.approx(0.01)

    def test_custom_entropy_gamma_flows_to_strategy(self) -> None:
        config = DiscoverConfig(entropy_gamma=0.7)
        strategy = build_strategy(config)
        assert strategy.entropy_gamma == pytest.approx(0.7)

    def test_full_engine_uses_custom_learning_rate(self) -> None:
        config = DiscoverConfig(learning_rate=0.05)
        torch.manual_seed(SEED)
        engine = build_engine(config)
        assert engine._strategy._learning_rate == pytest.approx(0.05)

    def test_full_engine_uses_custom_entropy_gamma(self) -> None:
        config = DiscoverConfig(entropy_gamma=0.7)
        torch.manual_seed(SEED)
        engine = build_engine(config)
        assert engine._strategy.entropy_gamma == pytest.approx(0.7)


class TestConfigUnification:

    def test_plugin_uses_canonical_config(self) -> None:
        from kd.search.discover import plugin
        from kd.search.discover.config import DiscoverConfig as CanonicalConfig


        plugin_config_cls = getattr(plugin, "DiscoverConfig", None)
        assert plugin_config_cls is CanonicalConfig

    def test_config_has_reward_alpha(self) -> None:
        config = DiscoverConfig()
        assert hasattr(config, "reward_alpha")
        assert config.reward_alpha == pytest.approx(0.01)

    def test_config_has_n_iterations(self) -> None:
        config = DiscoverConfig()
        assert hasattr(config, "n_iterations")
        assert config.n_iterations == 2000







class TestPluginBuilderIntegration:

    def test_plugin_prepare_produces_engine(self) -> None:
        from kd.search.discover.plugin import DISCOVERPlugin

        mock_evaluator = MockEvaluator()
        components = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=mock_evaluator,
            context=MagicMock(training_result=None),
            registry=MagicMock(),
        )

        torch.manual_seed(SEED)
        plugin = DISCOVERPlugin()
        plugin.prepare(components)

        candidates = plugin.propose(n=8)
        assert len(candidates) > 0

    def test_plugin_no_private_build_methods(self) -> None:
        from kd.search.discover.plugin import DISCOVERPlugin


        assert not hasattr(DISCOVERPlugin, "_build_prior_system")
        assert not hasattr(DISCOVERPlugin, "_build_controller")
        assert not hasattr(DISCOVERPlugin, "_build_strategy")

    def test_plugin_prepare_delegates_to_builder(self) -> None:
        from kd.search.discover.plugin import DISCOVERPlugin

        mock_evaluator = MockEvaluator()
        components = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=mock_evaluator,
            context=MagicMock(training_result=None),
            registry=MagicMock(),
        )

        torch.manual_seed(SEED)
        plugin = DISCOVERPlugin()
        with patch("kd.search.discover.plugin.build_engine") as mock_build:

            real_engine = build_engine(DiscoverConfig())
            mock_build.return_value = real_engine
            plugin.prepare(components)
            mock_build.assert_called_once()







class TestMainBuilderIntegration:

    def test_main_imports_builder(self) -> None:
        import kd.search.discover.__main__ as main_mod


        assert hasattr(main_mod, "build_engine")
        from kd.search.discover.builder import build_engine as builder_fn

        assert main_mod.build_engine is builder_fn







class TestPINNConfig:

    @pytest.mark.unit
    def test_creation_with_defaults(self) -> None:
        cfg = PINNConfig()
        assert cfg.number_layer == 8
        assert cfg.n_hidden == 20
        assert cfg.activation == "tanh"
        assert cfg.pretrain_epoch == 200_000
        assert cfg.pinn_epoch == 1_000
        assert cfg.lr == pytest.approx(0.001)
        assert cfg.coef_pde == pytest.approx(0.0)
        assert cfg.n_cycles == 2
        assert cfg.n_collocation == 50_000
        assert cfg.local_sample is True
        assert cfg.early_stop_patience == 500

    @pytest.mark.unit
    def test_is_frozen(self) -> None:
        cfg = PINNConfig()
        with pytest.raises(AttributeError):
            cfg.n_hidden = 64

    @pytest.mark.unit
    def test_has_slots(self) -> None:
        assert hasattr(PINNConfig, "__slots__")

    @pytest.mark.unit
    def test_custom_values(self) -> None:
        cfg = PINNConfig(
            number_layer=4,
            n_hidden=64,
            activation="sin",
            pretrain_epoch=100_000,
            pinn_epoch=500,
            lr=0.0001,
            coef_pde=0.1,
            n_cycles=5,
        )
        assert cfg.number_layer == 4
        assert cfg.n_hidden == 64
        assert cfg.activation == "sin"
        assert cfg.n_cycles == 5

    @pytest.mark.unit
    def test_activation_accepts_valid_literals(self) -> None:
        for act in ("tanh", "sin", "relu"):
            cfg = PINNConfig(activation=act)
            assert cfg.activation == act







class TestDiscoverConfigPINN:

    @pytest.mark.unit
    def test_pinn_default_none(self) -> None:
        cfg = DiscoverConfig()
        assert cfg.pinn is None

    @pytest.mark.unit
    def test_pinn_with_config(self) -> None:
        pinn = PINNConfig()
        cfg = DiscoverConfig(pinn=pinn)
        assert cfg.pinn is pinn
        assert cfg.pinn.n_hidden == 20

    @pytest.mark.unit
    def test_pinn_with_custom_config(self) -> None:
        pinn = PINNConfig(n_cycles=3, coef_pde=0.1)
        cfg = DiscoverConfig(pinn=pinn)
        assert cfg.pinn is not None
        assert cfg.pinn.n_cycles == 3
        assert cfg.pinn.coef_pde == pytest.approx(0.1)
