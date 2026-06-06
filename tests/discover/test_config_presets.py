
from __future__ import annotations

import pytest
import torch

from kd.search.discover.config import DiscoverConfig








BURGERS_OPERATORS = ["add", "mul", "div", "diff_x", "diff2_x", "diff3_x", "n2", "n3"]
BURGERS_STATE_VARS = ["u"]
BURGERS_COORD_VARS = ["x", "t"]


BURGERS_BATCH_SIZE = 500
BURGERS_EPSILON = 0.02


BURGERS_LEARNING_RATE = 0.0025
BURGERS_ENTROPY_WEIGHT = 0.03
BURGERS_ENTROPY_GAMMA = 0.7
BURGERS_NUM_UNITS = 32
BURGERS_INITIALIZER = "zeros"




BURGERS_MIN_LENGTH = 2
BURGERS_MAX_LENGTH = 64
BURGERS_SOFT_LENGTH_LOC = 10.0
BURGERS_SOFT_LENGTH_SCALE = 5.0


CHAFEE_MAX_LENGTH = 256

SEED = 42







class TestBurgersPreset:

    def test_returns_discover_config(self) -> None:
        cfg = DiscoverConfig.burgers_preset()
        assert isinstance(cfg, DiscoverConfig)

    def test_is_frozen(self) -> None:
        cfg = DiscoverConfig.burgers_preset()
        with pytest.raises(AttributeError):
            cfg.batch_size = 1

    def test_n_iterations(self) -> None:
        cfg = DiscoverConfig.burgers_preset()
        assert cfg.n_iterations == 100



    def test_operators(self) -> None:
        cfg = DiscoverConfig.burgers_preset()
        assert cfg.library.operators == BURGERS_OPERATORS

    def test_operators_no_sub_sin_cos(self) -> None:
        cfg = DiscoverConfig.burgers_preset()
        for excluded in ("sub", "sin", "cos"):
            assert excluded not in cfg.library.operators

    def test_operators_include_n2_n3(self) -> None:
        cfg = DiscoverConfig.burgers_preset()
        assert "n2" in cfg.library.operators
        assert "n3" in cfg.library.operators

    def test_state_vars(self) -> None:
        cfg = DiscoverConfig.burgers_preset()
        assert cfg.library.state_vars == BURGERS_STATE_VARS

    def test_coord_vars(self) -> None:
        cfg = DiscoverConfig.burgers_preset()
        assert cfg.library.coord_vars == BURGERS_COORD_VARS



    def test_batch_size(self) -> None:
        cfg = DiscoverConfig.burgers_preset()
        assert cfg.batch_size == BURGERS_BATCH_SIZE

    def test_epsilon(self) -> None:
        cfg = DiscoverConfig.burgers_preset()
        assert cfg.epsilon == pytest.approx(BURGERS_EPSILON)



    def test_learning_rate(self) -> None:
        cfg = DiscoverConfig.burgers_preset()
        assert cfg.learning_rate == pytest.approx(BURGERS_LEARNING_RATE)

    def test_entropy_weight(self) -> None:
        cfg = DiscoverConfig.burgers_preset()
        assert cfg.entropy_weight == pytest.approx(BURGERS_ENTROPY_WEIGHT)

    def test_entropy_gamma(self) -> None:
        cfg = DiscoverConfig.burgers_preset()
        assert cfg.entropy_gamma == pytest.approx(BURGERS_ENTROPY_GAMMA)

    def test_num_units(self) -> None:
        cfg = DiscoverConfig.burgers_preset()
        assert cfg.num_units == BURGERS_NUM_UNITS

    def test_initializer(self) -> None:
        cfg = DiscoverConfig.burgers_preset()
        assert cfg.initializer == BURGERS_INITIALIZER



    def test_min_length(self) -> None:
        cfg = DiscoverConfig.burgers_preset()
        assert cfg.min_length == BURGERS_MIN_LENGTH

    def test_max_length(self) -> None:
        cfg = DiscoverConfig.burgers_preset()
        assert cfg.max_length == BURGERS_MAX_LENGTH

    def test_soft_length_loc(self) -> None:
        cfg = DiscoverConfig.burgers_preset()
        assert cfg.soft_length_loc == pytest.approx(BURGERS_SOFT_LENGTH_LOC)

    def test_soft_length_scale(self) -> None:
        cfg = DiscoverConfig.burgers_preset()
        assert cfg.soft_length_scale == pytest.approx(BURGERS_SOFT_LENGTH_SCALE)

    def test_all_priors_enabled(self) -> None:
        cfg = DiscoverConfig.burgers_preset()
        assert cfg.use_repeat_prior is True
        assert cfg.use_trig_prior is True
        assert cfg.use_inverse_prior is True
        assert cfg.use_diff_descendant_prior is True



    def test_override_batch_size(self) -> None:
        cfg = DiscoverConfig.burgers_preset(batch_size=64)
        assert cfg.batch_size == 64

        assert cfg.epsilon == pytest.approx(BURGERS_EPSILON)

    def test_override_learning_rate(self) -> None:
        cfg = DiscoverConfig.burgers_preset(learning_rate=0.001)
        assert cfg.learning_rate == pytest.approx(0.001)

    def test_override_n_iterations(self) -> None:
        cfg = DiscoverConfig.burgers_preset(n_iterations=500)
        assert cfg.n_iterations == 500

    def test_override_pinn(self) -> None:
        from kd.search.discover.config import PINNConfig

        pinn = PINNConfig(n_cycles=3)
        cfg = DiscoverConfig.burgers_preset(pinn=pinn)
        assert cfg.pinn is not None
        assert cfg.pinn.n_cycles == 3



    def test_differs_from_default_batch_size(self) -> None:
        default = DiscoverConfig()
        preset = DiscoverConfig.burgers_preset()
        assert preset.batch_size != default.batch_size

    def test_differs_from_default_epsilon(self) -> None:
        default = DiscoverConfig()
        preset = DiscoverConfig.burgers_preset()
        assert preset.epsilon != default.epsilon

    def test_differs_from_default_learning_rate(self) -> None:
        default = DiscoverConfig()
        preset = DiscoverConfig.burgers_preset()
        assert preset.learning_rate != default.learning_rate







class TestChafeePreset:

    def test_returns_discover_config(self) -> None:
        cfg = DiscoverConfig.chafee_preset()
        assert isinstance(cfg, DiscoverConfig)

    def test_max_length_256(self) -> None:
        cfg = DiscoverConfig.chafee_preset()
        assert cfg.max_length == CHAFEE_MAX_LENGTH

    def test_same_operators_as_burgers(self) -> None:
        cfg = DiscoverConfig.chafee_preset()
        assert cfg.library.operators == BURGERS_OPERATORS

    def test_same_training_params_as_burgers(self) -> None:
        cfg = DiscoverConfig.chafee_preset()
        assert cfg.batch_size == BURGERS_BATCH_SIZE
        assert cfg.epsilon == pytest.approx(BURGERS_EPSILON)
        assert cfg.learning_rate == pytest.approx(BURGERS_LEARNING_RATE)
        assert cfg.entropy_weight == pytest.approx(BURGERS_ENTROPY_WEIGHT)
        assert cfg.entropy_gamma == pytest.approx(BURGERS_ENTROPY_GAMMA)

    def test_same_controller_params_as_burgers(self) -> None:
        cfg = DiscoverConfig.chafee_preset()
        assert cfg.num_units == BURGERS_NUM_UNITS
        assert cfg.initializer == BURGERS_INITIALIZER

    def test_override_max_length(self) -> None:
        cfg = DiscoverConfig.chafee_preset(max_length=128)
        assert cfg.max_length == 128







class TestPresetBuildIntegration:

    def test_burgers_builds_engine(self) -> None:
        from kd.search.discover.builder import build_engine

        torch.manual_seed(SEED)
        cfg = DiscoverConfig.burgers_preset()
        engine = build_engine(cfg)
        assert engine.batch_size == BURGERS_BATCH_SIZE

    def test_burgers_engine_can_propose(self) -> None:
        from kd.search.discover.builder import build_engine

        torch.manual_seed(SEED)
        engine = build_engine(DiscoverConfig.burgers_preset())
        candidates = engine.propose()
        assert len(candidates) > 0

    def test_chafee_builds_engine(self) -> None:
        from kd.search.discover.builder import build_engine

        torch.manual_seed(SEED)
        cfg = DiscoverConfig.chafee_preset()
        engine = build_engine(cfg)
        assert engine.batch_size == BURGERS_BATCH_SIZE

    def test_burgers_prior_count(self) -> None:
        from kd.search.discover.builder import build_library, build_prior_system
        from kd.search.discover.tokens.prior import SoftLengthPrior

        cfg = DiscoverConfig.burgers_preset()
        library = build_library(cfg)
        ps = build_prior_system(library, cfg)

        assert len(ps.priors) == 7
        soft = [p for p in ps.priors if isinstance(p, SoftLengthPrior)]
        assert len(soft) == 1

    def test_burgers_initializer_zeros(self) -> None:
        from kd.search.discover.builder import (
            build_controller,
            build_library,
            build_prior_system,
        )

        torch.manual_seed(SEED)
        cfg = DiscoverConfig.burgers_preset()
        library = build_library(cfg)
        ps = build_prior_system(library, cfg)
        controller = build_controller(library, ps, cfg)
        assert controller.initializer == "zeros"
        for name, param in controller.rnn.named_parameters():
            assert torch.all(param == 0.0), f"rnn.{name} should be zero"
