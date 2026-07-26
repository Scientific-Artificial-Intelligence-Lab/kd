
from __future__ import annotations

import numpy as np
import pytest
import torch

from kd.search.discover.controller.lstm import LSTMController
from kd.search.discover.core.batch import Batch
from kd.search.discover.tokens.library import Library, LibraryConfig
from kd.search.discover.tokens.prior import (
    DiffChildConstraint,
    LengthConstraint,
    PriorSystem,
)
from kd.search.discover.training.strategy import BaselineState, RSPGStrategy





BURGERS_CONFIG = LibraryConfig(
    operators=["add", "mul", "sub", "div", "sin", "cos", "diff_x", "diff2_x"],
    state_vars=["u"],
    coord_vars=["x", "t"],
)

MAX_LENGTH = 15
MIN_LENGTH = 4
BATCH_SIZE = 20
SEED = 42






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
def batch(controller: LSTMController) -> Batch:
    torch.manual_seed(SEED + 1)
    np.random.seed(SEED + 1)
    return controller.sample(BATCH_SIZE)


def _make_strategy(**kwargs: object) -> RSPGStrategy:
    defaults: dict[str, object] = {
        "epsilon": 0.05,
        "baseline": "R_e",
        "entropy_weight": 0.005,
    }
    defaults.update(kwargs)
    return RSPGStrategy(**defaults)


def _quantile(rewards: torch.Tensor, epsilon: float) -> float:
    return float(np.quantile(
        rewards.numpy(), 1 - epsilon, method="higher",
    ))







class TestQuantileFiltering:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_baseline_R_e_matches_quantile_B100(
        self, controller: LSTMController,
    ) -> None:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        big_batch = controller.sample(100)
        rewards = torch.arange(100, dtype=torch.float32)
        strategy = _make_strategy(epsilon=0.05, baseline="R_e")

        loss_info, _ = strategy.train_step(
            controller, big_batch, rewards, BaselineState(),
        )

        expected_q = _quantile(rewards, 0.05)
        assert loss_info["baseline"] == pytest.approx(expected_q)

    @pytest.mark.unit
    def test_all_same_reward_keeps_all(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy(epsilon=0.05, baseline="R_e")
        rewards = torch.ones(BATCH_SIZE)

        loss_info, _ = strategy.train_step(
            controller, batch, rewards, BaselineState(),
        )
        assert loss_info["baseline"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_epsilon_one_keeps_all(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy(epsilon=1.0, baseline="R_e")
        rewards = torch.arange(BATCH_SIZE, dtype=torch.float32)

        loss_info, _ = strategy.train_step(
            controller, batch, rewards, BaselineState(),
        )

        assert loss_info["baseline"] == pytest.approx(0.0)







class TestBaselineComputation:

    @pytest.mark.unit
    def test_R_e_equals_quantile(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy(epsilon=0.2, baseline="R_e")
        rewards = torch.arange(BATCH_SIZE, dtype=torch.float32)

        loss_info, _ = strategy.train_step(
            controller, batch, rewards, BaselineState(),
        )
        assert loss_info["baseline"] == pytest.approx(
            _quantile(rewards, 0.2),
        )

    @pytest.mark.unit
    def test_ewma_R_initializes_to_mean_filtered(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy(
            epsilon=0.2, baseline="ewma_R", gamma=0.99,
        )
        rewards = torch.arange(BATCH_SIZE, dtype=torch.float32)

        _, new_state = strategy.train_step(
            controller, batch, rewards, BaselineState(),
        )

        q = _quantile(rewards, 0.2)
        filtered = rewards[rewards >= q].numpy()
        assert new_state.ewma_reward == pytest.approx(
            float(np.mean(filtered)), rel=1e-5,
        )
        assert new_state.n_updates == 1

    @pytest.mark.unit
    def test_ewma_R_decays_on_subsequent_call(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        gamma = 0.9
        strategy = _make_strategy(
            epsilon=0.2, baseline="ewma_R", gamma=gamma,
        )
        rewards = torch.arange(BATCH_SIZE, dtype=torch.float32)


        _, state1 = strategy.train_step(
            controller, batch, rewards, BaselineState(),
        )


        rewards2 = torch.arange(BATCH_SIZE, dtype=torch.float32) * 2.0
        _, state2 = strategy.train_step(
            controller, batch, rewards2, state1,
        )

        q2 = _quantile(rewards2, 0.2)
        filtered2 = rewards2[rewards2 >= q2].numpy()

        expected = (
            (1 - gamma) * float(np.mean(filtered2))
            + gamma * state1.ewma_reward
        )
        assert state2.ewma_reward == pytest.approx(expected, rel=1e-5)
        assert state2.n_updates == 2

    @pytest.mark.unit
    def test_combined_baseline_two_calls(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        gamma = 0.9
        strategy = _make_strategy(
            epsilon=0.2, baseline="combined", gamma=gamma,
        )
        rewards = torch.arange(BATCH_SIZE, dtype=torch.float32)


        loss_info1, state1 = strategy.train_step(
            controller, batch, rewards, BaselineState(),
        )
        q1 = _quantile(rewards, 0.2)
        filtered1 = rewards[rewards >= q1].numpy()
        expected_ewma1 = float(np.mean(filtered1)) - q1
        expected_b1 = q1 + expected_ewma1
        assert loss_info1["baseline"] == pytest.approx(expected_b1, rel=1e-5)


        rewards2 = torch.arange(BATCH_SIZE, dtype=torch.float32) * 3.0
        loss_info2, state2 = strategy.train_step(
            controller, batch, rewards2, state1,
        )
        q2 = _quantile(rewards2, 0.2)
        filtered2 = rewards2[rewards2 >= q2].numpy()

        expected_ewma2 = (
            (1 - gamma) * (float(np.mean(filtered2)) - q2)
            + gamma * expected_ewma1
        )
        expected_b2 = q2 + expected_ewma2
        assert loss_info2["baseline"] == pytest.approx(expected_b2, rel=1e-5)

    @pytest.mark.unit
    def test_baseline_state_persists(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy(epsilon=0.2, baseline="ewma_R")
        rewards = torch.rand(BATCH_SIZE)

        state = BaselineState()
        assert state.n_updates == 0
        assert state.ewma_reward == 0.0

        _, state = strategy.train_step(controller, batch, rewards, state)
        assert state.n_updates == 1
        assert state.ewma_reward != 0.0

        _, state = strategy.train_step(controller, batch, rewards, state)
        assert state.n_updates == 2







class TestLossComputation:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_total_equals_pg_plus_entropy(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy(epsilon=0.5)
        rewards = torch.rand(BATCH_SIZE)

        loss_info, _ = strategy.train_step(
            controller, batch, rewards, BaselineState(),
        )

        expected = loss_info["pg_loss"] + loss_info["entropy_loss"]
        assert loss_info["total_loss"] == pytest.approx(expected, rel=1e-5)

    @pytest.mark.unit
    def test_loss_values_are_finite_floats(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy(epsilon=0.2)
        rewards = torch.rand(BATCH_SIZE)

        loss_info, _ = strategy.train_step(
            controller, batch, rewards, BaselineState(),
        )

        for key in ("pg_loss", "entropy_loss", "total_loss", "baseline"):
            assert isinstance(loss_info[key], float), f"{key} not float"
            assert np.isfinite(loss_info[key]), f"{key} not finite"

    @pytest.mark.unit
    def test_loss_matches_manual_computation(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        epsilon = 0.3
        ew = 0.01
        strategy = _make_strategy(
            epsilon=epsilon, baseline="R_e", entropy_weight=ew,
        )
        rewards = torch.arange(BATCH_SIZE, dtype=torch.float32)





        q = _quantile(rewards, epsilon)
        keep = (rewards >= q).numpy()
        filtered_batch = Batch(
            actions=batch.actions[keep],
            obs=batch.obs[keep],
            priors=batch.priors[keep],
            lengths=batch.lengths[keep],
        )
        with torch.no_grad():
            neglogp, entropy = controller.make_neglogp_and_entropy(
                filtered_batch,
            )
        filtered_r = rewards[rewards >= q]

        expected_pg = float(torch.mean((filtered_r - q) * neglogp))
        expected_ent = float(-ew * torch.mean(entropy))


        loss_info, _ = strategy.train_step(
            controller, batch, rewards, BaselineState(),
        )

        assert loss_info["pg_loss"] == pytest.approx(expected_pg, rel=1e-4)
        assert loss_info["entropy_loss"] == pytest.approx(
            expected_ent, rel=1e-4,
        )







class TestGradientFlow:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_parameters_change_after_train_step(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy(epsilon=0.5)
        rewards = torch.rand(BATCH_SIZE)

        params_before = {
            name: p.clone().detach()
            for name, p in controller.named_parameters()
        }

        strategy.train_step(controller, batch, rewards, BaselineState())

        any_changed = any(
            not torch.equal(params_before[name], p)
            for name, p in controller.named_parameters()
        )
        assert any_changed, "No parameters changed after train_step"

    @pytest.mark.unit
    def test_grad_norm_is_positive(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy(epsilon=0.5)

        rewards = torch.arange(BATCH_SIZE, dtype=torch.float32)

        loss_info, _ = strategy.train_step(
            controller, batch, rewards, BaselineState(),
        )

        assert "grad_norm" in loss_info
        assert loss_info["grad_norm"] > 0.0







class TestInvariants:

    @pytest.mark.unit
    def test_all_zero_rewards_zero_pg_loss(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy(epsilon=0.5, baseline="R_e")
        rewards = torch.zeros(BATCH_SIZE)

        loss_info, _ = strategy.train_step(
            controller, batch, rewards, BaselineState(),
        )

        assert loss_info["pg_loss"] == pytest.approx(0.0, abs=1e-7)

    @pytest.mark.unit
    def test_entropy_weight_zero_removes_entropy(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy(epsilon=0.5, entropy_weight=0.0)
        rewards = torch.rand(BATCH_SIZE)

        loss_info, _ = strategy.train_step(
            controller, batch, rewards, BaselineState(),
        )
        assert loss_info["entropy_loss"] == pytest.approx(0.0, abs=1e-7)

    @pytest.mark.unit
    def test_return_types(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy()
        rewards = torch.rand(BATCH_SIZE)

        result = strategy.train_step(
            controller, batch, rewards, BaselineState(),
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        loss_info, new_state = result
        assert isinstance(loss_info, dict)
        assert isinstance(new_state, BaselineState)

    @pytest.mark.unit
    def test_nan_reward_raises(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy(epsilon=0.5)
        rewards = torch.ones(BATCH_SIZE)
        rewards[3] = float("nan")

        with pytest.raises(ValueError, match="finite"):
            strategy.train_step(
                controller, batch, rewards, BaselineState(),
            )

    @pytest.mark.unit
    def test_inf_reward_raises(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy(epsilon=0.5)
        rewards = torch.ones(BATCH_SIZE)
        rewards[0] = float("inf")

        with pytest.raises(ValueError, match="finite"):
            strategy.train_step(
                controller, batch, rewards, BaselineState(),
            )

    @pytest.mark.unit
    def test_R_e_increments_n_updates(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy(epsilon=0.5, baseline="R_e")
        rewards = torch.rand(BATCH_SIZE)

        state = BaselineState()
        _, state = strategy.train_step(controller, batch, rewards, state)
        assert state.n_updates == 1

        _, state = strategy.train_step(controller, batch, rewards, state)
        assert state.n_updates == 2







class TestValidMaskFiltering:

    @pytest.mark.unit
    def test_apply_valid_mask_removes_invalid_rows(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy(epsilon=0.5)
        rewards = torch.arange(BATCH_SIZE, dtype=torch.float32)
        valid_mask = np.zeros(BATCH_SIZE, dtype=bool)
        valid_mask[0] = True
        valid_mask[2] = True
        valid_mask[4] = True

        filtered_batch, filtered_rewards = strategy._apply_valid_mask(
            batch, rewards, valid_mask,
        )

        assert filtered_rewards.shape[0] == 3
        assert filtered_batch.actions.shape[0] == 3

        assert filtered_rewards[0].item() == pytest.approx(0.0)
        assert filtered_rewards[1].item() == pytest.approx(2.0)
        assert filtered_rewards[2].item() == pytest.approx(4.0)

    @pytest.mark.unit
    def test_empty_batch_after_valid_mask_returns_empty_loss_info(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy(epsilon=0.5)
        rewards = torch.arange(BATCH_SIZE, dtype=torch.float32)
        valid_mask = np.zeros(BATCH_SIZE, dtype=bool)
        baseline = BaselineState(ewma_reward=1.5, n_updates=3)

        loss_info, new_state = strategy.train_step(
            controller, batch, rewards, baseline,
            valid_mask=valid_mask,
        )


        assert new_state is baseline

        assert loss_info["pg_loss"] == pytest.approx(0.0)
        assert loss_info["entropy_loss"] == pytest.approx(0.0)
        assert loss_info["total_loss"] == pytest.approx(0.0)

        for key in ("pg_loss", "entropy_loss", "total_loss",
                     "baseline", "reward", "grad_norm"):
            assert key in loss_info, f"Missing key in empty loss_info: {key}"

    @pytest.mark.unit
    def test_empty_batch_does_not_call_optimizer_step(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy(epsilon=0.5)
        rewards = torch.ones(BATCH_SIZE)


        strategy.train_step(
            controller, batch, rewards, BaselineState(),
        )


        params_before = {
            name: p.clone().detach()
            for name, p in controller.named_parameters()
        }


        all_invalid = np.zeros(BATCH_SIZE, dtype=bool)
        strategy.train_step(
            controller, batch, rewards, BaselineState(),
            valid_mask=all_invalid,
        )


        for name, p in controller.named_parameters():
            assert torch.equal(params_before[name], p), (
                f"Parameter {name} changed despite empty filtered batch"
            )

    @pytest.mark.unit
    def test_all_valid_mask_passes_through(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy(epsilon=0.5)
        rewards = torch.arange(BATCH_SIZE, dtype=torch.float32)
        valid_mask = np.ones(BATCH_SIZE, dtype=bool)

        loss_info, _ = strategy.train_step(
            controller, batch, rewards, BaselineState(),
            valid_mask=valid_mask,
        )


        assert loss_info["baseline"] == pytest.approx(
            _quantile(rewards, 0.5),
        )

    @pytest.mark.unit
    def test_mixed_valid_invalid_trains_only_on_valid(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy(epsilon=1.0)

        rewards = torch.zeros(BATCH_SIZE)
        valid_mask = np.zeros(BATCH_SIZE, dtype=bool)

        valid_mask[:5] = True
        rewards[:5] = 10.0


        loss_info, _ = strategy.train_step(
            controller, batch, rewards, BaselineState(),
            valid_mask=valid_mask,
        )



        assert loss_info["baseline"] == pytest.approx(10.0)


    @pytest.mark.unit
    def test_quantile_computed_on_valid_only(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy(epsilon=0.5, baseline="R_e")
        rewards = torch.zeros(BATCH_SIZE)
        valid_mask = np.zeros(BATCH_SIZE, dtype=bool)

        valid_mask[:10] = True
        rewards[:10] = torch.arange(1, 11, dtype=torch.float32)


        loss_info, _ = strategy.train_step(
            controller, batch, rewards, BaselineState(),
            valid_mask=valid_mask,
        )


        valid_rewards = torch.arange(1, 11, dtype=torch.float32)
        expected_q = _quantile(valid_rewards, 0.5)
        assert loss_info["baseline"] == pytest.approx(expected_q)

    @pytest.mark.unit
    def test_none_valid_mask_behaves_as_all_valid(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        strategy = _make_strategy(epsilon=0.5)
        rewards = torch.arange(BATCH_SIZE, dtype=torch.float32)


        loss_info, _ = strategy.train_step(
            controller, batch, rewards, BaselineState(),
            valid_mask=None,
        )

        expected_q = _quantile(rewards, 0.5)
        assert loss_info["baseline"] == pytest.approx(expected_q)







class TestLearningRateParam:

    @pytest.mark.unit
    def test_strategy_accepts_learning_rate(self) -> None:
        strategy = RSPGStrategy(
            epsilon=0.05,
            baseline="R_e",
            entropy_weight=0.005,
            learning_rate=0.01,
        )
        assert strategy._learning_rate == 0.01

    @pytest.mark.unit
    def test_default_learning_rate(self) -> None:
        strategy = RSPGStrategy(epsilon=0.05, baseline="R_e")
        assert strategy._learning_rate == pytest.approx(0.001)

    @pytest.mark.unit
    def test_custom_learning_rate_used_by_optimizer(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        custom_lr = 0.05
        strategy = RSPGStrategy(
            epsilon=0.5,
            baseline="R_e",
            entropy_weight=0.005,
            learning_rate=custom_lr,
        )
        rewards = torch.arange(BATCH_SIZE, dtype=torch.float32)
        strategy.train_step(controller, batch, rewards, BaselineState())


        assert strategy._optimizer is not None
        actual_lr = strategy._optimizer.param_groups[0]["lr"]
        assert actual_lr == pytest.approx(custom_lr)

    @pytest.mark.unit
    def test_reset_optimizer(self) -> None:
        strategy = RSPGStrategy(epsilon=0.05, baseline="R_e")
        strategy._pending_optimizer_state = {"fake": True}
        strategy.reset_optimizer()
        assert strategy._optimizer is None
        assert strategy._pending_optimizer_state is None







class TestObservabilityWarnings:

    @pytest.mark.unit
    def test_filter_down_to_single_sample_warns(
        self,
        controller: LSTMController,
        batch: Batch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        strategy = _make_strategy(epsilon=0.001, baseline="R_e")
        rewards = torch.arange(BATCH_SIZE, dtype=torch.float32)

        with caplog.at_level(
            logging.WARNING, logger="kd.search.discover.training.strategy",
        ):
            strategy.train_step(controller, batch, rewards, BaselineState())

        assert any(
            "quantile" in rec.message.lower()
            and "sample" in rec.message.lower()
            for rec in caplog.records
        ), (
            "Expected a warning about degenerate quantile filtering; "
            f"got: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.unit
    def test_grad_norm_all_none_logs_debug(
        self,
        controller: LSTMController,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        strategy = _make_strategy()
        for parameter in controller.parameters():
            parameter.grad = None

        with caplog.at_level(
            logging.DEBUG, logger="kd.search.discover.training.strategy",
        ):
            result = strategy._grad_norm(controller)

        assert result == 0.0
        assert any(
            "grad" in rec.message.lower() and "none" in rec.message.lower()
            for rec in caplog.records
        ), (
            "Expected a DEBUG trace when every parameter lacks a gradient; "
            f"got: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.unit
    def test_grad_norm_with_gradients_does_not_log(
        self,
        controller: LSTMController,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        strategy = _make_strategy()

        for parameter in controller.parameters():
            parameter.grad = torch.zeros_like(parameter)

        with caplog.at_level(
            logging.DEBUG, logger="kd.search.discover.training.strategy",
        ):
            strategy._grad_norm(controller)

        assert not any(
            "grad" in rec.message.lower() and "none" in rec.message.lower()
            for rec in caplog.records
        ), (
            "_grad_norm must not log the all-None trace when gradients "
            f"exist; got: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.unit
    def test_filter_batch_warning_is_sticky(
        self,
        controller: LSTMController,
        batch: Batch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        strategy = _make_strategy(epsilon=0.001, baseline="R_e")
        rewards = torch.arange(BATCH_SIZE, dtype=torch.float32)

        with caplog.at_level(
            logging.WARNING, logger="kd.search.discover.training.strategy",
        ):
            state = BaselineState()
            for _ in range(3):
                _, state = strategy.train_step(
                    controller, batch, rewards, state,
                )

        warnings = [
            rec
            for rec in caplog.records
            if "quantile" in rec.message.lower()
            and "sample" in rec.message.lower()
        ]
        assert len(warnings) == 1, (
            "degenerate-filter warning should be sticky (one per strategy); "
            f"got {len(warnings)} warnings: {[w.message for w in warnings]}"
        )

    @pytest.mark.unit
    def test_grad_norm_debug_is_sticky(
        self,
        controller: LSTMController,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        strategy = _make_strategy()
        for parameter in controller.parameters():
            parameter.grad = None

        with caplog.at_level(
            logging.DEBUG, logger="kd.search.discover.training.strategy",
        ):
            for _ in range(3):
                strategy._grad_norm(controller)

        traces = [
            rec
            for rec in caplog.records
            if "grad" in rec.message.lower() and "none" in rec.message.lower()
        ]
        assert len(traces) == 1, (
            "_grad_norm DEBUG should be sticky (one per strategy); "
            f"got {len(traces)} traces: {[t.message for t in traces]}"
        )







class TestOptimizerRestoreReassertsLiveLR:

    @staticmethod
    def _donor_state(
        controller: LSTMController, batch: Batch, lr: float
    ) -> dict[str, object]:


        donor = _make_strategy(epsilon=0.5, learning_rate=lr)
        rewards = torch.linspace(0.1, 1.0, BATCH_SIZE)
        donor.train_step(controller, batch, rewards, BaselineState())
        state = donor.optimizer_state
        assert state is not None
        return state

    @pytest.mark.unit
    def test_deferred_load_reasserts_live_lr_and_keeps_moments(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        donor_lr, live_lr = 0.001, 0.01
        donor_state = self._donor_state(controller, batch, donor_lr)


        assert all(g["lr"] == donor_lr for g in donor_state["param_groups"])
        assert donor_state["state"]
        assert donor_lr != live_lr

        subject = _make_strategy(epsilon=0.5, learning_rate=live_lr)


        subject.optimizer_state = donor_state




        optimizer = subject._get_optimizer(controller)
        assert all(g["lr"] == live_lr for g in optimizer.param_groups)
        loaded = optimizer.state_dict()["state"]
        assert loaded
        for idx, param_state in donor_state["state"].items():
            torch.testing.assert_close(
                loaded[idx]["exp_avg"], param_state["exp_avg"]
            )

    @pytest.mark.unit
    def test_immediate_load_reasserts_live_lr_and_keeps_moments(
        self, controller: LSTMController, batch: Batch,
    ) -> None:
        donor_lr, live_lr = 0.001, 0.01
        donor_state = self._donor_state(controller, batch, donor_lr)

        subject = _make_strategy(epsilon=0.5, learning_rate=live_lr)


        optimizer = subject._get_optimizer(controller)
        subject.optimizer_state = donor_state
        assert all(g["lr"] == live_lr for g in optimizer.param_groups)
        loaded = optimizer.state_dict()["state"]
        assert loaded
        for idx, param_state in donor_state["state"].items():
            torch.testing.assert_close(
                loaded[idx]["exp_avg"], param_state["exp_avg"]
            )
