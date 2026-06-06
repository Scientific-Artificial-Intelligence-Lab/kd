
from __future__ import annotations

import numpy as np
import pytest
import torch

from kd.search.discover.controller.lstm import LSTMController
from kd.search.discover.controller.tree_state import BatchTracker, IncrementalTracker
from kd.search.discover.core.batch import Batch
from kd.search.discover.tokens.library import Library, LibraryConfig
from kd.search.discover.tokens.prior import (
    DiffChildConstraint,
    LengthConstraint,
    Prior,
    PriorContext,
    PriorSystem,
)





BURGERS_CONFIG = LibraryConfig(
    operators=["add", "mul", "sub", "div", "sin", "cos", "diff_x", "diff2_x"],
    state_vars=["u"],
    coord_vars=["x", "t"],
)

MAX_LENGTH = 30
MIN_LENGTH = 4
BATCH_SIZE = 16
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
def controller(lib: Library, prior_system: PriorSystem) -> LSTMController:
    return LSTMController(
        library=lib,
        prior_system=prior_system,
        num_units=32,
        num_layers=1,
        embedding_dim=8,
    )






class TestBatch:

    @pytest.mark.unit
    def test_batch_fields_exist(self) -> None:
        actions = np.zeros((2, 5), dtype=np.int32)
        obs = np.zeros((2, 4, 5), dtype=np.float32)
        priors = np.zeros((2, 5, 10), dtype=np.float32)
        lengths = np.array([3, 5], dtype=np.int32)
        batch = Batch(
            actions=actions,
            obs=obs,
            priors=priors,
            lengths=lengths,
        )
        assert batch.actions.shape == (2, 5)
        assert batch.obs.shape == (2, 4, 5)
        assert batch.priors.shape == (2, 5, 10)
        assert batch.lengths.shape == (2,)






class TestSampleShapes:

    @pytest.mark.smoke
    def test_sample_returns_batch(self, controller: LSTMController) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        assert isinstance(batch, Batch)

    @pytest.mark.unit
    def test_actions_shape(self, controller: LSTMController,
                           lib: Library) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        assert batch.actions.shape[0] == BATCH_SIZE
        assert batch.actions.shape[1] == int(batch.lengths.max())
        assert batch.actions.dtype == np.int32

    @pytest.mark.unit
    def test_obs_shape(self, controller: LSTMController) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        B, L = batch.actions.shape
        assert batch.obs.shape == (B, 4, L)
        assert batch.obs.dtype == np.float32

    @pytest.mark.unit
    def test_priors_shape(self, controller: LSTMController,
                          lib: Library) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        B, L = batch.actions.shape
        n_tokens = len(lib.tokens)
        assert batch.priors.shape == (B, L, n_tokens)
        assert batch.priors.dtype == np.float32

    @pytest.mark.unit
    def test_lengths_shape(self, controller: LSTMController) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        assert batch.lengths.shape == (BATCH_SIZE,)
        assert batch.lengths.dtype == np.int32

    @pytest.mark.unit
    def test_lengths_within_bounds(self, controller: LSTMController) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        assert np.all(batch.lengths >= 1)
        assert np.all(batch.lengths <= MAX_LENGTH)






class TestExpressionValidity:

    @pytest.mark.smoke
    def test_dangling_zero_at_length(self, controller: LSTMController,
                                     lib: Library) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        for i in range(BATCH_SIZE):
            length = int(batch.lengths[i])
            tokens = batch.actions[i, :length].tolist()

            dangling = 1
            for token_idx in tokens:
                dangling += lib[token_idx].arity - 1
            assert dangling == 0, (
                f"Sequence {i} has dangling={dangling} at length={length}, "
                f"tokens={tokens}"
            )

    @pytest.mark.unit
    def test_actions_in_valid_range(self, controller: LSTMController,
                                    lib: Library) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        n_tokens = len(lib.tokens)
        for i in range(BATCH_SIZE):
            length = int(batch.lengths[i])
            actions = batch.actions[i, :length]
            assert np.all(actions >= 0)
            assert np.all(actions < n_tokens)






class TestPriorApplication:

    @pytest.mark.smoke
    def test_forbidden_tokens_not_sampled(self, controller: LSTMController,
                                          lib: Library) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        for i in range(BATCH_SIZE):
            length = int(batch.lengths[i])
            for t in range(length):
                action = batch.actions[i, t]
                adjustment = batch.priors[i, t, action]
                assert adjustment == 0.0, (
                    f"Seq {i}, step {t}: action={action} has adjustment={adjustment}"
                )

    @pytest.mark.unit
    def test_priors_are_logit_adjustments(
        self, controller: LSTMController,
    ) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        unique_vals = set(np.unique(batch.priors).tolist())
        assert unique_vals.issubset({0.0, float("-inf")})

class TestSamplingLoopBatchedTracker:

    @pytest.mark.unit
    def test_sample_with_seed_produces_consistent_shape(
        self, controller: LSTMController,
    ) -> None:
        torch.manual_seed(SEED)
        batch1 = controller.sample(BATCH_SIZE)
        torch.manual_seed(SEED)
        batch2 = controller.sample(BATCH_SIZE)
        assert batch1.actions.shape == batch2.actions.shape
        assert batch1.obs.shape == batch2.obs.shape
        assert batch1.priors.shape == batch2.priors.shape
        assert batch1.lengths.shape == batch2.lengths.shape

    @pytest.mark.unit
    def test_sample_actions_match_bit_exact_given_seed(
        self, controller: LSTMController,
    ) -> None:
        torch.manual_seed(SEED)
        batch1 = controller.sample(BATCH_SIZE)
        torch.manual_seed(SEED)
        batch2 = controller.sample(BATCH_SIZE)
        np.testing.assert_array_equal(batch1.actions, batch2.actions)

    @pytest.mark.unit
    def test_obs_from_sample_equals_batch_tracker(
        self, controller: LSTMController, lib: Library,
    ) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        obs_rebuilt = BatchTracker(lib).compute_obs(batch.actions)
        np.testing.assert_array_equal(batch.obs, obs_rebuilt)

    @pytest.mark.unit
    def test_sample_does_not_require_tracker_parameter(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        controller = LSTMController(
            library=lib,
            prior_system=prior_system,
            num_units=16,
            num_layers=1,
            embedding_dim=4,
        )
        assert isinstance(controller, LSTMController)

    @pytest.mark.unit
    def test_sample_finished_rows_frozen(self, lib: Library) -> None:
        prior_system = PriorSystem(lib, [])
        controller = LSTMController(
            library=lib,
            prior_system=prior_system,
            num_units=8,
            num_layers=1,
            embedding_dim=4,
        )
        planned_actions = [
            np.array([0, 3], dtype=np.int32),
            np.array([lib.EMPTY_ACTION, 0], dtype=np.int32),
            np.array([lib.EMPTY_ACTION, 1], dtype=np.int32),
        ]

        def scripted_sample_step(
            tracker: IncrementalTracker,
            obs: np.ndarray,
            hidden: list[tuple[torch.Tensor, torch.Tensor]],
            finished: np.ndarray,
            step_idx: int,
        ) -> tuple[np.ndarray, list[tuple[torch.Tensor, torch.Tensor]]]:
            del tracker, obs, finished
            return planned_actions[step_idx], hidden

        controller._sample_step = scripted_sample_step

        batch = controller.sample(2)

        np.testing.assert_array_equal(batch.lengths, np.array([1, 3], dtype=np.int32))
        assert batch.actions[0, 0] == 0
        assert np.all(batch.actions[0, 1:] == lib.EMPTY_ACTION)
        np.testing.assert_array_equal(
            batch.actions[1, :3],
            np.array([3, 0, 1], dtype=np.int32),
        )
        assert batch.obs[0, 3, int(batch.lengths[0]) - 1] == 1.0
        assert batch.obs[1, 3, int(batch.lengths[1]) - 1] == 1.0






class TestTermination:

    @pytest.mark.unit
    def test_padding_after_length(self, controller: LSTMController,
                                  lib: Library) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        _, L = batch.actions.shape
        for i in range(BATCH_SIZE):
            length = int(batch.lengths[i])
            if length < L:
                padding = batch.actions[i, length:]
                assert np.all(padding == lib.EMPTY_ACTION), (
                    f"Seq {i}: padding after length={length} is not EMPTY_ACTION"
                )

    @pytest.mark.unit
    def test_batch_size_one(self, controller: LSTMController) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(1)
        assert batch.actions.shape[0] == 1
        assert int(batch.lengths[0]) >= 1

    @pytest.mark.unit
    def test_max_length_boundary(self, lib: Library) -> None:
        small_max = 6
        ps = PriorSystem(lib, [
            LengthConstraint(lib, min_=2, max_=small_max),
        ])
        ctrl = LSTMController(
            library=lib,
            prior_system=ps,
            num_units=16,
            num_layers=1,
            embedding_dim=4,
        )
        torch.manual_seed(SEED)
        batch = ctrl.sample(64)

        for i in range(64):
            length = int(batch.lengths[i])
            tokens = batch.actions[i, :length].tolist()
            dangling = 1
            for token_idx in tokens:
                dangling += lib[token_idx].arity - 1
            assert dangling == 0, (
                f"Seq {i} at length={length} has dangling={dangling}"
            )

        assert np.any(batch.lengths <= small_max)






class TestNeglogpAndEntropy:

    @pytest.mark.smoke
    def test_gradients_flow(self, controller: LSTMController) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        controller.zero_grad()
        neglogp, entropy = controller.make_neglogp_and_entropy(
            batch, entropy_gamma=1.0,
        )
        loss = neglogp.mean()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in controller.parameters()
        )
        assert has_grad, "No gradients flowed to controller parameters"

    @pytest.mark.unit
    def test_neglogp_shape(self, controller: LSTMController) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        neglogp, _ = controller.make_neglogp_and_entropy(
            batch, entropy_gamma=1.0,
        )
        assert neglogp.shape == (BATCH_SIZE,)

    @pytest.mark.unit
    def test_entropy_shape(self, controller: LSTMController) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        _, entropy = controller.make_neglogp_and_entropy(
            batch, entropy_gamma=1.0,
        )
        assert entropy.shape == (BATCH_SIZE,)

    @pytest.mark.unit
    def test_neglogp_finite_positive(self, controller: LSTMController) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        neglogp, _ = controller.make_neglogp_and_entropy(
            batch, entropy_gamma=1.0,
        )
        assert torch.all(torch.isfinite(neglogp)), "neglogp contains non-finite"
        assert torch.all(neglogp > 0), "neglogp must be positive"

    @pytest.mark.unit
    def test_entropy_finite_nonneg(self, controller: LSTMController) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        _, entropy = controller.make_neglogp_and_entropy(
            batch, entropy_gamma=1.0,
        )
        assert torch.all(torch.isfinite(entropy)), "entropy contains non-finite"
        assert torch.all(entropy >= 0), "entropy must be non-negative"

    @pytest.mark.unit
    def test_entropy_gamma_decay(self, controller: LSTMController) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        _, entropy_nodecay = controller.make_neglogp_and_entropy(
            batch, entropy_gamma=1.0,
        )
        _, entropy_decay = controller.make_neglogp_and_entropy(
            batch, entropy_gamma=0.9,
        )

        assert torch.all(entropy_decay <= entropy_nodecay + 1e-6)

        multi_step = torch.as_tensor(batch.lengths > 1)
        if multi_step.any():
            assert torch.any(
                entropy_decay[multi_step] < entropy_nodecay[multi_step] - 1e-6
            ), "entropy_gamma not applied: no difference for multi-step sequences"

    @pytest.mark.unit
    def test_neglogp_on_filtered_batch(self, controller: LSTMController) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)

        k = BATCH_SIZE // 2
        filtered = Batch(
            actions=batch.actions[:k],
            obs=batch.obs[:k],
            priors=batch.priors[:k],
            lengths=batch.lengths[:k],
        )
        neglogp, entropy = controller.make_neglogp_and_entropy(
            filtered, entropy_gamma=1.0,
        )
        assert neglogp.shape == (k,)
        assert entropy.shape == (k,)

    @pytest.mark.unit
    def test_neglogp_varies_across_sequences(
        self, controller: LSTMController,
    ) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        neglogp, _ = controller.make_neglogp_and_entropy(
            batch, entropy_gamma=1.0,
        )
        assert neglogp.std() > 1e-6, (
            "neglogp has zero variance — likely ignoring batch content"
        )






class TestInvariants:

    @pytest.mark.unit
    def test_obs_matches_batch_tracker(self, controller: LSTMController,
                                       lib: Library) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        bt = BatchTracker(lib)
        expected_obs = bt.compute_obs(batch.actions)

        for i in range(BATCH_SIZE):
            length = int(batch.lengths[i])
            np.testing.assert_array_equal(
                batch.obs[i, :, :length],
                expected_obs[i, :, :length],
                err_msg=f"obs mismatch at sequence {i}",
            )

    @pytest.mark.unit
    def test_priors_match_prior_system_batch(
        self, controller: LSTMController, lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        expected_priors = prior_system.compute_batch(batch.actions, batch.obs)

        for i in range(BATCH_SIZE):
            length = int(batch.lengths[i])
            np.testing.assert_array_equal(
                batch.priors[i, :length,:],
                expected_priors[i, :length,:],
                err_msg=f"priors mismatch at sequence {i}",
            )

    @pytest.mark.unit
    def test_deterministic_with_seed(self, controller: LSTMController) -> None:
        torch.manual_seed(SEED)
        batch1 = controller.sample(BATCH_SIZE)
        torch.manual_seed(SEED)
        batch2 = controller.sample(BATCH_SIZE)
        np.testing.assert_array_equal(batch1.actions, batch2.actions)
        np.testing.assert_array_equal(batch1.lengths, batch2.lengths)
        np.testing.assert_array_equal(batch1.obs, batch2.obs)
        np.testing.assert_array_equal(batch1.priors, batch2.priors)

    @pytest.mark.unit
    def test_obs_channel_ranges(self, controller: LSTMController,
                                lib: Library) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        for i in range(BATCH_SIZE):
            length = int(batch.lengths[i])
            obs = batch.obs[i, :, :length]

            assert np.all(obs[0] >= 0)
            assert np.all(obs[0] < lib.n_action_inputs)

            assert np.all(obs[1] >= 0)
            assert np.all(obs[1] < lib.n_parent_inputs)

            assert np.all(obs[2] >= 0)
            assert np.all(obs[2] < lib.n_sibling_inputs)

            assert np.all(obs[3, :length - 1] >= 1), (
                "dangling must be >=1 before last token"
            )

    @pytest.mark.unit
    def test_min_length_enforced(self, controller: LSTMController) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)
        assert np.all(batch.lengths >= MIN_LENGTH), (
            f"Some sequences shorter than min_length={MIN_LENGTH}: "
            f"{batch.lengths[batch.lengths < MIN_LENGTH]}"
        )






class TestModuleIntegration:

    @pytest.mark.unit
    def test_is_nn_module(self, controller: LSTMController) -> None:
        assert isinstance(controller, torch.nn.Module)

    @pytest.mark.unit
    def test_has_parameters(self, controller: LSTMController) -> None:
        params = list(controller.parameters())
        assert len(params) > 0

    @pytest.mark.unit
    def test_embedding_dimensions(
        self,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        controller = LSTMController(
            library=lib,
            prior_system=prior_system,
            num_units=32,
            num_layers=1,
            embedding_dim=8,
            observe_action=True,
            observe_parent=True,
            observe_sibling=True,
            use_embedding=True,
        )
        emb_modules = [
            m for m in controller.modules()
            if isinstance(m, torch.nn.Embedding)
        ]
        assert len(emb_modules) == 3, (
            f"Expected 3 nn.Embedding modules, got {len(emb_modules)}"
        )
        emb_sizes = sorted(m.num_embeddings for m in emb_modules)

        assert lib.n_action_inputs in emb_sizes, (
            f"No embedding with n_action_inputs={lib.n_action_inputs}"
        )
        assert lib.n_parent_inputs in emb_sizes, (
            f"No embedding with n_parent_inputs={lib.n_parent_inputs}"
        )
        assert lib.n_sibling_inputs in emb_sizes, (
            f"No embedding with n_sibling_inputs={lib.n_sibling_inputs}"
        )

    @pytest.mark.unit
    def test_device_consistency(self, controller: LSTMController) -> None:
        devices = {p.device for p in controller.parameters()}
        assert len(devices) == 1, f"Parameters on multiple devices: {devices}"

    @pytest.mark.unit
    def test_sample_no_grad(self, controller: LSTMController) -> None:
        torch.manual_seed(SEED)
        controller.sample(BATCH_SIZE)

        for name, p in controller.named_parameters():
            assert p.grad is None or p.grad.abs().sum() == 0, (
                f"Parameter {name} has gradient after sample()"
            )






class TestEdgeCases:

    @pytest.mark.unit
    def test_large_batch(self, controller: LSTMController) -> None:
        torch.manual_seed(SEED)
        batch = controller.sample(64)
        assert batch.actions.shape[0] == 64
        assert np.all(batch.lengths >= 1)

    @pytest.mark.unit
    def test_multiple_samples_independent(
        self, controller: LSTMController,
    ) -> None:
        torch.manual_seed(SEED)
        batch1 = controller.sample(BATCH_SIZE)

        batch2 = controller.sample(BATCH_SIZE)

        assert not np.array_equal(batch1.actions, batch2.actions)

    @pytest.mark.unit
    def test_right_nested_obs_correct(self, lib: Library) -> None:

        add_idx = lib.name_to_index("add")
        x_idx = lib.name_to_index("x")
        u_idx = lib.name_to_index("u")
        tokens = np.array([[add_idx, x_idx, add_idx, x_idx, u_idx]],
                          dtype=np.int32)
        bt = BatchTracker(lib)
        obs = bt.compute_obs(tokens)

        assert obs.shape == (1, 4, 5)

        dangling = 1
        for t_idx in tokens[0]:
            dangling += lib[t_idx].arity - 1
        assert dangling == 0







class TestInitializer:

    @pytest.mark.unit
    def test_default_initializer_is_xavier(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        torch.manual_seed(SEED)
        ctrl = LSTMController(
            library=lib,
            prior_system=prior_system,
            num_units=32,
            num_layers=1,
            embedding_dim=8,
        )

        has_nonzero = any(
            p.dim() >= 2 and not torch.all(p == 0.0)
            for p in ctrl.parameters()
        )
        assert has_nonzero, "xavier default must produce at least one non-zero weight"

    @pytest.mark.unit
    def test_initializer_zeros_sets_rnn_cells_to_zero(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = LSTMController(
            library=lib,
            prior_system=prior_system,
            num_units=32,
            num_layers=1,
            embedding_dim=8,
            initializer="zeros",
        )
        for name, param in ctrl.rnn.named_parameters():
            assert torch.all(param == 0.0), (
                f"rnn parameter '{name}' is not zero under initializer='zeros'"
            )

    @pytest.mark.unit
    def test_initializer_zeros_output_layer_weight_is_xavier(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = LSTMController(
            library=lib,
            prior_system=prior_system,
            num_units=32,
            num_layers=1,
            embedding_dim=8,
            initializer="zeros",
        )
        w = ctrl.output_layer.weight
        assert not torch.all(w == 0.0), (
            "output_layer.weight must NOT be zero under initializer='zeros' "
            "(gradient flow to LSTM depends on W_out being non-trivial)"
        )



        assert w.abs().max().item() < 1.0, (
            f"output_layer.weight exceeds xavier bound: {w.abs().max().item()}"
        )

    @pytest.mark.unit
    def test_initializer_zeros_output_layer_bias_is_zero(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = LSTMController(
            library=lib,
            prior_system=prior_system,
            num_units=32,
            num_layers=1,
            embedding_dim=8,
            initializer="zeros",
        )
        b = ctrl.output_layer.bias
        assert b is not None
        assert torch.all(b == 0.0), (
            f"output_layer.bias must be zero under initializer='zeros': "
            f"max abs {b.abs().max().item()}"
        )

    @pytest.mark.unit
    def test_initializer_zeros_produces_uniform_logits(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = LSTMController(
            library=lib,
            prior_system=prior_system,
            num_units=32,
            num_layers=1,
            embedding_dim=8,
            initializer="zeros",
        )
        torch.manual_seed(SEED)
        batch = ctrl.sample(BATCH_SIZE)


        obs = torch.as_tensor(batch.obs, dtype=torch.float32)
        inputs = ctrl._embed_obs(obs)
        outputs, _ = ctrl.rnn(inputs)
        raw_logits = ctrl.output_layer(outputs)
        assert torch.allclose(
            raw_logits, torch.zeros_like(raw_logits), atol=1e-6,
        ), f"raw logits max abs: {raw_logits.abs().max().item()}"

    @pytest.mark.unit
    def test_initializer_zeros_lstm_receives_nonzero_gradient(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = LSTMController(
            library=lib,
            prior_system=prior_system,
            num_units=32,
            num_layers=1,
            embedding_dim=8,
            initializer="zeros",
        )
        torch.manual_seed(SEED)
        batch = ctrl.sample(BATCH_SIZE)
        ctrl.zero_grad()
        neglogp, _ = ctrl.make_neglogp_and_entropy(batch, entropy_gamma=1.0)
        neglogp.sum().backward()

        lstm_params = dict(ctrl.rnn.named_parameters())
        assert lstm_params, "controller has no LSTM parameters to check"
        grads = {
            name: param.grad for name, param in lstm_params.items()
        }

        has_grad = any(
            g is not None and g.abs().sum().item() > 0.0
            for g in grads.values()
        )
        assert has_grad, (
            "LSTM kernel parameters received zero gradient — the pre-Gemini "
            " dead-gradient bug is back. Gradient summary: "
            + ", ".join(
                f"{n}:{(g.abs().sum().item() if g is not None else 'None')}"
                for n, g in grads.items()
            )
        )

    @pytest.mark.unit
    def test_initializer_zeros_output_layer_weight_receives_zero_gradient(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = LSTMController(
            library=lib,
            prior_system=prior_system,
            num_units=32,
            num_layers=1,
            embedding_dim=8,
            initializer="zeros",
        )
        torch.manual_seed(SEED)
        batch = ctrl.sample(BATCH_SIZE)
        ctrl.zero_grad()
        neglogp, _ = ctrl.make_neglogp_and_entropy(batch, entropy_gamma=1.0)
        neglogp.sum().backward()
        w_grad = ctrl.output_layer.weight.grad
        assert w_grad is not None
        assert w_grad.abs().sum().item() == pytest.approx(0.0, abs=1e-12), (
            "output_layer.weight gradient should be exactly zero on the "
            f"first backward pass with zero-init LSTM; got "
            f"{w_grad.abs().sum().item()}"
        )

    @pytest.mark.unit
    def test_initializer_invalid_raises(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        with pytest.raises(ValueError, match="initializer"):
            LSTMController(
                library=lib,
                prior_system=prior_system,
                num_units=32,
                num_layers=1,
                embedding_dim=8,
                initializer="glorot",
            )

    @pytest.mark.unit
    def test_initializer_xavier_matches_default(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        torch.manual_seed(SEED)
        default_ctrl = LSTMController(
            library=lib,
            prior_system=prior_system,
            num_units=32,
            num_layers=1,
            embedding_dim=8,
        )
        torch.manual_seed(SEED)
        explicit_ctrl = LSTMController(
            library=lib,
            prior_system=prior_system,
            num_units=32,
            num_layers=1,
            embedding_dim=8,
            initializer="xavier",
        )
        default_params = dict(default_ctrl.named_parameters())
        for name, p in explicit_ctrl.named_parameters():
            assert torch.allclose(p, default_params[name]), (
                f"xavier-explicit diverges from default at '{name}'"
            )







class _DeadOnEmptyActionHistoryPrior(Prior):

    def __init__(self, library: Library) -> None:
        super().__init__(library)
        self.fire_count: int = 0

    def __call__(self, ctx: PriorContext) -> np.ndarray:
        adjustment = np.zeros(
            (ctx.actions.shape[0], self.n_choices), dtype=np.float32,
        )
        if ctx.actions.shape[1] == 0:
            return adjustment
        last_token = ctx.actions[:, -1]
        mask = last_token == self.library.EMPTY_ACTION
        if np.any(mask):
            self.fire_count += 1
            adjustment[mask,:] = -np.inf
        return adjustment


class TestAdversarialPriorFinishedRowSafety:

    @pytest.mark.unit
    def test_sample_with_adversarial_finished_row_prior_does_not_crash(
        self, lib: Library,
    ) -> None:




        adversarial = _DeadOnEmptyActionHistoryPrior(lib)
        adv_prior_system = PriorSystem(
            lib,
            [
                LengthConstraint(lib, min_=2, max_=MAX_LENGTH),
                DiffChildConstraint(lib),
                adversarial,
            ],
        )
        controller = LSTMController(
            library=lib,
            prior_system=adv_prior_system,
            num_units=32,
            num_layers=1,
            embedding_dim=8,
        )

        torch.manual_seed(SEED)
        batch = controller.sample(BATCH_SIZE)

        assert batch.actions.shape[0] == BATCH_SIZE
        assert int(batch.lengths.min()) >= 2
        assert int(batch.lengths.max()) <= MAX_LENGTH
        assert not np.any(np.isnan(batch.priors))






        assert (
            int(batch.lengths.max()) >= int(batch.lengths.min()) + 2
        ), (
            "adversarial test needs lengths.max() >= lengths.min() + 2 to "
            "exercise the finished-row path; re-seed if this ever fires."
        )




        assert adversarial.fire_count > 0, (
            "adversarial prior never fired — test is a vacuous pass; "
            "revisit exercise condition"
        )
