
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from kd.search.discover.controller.attention import AttentionState, BahdanauAttention
from kd.search.discover.controller.lstm import LSTMController
from kd.search.discover.core.batch import Batch
from kd.search.discover.tokens.library import Library, LibraryConfig
from kd.search.discover.tokens.prior import (
    DiffChildConstraint,
    LengthConstraint,
    PriorSystem,
)





BURGERS_CONFIG = LibraryConfig(
    operators=["add", "mul", "sub", "div", "sin", "cos", "diff_x", "diff2_x"],
    state_vars=["u"],
    coord_vars=["x", "t"],
)

MAX_LENGTH = 30
MIN_LENGTH = 4
BATCH_SIZE = 8
HIDDEN_SIZE = 16
ATTN_LENGTH = 5
NUM_LAYERS = 1
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
def attention() -> BahdanauAttention:
    return BahdanauAttention(
        attn_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        attn_length=ATTN_LENGTH,
    )


@pytest.fixture
def attention_2layer() -> BahdanauAttention:
    return BahdanauAttention(
        attn_size=HIDDEN_SIZE,
        num_layers=2,
        attn_length=ATTN_LENGTH,
    )


@pytest.fixture
def controller_no_attn(lib: Library, prior_system: PriorSystem) -> LSTMController:
    return LSTMController(
        library=lib,
        prior_system=prior_system,
        num_units=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        embedding_dim=4,
        attention=False,
    )


@pytest.fixture
def controller_attn(lib: Library, prior_system: PriorSystem) -> LSTMController:
    return LSTMController(
        library=lib,
        prior_system=prior_system,
        num_units=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        embedding_dim=4,
        attention=True,
        attn_length=ATTN_LENGTH,
    )


def _manual_batch(
    controller: LSTMController,
    lengths: list[int],
    sequence_length: int,
) -> Batch:
    batch_size = len(lengths)
    return Batch(
        actions=np.zeros((batch_size, sequence_length), dtype=np.int32),
        obs=np.zeros((batch_size, 4, sequence_length), dtype=np.float32),
        priors=np.zeros(
            (batch_size, sequence_length, len(controller.library.tokens)),
            dtype=np.float32,
        ),
        lengths=np.asarray(lengths, dtype=np.int32),
    )






class TestBahdanauAttentionUnit:

    @pytest.mark.unit
    def test_init_state_shapes(self, attention: BahdanauAttention) -> None:
        state = attention.init_state(BATCH_SIZE, torch.device("cpu"))
        assert isinstance(state, AttentionState)
        assert state.context.shape == (BATCH_SIZE, HIDDEN_SIZE)
        assert state.history.shape == (BATCH_SIZE, ATTN_LENGTH, HIDDEN_SIZE)
        assert torch.all(state.context == 0)
        assert torch.all(state.history == 0)

    @pytest.mark.unit
    def test_forward_output_shape(self, attention: BahdanauAttention) -> None:
        state = attention.init_state(BATCH_SIZE, torch.device("cpu"))
        cell_output = torch.randn(BATCH_SIZE, HIDDEN_SIZE)

        cell_state_flat = torch.randn(BATCH_SIZE, 2 * HIDDEN_SIZE)

        output, new_state = attention(cell_output, cell_state_flat, state)

        assert output.shape == (BATCH_SIZE, HIDDEN_SIZE)
        assert isinstance(new_state, AttentionState)
        assert new_state.context.shape == (BATCH_SIZE, HIDDEN_SIZE)
        assert new_state.history.shape == (BATCH_SIZE, ATTN_LENGTH, HIDDEN_SIZE)

    @pytest.mark.unit
    def test_forward_output_shape_2layer(
        self, attention_2layer: BahdanauAttention
    ) -> None:
        state = attention_2layer.init_state(BATCH_SIZE, torch.device("cpu"))
        cell_output = torch.randn(BATCH_SIZE, HIDDEN_SIZE)

        cell_state_flat = torch.randn(BATCH_SIZE, 4 * HIDDEN_SIZE)

        output, new_state = attention_2layer(cell_output, cell_state_flat, state)

        assert output.shape == (BATCH_SIZE, HIDDEN_SIZE)
        assert new_state.history.shape == (BATCH_SIZE, ATTN_LENGTH, HIDDEN_SIZE)

    @pytest.mark.unit
    def test_sliding_window_shift(self, attention: BahdanauAttention) -> None:
        state = attention.init_state(BATCH_SIZE, torch.device("cpu"))

        state = AttentionState(
            context=state.context,
            history=torch.arange(
                BATCH_SIZE * ATTN_LENGTH * HIDDEN_SIZE, dtype=torch.float32
            ).reshape(BATCH_SIZE, ATTN_LENGTH, HIDDEN_SIZE),
        )
        old_history = state.history.clone()

        cell_output = torch.randn(BATCH_SIZE, HIDDEN_SIZE)
        cell_state_flat = torch.randn(BATCH_SIZE, 2 * HIDDEN_SIZE)
        output, new_state = attention(cell_output, cell_state_flat, state)


        assert torch.allclose(new_state.history[:, :-1,:], old_history[:, 1:,:])

        assert torch.allclose(new_state.history[:, -1,:], output)

        assert not torch.allclose(output, cell_output, atol=1e-6)

    @pytest.mark.unit
    def test_attention_weights_sum_to_one(
        self, attention: BahdanauAttention
    ) -> None:
        torch.manual_seed(SEED)
        state = attention.init_state(BATCH_SIZE, torch.device("cpu"))

        state = AttentionState(
            context=state.context,
            history=torch.randn(BATCH_SIZE, ATTN_LENGTH, HIDDEN_SIZE),
        )
        cell_output = torch.randn(BATCH_SIZE, HIDDEN_SIZE)
        cell_state_flat = torch.randn(BATCH_SIZE, 2 * HIDDEN_SIZE)

        output, new_state, weights = attention(
            cell_output, cell_state_flat, state, return_weights=True,
        )

        assert weights.shape == (BATCH_SIZE, ATTN_LENGTH)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(BATCH_SIZE), atol=1e-6)
        assert (weights >= 0).all()
        assert torch.isfinite(output).all()

    @pytest.mark.unit
    def test_zero_history_uniform_weights(
        self, attention: BahdanauAttention
    ) -> None:
        state = attention.init_state(BATCH_SIZE, torch.device("cpu"))
        cell_output = torch.zeros(BATCH_SIZE, HIDDEN_SIZE)
        cell_state_flat = torch.zeros(BATCH_SIZE, 2 * HIDDEN_SIZE)

        _, _, weights = attention(
            cell_output, cell_state_flat, state, return_weights=True,
        )

        expected = torch.full((BATCH_SIZE, ATTN_LENGTH), 1.0 / ATTN_LENGTH)
        assert torch.allclose(weights, expected, atol=1e-5)

    @pytest.mark.unit
    def test_nonzero_history_nonuniform_weights(
        self, attention: BahdanauAttention
    ) -> None:
        torch.manual_seed(SEED)
        state = attention.init_state(BATCH_SIZE, torch.device("cpu"))

        history = torch.zeros(BATCH_SIZE, ATTN_LENGTH, HIDDEN_SIZE)
        history[:, 0,:] = 5.0
        state = AttentionState(context=state.context, history=history)

        cell_output = torch.randn(BATCH_SIZE, HIDDEN_SIZE)
        cell_state_flat = torch.randn(BATCH_SIZE, 2 * HIDDEN_SIZE)

        _, _, weights = attention(
            cell_output, cell_state_flat, state, return_weights=True,
        )

        uniform = torch.full((BATCH_SIZE, ATTN_LENGTH), 1.0 / ATTN_LENGTH)
        assert not torch.allclose(weights, uniform, atol=0.01), (
            "Weights should be non-uniform when history is asymmetric"
        )

    @pytest.mark.unit
    def test_output_depends_on_cell_output_and_context(
        self, attention: BahdanauAttention
    ) -> None:
        torch.manual_seed(SEED)
        state = attention.init_state(BATCH_SIZE, torch.device("cpu"))
        state_with_history = AttentionState(
            context=state.context,
            history=torch.randn(BATCH_SIZE, ATTN_LENGTH, HIDDEN_SIZE),
        )
        cell_state_flat = torch.randn(BATCH_SIZE, 2 * HIDDEN_SIZE)

        cell_output_a = torch.randn(BATCH_SIZE, HIDDEN_SIZE)
        cell_output_b = torch.randn(BATCH_SIZE, HIDDEN_SIZE)

        out_a, _ = attention(cell_output_a, cell_state_flat, state_with_history)
        out_b, _ = attention(cell_output_b, cell_state_flat, state_with_history)

        assert not torch.allclose(out_a, out_b, atol=1e-6)


        state_alt = AttentionState(
            context=state.context,
            history=torch.randn(BATCH_SIZE, ATTN_LENGTH, HIDDEN_SIZE),
        )
        out_c, _ = attention(cell_output_a, cell_state_flat, state_alt)
        assert not torch.allclose(out_a, out_c, atol=1e-6)

    @pytest.mark.unit
    def test_gradient_flows_through_all_params(
        self, attention: BahdanauAttention
    ) -> None:
        torch.manual_seed(SEED)
        state = attention.init_state(BATCH_SIZE, torch.device("cpu"))
        state = AttentionState(
            context=state.context,
            history=torch.randn(BATCH_SIZE, ATTN_LENGTH, HIDDEN_SIZE),
        )
        cell_output = torch.randn(BATCH_SIZE, HIDDEN_SIZE, requires_grad=True)
        cell_state_flat = torch.randn(
            BATCH_SIZE, 2 * HIDDEN_SIZE, requires_grad=True
        )

        output, _ = attention(cell_output, cell_state_flat, state)
        loss = output.sum()
        loss.backward()

        for name, param in attention.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    @pytest.mark.unit
    def test_numerical_reference_known_weights(self) -> None:
        batch_size, hidden_size, attn_length = 2, 4, 3
        attn = BahdanauAttention(
            attn_size=hidden_size,
            num_layers=1,
            attn_length=attn_length,
        )


        with torch.no_grad():

            attn.key_proj.weight.copy_(torch.eye(hidden_size))

            attn.query_proj.weight.zero_()
            attn.query_proj.weight[:, :hidden_size] = torch.eye(hidden_size)
            attn.query_proj.bias.zero_()

            attn.v.fill_(1.0)

            attn.output_proj.weight.zero_()
            attn.output_proj.weight[:, :hidden_size] = torch.eye(hidden_size)
            attn.output_proj.weight[:, hidden_size:] = torch.eye(hidden_size)
            attn.output_proj.bias.zero_()


        cell_output = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0, 0.0]])
        cell_state_flat = torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        history = torch.zeros(batch_size, attn_length, hidden_size)
        history[0, 0,:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        history[0, 1,:] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        history[0, 2,:] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        history[1, 0,:] = torch.tensor([0.5, 0.5, 0.0, 0.0])
        history[1, 1,:] = torch.tensor([0.0, 0.5, 0.5, 0.0])
        history[1, 2,:] = torch.tensor([0.0, 0.0, 0.5, 0.5])
        state = AttentionState(
            context=torch.zeros(batch_size, hidden_size),
            history=history,
        )

        output, new_state, weights = attn(
            cell_output, cell_state_flat, state, return_weights=True,
        )





        tanh_zero = torch.tanh(torch.tensor(0.0)).item()
        s0 = torch.tanh(torch.tensor(2.0)).item() + 3 * tanh_zero
        s1 = 2 * torch.tanh(torch.tensor(1.0)).item() + 2 * tanh_zero
        s2 = s1
        scores_b0 = torch.tensor([s0, s1, s2])
        expected_w0 = torch.softmax(scores_b0, dim=0)

        assert torch.allclose(weights[0], expected_w0, atol=1e-5), (
            f"Batch 0 weights: got {weights[0]}, expected {expected_w0}"
        )


        expected_context_b0 = (expected_w0.unsqueeze(1) * history[0]).sum(dim=0)

        expected_output_b0 = cell_output[0] + expected_context_b0
        assert torch.allclose(output[0], expected_output_b0, atol=1e-5), (
            f"Batch 0 output: got {output[0]}, expected {expected_output_b0}"
        )

    @pytest.mark.unit
    def test_attn_length_one_boundary(self) -> None:
        attn = BahdanauAttention(attn_size=HIDDEN_SIZE, num_layers=1, attn_length=1)
        state = attn.init_state(BATCH_SIZE, torch.device("cpu"))
        assert state.history.shape == (BATCH_SIZE, 1, HIDDEN_SIZE)

        cell_output = torch.randn(BATCH_SIZE, HIDDEN_SIZE)
        cell_state_flat = torch.randn(BATCH_SIZE, 2 * HIDDEN_SIZE)

        output, new_state, weights = attn(
            cell_output, cell_state_flat, state, return_weights=True,
        )

        assert output.shape == (BATCH_SIZE, HIDDEN_SIZE)
        assert new_state.history.shape == (BATCH_SIZE, 1, HIDDEN_SIZE)

        assert torch.allclose(weights, torch.ones(BATCH_SIZE, 1), atol=1e-6)
        assert torch.isfinite(output).all()






class TestControllerAttentionIntegration:

    @pytest.mark.smoke
    def test_attention_off_sample_unchanged(
        self, controller_no_attn: LSTMController
    ) -> None:
        torch.manual_seed(SEED)
        batch = controller_no_attn.sample(BATCH_SIZE)

        assert batch.actions.shape[0] == BATCH_SIZE
        assert batch.obs.shape[0] == BATCH_SIZE
        assert batch.priors.shape[0] == BATCH_SIZE
        assert batch.lengths.shape == (BATCH_SIZE,)
        assert (batch.lengths >= MIN_LENGTH).all()
        assert (batch.lengths <= MAX_LENGTH).all()

    @pytest.mark.smoke
    def test_attention_on_sample_shape(
        self, controller_attn: LSTMController
    ) -> None:
        torch.manual_seed(SEED)
        batch = controller_attn.sample(BATCH_SIZE)

        assert batch.actions.shape[0] == BATCH_SIZE
        assert batch.obs.shape[0] == BATCH_SIZE
        assert batch.priors.shape[0] == BATCH_SIZE
        assert batch.lengths.shape == (BATCH_SIZE,)
        assert (batch.lengths >= MIN_LENGTH).all()
        assert (batch.lengths <= MAX_LENGTH).all()

        n_tokens = len(controller_attn.library.tokens)
        for i in range(BATCH_SIZE):
            valid = batch.actions[i,: batch.lengths[i]]
            assert (valid >= 0).all()
            assert (valid < n_tokens).all()

    @pytest.mark.smoke
    def test_attention_on_neglogp_entropy(
        self, controller_attn: LSTMController
    ) -> None:
        torch.manual_seed(SEED)
        batch = controller_attn.sample(BATCH_SIZE)
        neglogp, entropy = controller_attn.make_neglogp_and_entropy(batch)

        assert neglogp.shape == (BATCH_SIZE,)
        assert entropy.shape == (BATCH_SIZE,)
        assert torch.isfinite(neglogp).all()
        assert torch.isfinite(entropy).all()
        assert (neglogp > 0).all()
        assert (entropy >= 0).all()

    @pytest.mark.smoke
    def test_attention_off_neglogp_entropy(
        self, controller_no_attn: LSTMController
    ) -> None:
        torch.manual_seed(SEED)
        batch = controller_no_attn.sample(BATCH_SIZE)
        neglogp, entropy = controller_no_attn.make_neglogp_and_entropy(batch)

        assert neglogp.shape == (BATCH_SIZE,)
        assert entropy.shape == (BATCH_SIZE,)
        assert torch.isfinite(neglogp).all()
        assert torch.isfinite(entropy).all()

    @pytest.mark.unit
    def test_attention_changes_logits(
        self,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        torch.manual_seed(SEED)
        ctrl_on = LSTMController(
            library=lib,
            prior_system=prior_system,
            num_units=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            embedding_dim=4,
            attention=True,
            attn_length=ATTN_LENGTH,
        )


        torch.manual_seed(SEED + 1)
        batch = ctrl_on.sample(BATCH_SIZE)


        logits_on = ctrl_on._masked_logits_from_batch(batch)


        ctrl_off = LSTMController(
            library=lib,
            prior_system=prior_system,
            num_units=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            embedding_dim=4,
            attention=False,
        )


        with torch.no_grad():
            for (_name_off, p_off), (_name_on, p_on) in zip(
                ctrl_off.rnn.named_parameters(),
                ctrl_on.rnn.named_parameters(),
                strict=False,
            ):
                p_off.copy_(p_on)
            ctrl_off.output_layer.weight.copy_(ctrl_on.output_layer.weight)
            ctrl_off.output_layer.bias.copy_(ctrl_on.output_layer.bias)

        logits_off = ctrl_off._masked_logits_from_batch(batch)

        assert logits_off.shape == logits_on.shape

        assert not torch.allclose(logits_off, logits_on, atol=1e-4), (
            "Logits should differ when attention is active vs bypassed"
        )

    @pytest.mark.unit
    def test_attention_state_resets_between_samples(
        self, controller_attn: LSTMController
    ) -> None:
        torch.manual_seed(SEED)
        batch1 = controller_attn.sample(BATCH_SIZE)

        torch.manual_seed(SEED)
        batch2 = controller_attn.sample(BATCH_SIZE)


        np.testing.assert_array_equal(batch1.actions, batch2.actions)
        np.testing.assert_array_equal(batch1.lengths, batch2.lengths)

    @pytest.mark.unit
    def test_training_path_idempotent(
        self, controller_attn: LSTMController
    ) -> None:
        torch.manual_seed(SEED)
        batch = controller_attn.sample(BATCH_SIZE)

        logits_1 = controller_attn._masked_logits_from_batch(batch)
        logits_2 = controller_attn._masked_logits_from_batch(batch)

        assert torch.allclose(logits_1, logits_2), (
            "Training path should be deterministic — attention state must reset"
        )

    @pytest.mark.unit
    def test_training_path_slices_padded_rnn_steps(
        self,
        controller_no_attn: LSTMController,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        batch = _manual_batch(controller_no_attn, lengths=[1, 3], sequence_length=3)
        step_batch_sizes: list[int] = []

        def fake_step(
            x: Tensor,
            hidden: list[tuple[Tensor, Tensor]],
            active_mask: Tensor | None = None,
        ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
            assert active_mask is None
            step_batch_sizes.append(x.shape[0])
            next_hidden = [
                (torch.zeros_like(h), torch.zeros_like(c))
                for h, c in hidden
            ]
            return next_hidden[-1][0], next_hidden

        monkeypatch.setattr(controller_no_attn.rnn, "step", fake_step)

        controller_no_attn._masked_logits_from_batch(batch)

        assert step_batch_sizes == [2, 1, 1]

    @pytest.mark.unit
    def test_training_path_slices_padded_attention_steps(
        self,
        controller_attn: LSTMController,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        batch = _manual_batch(controller_attn, lengths=[1, 3], sequence_length=3)
        step_batch_sizes: list[int] = []

        def fake_apply_attention_step(
            output: Tensor,
            hidden: list[tuple[Tensor, Tensor]],
            attn_state: AttentionState | None,
            active_mask: Tensor | None = None,
        ) -> tuple[Tensor, AttentionState | None]:
            del hidden
            assert active_mask is None
            step_batch_sizes.append(output.shape[0])
            return output, attn_state

        monkeypatch.setattr(
            controller_attn,
            "_apply_attention_step",
            fake_apply_attention_step,
        )

        controller_attn._masked_logits_from_batch(batch)

        assert step_batch_sizes == [2, 1, 1]

    @pytest.mark.unit
    def test_valid_prefix_logits_match_cropped_single_row_run(
        self,
        controller_attn: LSTMController,
    ) -> None:
        torch.manual_seed(SEED)
        batch = controller_attn.sample(BATCH_SIZE)
        logits_full = controller_attn._masked_logits_from_batch(batch)

        for row_idx, length in enumerate(batch.lengths.tolist()):
            cropped = Batch(
                actions=batch.actions[row_idx: row_idx + 1, :length].copy(),
                obs=batch.obs[row_idx: row_idx + 1, :, :length].copy(),
                priors=batch.priors[row_idx: row_idx + 1, :length,:].copy(),
                lengths=np.asarray([length], dtype=np.int32),
            )
            logits_cropped = controller_attn._masked_logits_from_batch(cropped)
            torch.testing.assert_close(
                logits_full[row_idx, :length,:],
                logits_cropped[0, :length,:],
            )

    @pytest.mark.unit
    def test_zeros_initializer_preserves_attention_params(
        self,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        ctrl = LSTMController(
            library=lib,
            prior_system=prior_system,
            num_units=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            embedding_dim=4,
            attention=True,
            attn_length=ATTN_LENGTH,
            initializer="zeros",
        )


        for cell in ctrl.rnn.cells:
            for param in cell.parameters():
                assert param.abs().sum() == 0, "RNN cell params should be zero"


        attn_mod = next(
            m for m in ctrl.modules() if isinstance(m, BahdanauAttention)
        )
        has_nonzero_weight = False
        for name, param in attn_mod.named_parameters():
            if "weight" in name and param.abs().sum() > 0:
                has_nonzero_weight = True
        assert has_nonzero_weight, (
            "Attention weights should be non-zero under 'zeros' initializer"
        )






class TestConfigAttention:

    @pytest.mark.unit
    def test_config_attention_defaults(self) -> None:
        from kd.search.discover.config import DiscoverConfig

        config = DiscoverConfig()
        assert config.attention is False
        assert config.attn_length == 10

    @pytest.mark.unit
    def test_burgers_preset_enables_attention(self) -> None:
        from kd.search.discover.config import DiscoverConfig

        config = DiscoverConfig.burgers_preset()
        assert config.attention is True
        assert config.attn_length == 10

    @pytest.mark.unit
    def test_chafee_preset_enables_attention(self) -> None:
        from kd.search.discover.config import DiscoverConfig

        config = DiscoverConfig.chafee_preset()
        assert config.attention is True
        assert config.attn_length == 10

    @pytest.mark.unit
    def test_builder_passes_attention_to_controller(self) -> None:
        from kd.search.discover.builder import (
            build_controller,
            build_library,
            build_prior_system,
        )
        from kd.search.discover.config import DiscoverConfig

        config = DiscoverConfig(
            attention=True,
            attn_length=7,
            num_units=HIDDEN_SIZE,
        )
        lib = build_library(config)
        ps = build_prior_system(lib, config)
        ctrl = build_controller(lib, ps, config)


        assert any(isinstance(m, BahdanauAttention) for m in ctrl.modules()), (
            "Controller should contain a BahdanauAttention submodule"
        )

        torch.manual_seed(SEED)
        batch = ctrl.sample(4)
        assert batch.actions.shape[0] == 4

    @pytest.mark.unit
    def test_builder_no_attention_by_default(self) -> None:
        from kd.search.discover.builder import (
            build_controller,
            build_library,
            build_prior_system,
        )
        from kd.search.discover.config import DiscoverConfig

        config = DiscoverConfig()
        lib = build_library(config)
        ps = build_prior_system(lib, config)
        ctrl = build_controller(lib, ps, config)

        assert not any(isinstance(m, BahdanauAttention) for m in ctrl.modules()), (
            "Default controller should not contain BahdanauAttention"
        )
        torch.manual_seed(SEED)
        batch = ctrl.sample(4)
        assert batch.actions.shape[0] == 4
