
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from kd.search.discover.controller.attention import AttentionState, BahdanauAttention
from kd.search.discover.controller.tree_state import BatchTracker, IncrementalTracker
from kd.search.discover.core.batch import Batch
from kd.search.discover.core.tree import finish_tokens
from kd.search.discover.tokens.library import Library
from kd.search.discover.tokens.prior import LengthConstraint, PriorSystem

InitializerName = Literal["xavier", "zeros"]
_VALID_INITIALIZERS: frozenset[str] = frozenset({"xavier", "zeros"})

_OBS_CHANNELS = 4
_ACTION_CHANNEL = 0
_PARENT_CHANNEL = 1
_SIBLING_CHANNEL = 2
_DANGLING_CHANNEL = 3
_DANGLING_COMPLETE = 0
_DANGLING_INPUT_DIM = 1
_MIN_SEQUENCE_LENGTH = 1
_STEP_OFFSET = 1

Int32Array = npt.NDArray[np.int32]
Float32Array = npt.NDArray[np.float32]
HiddenState = list[tuple[Tensor, Tensor]]
ControllerAttentionState = AttentionState | None


@dataclass(frozen=True, slots=True)
class _EmbeddingMetadata:

    embedding_dim: int


def safe_cross_entropy(p: Tensor, logq: Tensor, axis: int = -1) -> Tensor:
    safe_logq = torch.where(p == 0, torch.ones_like(logq), logq)
    return -(p * safe_logq).sum(dim=axis)


class StackedRNN(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError("input_size, hidden_size, and num_layers must be > 0.")
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList([
            nn.LSTMCell(input_size if layer == 0 else hidden_size, hidden_size)
            for layer in range(num_layers)
        ])

    def init_hidden(self, batch_size: int, device: torch.device) -> HiddenState:
        return [
            (
                torch.zeros(batch_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device),
            )
            for _ in self.cells
        ]

    def forward(self, inputs: Tensor) -> tuple[Tensor, HiddenState]:
        hidden = self.init_hidden(inputs.shape[0], inputs.device)
        outputs: list[Tensor] = []
        for step_idx in range(inputs.shape[1]):
            output, hidden = self.step(inputs[:, step_idx,:], hidden)
            outputs.append(output)
        return torch.stack(outputs, dim=1), hidden

    def step(
        self,
        x: Tensor,
        hidden: HiddenState,
        active_mask: Tensor | None = None,
    ) -> tuple[Tensor, HiddenState]:
        mask = None if active_mask is None else active_mask.to(x.dtype).unsqueeze(1)
        next_hidden: HiddenState = []
        for layer_idx, cell in enumerate(self.cells):
            h_prev, c_prev = hidden[layer_idx]
            h_next, c_next = cell(x, (h_prev, c_prev))
            if mask is not None:
                h_next = h_next * mask + h_prev * (1.0 - mask)
                c_next = c_next * mask + c_prev * (1.0 - mask)
            x = h_next
            next_hidden.append((h_next, c_next))
        return x, next_hidden


class LSTMController(nn.Module):

    def __init__(
        self,
        library: Library,
        prior_system: PriorSystem,
        num_units: int = 32,
        num_layers: int = 1,
        embedding_dim: int = 8,
        observe_parent: bool = True,
        observe_sibling: bool = True,
        observe_action: bool = False,
        observe_dangling: bool = False,
        use_embedding: bool = False,
        attention: bool = False,
        attn_length: int = 10,
        initializer: InitializerName = "xavier",
    ) -> None:
        super().__init__()
        if num_units <= 0 or num_layers <= 0 or embedding_dim <= 0:
            raise ValueError("num_units, num_layers, and embedding_dim must be > 0.")
        if attn_length <= 0:
            raise ValueError("attn_length must be > 0.")
        if initializer not in _VALID_INITIALIZERS:
            raise ValueError(
                f"initializer must be one of {sorted(_VALID_INITIALIZERS)}; "
                f"got '{initializer}'."
            )
        self.library = library
        self.prior_system = prior_system
        self.observe_parent = observe_parent
        self.observe_sibling = observe_sibling
        self.observe_action = observe_action
        self.observe_dangling = observe_dangling
        self.use_embedding = use_embedding
        self.attn_length = attn_length
        self.initializer: InitializerName = initializer
        self._validate_observation_config()
        self._n_choices = len(library.tokens)
        self._min_length, self._max_length = self._infer_length_bounds()
        self.action_embedding = self._make_channel_encoder(
            observe_action,
            library.n_action_inputs,
            embedding_dim,
        )
        self.parent_embedding = self._make_channel_encoder(
            observe_parent,
            library.n_parent_inputs,
            embedding_dim,
        )
        self.sibling_embedding = self._make_channel_encoder(
            observe_sibling,
            library.n_sibling_inputs,
            embedding_dim,
        )
        self._enabled_categorical_specs: list[
            tuple[int, int, nn.Embedding | _EmbeddingMetadata]
        ] = []
        if self.observe_action:
            self._enabled_categorical_specs.append(
                (_ACTION_CHANNEL, library.n_action_inputs, self.action_embedding),
            )
        if self.observe_parent:
            self._enabled_categorical_specs.append(
                (_PARENT_CHANNEL, library.n_parent_inputs, self.parent_embedding),
            )
        if self.observe_sibling:
            self._enabled_categorical_specs.append(
                (_SIBLING_CHANNEL, library.n_sibling_inputs, self.sibling_embedding),
            )
        input_dim = self._compute_input_dim(embedding_dim)
        self.rnn = StackedRNN(input_dim, num_units, num_layers)
        self.attention_module = (
            BahdanauAttention(
                attn_size=num_units,
                num_layers=num_layers,
                attn_length=attn_length,
            )
            if attention else None
        )
        self.output_layer = nn.Linear(num_units, self._n_choices)
        self._sampling_attention_state: ControllerAttentionState = None
        self._initialize_parameters()

    def sample(self, batch_size: int) -> Batch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        with torch.no_grad():
            actions, lengths = self._sample_actions(batch_size)
        obs = BatchTracker(self.library).compute_obs(actions)
        priors = self.prior_system.compute_batch(actions, obs)
        return Batch(actions=actions, obs=obs, priors=priors, lengths=lengths)

    def make_neglogp_and_entropy(
        self,
        batch: Batch,
        entropy_gamma: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        logits = self._masked_logits_from_batch(batch)
        log_probs = functional.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        lengths = torch.as_tensor(batch.lengths, dtype=torch.long, device=self.device)
        valid_mask = self._length_mask(lengths, actions.shape[1])
        mask_f = valid_mask.to(log_probs.dtype)

        safe_actions = torch.where(valid_mask, actions, torch.zeros_like(actions))
        actions_one_hot = functional.one_hot(
            safe_actions, num_classes=self._n_choices,
        ).float()
        neglogp_per_step = safe_cross_entropy(actions_one_hot, log_probs, axis=2)
        neglogp = (neglogp_per_step * mask_f).sum(dim=1)
        step_weights = self._entropy_weights(actions.shape[1], entropy_gamma)
        entropy_steps = safe_cross_entropy(probs, log_probs, axis=2)
        entropy = (entropy_steps * mask_f * step_weights).sum(dim=1)
        return neglogp, entropy

    @property
    def device(self) -> torch.device:
        return self.output_layer.weight.device

    def _initialize_parameters(self) -> None:

        for param in self.parameters():
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param, a=-0.1, b=0.1)



        if self.attention_module is not None:
            self.attention_module.reset_parameters()
        if self.initializer == "xavier":
            return

        for cell in self.rnn.cells:
            for param in cell.parameters():
                nn.init.zeros_(param)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def _infer_length_bounds(self) -> tuple[int, int]:
        min_values = [
            prior.min_ for prior in self.prior_system.priors
            if isinstance(prior, LengthConstraint) and prior.min_ is not None
        ]
        max_values = [
            prior.max_ for prior in self.prior_system.priors
            if isinstance(prior, LengthConstraint) and prior.max_ is not None
        ]
        min_length = max(min_values, default=_MIN_SEQUENCE_LENGTH)
        max_length = min(max_values) if max_values else max(self._n_choices, min_length)
        return min_length, max_length

    def _validate_observation_config(self) -> None:
        if any((
            self.observe_action,
            self.observe_parent,
            self.observe_sibling,
            self.observe_dangling,
        )):
            return
        raise ValueError("At least one observation channel must be enabled.")

    def _make_channel_encoder(
        self,
        enabled: bool,
        num_embeddings: int,
        embedding_dim: int,
    ) -> nn.Embedding | _EmbeddingMetadata:
        if self.use_embedding and enabled:
            return nn.Embedding(num_embeddings, embedding_dim)
        return _EmbeddingMetadata(embedding_dim=embedding_dim)

    def _compute_input_dim(self, embedding_dim: int) -> int:
        input_dim = 0
        for _, n_classes, _ in self._enabled_categorical_specs:
            input_dim += embedding_dim if self.use_embedding else n_classes
        if self.observe_dangling:
            input_dim += _DANGLING_INPUT_DIM
        return input_dim

    def _encode_categorical(
        self,
        values: Tensor,
        n_classes: int,
        encoder: nn.Embedding | _EmbeddingMetadata,
    ) -> Tensor:
        if self.use_embedding:
            if not isinstance(encoder, nn.Embedding):
                raise RuntimeError("Enabled embedding channel is missing nn.Embedding.")
            return cast(Tensor, encoder(values))
        return functional.one_hot(values, num_classes=n_classes).to(dtype=torch.float32)

    def _masked_logits_from_batch(self, batch: Batch) -> Tensor:
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        priors = torch.as_tensor(batch.priors, dtype=torch.float32, device=self.device)
        lengths = torch.as_tensor(
            batch.lengths,
            dtype=torch.long,
            device=self.device,
        )
        inputs = self._embed_obs(obs)
        outputs, hidden_history = self._run_rnn_with_hidden_history(inputs, lengths)
        outputs = self._apply_attention_sequence(outputs, hidden_history, lengths)
        logits = self.output_layer(outputs)
        return cast(Tensor, logits + priors)

    def _embed_obs(self, obs: Tensor) -> Tensor:
        if obs.dim() not in {2, 3} or obs.shape[1] != _OBS_CHANNELS:
            raise ValueError("obs must have shape (B, 4) or (B, 4, L).")
        pieces: list[Tensor] = []
        for channel_idx, n_classes, encoder in self._enabled_categorical_specs:
            values = obs[:, channel_idx].long()
            pieces.append(self._encode_categorical(values, n_classes, encoder))
        if self.observe_dangling:
            pieces.append(obs[:, _DANGLING_CHANNEL].unsqueeze(-1))
        return torch.cat(pieces, dim=-1)

    def _sample_actions(self, batch_size: int) -> tuple[Int32Array, Int32Array]:
        actions = np.full(
            (batch_size, self._max_length),
            self.library.EMPTY_ACTION,
            dtype=np.int32,
        )
        lengths = np.zeros(batch_size, dtype=np.int32)
        tracker = IncrementalTracker(self.library, max_length=self._max_length)
        obs = tracker.reset(batch_size)
        hidden = self.rnn.init_hidden(batch_size, self.device)
        self._sampling_attention_state = self._init_attention_state(batch_size)
        try:
            steps_used = self._sampling_loop(
                tracker,
                obs,
                hidden,
                actions,
                lengths,
            )
        finally:
            self._sampling_attention_state = None
        return self._finalize_actions(actions, lengths, steps_used)

    def _sampling_loop(
        self,
        tracker: IncrementalTracker,
        obs: Float32Array,
        hidden: HiddenState,
        actions: Int32Array,
        lengths: Int32Array,
    ) -> int:
        finished = np.zeros(actions.shape[0], dtype=bool)
        for step_idx in range(self._max_length):
            step_actions, hidden = self._sample_step(
                tracker,
                obs,
                hidden,
                finished,
                step_idx,
            )
            actions[:, step_idx] = step_actions
            self._advance_sample_state(
                tracker,
                obs,
                step_actions,
                finished,
                lengths,
                step_idx,
            )
            if finished.all():
                return step_idx + _STEP_OFFSET
        return self._max_length

    def _sample_step(
        self,
        tracker: IncrementalTracker,
        obs: Float32Array,
        hidden: HiddenState,
        finished: npt.NDArray[np.bool_],
        step_idx: int,
    ) -> tuple[Int32Array, HiddenState]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        inputs = self._embed_obs(obs_tensor)
        active = ~finished
        active_tensor = torch.as_tensor(active, device=self.device)
        outputs, next_hidden = self.rnn.step(inputs, hidden, active_tensor)
        outputs, next_attn_state = self._apply_attention_step(
            outputs,
            next_hidden,
            self._sampling_attention_state,
            active_tensor,
        )
        self._sampling_attention_state = next_attn_state
        logits = self.output_layer(outputs)
        action_history = tracker.history


        prior_adjust = self.prior_system.step(action_history, obs, step_idx)
        logits = logits + torch.as_tensor(
            prior_adjust, dtype=torch.float32, device=self.device,
        )
        return self._draw_actions(logits, active), next_hidden

    def _run_rnn_with_hidden_history(
        self,
        inputs: Tensor,
        lengths: Tensor,
    ) -> tuple[Tensor, list[HiddenState]]:
        hidden = self.rnn.init_hidden(inputs.shape[0], inputs.device)
        outputs: list[Tensor] = []
        hidden_history: list[HiddenState] = []
        for step_idx in range(inputs.shape[1]):
            active_idx = self._active_indices(lengths, step_idx)
            output = hidden[-1][0]
            if active_idx.numel() > 0:
                active_output, active_hidden = self.rnn.step(
                    inputs.index_select(0, active_idx)[:, step_idx,:],
                    self._select_hidden(hidden, active_idx),
                )
                output = self._scatter_rows(output, active_idx, active_output)
                hidden = self._scatter_hidden(hidden, active_idx, active_hidden)
            outputs.append(output)
            hidden_history.append(hidden)
        return torch.stack(outputs, dim=1), hidden_history

    def _apply_attention_sequence(
        self,
        outputs: Tensor,
        hidden_history: list[HiddenState],
        lengths: Tensor,
    ) -> Tensor:
        if self.attention_module is None:
            return outputs
        attn_state = self.attention_module.init_state(outputs.shape[0], outputs.device)
        attended_outputs: list[Tensor] = []
        for step_idx, hidden in enumerate(hidden_history):
            active_idx = self._active_indices(lengths, step_idx)
            attended = outputs[:, step_idx,:]
            if active_idx.numel() > 0:
                active_attended, active_next_state = self._apply_attention_step(
                    attended.index_select(0, active_idx),
                    self._select_hidden(hidden, active_idx),
                    self._select_attention_state(attn_state, active_idx),
                )
                if active_next_state is None:
                    raise RuntimeError("attention state unexpectedly missing.")
                attended = self._scatter_rows(attended, active_idx, active_attended)
                attn_state = self._scatter_attention_state(
                    attn_state,
                    active_idx,
                    active_next_state,
                )
            attended_outputs.append(attended)
        return torch.stack(attended_outputs, dim=1)

    def _active_indices(self, lengths: Tensor, step_idx: int) -> Tensor:
        return torch.nonzero(lengths > step_idx, as_tuple=False).flatten()

    def _select_hidden(self, hidden: HiddenState, indices: Tensor) -> HiddenState:
        return [
            (h.index_select(0, indices), c.index_select(0, indices))
            for h, c in hidden
        ]

    def _scatter_hidden(
        self,
        previous: HiddenState,
        indices: Tensor,
        updated: HiddenState,
    ) -> HiddenState:
        return [
            (
                self._scatter_rows(h_prev, indices, h_next),
                self._scatter_rows(c_prev, indices, c_next),
            )
            for (h_prev, c_prev), (h_next, c_next) in zip(
                previous,
                updated,
                strict=True,
            )
        ]

    def _select_attention_state(
        self,
        state: AttentionState,
        indices: Tensor,
    ) -> AttentionState:
        return AttentionState(
            context=state.context.index_select(0, indices),
            history=state.history.index_select(0, indices),
        )

    def _scatter_attention_state(
        self,
        previous: AttentionState,
        indices: Tensor,
        updated: AttentionState,
    ) -> AttentionState:
        return AttentionState(
            context=self._scatter_rows(previous.context, indices, updated.context),
            history=self._scatter_rows(previous.history, indices, updated.history),
        )

    def _scatter_rows(self, base: Tensor, indices: Tensor, values: Tensor) -> Tensor:
        return base.index_copy(0, indices, values)

    def _init_attention_state(self, batch_size: int) -> ControllerAttentionState:
        if self.attention_module is None:
            return None
        return self.attention_module.init_state(batch_size, self.device)

    def _apply_attention_step(
        self,
        output: Tensor,
        hidden: HiddenState,
        attn_state: ControllerAttentionState,
        active_mask: Tensor | None = None,
    ) -> tuple[Tensor, ControllerAttentionState]:
        if self.attention_module is None:
            return output, None
        if attn_state is None:
            raise RuntimeError("attention state must be initialized before use.")
        state_flat = self._flatten_hidden_state(hidden)
        attended, next_state = self.attention_module(output, state_flat, attn_state)
        if active_mask is not None:
            mask = active_mask.to(dtype=torch.bool).unsqueeze(1)
            attended = torch.where(mask, attended, output)
        frozen_state = self._freeze_attention_state(
            attn_state,
            next_state,
            active_mask,
        )
        return attended, frozen_state

    def _freeze_attention_state(
        self,
        previous: AttentionState,
        updated: AttentionState,
        active_mask: Tensor | None,
    ) -> AttentionState:
        if active_mask is None:
            return updated
        mask = active_mask.to(dtype=torch.bool)
        context = torch.where(mask.unsqueeze(1), updated.context, previous.context)
        history = torch.where(
            mask.unsqueeze(1).unsqueeze(2),
            updated.history,
            previous.history,
        )
        return AttentionState(context=context, history=history)

    def _flatten_hidden_state(self, hidden: HiddenState) -> Tensor:
        pieces: list[Tensor] = []
        for h_state, c_state in hidden:
            pieces.extend([c_state, h_state])
        return torch.cat(pieces, dim=1)

    def _draw_actions(
        self,
        logits: Tensor,
        active: npt.NDArray[np.bool_],
    ) -> Int32Array:
        actions = np.full(logits.shape[0], self.library.EMPTY_ACTION, dtype=np.int32)
        if not np.any(active):
            return actions
        active_logits = logits[torch.as_tensor(active, device=self.device)]
        sampled = torch.distributions.Categorical(
            logits=active_logits
        ).sample()
        actions[active] = sampled.cpu().numpy().astype(np.int32)
        return actions

    def _advance_sample_state(
        self,
        tracker: IncrementalTracker,
        obs: Float32Array,
        actions: Int32Array,
        finished: npt.NDArray[np.bool_],
        lengths: Int32Array,
        step_idx: int,
    ) -> None:
        next_obs = tracker.step(actions)
        active = ~finished
        if not np.any(active):
            return
        obs[active] = next_obs[active]
        active_next_dangling = next_obs[active, _DANGLING_CHANNEL].astype(np.int32)
        just_finished_local = active_next_dangling == _DANGLING_COMPLETE
        if not np.any(just_finished_local):
            return
        just_finished_global = np.flatnonzero(active)[just_finished_local]
        finished[just_finished_global] = True
        lengths[just_finished_global] = np.int32(step_idx + _STEP_OFFSET)

    def _finalize_actions(
        self,
        actions: Int32Array,
        lengths: Int32Array,
        steps_used: int,
    ) -> tuple[Int32Array, Int32Array]:
        final_actions = actions[:, :steps_used].copy()
        final_lengths = lengths.copy()
        for row_idx in range(final_actions.shape[0]):
            if final_lengths[row_idx] != 0:
                continue
            repaired = self._repair_sequence(final_actions[row_idx])
            final_actions[row_idx].fill(self.library.EMPTY_ACTION)
            final_actions[row_idx,: repaired.shape[0]] = repaired
            final_lengths[row_idx] = np.int32(repaired.shape[0])
        max_length = int(final_lengths.max())
        return final_actions[:, :max_length].copy(), final_lengths

    def _repair_sequence(self, sequence: Int32Array) -> Int32Array:
        tokens = [int(token) for token in sequence.tolist()]
        fallback: Int32Array | None = None
        for prefix_end in range(len(tokens), -1, -1):
            completed = finish_tokens(tokens[:prefix_end], self.library)
            candidate = np.asarray(completed, dtype=np.int32)
            if candidate.shape[0] > self._max_length:
                continue
            if candidate.shape[0] >= self._min_length:
                return candidate
            if fallback is None:
                fallback = candidate
        if fallback is None:
            raise RuntimeError("repair_sequence could not construct a valid prefix.")
        return fallback

    def _length_mask(self, lengths: Tensor, max_length: int) -> Tensor:
        steps = torch.arange(max_length, device=self.device).unsqueeze(0)
        return steps < lengths.unsqueeze(1)

    def _entropy_weights(self, max_length: int, entropy_gamma: float) -> Tensor:
        weights = torch.tensor(
            [float(entropy_gamma) ** step for step in range(max_length)],
            dtype=torch.float32,
            device=self.device,
        )
        return weights.unsqueeze(0)


__all__ = ["LSTMController", "StackedRNN"]
