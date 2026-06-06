
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, overload

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class AttentionState:

    context: Tensor
    history: Tensor


class BahdanauAttention(nn.Module):

    def __init__(
        self,
        attn_size: int,
        num_layers: int,
        attn_length: int = 10,
    ) -> None:
        super().__init__()
        if attn_size <= 0 or num_layers <= 0 or attn_length <= 0:
            raise ValueError("attn_size, num_layers, and attn_length must be > 0.")
        state_dim = 2 * num_layers * attn_size
        self.attn_size = attn_size
        self.attn_length = attn_length
        self.key_proj = nn.Linear(attn_size, attn_size, bias=False)
        self.query_proj = nn.Linear(state_dim, attn_size)
        self.output_proj = nn.Linear(2 * attn_size, attn_size)
        self.v = nn.Parameter(torch.empty(attn_size))
        self.reset_parameters()

    def init_state(self, batch_size: int, device: torch.device) -> AttentionState:
        context = torch.zeros(batch_size, self.attn_size, device=device)
        history = torch.zeros(
            batch_size,
            self.attn_length,
            self.attn_size,
            device=device,
        )
        return AttentionState(context=context, history=history)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        limit = math.sqrt(3.0 / self.attn_size)
        nn.init.uniform_(self.v, -limit, limit)

    @overload
    def forward(
        self,
        cell_output: Tensor,
        cell_state_flat: Tensor,
        attn_state: AttentionState,
        return_weights: Literal[False] = False,
    ) -> tuple[Tensor, AttentionState]: ...

    @overload
    def forward(
        self,
        cell_output: Tensor,
        cell_state_flat: Tensor,
        attn_state: AttentionState,
        return_weights: Literal[True],
    ) -> tuple[Tensor, AttentionState, Tensor]: ...

    def forward(
        self,
        cell_output: Tensor,
        cell_state_flat: Tensor,
        attn_state: AttentionState,
        return_weights: bool = False,
    ) -> tuple[Tensor, AttentionState] | tuple[Tensor, AttentionState, Tensor]:
        weights = self._score(cell_state_flat, attn_state.history)
        context = torch.sum(weights.unsqueeze(-1) * attn_state.history, dim=1)
        output_input = torch.cat([cell_output, context], dim=1)
        output = self.output_proj(output_input)
        next_history = torch.cat(
            [attn_state.history[:, 1:,:], output.unsqueeze(1)],
            dim=1,
        )
        next_state = AttentionState(context=context, history=next_history)
        if return_weights:
            return output, next_state, weights
        return output, next_state

    def _score(self, query: Tensor, history: Tensor) -> Tensor:
        key_features = self.key_proj(history)
        query_features = self.query_proj(query).unsqueeze(1)
        scores = torch.sum(self.v * torch.tanh(key_features + query_features), dim=-1)
        return functional.softmax(scores, dim=-1)


__all__ = ["AttentionState", "BahdanauAttention"]
