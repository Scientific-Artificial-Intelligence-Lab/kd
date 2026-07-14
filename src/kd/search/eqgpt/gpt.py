
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GPTConfig:

    vocab_size: int
    n_layers: int = 6
    n_heads: int = 8
    d_model: int = 768
    d_head: int = 64
    d_ff: int = 2048
    max_pos: int = 50

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "n_layers",
            "n_heads",
            "d_model",
            "d_head",
            "d_ff",
            "max_pos",
        ):
            value = getattr(self, name)


            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")

    @property
    def attn_inner_dim(self) -> int:
        return self.n_heads * self.d_head


def _pad_mask(tokens: Tensor) -> Tensor:
    seq_len = tokens.size(1)
    return tokens.eq(0).unsqueeze(1).expand(-1, seq_len, -1)


def _causal_mask(seq_len: int, device: torch.device) -> Tensor:
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    )


def _decoder_self_attn_mask(tokens: Tensor) -> Tensor:
    seq_len = tokens.size(1)
    return _pad_mask(tokens) | _causal_mask(seq_len, tokens.device).unsqueeze(0)


class _ScaledDotProductAttention(nn.Module):

    def __init__(self, d_head: int) -> None:
        super().__init__()
        self._scale = math.sqrt(d_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor) -> Tensor:
        scores = torch.matmul(q, k.transpose(-1, -2)) / self._scale
        scores = scores.masked_fill(attn_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)


class _MultiHeadAttention(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        inner = config.attn_inner_dim
        self.W_Q = nn.Linear(config.d_model, inner, bias=False)
        self.W_K = nn.Linear(config.d_model, inner, bias=False)
        self.W_V = nn.Linear(config.d_model, inner, bias=False)
        self.fc = nn.Linear(inner, config.d_model, bias=False)
        self.layernorm = nn.LayerNorm(config.d_model)
        self._attention = _ScaledDotProductAttention(config.d_head)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        batch = x.size(0)
        q = self.W_Q(x).view(batch, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_K(x).view(batch, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_V(x).view(batch, -1, self.n_heads, self.d_head).transpose(1, 2)
        context = self._attention(q, k, v, attn_mask.unsqueeze(1))
        context = context.transpose(1, 2).reshape(batch, -1, self.n_heads * self.d_head)
        return cast(Tensor, self.layernorm(self.fc(context) + x))


class _PoswiseFeedForward(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(config.d_ff, config.d_model, bias=False),
        )
        self.layernorm = nn.LayerNorm(config.d_model)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.layernorm(self.fc(x) + x))


class _DecoderLayer(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.dec_self_attn = _MultiHeadAttention(config)
        self.pos_ffn = _PoswiseFeedForward(config)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        x = self.dec_self_attn(x, attn_mask)
        return cast(Tensor, self.pos_ffn(x))


class _Decoder(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.tgt_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_pos, config.d_model)
        self.layers = nn.ModuleList(
            [_DecoderLayer(config) for _ in range(config.n_layers)]
        )

    def forward(self, tokens: Tensor) -> Tensor:
        batch, seq_len = tokens.shape
        pos = torch.arange(seq_len, device=tokens.device)
        pos = pos.unsqueeze(0).expand(batch, seq_len)
        x = self.tgt_emb(tokens) + self.pos_emb(pos)
        attn_mask = _decoder_self_attn_mask(tokens)
        for layer in self.layers:
            x = layer(x, attn_mask)
        return cast(Tensor, x)


class EqGPT(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.decoder = _Decoder(config)
        self.projection = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        seq_len = tokens.size(1)
        if seq_len > self.config.max_pos:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_pos={self.config.max_pos}"
            )
        hidden = self.decoder(tokens)
        return cast(Tensor, self.projection(hidden))


def adapt_pretrained_state_dict(external_sd: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {
        key: tensor for key, tensor in external_sd.items() if "dec_enc_attn" not in key
    }
