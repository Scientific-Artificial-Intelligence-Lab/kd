
from __future__ import annotations

import torch
from torch import Tensor

from kd.search.eqgpt.gpt import GPTConfig



REAL_CONFIG = GPTConfig(vocab_size=57)


SMALL_CONFIG = GPTConfig(
    vocab_size=8,
    n_layers=2,
    n_heads=2,
    d_model=16,
    d_head=4,
    d_ff=32,
    max_pos=12,
)





_ATTN_SUBMODULES = ("dec_self_attn", "dec_enc_attn")


def external_checkpoint_keys(config: GPTConfig) -> dict[str, tuple[int, ...]]:
    attn_inner = config.attn_inner_dim
    d_model = config.d_model
    schema: dict[str, tuple[int, ...]] = {
        "decoder.tgt_emb.weight": (config.vocab_size, d_model),
        "decoder.pos_emb.weight": (config.max_pos, d_model),
    }
    for i in range(config.n_layers):
        for attn in _ATTN_SUBMODULES:
            base = f"decoder.layers.{i}.{attn}"
            schema[f"{base}.W_Q.weight"] = (attn_inner, d_model)
            schema[f"{base}.W_K.weight"] = (attn_inner, d_model)
            schema[f"{base}.W_V.weight"] = (attn_inner, d_model)
            schema[f"{base}.fc.weight"] = (d_model, attn_inner)
            schema[f"{base}.layernorm.weight"] = (d_model,)
            schema[f"{base}.layernorm.bias"] = (d_model,)
        ffn = f"decoder.layers.{i}.pos_ffn"
        schema[f"{ffn}.fc.0.weight"] = (config.d_ff, d_model)
        schema[f"{ffn}.fc.2.weight"] = (d_model, config.d_ff)
        schema[f"{ffn}.layernorm.weight"] = (d_model,)
        schema[f"{ffn}.layernorm.bias"] = (d_model,)
    schema["projection.weight"] = (config.vocab_size, d_model)
    schema["projection.bias"] = (config.vocab_size,)
    return schema


def source_fingerprints(config: GPTConfig) -> dict[str, float]:
    return {key: float(i + 1) for i, key in enumerate(external_checkpoint_keys(config))}


def build_external_state_dict(config: GPTConfig) -> dict[str, Tensor]:
    fingerprints = source_fingerprints(config)
    return {
        key: torch.full(shape, fingerprints[key], dtype=torch.float32)
        for key, shape in external_checkpoint_keys(config).items()
    }
