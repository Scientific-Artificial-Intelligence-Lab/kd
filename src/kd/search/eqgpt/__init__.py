"""EqGPT proposer plugin (foundation layer: vocab + GPT + backend).

This facade is the PLUGIN level of the two-tier export surface: it publishes
the package's own machinery (config, plugin, GPT + backend seam, sampler,
reward, gates, vocabulary and token mapping) for plugin-level consumers such as
``examples/16_eqgpt.py`` and this package's tests. The ENGINE level
(``kd.search``) carries
only :class:`EqGPTConfig` + :class:`EqGPTPlugin`; the top-level ``kd``
namespace carries :class:`EqGPTConfig` alone.

The vocabulary tokens stay at this level on purpose (SPEC L-3): they are an
internal representation, are never re-exported into ``kd`` or ``kd.search``,
and anything leaving the plugin converts to platform funcall-IR strings at the
boundary.
"""

from __future__ import annotations

from kd.search.eqgpt.backend import (
    ASSET_ENV_VAR,
    DEFAULT_WEIGHTS_FILENAME,
    FakeGPTBackend,
    GPTBackend,
    RealGPTBackend,
    resolve_asset_path,
)
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.gates import REDUNDANT_COMBOS, should_zero_reward
from kd.search.eqgpt.gpt import EqGPT, GPTConfig, adapt_pretrained_state_dict
from kd.search.eqgpt.ir_map import (
    TOKEN_IR_ATOM,
    MalformedSentenceError,
    UnmappedTokenError,
    ir_inexpressible_tokens,
    order_masked_tokens,
    sentence_to_rhs_terms,
    token_to_ir_atom,
)
from kd.search.eqgpt.plugin import EqGPTPlugin
from kd.search.eqgpt.pool import dedup_sentence, merge_top_k
from kd.search.eqgpt.reward import (
    INVALID_REWARD,
    WAVE_SPARSITY_ALPHA,
    RewardResult,
    compute_reward,
)
from kd.search.eqgpt.sampling import (
    AllTokensMaskedError,
    Sampler,
    SamplingConfig,
    default_start_tokens,
    dimension_masked_tokens,
)
from kd.search.eqgpt.vocab import (
    UnknownTokenError,
    Vocab,
    load_vocab,
    vocab_asset_path,
)

__all__ = [
    "ASSET_ENV_VAR",
    "DEFAULT_WEIGHTS_FILENAME",
    "INVALID_REWARD",
    "REDUNDANT_COMBOS",
    "TOKEN_IR_ATOM",
    "WAVE_SPARSITY_ALPHA",
    "AllTokensMaskedError",
    "EqGPT",
    "EqGPTConfig",
    "EqGPTPlugin",
    "FakeGPTBackend",
    "GPTBackend",
    "GPTConfig",
    "MalformedSentenceError",
    "RealGPTBackend",
    "RewardResult",
    "Sampler",
    "SamplingConfig",
    "UnknownTokenError",
    "UnmappedTokenError",
    "Vocab",
    "adapt_pretrained_state_dict",
    "compute_reward",
    "dedup_sentence",
    "default_start_tokens",
    "dimension_masked_tokens",
    "ir_inexpressible_tokens",
    "load_vocab",
    "merge_top_k",
    "order_masked_tokens",
    "resolve_asset_path",
    "sentence_to_rhs_terms",
    "should_zero_reward",
    "token_to_ir_atom",
    "vocab_asset_path",
]
