
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import numpy as np
import pytest
import torch
from torch import Tensor

from kd.search.eqgpt import vocab as vmod
from kd.search.eqgpt.backend import FakeGPTBackend
from kd.search.eqgpt.sampling import (
    AllTokensMaskedError,
    Sampler,
    SamplingConfig,
    default_start_tokens,
    dimension_masked_tokens,
)
from kd.search.eqgpt.vocab import load_vocab

_OPERATOR_IDS = {vmod.E_ID, vmod.PLUS_ID, vmod.MUL_ID, vmod.DIV_ID}
_MAX_LEN = 16


class _BiasedBackend:

    def __init__(self, vocab_size: int, favoured: int) -> None:
        self.vocab_size = vocab_size
        self.favoured = favoured

    def _row(self) -> Tensor:
        row = torch.zeros(self.vocab_size)
        row[self.favoured] = 100.0
        return row

    def forward_logits(self, tokens: Tensor) -> Tensor:
        batch, seq = tokens.shape
        return self._row().expand(batch, seq, self.vocab_size)

    def next_token_logits(self, prefix: Sequence[int]) -> Tensor:
        if not prefix:
            raise ValueError("empty prefix")
        return self._row()

    def state_dict(self) -> dict[str, Tensor]:
        return {}

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        return None

    def parameters(self):
        return iter(())


@pytest.fixture(scope="module")
def vocab():
    return load_vocab()


def _sampler(
    vocab,
    *,
    backend=None,
    masked=frozenset(),
    exploration_rate=0.2,
    max_length=_MAX_LEN,
):
    config = SamplingConfig(
        start_tokens=default_start_tokens(vocab),
        masked_tokens=frozenset(masked),
        exploration_rate=exploration_rate,
        max_length=max_length,
    )
    if backend is None:
        backend = FakeGPTBackend(vocab.size, seed=0)
    return Sampler(backend, vocab, config)







def test_default_start_tokens_is_pinned_s_ut_plus(vocab) -> None:
    assert default_start_tokens(vocab) == (vmod.S_ID, vocab.word2id["ut"], vmod.PLUS_ID)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_tokens": ()},
        {"start_tokens": (5, 6, 2), "exploration_rate": 1.5},
        {"start_tokens": (5, 6, 2), "exploration_rate": -0.1},
        {"start_tokens": (5, 6, 2), "max_length": 3},
    ],
)
def test_sampling_config_rejects_invalid(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SamplingConfig(**kwargs)







def test_samples_are_sat_legal(vocab) -> None:
    sampler = _sampler(vocab)
    for sentence in sampler.sample_batch(24, seed=0):
        assert sentence[0] == vmod.S_ID
        assert all(0 <= tok < vocab.size for tok in sentence)
        for i in range(1, len(sentence)):
            if i % 2 == 1:
                assert sentence[i] >= vmod.FIRST_TERM_ID, (i, sentence)
            else:
                assert sentence[i] in _OPERATOR_IDS, (i, sentence)

        if vmod.E_ID in sentence:
            assert sentence.index(vmod.E_ID) == len(sentence) - 1


def test_masked_tokens_never_appear(vocab) -> None:
    masked = frozenset({7, 8, 9, 10, vmod.MUL_ID})
    sampler = _sampler(vocab, masked=masked)
    for sentence in sampler.sample_batch(24, seed=1):
        assert masked.isdisjoint(sentence), sentence


def test_sampling_terminates_within_cap(vocab) -> None:
    sampler = _sampler(vocab)
    for sentence in sampler.sample_batch(24, seed=2):
        assert len(sentence) <= _MAX_LEN
        assert sentence[-1] == vmod.E_ID or len(sentence) == _MAX_LEN


def test_same_seed_same_batch_different_seed_differs(vocab) -> None:
    sampler = _sampler(vocab)
    a = sampler.sample_batch(16, seed=0)
    b = sampler.sample_batch(16, seed=0)
    c = sampler.sample_batch(16, seed=1)
    assert a == b
    assert a != c


def test_batch_has_intra_batch_diversity(vocab) -> None:
    sampler = _sampler(vocab)
    batch = sampler.sample_batch(24, seed=0)
    assert len({tuple(s) for s in batch}) > 1


def test_sampler_consumes_backend_logits_when_greedy(vocab) -> None:
    favoured = vocab.word2id["u"]
    sampler = _sampler(
        vocab, backend=_BiasedBackend(vocab.size, favoured), exploration_rate=0.0
    )


    start_len = len(default_start_tokens(vocab))
    for sentence in sampler.sample_batch(16, seed=0):
        for i in range(start_len, len(sentence)):
            if i % 2 == 1:
                assert sentence[i] == favoured, sentence


def test_exploration_rate_one_ignores_backend_bias(vocab) -> None:
    favoured = vocab.word2id["u"]
    sampler = _sampler(
        vocab, backend=_BiasedBackend(vocab.size, favoured), exploration_rate=1.0
    )


    start_len = len(default_start_tokens(vocab))
    seen_terms = {
        sentence[i]
        for sentence in sampler.sample_batch(24, seed=0)
        for i in range(start_len, len(sentence))
        if i % 2 == 1
    }
    assert seen_terms - {favoured}


def test_sample_single_is_legal_and_seed_reproducible(vocab) -> None:
    sampler = _sampler(vocab)
    sentence = sampler.sample(seed=5)
    assert sentence[0] == vmod.S_ID
    for i in range(1, len(sentence)):
        if i % 2 == 1:
            assert sentence[i] >= vmod.FIRST_TERM_ID, sentence
        else:
            assert sentence[i] in _OPERATOR_IDS, sentence
    assert sampler.sample(seed=5) == sentence


def test_length_cap_termination_when_e_masked(vocab) -> None:
    sampler = _sampler(vocab, masked=frozenset({vmod.E_ID}))
    for sentence in sampler.sample_batch(16, seed=3):
        assert len(sentence) == _MAX_LEN
        assert vmod.E_ID not in sentence
        for i in range(1, len(sentence)):
            if i % 2 == 1:
                assert sentence[i] >= vmod.FIRST_TERM_ID, sentence
            else:
                assert sentence[i] in (_OPERATOR_IDS - {vmod.E_ID}), sentence


def test_sample_batch_zero_returns_empty(vocab) -> None:
    assert _sampler(vocab).sample_batch(0, seed=0) == []


def test_sampling_does_not_perturb_global_rng(vocab) -> None:
    sampler = _sampler(vocab)
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch_before = torch.random.get_rng_state()
    np_before = np.random.get_state()
    py_before = random.getstate()
    sampler.sample_batch(8, seed=7)
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert np.random.get_state()[1].tolist() == np_before[1].tolist()
    assert random.getstate() == py_before







def test_all_terms_masked_raises_loudly(vocab) -> None:
    all_terms = frozenset(range(vmod.FIRST_TERM_ID, vocab.size))
    sampler = _sampler(vocab, masked=all_terms)
    with pytest.raises(AllTokensMaskedError):
        sampler.sample_batch(1, seed=0)


def test_all_operators_masked_raises_loudly(vocab) -> None:
    sampler = _sampler(vocab, masked=frozenset(_OPERATOR_IDS))
    with pytest.raises(AllTokensMaskedError):
        sampler.sample_batch(1, seed=0)







def test_dimension_mask_masks_absent_axis_tokens(vocab) -> None:
    masked = dimension_masked_tokens(vocab, ["x", "t"])
    assert vocab.word2id["uy"] in masked
    assert vocab.word2id["uz"] in masked
    assert vocab.word2id["ux"] not in masked


def test_dimension_mask_always_masks_time_tokens_quirk(vocab) -> None:
    masked = dimension_masked_tokens(vocab, ["x", "y", "z", "t"])
    assert vocab.word2id["ut"] in masked
    assert vocab.word2id["ux"] not in masked
    assert vocab.word2id["uy"] not in masked


def test_dimension_mask_masks_laplace_family_when_y_absent(vocab) -> None:
    masked = dimension_masked_tokens(vocab, ["x", "t"])
    assert vocab.word2id["Laplace(u)"] in masked
    assert vocab.word2id["BiLaplace(u)"] in masked


def test_dimension_mask_t_quirk_blast_radius(vocab) -> None:
    masked = dimension_masked_tokens(vocab, ["t", "x", "y", "z"])
    for token in ("sqrt(u)", "sqrt(x)", "sint"):
        assert vocab.word2id[token] in masked, token
