
from __future__ import annotations

import pytest

from kd.search.eqgpt.ir_map import (
    TOKEN_IR_ATOM,
    MalformedSentenceError,
    UnmappedTokenError,
    ir_inexpressible_tokens,
    sentence_to_rhs_terms,
    token_to_ir_atom,
)
from kd.search.eqgpt.vocab import FIRST_TERM_ID, load_vocab

_START_LEN = 3


@pytest.fixture(scope="module")
def vocab():
    return load_vocab()


def _ids(vocab, words: list[str]) -> list[int]:
    return [vocab.word2id[w] for w in words]







@pytest.mark.parametrize(
    ("word", "ir"),
    [
        ("u", "u"),
        ("ux", "u_x"),
        ("uxx", "u_xx"),
        ("uxxx", "u_xxx"),
        ("uxxxx", "u_xxxx"),
        ("uxxxxx", "u_xxxxx"),
        ("u^2", "n2(u)"),
        ("u^3", "n3(u)"),
        ("x", "x"),
        ("(uux)x", "diff_x(mul(u, u_x))"),
        ("(uux)xx", "diff2_x(mul(u, u_x))"),
        ("Laplace(u)", "lap(u)"),
    ],
)
def test_token_to_ir_atom(word: str, ir: str) -> None:
    assert token_to_ir_atom(word) == ir


@pytest.mark.parametrize("word", ["sqrt(u)", "sqrt(x)", "sinh(u)"])
def test_token_to_ir_atom_unmapped_raises(word: str) -> None:
    with pytest.raises(UnmappedTokenError):
        token_to_ir_atom(word)







def test_sentence_wave_family_to_rhs_terms(vocab) -> None:
    sentence = _ids(vocab, ["S", "ut", "+", "ux", "+", "uxxx", "+", "(uux)xx", "E"])
    terms = sentence_to_rhs_terms(vocab, sentence, start_len=_START_LEN)
    assert terms == ["u_x", "u_xxx", "diff2_x(mul(u, u_x))"]


def test_sentence_product_term(vocab) -> None:
    sentence = _ids(vocab, ["S", "ut", "+", "u", "*", "ux", "E"])
    assert sentence_to_rhs_terms(vocab, sentence, start_len=_START_LEN) == [
        "mul(u, u_x)"
    ]


def test_sentence_quotient_term(vocab) -> None:
    sentence = _ids(vocab, ["S", "ut", "+", "uxx", "/", "x", "E"])
    assert sentence_to_rhs_terms(vocab, sentence, start_len=_START_LEN) == [
        "div(u_xx, x)"
    ]


def test_sentence_three_factor_left_fold(vocab) -> None:
    sentence = _ids(vocab, ["S", "ut", "+", "uxx", "/", "x", "*", "u", "E"])
    assert sentence_to_rhs_terms(vocab, sentence, start_len=_START_LEN) == [
        "mul(div(u_xx, x), u)"
    ]


def test_ir_inexpressible_tokens_complement_of_mappable(vocab) -> None:
    masked = ir_inexpressible_tokens(vocab)
    assert vocab.word2id["(uux)xx"] not in masked
    assert vocab.word2id["sinh(u)"] in masked
    for word in TOKEN_IR_ATOM:
        if vocab.word2id[word] >= FIRST_TERM_ID:
            assert vocab.word2id[word] not in masked


def test_rhs_terms_carry_no_private_vocab_tokens(vocab) -> None:
    sentence = _ids(vocab, ["S", "ut", "+", "ux", "+", "(uux)xx", "E"])
    joined = " ".join(sentence_to_rhs_terms(vocab, sentence, start_len=_START_LEN))
    for leak in ("(uux)xx", "ut", "S", "E"):
        assert leak not in joined







def test_operator_terminated_sentence_raises(vocab) -> None:

    sentence = _ids(vocab, ["S", "ut", "+", "ux", "+"])
    with pytest.raises(MalformedSentenceError):
        sentence_to_rhs_terms(vocab, sentence, start_len=_START_LEN)


def test_unmapped_token_in_sentence_raises(vocab) -> None:

    sentence = _ids(vocab, ["S", "ut", "+", "sqrt(u)", "E"])
    with pytest.raises(UnmappedTokenError):
        sentence_to_rhs_terms(vocab, sentence, start_len=_START_LEN)
