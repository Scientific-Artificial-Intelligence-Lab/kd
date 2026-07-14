
from __future__ import annotations

import pytest
import torch

from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.search.eqgpt.backend import FakeGPTBackend
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.ir_map import (
    TOKEN_IR_ATOM,
    UnmappedTokenError,
    ir_inexpressible_tokens,
    order_masked_tokens,
    token_to_ir_atom,
)
from kd.search.eqgpt.plugin import EqGPTPlugin
from kd.search.eqgpt.sampling import dimension_masked_tokens
from kd.search.eqgpt.vocab import FIRST_TERM_ID, load_vocab






CONVERSION_CLASS: dict[int, str] = {
    0: "framing",
    1: "framing",
    2: "combinator",
    3: "combinator",
    4: "combinator",
    5: "framing",
    6: "framing",

    7: "direct", 8: "direct", 9: "direct", 10: "direct", 11: "direct",
    12: "direct", 13: "direct", 14: "direct", 15: "direct", 16: "direct",
    17: "direct", 18: "direct", 19: "direct", 20: "direct", 21: "direct",
    22: "direct", 23: "direct", 24: "direct", 25: "direct", 26: "direct",
    27: "direct",
    28: "hardfail",
    29: "direct",
    30: "hardfail",
    31: "direct", 32: "direct", 33: "direct", 34: "direct", 35: "direct",
    36: "direct", 37: "direct", 38: "direct", 39: "direct", 40: "direct",
    41: "direct", 42: "direct", 43: "direct", 44: "direct", 45: "direct",
    46: "direct", 47: "direct", 48: "direct",
    49: "hardfail",
    50: "direct", 51: "direct", 52: "direct", 53: "direct", 54: "direct",
    55: "direct", 56: "direct",
}








EXACT_IR: dict[str, str] = {
    "u": "u",
    "ux": "u_x",
    "uxx": "u_xx",
    "ux^2": "n2(u_x)",
    "uxxxx": "u_xxxx",
    "(uux)x": "diff_x(mul(u, u_x))",
    "u^2": "n2(u)",
    "uxxx": "u_xxx",
    "u^3": "n3(u)",
    "x": "x",
    "uxxxxx": "u_xxxxx",
    "sin(u)": "sin(u)",
    "Laplace(u)": "lap(u)",
    "BiLaplace(u)": "lap(lap(u))",
    "x^2": "n2(x)",
    "exp(x)": "exp(x)",
    "sinx": "sin(x)",
    "t": "t",
    "(uux)xx": "diff2_x(mul(u, u_x))",
    "(u^3)xx": "diff2_x(n3(u))",

    "(1/u)xx": "diff2_x(recip(u))",

    "(u^-2*ux)x": "diff_x(mul(recip(n2(u)), u_x))",

    "(uxx+ux/x)^2": "n2(add(u_xx, mul(recip(x), u_x)))",
    "x^4": "n2(n2(x))",
    "(u^4)xx": "diff2_x(n2(n2(u)))",

    "(u(u^2)xx)xx": "diff2_x(mul(u, diff2_x(mul(u, u_x))))",


    "uxxt": "diff_t(u_xx)",
    "ut^2": "n2(u_t)",
    "uxt": "diff_t(u_x)",
    "utt": "u_tt",
    "uxxtt": "diff2_t(u_xx)",
    "uyy": "u_yy",
    "ut^3": "n3(u_t)",
    "uy": "u_y",
    "y": "y",
    "y^2": "n2(y)",
    "uy^2": "n2(u_y)",
    "uxy": "diff_y(u_x)",
    "uz": "u_z",
    "uzz": "u_zz",
    "Laplace(utt)": "lap(u_tt)",
    "(x+y)": "add(x, y)",
    "exp(-y)": "exp(neg(y))",
    "uyyy": "u_yyy",
    "(uux)t": "diff_t(mul(u, u_x))",




    "sint": "sin(t)",
    "uyyt": "diff_t(u_yy)",
}


HARDFAIL_WORDS: tuple[str, ...] = ("sqrt(u)", "sinh(u)", "sqrt(x)")




SAMPLEABLE_WORDS: tuple[str, ...] = (
    "u", "ux", "uxx", "ux^2", "(uux)x", "(u^4)xx", "u^2", "uxxx", "u^3", "x",
    "(1/u)xx", "(u^-2*ux)x", "sin(u)", "x^2", "exp(x)", "sinx",
    "(uxx+ux/x)^2", "x^4", "(uux)xx", "(u^3)xx", "(u(u^2)xx)xx",
)


@pytest.fixture(scope="module")
def vocab():
    return load_vocab()







def test_table_covers_every_vocab_id_exactly_once(vocab) -> None:
    assert set(CONVERSION_CLASS) == set(range(vocab.size))


def test_conversion_class_counts_match_adjudication() -> None:
    from collections import Counter

    counts = Counter(CONVERSION_CLASS.values())
    assert counts == {"framing": 4, "combinator": 3, "hardfail": 3, "direct": 47}


def test_hardfail_set_is_exactly_the_unregistered_function_tokens(vocab) -> None:
    hardfail_ids = {i for i, c in CONVERSION_CLASS.items() if c == "hardfail"}
    assert hardfail_ids == {vocab.word2id[w] for w in HARDFAIL_WORDS}


def test_exact_ir_pins_every_direct_token(vocab) -> None:
    direct_words = {
        vocab.id2word[i] for i, c in CONVERSION_CLASS.items() if c == "direct"
    }
    assert set(EXACT_IR) == direct_words


def test_token_ir_atom_maps_all_47_direct_tokens(vocab) -> None:
    direct_ids = {i for i, c in CONVERSION_CLASS.items() if c == "direct"}
    mapped_ids = {
        vocab.word2id[w] for w in TOKEN_IR_ATOM if vocab.word2id[w] >= FIRST_TERM_ID
    }
    assert direct_ids <= mapped_ids







@pytest.mark.parametrize(("word", "ir"), sorted(EXACT_IR.items()))
def test_direct_token_maps_to_exact_ir(word: str, ir: str) -> None:
    assert token_to_ir_atom(word) == ir


@pytest.mark.parametrize("word", HARDFAIL_WORDS)
def test_hardfail_token_raises_unmapped(word: str) -> None:
    with pytest.raises(UnmappedTokenError):
        token_to_ir_atom(word)


def test_hardfail_tokens_are_in_ir_inexpressible(vocab) -> None:
    inexpressible = ir_inexpressible_tokens(vocab)
    for word in HARDFAIL_WORDS:
        assert vocab.word2id[word] in inexpressible


def test_sampleable_direct_tokens_are_not_inexpressible_masked(vocab) -> None:
    inexpressible = ir_inexpressible_tokens(vocab)
    for word in SAMPLEABLE_WORDS:
        assert vocab.word2id[word] not in inexpressible, word







def _tx_components():
    x = torch.linspace(0.0, 1.0, 24)
    t = torch.linspace(0.0, 0.5, 12)
    gx, gt = torch.meshgrid(x, t, indexing="ij")
    dataset = PDEDataset(
        name="tt_tx",
        task_type=TaskType.PDE,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=torch.sin(gx) * torch.cos(gt))},
        lhs_field="u",
        lhs_axis="t",
    )
    return PlatformBuilder(dataset, DerivativeReqs(max_atomic_order=3)).build()


@pytest.fixture(scope="module")
def tx_components():
    return _tx_components()


@pytest.mark.parametrize("word", SAMPLEABLE_WORDS)
def test_sampleable_direct_token_ir_executes_finitely(word, tx_components) -> None:
    ir = token_to_ir_atom(word)
    result = tx_components.executor.execute(ir, tx_components.context).value
    assert torch.isfinite(result).all()

    assert ir == EXACT_IR[word]







Y_FAMILY_WORDS: tuple[str, ...] = (
    "BiLaplace(u)", "uyy", "uy", "y", "y^2", "uy^2", "uxy",
    "(x+y)", "exp(-y)", "uyyy", "Laplace(u)",
)


def _txy_components():
    x = torch.linspace(0.1, 1.1, 8)
    t = torch.linspace(0.0, 0.5, 5)
    y = torch.linspace(0.1, 1.1, 6)
    gx, gt, gy = torch.meshgrid(x, t, y, indexing="ij")
    dataset = PDEDataset(
        name="tt_txy",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
            "y": AxisInfo(name="y", values=y),
        },
        axis_order=["x", "t", "y"],
        fields={
            "u": FieldData(
                name="u", values=torch.sin(gx) * torch.cos(gt) * torch.sin(gy)
            )
        },
        lhs_field="u",
        lhs_axis="t",
    )
    return PlatformBuilder(dataset, DerivativeReqs(max_atomic_order=3)).build()


@pytest.fixture(scope="module")
def txy_components():
    return _txy_components()


@pytest.mark.parametrize("word", Y_FAMILY_WORDS)
def test_y_family_direct_token_executes_under_txy(word, txy_components) -> None:
    ir = token_to_ir_atom(word)
    result = txy_components.executor.execute(ir, txy_components.context).value
    assert torch.isfinite(result).all()
    assert ir == EXACT_IR[word]






_BATCH = 8


def _prepared_tx() -> EqGPTPlugin:
    config = EqGPTConfig(
        sparsity_alpha=0.02,
        seed=0,
        samples_per_epoch=_BATCH,
        top_k=4,
        max_length=12,
        variables=("t", "x"),
    )
    plugin = EqGPTPlugin(config, backend=FakeGPTBackend(57, seed=0))
    plugin.prepare(_tx_components())
    return plugin


def test_sampled_term_ids_under_tx_equal_the_adjudicated_21(vocab) -> None:
    plugin = _prepared_tx()
    masked = frozenset(plugin._sampler.config.masked_tokens)
    legal = frozenset(range(FIRST_TERM_ID, vocab.size)) - masked
    expected = frozenset(vocab.word2id[w] for w in SAMPLEABLE_WORDS)
    assert legal == expected


def test_prepare_mask_equals_dimension_order_inexpressible_union(vocab) -> None:
    plugin = _prepared_tx()
    expected = (
        dimension_masked_tokens(vocab, ("t", "x"))
        | order_masked_tokens(vocab, 3)
        | ir_inexpressible_tokens(vocab)
    )
    assert frozenset(plugin._sampler.config.masked_tokens) == expected


def test_sampleable_count_is_21_not_the_s2c_partial(vocab) -> None:
    assert len(SAMPLEABLE_WORDS) == 21
    assert len(set(SAMPLEABLE_WORDS)) == 21







def test_laplace_masked_under_tx_but_sampleable_under_txy() -> None:
    vocab = load_vocab()
    lap_id = vocab.word2id["Laplace(u)"]
    assert lap_id in dimension_masked_tokens(vocab, ("t", "x"))
    assert lap_id not in dimension_masked_tokens(vocab, ("t", "x", "y"))
