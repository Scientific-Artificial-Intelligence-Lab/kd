
from __future__ import annotations

import pytest
import torch

from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.search.eqgpt.backend import FakeGPTBackend
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.ir_map import order_masked_tokens
from kd.search.eqgpt.plugin import EqGPTPlugin
from kd.search.eqgpt.vocab import load_vocab


@pytest.fixture(scope="module")
def vocab():
    return load_vocab()







def test_order_mask_at_three_masks_fourth_and_fifth_order(vocab) -> None:
    masked = order_masked_tokens(vocab, 3)
    assert masked == frozenset({vocab.word2id["uxxxx"], vocab.word2id["uxxxxx"]})


def test_order_mask_at_four_masks_only_fifth_order(vocab) -> None:
    assert order_masked_tokens(vocab, 4) == frozenset({vocab.word2id["uxxxxx"]})


def test_order_mask_at_five_is_empty(vocab) -> None:
    assert order_masked_tokens(vocab, 5) == frozenset()


def test_order_mask_leaves_low_order_and_composites_untouched(vocab) -> None:
    masked = order_masked_tokens(vocab, 3)
    for word in ("u", "ux", "uxx", "uxxx", "(uux)x", "(uux)xx", "(u^4)xx", "u^2"):
        assert vocab.word2id[word] not in masked


def test_order_mask_is_monotone_in_max_order(vocab) -> None:
    assert order_masked_tokens(vocab, 3) >= order_masked_tokens(vocab, 4)
    assert order_masked_tokens(vocab, 4) >= order_masked_tokens(vocab, 5)






_BATCH = 8


def _components() -> object:
    x = torch.linspace(0.0, 1.0, 12)
    t = torch.linspace(0.0, 0.5, 6)
    gx, gt = torch.meshgrid(x, t, indexing="ij")
    dataset = PDEDataset(
        name="order_mask_tiny",
        task_type=TaskType.PDE,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=torch.sin(gx) * torch.cos(gt))},
        lhs_field="u",
        lhs_axis="t",
    )

    return PlatformBuilder(dataset, DerivativeReqs(max_atomic_order=3)).build()


def _prepared() -> EqGPTPlugin:
    config = EqGPTConfig(
        sparsity_alpha=0.02,
        seed=0,
        samples_per_epoch=_BATCH,
        top_k=4,
        max_length=12,
        variables=("t", "x"),
    )
    plugin = EqGPTPlugin(config, backend=FakeGPTBackend(57, seed=0))
    plugin.prepare(_components())
    return plugin


def test_prepare_wires_order_mask_from_max_atomic_order() -> None:
    plugin = _prepared()
    vocab = load_vocab()
    masked = plugin._sampler.config.masked_tokens
    assert plugin.derivative_requirements.max_atomic_order == 3
    assert vocab.word2id["uxxxx"] in masked
    assert vocab.word2id["uxxxxx"] in masked


def test_prepare_order_mask_does_not_swallow_third_order() -> None:
    plugin = _prepared()
    vocab = load_vocab()
    masked = plugin._sampler.config.masked_tokens
    assert vocab.word2id["uxxx"] not in masked
