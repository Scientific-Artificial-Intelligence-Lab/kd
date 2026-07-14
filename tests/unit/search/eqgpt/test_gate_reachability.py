
from __future__ import annotations

import pytest
import torch

from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.search.eqgpt.backend import FakeGPTBackend
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.gates import should_zero_reward
from kd.search.eqgpt.ir_map import (
    UnmappedTokenError,
    ir_inexpressible_tokens,
    token_to_ir_atom,
)
from kd.search.eqgpt.plugin import EqGPTPlugin
from kd.search.eqgpt.vocab import FIRST_TERM_ID, load_vocab


_LIVE_COMBOS: dict[str, tuple[tuple[str, str], ...]] = {
    "u_sin_u": (("u", "u"), ("sin(u)", "sin(u)")),
    "x_sinx": (("x", "x"), ("sinx", "sin(x)")),
    "u_u2_u3": (("u", "u"), ("u^2", "n2(u)"), ("u^3", "n3(u)")),
}


def _components():
    x = torch.linspace(0.0, 1.0, 16)
    t = torch.linspace(0.0, 0.5, 8)
    gx, gt = torch.meshgrid(x, t, indexing="ij")
    dataset = PDEDataset(
        name="gate_tiny",
        task_type=TaskType.PDE,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=torch.sin(gx) * torch.cos(gt))},
        lhs_field="u",
        lhs_axis="t",
    )
    return PlatformBuilder(dataset, DerivativeReqs()).build()


def _mask_all_but(*words: str) -> frozenset[int]:
    vocab = load_vocab()
    keep = {vocab.word2id[word] for word in words}
    return frozenset(set(range(FIRST_TERM_ID, vocab.size)) - keep)


def _prepared(masked: frozenset[int] = frozenset()) -> EqGPTPlugin:
    config = EqGPTConfig(
        sparsity_alpha=0.02,
        seed=0,
        samples_per_epoch=64,
        top_k=4,
        max_length=16,
        variables=("t", "x"),
        masked_tokens=masked,
    )
    plugin = EqGPTPlugin(config, backend=FakeGPTBackend(57, seed=0))
    plugin.prepare(_components())
    return plugin








@pytest.mark.parametrize("combo", ["u_sin_u", "x_sinx", "u_u2_u3"])
def test_live_combo_tokens_are_sampleable(combo: str) -> None:
    plugin = _prepared()
    masked = plugin._sampler.config.masked_tokens
    vocab = load_vocab()
    for word, _ir in _LIVE_COMBOS[combo]:
        assert vocab.word2id[word] not in masked, word


@pytest.mark.parametrize("combo", ["u_sin_u", "x_sinx", "u_u2_u3"])
def test_live_combo_tokens_are_convertible(combo: str) -> None:
    for word, ir in _LIVE_COMBOS[combo]:
        assert token_to_ir_atom(word) == ir








@pytest.mark.parametrize(
    ("keep", "ir_term"),
    [(("u", "sin(u)"), "sin(u)"), (("x", "sinx"), "sin(x)")],
)
def test_newly_live_token_appears_in_real_propose_output(keep, ir_term) -> None:
    plugin = _prepared(_mask_all_but(*keep))
    seen_terms: set[str] = set()
    for _ in range(4):
        candidates = plugin.propose(64)
        for candidate in candidates:
            seen_terms.update(candidate.split(" + "))
        plugin.update(plugin.evaluate(candidates))
    assert ir_term in seen_terms








def test_gate_zeros_u_sin_combo() -> None:
    assert should_zero_reward(["ut", "+", "u", "+", "sin(u)", "+", "x"], ("t", "x"))


def test_gate_zeros_x_sinx_combo() -> None:
    assert should_zero_reward(["ut", "+", "x", "+", "sinx"], ("t", "x"))


def test_gate_zeros_u_u2_u3_combo() -> None:
    assert should_zero_reward(
        ["ut", "+", "u", "+", "u^2", "+", "u^3", "+", "x"], ("t", "x")
    )


def test_gate_control_non_combo_not_zeroed() -> None:
    assert not should_zero_reward(["ut", "+", "u", "+", "x"], ("t", "x"))







def test_sinh_combos_still_zero_at_the_gate() -> None:
    assert should_zero_reward(["ut", "+", "u", "+", "sinh(u)", "+", "x"], ("t", "x"))
    assert should_zero_reward(
        ["ut", "+", "sin(u)", "+", "sinh(u)", "+", "x"], ("t", "x")
    )


def test_sinh_is_inexpressible_and_unconvertible() -> None:
    vocab = load_vocab()
    assert vocab.word2id["sinh(u)"] in ir_inexpressible_tokens(vocab)
    with pytest.raises(UnmappedTokenError):
        token_to_ir_atom("sinh(u)")
