
from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.search.eqgpt import _scoring
from kd.search.eqgpt.gates import should_zero_reward
from kd.search.eqgpt.vocab import Vocab, load_vocab

pytestmark = pytest.mark.unit





_GATE_WORDS: list[str] = ["ut", "+", "u", "+", "sin(u)", "+", "ux"]
_CANDIDATE = "u + sin(u)"
_VARIABLES: tuple[str, ...] = ("t", "x")


def _tiny_platform():
    x = torch.linspace(0.0, 1.0, 12)
    t = torch.linspace(0.0, 0.5, 6)
    gx, gt = torch.meshgrid(x, t, indexing="ij")
    dataset = PDEDataset(
        name="gate_zero_tiny",
        task_type=TaskType.PDE,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=torch.sin(gx) * torch.cos(gt))},
        lhs_field="u",
        lhs_axis="t",
    )
    return PlatformBuilder(dataset, DerivativeReqs()).build()


def _gate_zero_result():
    vocab: Vocab = load_vocab()
    sentence: list[int] = vocab.encode(_GATE_WORDS)


    assert should_zero_reward(vocab.decode(sentence), _VARIABLES) is True
    components = _tiny_platform()
    lhs_flat: Tensor = torch.zeros(12 * 6)
    return _scoring.score_candidate(
        candidate=_CANDIDATE,
        sentence=sentence,
        vocab=vocab,
        variables=_VARIABLES,
        executor=components.executor,
        context=components.context,
        lhs_flat=lhs_flat,
        sparsity_alpha=0.02,
    )


def test_gate_zero_metrics_are_worst_case_sentinels() -> None:
    result = _gate_zero_result()
    assert math.isinf(result.mse) and result.mse > 0
    assert math.isinf(result.nmse) and result.nmse > 0
    assert result.r2 == -math.inf


def test_gate_zero_preserves_load_bearing_fields() -> None:
    result = _gate_zero_result()
    assert result.score == 0.0
    assert result.is_valid is True
    assert result.error_message == ""
    assert result.terms == ["u", "sin(u)"]
    assert result.complexity == 2


def test_gate_zero_score_is_exactly_zero_not_penalty() -> None:
    result = _gate_zero_result()
    assert result.score == 0.0
    assert math.isfinite(result.score)
