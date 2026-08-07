
from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.axes import Axes
from torch import Tensor

from kd.core.equation import Form
from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.search.eqgpt import _scoring
from kd.search.eqgpt import viz as eqgpt_viz
from kd.search.eqgpt._multicase import MultiCaseEvaluator, WaveCaseBundle
from kd.search.eqgpt.backend import FakeGPTBackend
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.plugin import EqGPTPlugin
from kd.search.eqgpt.vocab import load_vocab
from kd.search.protocol import PlatformComponents

pytestmark = pytest.mark.unit

_ALPHA = 0.02
_BATCH = 8








class _ExecOut:

    def __init__(self, value: Tensor) -> None:
        self.value = value


class FakeExecutor:

    def __init__(self, columns: dict[str, Tensor]) -> None:
        self._columns = columns

    def execute(self, term: str, context: Any = None) -> _ExecOut:
        return _ExecOut(self._columns[term])


class FakeContext:
    pass


def _bundle(
    case_name: str,
    columns: dict[str, Tensor],
    pinned: Tensor,
    generation: int,
) -> WaveCaseBundle:
    return WaveCaseBundle(
        case_name=case_name,
        executor=FakeExecutor(columns),
        context=FakeContext(),
        pinned_lhs=pinned,
        n_points=pinned.numel(),
        cache_generation=generation,
    )


def _single_case_evaluator(
    columns: dict[str, Tensor], pinned: Tensor
) -> MultiCaseEvaluator:
    return MultiCaseEvaluator(
        reward_bundles=[_bundle("caseA", columns, pinned, 0)],
        coeff_bundles=[_bundle("caseA", columns, pinned, 100)],
        primary_case="caseA",
        sparsity_alpha=_ALPHA,
    )





_COLUMN = torch.tensor([1.0, -1.0, 2.0, -2.0, 3.0, -3.0], dtype=torch.float64)
_K = 2.0
_OFFSET = torch.ones(6, dtype=torch.float64)
_PINNED = _K * _COLUMN + _OFFSET
_TERM = "u_x"








def _components() -> PlatformComponents:
    x = torch.linspace(0.0, 1.0, 12)
    t = torch.linspace(0.0, 0.5, 6)
    gx, gt = torch.meshgrid(x, t, indexing="ij")
    dataset = PDEDataset(
        name="sign_convention_tiny",
        task_type=TaskType.PDE,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=torch.sin(gx) * torch.cos(gt))},
        lhs_field="u",
        lhs_axis="t",
    )
    return PlatformBuilder(dataset, DerivativeReqs()).build()


def _prepared_plugin(components: PlatformComponents) -> EqGPTPlugin:
    config = EqGPTConfig(
        sparsity_alpha=_ALPHA,
        seed=0,
        samples_per_epoch=_BATCH,
        top_k=4,
        max_length=12,
        variables=("t", "x"),
    )
    plugin = EqGPTPlugin(config, backend=FakeGPTBackend(57, seed=0))
    plugin.prepare(components)
    return plugin


def _run(plugin: EqGPTPlugin, epochs: int) -> None:
    for _ in range(epochs):
        candidates = plugin.propose(_BATCH)
        plugin.update(plugin.evaluate(candidates))


def _platform_lhs(components: PlatformComponents) -> Tensor:
    assert components.evaluator is not None
    return components.evaluator.lhs_target.reshape(-1)


def _execute_columns(components: PlatformComponents, terms: list[str]) -> Tensor:
    context = components.context
    assert context is not None
    return torch.stack(
        [
            components.executor.execute(term, context).value.reshape(-1).detach()
            for term in terms
        ],
        dim=1,
    )







def test_multicase_final_result_publishes_ut_domain_coefficients() -> None:
    assert float(torch.dot(_COLUMN, _OFFSET)) == 0.0

    mc = _single_case_evaluator({_TERM: _COLUMN}, _PINNED)
    result = mc.build_final_result([_TERM], best_reward=0.77)

    assert result.is_valid is True


    assert result.form is Form.EVOLUTION
    assert result.coefficients is not None
    torch.testing.assert_close(
        result.coefficients.double(),
        torch.tensor([_K], dtype=torch.float64),
        rtol=1e-6,
        atol=1e-9,
    )


def test_multicase_result_target_is_unnegated_ut() -> None:
    mc = _single_case_evaluator({_TERM: _COLUMN}, _PINNED)
    result = mc.build_final_result([_TERM], best_reward=0.77)
    target = mc.result_target()

    torch.testing.assert_close(
        target.double(), _PINNED, rtol=1e-6, atol=1e-9
    )

    assert result.residuals is not None
    assert result.coefficients is not None


    assert float(result.residuals.double().abs().max()) > 0.5
    predicted = target.double() + result.residuals.double()
    torch.testing.assert_close(
        predicted,
        _COLUMN * result.coefficients.double()[0],
        rtol=1e-6,
        atol=1e-9,
    )


def test_burgers_advection_coefficient_is_negative_after_handoff() -> None:
    advection = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
    diffusion = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float64)
    columns = {"mul(u, u_x)": advection, "u_xx": diffusion}
    terms = list(columns)
    pinned = -1.0 * advection + 0.1 * diffusion

    mc = _single_case_evaluator(columns, pinned)
    result = mc.build_final_result(terms, best_reward=0.5)

    assert result.coefficients is not None
    torch.testing.assert_close(
        result.coefficients.double(),
        torch.tensor([-1.0, 0.1], dtype=torch.float64),
        rtol=1e-6,
        atol=1e-9,
    )







def test_plugin_result_target_is_the_platform_ut_not_its_negation() -> None:
    components = _components()
    plugin = _prepared_plugin(components)
    _run(plugin, epochs=1)

    lhs = _platform_lhs(components)
    assert not torch.allclose(lhs, -lhs)

    torch.testing.assert_close(
        plugin.build_result_target(), lhs, rtol=1e-6, atol=1e-9
    )


def test_plugin_final_coefficients_predict_ut_not_negated_ut() -> None:
    components = _components()
    plugin = _prepared_plugin(components)
    _run(plugin, epochs=1)

    final = plugin.build_final_result()
    assert final.is_valid is True
    assert final.coefficients is not None
    assert final.terms

    theta = _execute_columns(components, final.terms)
    predicted = theta @ final.coefficients
    assert float(torch.dot(predicted, predicted)) > 0.0
    assert float(torch.dot(predicted, _platform_lhs(components))) > 0.0







def test_score_candidate_publishes_ut_domain_coefficients() -> None:
    components = _components()
    context = components.context
    assert context is not None
    column = components.executor.execute(_TERM, context).value.reshape(-1).detach()
    lhs_flat = _K * column

    vocab = load_vocab()


    sentence = vocab.encode(["ut", "+", "ux"])

    result = _scoring.score_candidate(
        candidate=_TERM,
        sentence=sentence,
        vocab=vocab,
        variables=("t", "x"),
        executor=components.executor,
        context=context,
        lhs_flat=lhs_flat,
        sparsity_alpha=_ALPHA,
    )



    assert result.r2 == pytest.approx(1.0, abs=1e-9)
    assert result.score == pytest.approx(1.0 - _ALPHA * math.log10(2.0), abs=1e-9)

    assert result.form is Form.EVOLUTION
    assert result.coefficients is not None
    torch.testing.assert_close(
        result.coefficients,
        torch.tensor([_K], dtype=torch.float32),
        rtol=1e-5,
        atol=1e-6,
    )







@pytest.fixture
def ax() -> Iterator[Axes]:
    fig = plt.figure()
    try:
        yield fig.add_subplot()
    finally:
        plt.close(fig)


def _panel_text(ax: Axes) -> str:
    parts: list[str] = [ax.get_title(), *(text.get_text() for text in ax.texts)]
    figure = ax.get_figure()
    if figure is not None:
        parts.extend(text.get_text() for text in figure.texts)
    return " ".join(parts).lower()




_PER_CASE = {"N_a": 0.94, "N_b": 0.92, "N_c": 0.95}


def test_per_case_reward_returns_the_statistics_scope_note(ax: Axes) -> None:
    channel = eqgpt_viz.render_per_case_reward(ax, dict(_PER_CASE))

    assert any(
        "primary" in note.lower() and "mean" in note.lower() for note in channel
    ), channel


def test_per_case_reward_draws_the_statistics_scope_caption(ax: Axes) -> None:
    eqgpt_viz.render_per_case_reward(ax, dict(_PER_CASE))

    text = _panel_text(ax)
    assert "primary" in text, text
    assert "mean" in text, text
