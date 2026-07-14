
from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

import pytest
import torch
from torch import Tensor

from kd.core.evaluator import EvaluationResult
from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs
from kd.core.term_cache import TermColumnCache
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.search.eqgpt import _scoring
from kd.search.eqgpt.backend import FakeGPTBackend
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.plugin import EqGPTPlugin

if TYPE_CHECKING:
    from kd.core.expr.executor import PythonExecutor

_BATCH = 8







def _components():
    x = torch.linspace(0.0, 1.0, 12)
    t = torch.linspace(0.0, 0.5, 6)
    gx, gt = torch.meshgrid(x, t, indexing="ij")
    dataset = PDEDataset(
        name="scoring_cache_tiny",
        task_type=TaskType.PDE,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=torch.sin(gx) * torch.cos(gt))},
        lhs_field="u",
        lhs_axis="t",
    )
    return PlatformBuilder(dataset, DerivativeReqs()).build()


def _prepared_plugin() -> EqGPTPlugin:
    config = EqGPTConfig(
        sparsity_alpha=0.02,
        seed=0,
        samples_per_epoch=_BATCH,
        top_k=4,
        max_length=12,
    )
    plugin = EqGPTPlugin(config, backend=FakeGPTBackend(57, seed=0))
    plugin.prepare(_components())
    return plugin


class SpyExecutor:

    def __init__(self, inner: PythonExecutor) -> None:
        self._inner = inner
        self.calls: Counter[str] = Counter()

    def execute(self, code, context, force_diff_path=False):
        self.calls[code] += 1
        return self._inner.execute(code, context, force_diff_path)

    @property
    def registry(self):
        return self._inner.registry


def _floats_identical(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _tensors_identical(a: Tensor | None, b: Tensor | None) -> bool:
    if a is None or b is None:
        return a is b
    return bool(torch.equal(a, b))


def _assert_result_bitwise_equal(
    a: EvaluationResult, b: EvaluationResult, *, label: str
) -> None:
    assert a.is_valid == b.is_valid, f"{label}: is_valid"
    assert _floats_identical(a.mse, b.mse), f"{label}: mse"
    assert _floats_identical(a.nmse, b.nmse), f"{label}: nmse"
    assert _floats_identical(a.r2, b.r2), f"{label}: r2"
    assert _floats_identical(a.score, b.score), f"{label}: aic"
    assert _tensors_identical(a.coefficients, b.coefficients), f"{label}: coeffs"







@pytest.mark.unit
def test_execute_term_cache_dedup_and_equivalence() -> None:
    components = _components()
    ctx = components.context
    spy = SpyExecutor(components.executor)
    cache = TermColumnCache()


    col1 = _scoring.execute_term(spy, ctx, "u_x", cache=cache)
    col2 = _scoring.execute_term(spy, ctx, "u_x", cache=cache)
    assert spy.calls["u_x"] == 1, "cached term re-executed"
    assert torch.equal(col1, col2)


    spy_nocache = SpyExecutor(_components().executor)
    ref = _scoring.execute_term(spy_nocache, ctx, "u_x", cache=None)
    _scoring.execute_term(spy_nocache, ctx, "u_x", cache=None)
    assert spy_nocache.calls["u_x"] == 2
    assert torch.equal(col1, ref)







def _first_scorable_candidate(
    plugin: EqGPTPlugin,
) -> tuple[str, list[int]] | None:
    candidates = plugin.propose(_BATCH)
    for cand in candidates:
        sentence = plugin._pending_cache.get(cand)
        if sentence is not None and cand:
            return cand, sentence
    return None


def _score_kwargs(plugin: EqGPTPlugin):
    return dict(
        vocab=plugin._vocab,
        variables=plugin._variables,
        context=plugin._components.context,
        lhs_flat=plugin._lhs_flat,
        sparsity_alpha=plugin._config.sparsity_alpha,
    )


@pytest.mark.unit
def test_score_candidate_cache_bitwise_identical() -> None:
    plugin = _prepared_plugin()
    found = _first_scorable_candidate(plugin)
    if found is None:
        pytest.skip("FakeGPTBackend produced no scorable candidate")
    candidate, sentence = found
    kwargs = _score_kwargs(plugin)
    executor = plugin._components.executor

    ref = _scoring.score_candidate(
        candidate=candidate, sentence=sentence, executor=executor,
        term_cache=None, **kwargs,
    )
    cached = _scoring.score_candidate(
        candidate=candidate, sentence=sentence, executor=executor,
        term_cache=TermColumnCache(), **kwargs,
    )
    _assert_result_bitwise_equal(ref, cached, label=f"candidate={candidate!r}")


@pytest.mark.unit
def test_score_candidate_shared_cache_dedupes_across_calls() -> None:
    plugin = _prepared_plugin()
    found = _first_scorable_candidate(plugin)
    if found is None:
        pytest.skip("FakeGPTBackend produced no scorable candidate")
    candidate, sentence = found
    kwargs = _score_kwargs(plugin)
    terms = candidate.split(" + ")


    spy_nocache = SpyExecutor(plugin._components.executor)
    _scoring.score_candidate(
        candidate=candidate, sentence=sentence, executor=spy_nocache,
        term_cache=None, **kwargs,
    )
    _scoring.score_candidate(
        candidate=candidate, sentence=sentence, executor=spy_nocache,
        term_cache=None, **kwargs,
    )
    assert all(spy_nocache.calls[term] == 2 for term in terms)


    spy_cache = SpyExecutor(plugin._components.executor)
    shared = TermColumnCache()
    _scoring.score_candidate(
        candidate=candidate, sentence=sentence, executor=spy_cache,
        term_cache=shared, **kwargs,
    )
    _scoring.score_candidate(
        candidate=candidate, sentence=sentence, executor=spy_cache,
        term_cache=shared, **kwargs,
    )
    assert all(spy_cache.calls[term] == 1 for term in terms)







@pytest.mark.unit
def test_prepare_builds_cache_but_state_excludes_it() -> None:
    plugin = _prepared_plugin()

    assert hasattr(plugin, "_term_cache")

    state = plugin.state
    assert "term_cache" not in state
    assert not any(isinstance(v, TermColumnCache) for v in state.values())


@pytest.mark.unit
def test_execute_term_always_returns_detached_column() -> None:
    components = _components()
    ctx = components.context

    class _GradExecutor:

        def execute(self, term: str, context):
            base = torch.ones(4, requires_grad=True)
            value = base * 2.0
            return type("R", (), {"value": value})()

    grad_executor = _GradExecutor()

    no_cache = _scoring.execute_term(grad_executor, ctx, "u_x", cache=None)
    assert no_cache.requires_grad is False

    cache = TermColumnCache()
    miss = _scoring.execute_term(grad_executor, ctx, "u_x", cache=cache)
    hit = _scoring.execute_term(grad_executor, ctx, "u_x", cache=cache)
    assert miss.requires_grad is False
    assert hit.requires_grad is False
    assert torch.equal(miss, hit)
