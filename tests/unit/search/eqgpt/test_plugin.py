
from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import pytest
import torch

from kd.core.evaluator import EvaluationResult
from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.search.eqgpt import vocab as vmod
from kd.search.eqgpt.backend import FakeGPTBackend
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.plugin import STATE_KEYS, EqGPTPlugin
from kd.search.protocol import PlatformComponents

_PROTOCOL_MEMBERS = (
    "prepare",
    "propose",
    "evaluate",
    "update",
    "build_final_result",
    "build_result_target",
    "best_score",
    "best_expression",
    "state",
    "config",
    "score_kind",
    "score_direction",
)
_BATCH = 8


def _components() -> PlatformComponents:
    x = torch.linspace(0.0, 1.0, 12)
    t = torch.linspace(0.0, 0.5, 6)
    gx, gt = torch.meshgrid(x, t, indexing="ij")
    dataset = PDEDataset(
        name="plugin_tiny",
        task_type=TaskType.PDE,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=torch.sin(gx) * torch.cos(gt))},
        lhs_field="u",
        lhs_axis="t",
    )
    return PlatformBuilder(dataset, DerivativeReqs()).build()


def _plugin(*, variables=None, masked=frozenset(), max_length=12) -> EqGPTPlugin:
    config = EqGPTConfig(
        sparsity_alpha=0.02,
        seed=0,
        samples_per_epoch=_BATCH,
        top_k=4,
        max_length=max_length,
        variables=variables,
        masked_tokens=frozenset(masked),
    )
    return EqGPTPlugin(config, backend=FakeGPTBackend(57, seed=0))


def _prepared(**kwargs) -> EqGPTPlugin:
    plugin = _plugin(**kwargs)
    plugin.prepare(_components())
    return plugin


def _run(plugin: EqGPTPlugin, epochs: int) -> None:
    for _ in range(epochs):
        candidates = plugin.propose(_BATCH)
        plugin.update(plugin.evaluate(candidates))


def _reward(result: EvaluationResult) -> float:
    return result.score if result.score is not None else 0.0


def _survivor_count(results: list[EvaluationResult]) -> int:
    return sum(1 for r in results if _reward(r) != 0.0)







def test_score_contract_is_eqgpt_reward_max() -> None:

    assert EqGPTPlugin.score_kind == "EqGPT reward"
    assert EqGPTPlugin.score_direction == "max"


def test_plugin_exposes_full_search_algorithm_surface() -> None:


    plugin = _plugin()
    for member in _PROTOCOL_MEMBERS:
        inspect.getattr_static(plugin, member)


def test_state_key_contract_enumerates_h5_components() -> None:
    assert {
        "backend_state",
        "optimizer_state",
        "top_k",
        "rng_state",
        "reward_history",
        "fingerprints",
    } == STATE_KEYS







def test_propose_returns_bounded_rhs_only_ir() -> None:
    plugin = _prepared()
    candidates = plugin.propose(_BATCH)
    assert 0 < len(candidates) <= _BATCH
    joined = " ".join(candidates)
    for leak in ("(uux)xx", "ut", "S", "E", "uxxx"):
        assert leak not in joined


def test_default_pipeline_candidates_all_convert() -> None:
    plugin = _prepared()
    candidates = plugin.propose(_BATCH)
    assert candidates


def test_evaluate_is_one_to_one_and_order_preserving() -> None:
    plugin = _prepared()
    candidates = plugin.propose(_BATCH)
    results = plugin.evaluate(candidates)
    assert len(results) == len(candidates)
    for candidate, result in zip(candidates, results, strict=True):
        assert result.expression == candidate or (
            result.terms is not None and candidate == " + ".join(result.terms)
        )







def _mask_all_but(*words: str) -> frozenset[int]:
    w = vmod.load_vocab().word2id
    keep = {w[word] for word in words}
    return frozenset(set(range(vmod.FIRST_TERM_ID, 57)) - keep)


def test_wave_family_leaves_gate_survivors() -> None:
    wave = _prepared(variables=("t", "x"))
    wave_survivors = _survivor_count(wave.evaluate(wave.propose(_BATCH)))

    absent = _prepared(variables=("t", "x"), masked=_mask_all_but("u", "u^2"))
    absent_survivors = _survivor_count(absent.evaluate(absent.propose(_BATCH)))

    assert absent_survivors == 0
    assert wave_survivors > absent_survivors


def test_spatial_axis_absent_zeroes_all_candidates() -> None:
    plugin = _prepared(variables=("t", "x"), masked=_mask_all_but("u", "u^2"))
    _run(plugin, epochs=2)
    assert plugin.best_score == 0.0







def test_update_mutates_pool_history_and_weights() -> None:
    plugin = _prepared(variables=("t", "x"))
    initial_weights = {k: v.clone() for k, v in plugin.state["backend_state"].items()}
    _run(plugin, epochs=2)
    state = plugin.state
    assert state["top_k"]
    assert len(state["reward_history"]) == 2
    changed = any(
        not torch.equal(initial_weights[k], v)
        for k, v in state["backend_state"].items()
    )
    assert changed, "fine-tuning did not change any GPT weight (no-op update)"







def test_prepare_preserves_perturbed_restored_state() -> None:
    source = _prepared(variables=("t", "x"))
    perturbed = source.state

    key = next(iter(perturbed["backend_state"]))
    perturbed["backend_state"][key] = perturbed["backend_state"][key] + 1.0

    target = _plugin(variables=("t", "x"))
    target.state = perturbed
    target.prepare(_components())
    assert torch.equal(
        target.state["backend_state"][key], perturbed["backend_state"][key]
    )














def _selfbuilt_config() -> EqGPTConfig:
    return EqGPTConfig(
        sparsity_alpha=0.02,
        seed=0,
        samples_per_epoch=_BATCH,
        top_k=4,
        max_length=12,
        variables=("t", "x"),
    )


def _counting_builder():
    calls = {"n": 0}

    def fake_build(self, vocab):
        calls["n"] += 1
        return FakeGPTBackend(57, seed=0)

    return calls, fake_build


def test_fresh_second_prepare_rebuilds_self_built_backend(monkeypatch) -> None:
    calls, fake_build = _counting_builder()
    monkeypatch.setattr(EqGPTPlugin, "_build_default_backend", fake_build)

    plugin = EqGPTPlugin(_selfbuilt_config(), backend=None)
    plugin.prepare(_components())
    assert calls["n"] == 1
    _run(plugin, epochs=1)

    pristine = FakeGPTBackend(57, seed=0).state_dict()
    drifted = plugin.state["backend_state"]
    assert any(not torch.equal(drifted[k], pristine[k]) for k in pristine), (
        "epoch did not drift weights -> the rebuild assertion would be vacuous"
    )

    plugin.prepare(_components())
    assert calls["n"] == 2
    rebuilt = plugin.state["backend_state"]
    for key in pristine:
        assert torch.equal(rebuilt[key], pristine[key])


def test_injected_backend_never_rebuilt_on_reprepare(monkeypatch) -> None:
    calls, fake_build = _counting_builder()
    monkeypatch.setattr(EqGPTPlugin, "_build_default_backend", fake_build)

    injected = FakeGPTBackend(57, seed=0)
    plugin = EqGPTPlugin(_selfbuilt_config(), backend=injected)
    plugin.prepare(_components())
    plugin.prepare(_components())

    assert calls["n"] == 0
    assert plugin._backend is injected


def test_restore_prepare_does_not_rebuild_self_built_backend(monkeypatch) -> None:
    calls, fake_build = _counting_builder()
    monkeypatch.setattr(EqGPTPlugin, "_build_default_backend", fake_build)

    plugin = EqGPTPlugin(_selfbuilt_config(), backend=None)
    plugin.prepare(_components())
    assert calls["n"] == 1
    _run(plugin, epochs=1)

    captured = plugin.state
    captured_backend = {k: v.clone() for k, v in captured["backend_state"].items()}

    plugin.state = captured
    plugin.prepare(_components())

    assert calls["n"] == 1
    restored = plugin.state["backend_state"]
    for key in captured_backend:
        assert torch.equal(restored[key], captured_backend[key])







def test_config_variables_must_match_dataset_axes() -> None:
    plugin = _plugin(variables=("t", "x", "y"))
    with pytest.raises((ValueError, KeyError)):
        plugin.prepare(_components())


@pytest.mark.parametrize("over", [0, 8])
def test_prepare_rejects_max_length_at_or_above_backend_max_pos(over: int) -> None:
    backend = FakeGPTBackend(57, seed=0)
    max_pos = backend.max_pos()
    max_length = max_pos + over
    config = EqGPTConfig(
        sparsity_alpha=0.02,
        seed=0,
        samples_per_epoch=_BATCH,
        top_k=4,
        max_length=max_length,
        variables=("t", "x"),
    )
    plugin = EqGPTPlugin(config, backend=backend)

    with pytest.raises(ValueError) as excinfo:
        plugin.prepare(_components())

    msg = str(excinfo.value)
    assert str(max_length) in msg, f"error must name max_length; got {msg!r}"
    assert str(max_pos) in msg, f"error must name backend max_pos; got {msg!r}"


def test_high_order_derivative_candidate_scored_invalid_not_crash() -> None:
    plugin = _prepared(variables=("t", "x"), masked=_mask_all_but("uxxx"))
    _run(plugin, epochs=1)
    assert plugin.best_score == 0.0


    results = plugin.evaluate(plugin.propose(_BATCH))
    assert results
    assert any(
        (not r.is_valid) and "execution error" in (r.error_message or "")
        for r in results
    )


def test_truncation_prone_config_drops_malformed_and_completes() -> None:
    plugin = _prepared(max_length=5)
    candidates = plugin.propose(_BATCH)
    assert 0 < len(candidates) <= _BATCH
    joined = " ".join(candidates)
    for leak in ("ut", "(uux)xx", "S", "E"):
        assert leak not in joined
    _run(plugin, epochs=1)
    assert math.isfinite(plugin.best_score)


def test_build_final_result_and_target_share_domain() -> None:
    plugin = _prepared(variables=("t", "x"))
    _run(plugin, epochs=1)
    final = plugin.build_final_result()
    target = plugin.build_result_target()
    assert final.residuals is not None
    assert final.residuals.shape == target.shape









def test_assert_primary_dataset_accepts_the_matching_case() -> None:
    plugin = _plugin()
    plugin._multicase = SimpleNamespace(primary_case="N_G2Tp12A080_broad")
    plugin._assert_primary_dataset(
        SimpleNamespace(name="wave-breaking-N_G2Tp12A080_broad")
    )


@pytest.mark.parametrize(
    "dataset",
    [
        SimpleNamespace(name="wave-breaking-N_OTHER_case"),
        SimpleNamespace(name="plugin_tiny"),
        SimpleNamespace(),
    ],
)
def test_assert_primary_dataset_fails_loud_on_mismatch(dataset) -> None:
    plugin = _plugin()
    plugin._multicase = SimpleNamespace(primary_case="N_G2Tp12A080_broad")
    with pytest.raises(ValueError, match="primary dataset mismatch"):
        plugin._assert_primary_dataset(dataset)
