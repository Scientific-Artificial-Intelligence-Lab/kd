
from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock

import pytest
from kd.search.lifecycle import LifecycleState, SearchLifecycle

from kd.api import _PLUGIN_CLASS_BY_ALGORITHM
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.plugin import EqGPTPlugin
from kd.search.llm4ed.config import Llm4edConfig
from kd.search.protocol import PlatformComponents

_SENTINEL_EXPRESSION = "RESTORED_BEST_FROM_CHECKPOINT"

_REGISTERED_PLUGIN_CASES = tuple(
    pytest.param(algorithm, plugin_cls, id=algorithm)
    for algorithm, plugin_cls in _PLUGIN_CLASS_BY_ALGORITHM.items()
)


def _make_plugin(plugin_cls: type) -> Any:
    if plugin_cls.config_cls is EqGPTConfig:
        return plugin_cls(EqGPTConfig(sparsity_alpha=0.02))
    if plugin_cls.config_cls is Llm4edConfig:
        return plugin_cls(Llm4edConfig())
    return plugin_cls()







@pytest.mark.unit
@pytest.mark.parametrize(("algorithm", "plugin_cls"), _REGISTERED_PLUGIN_CASES)
def test_empty_state_reset_accepted(algorithm: str, plugin_cls: type) -> None:
    plugin = _make_plugin(plugin_cls)
    plugin.state = {}


@pytest.mark.unit
@pytest.mark.parametrize(("algorithm", "plugin_cls"), _REGISTERED_PLUGIN_CASES)
def test_every_plugin_exposes_the_state_setter_the_runner_drives(
    algorithm: str, plugin_cls: type
) -> None:
    descriptor = inspect.getattr_static(plugin_cls, "state")
    assert isinstance(descriptor, property)
    assert descriptor.fset is not None, "state must be settable (restore path)"


@pytest.mark.unit
def test_platform_machine_classifies_restore_vs_fresh() -> None:
    restored = SearchLifecycle()
    restored.restore({"algorithm_state": {"best_expression": "u_x"}})
    assert restored.restored is True

    fresh = SearchLifecycle()
    fresh.restore({})
    assert fresh.restored is False


@pytest.mark.unit
def test_eqgpt_empty_state_reset_drops_stashed_state() -> None:
    plugin = EqGPTPlugin(EqGPTConfig(sparsity_alpha=0.02))
    stashed = {"top_k": {"rewards": [9.9], "sentences": [[1, 2]]}, "stale": "best"}

    plugin.state = stashed
    assert plugin.state == stashed, "non-empty payload must stash as pending state"

    plugin.state = {}
    with pytest.raises(RuntimeError):
        _ = plugin.state


@pytest.mark.unit
def test_conformance_sweeps_the_full_live_registry() -> None:
    swept = {case.id for case in _REGISTERED_PLUGIN_CASES}
    assert swept == set(_PLUGIN_CLASS_BY_ALGORITHM)







def _pure_mock_components() -> PlatformComponents:
    return PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=MagicMock(),
        registry=MagicMock(),
        recorder=None,
    )


def _pysr_saved_payload() -> dict[str, Any]:
    from kd.search.pysr import PySRPlugin

    donor = PySRPlugin()
    donor.prepare(_pure_mock_components())
    payload = donor.state
    payload["best_expression"] = _SENTINEL_EXPRESSION
    payload["best_score"] = 0.125
    payload["fitted"] = True
    donor.state = payload
    return donor.state


@pytest.mark.unit
def test_restore_transition_preserves_best_across_prepare() -> None:
    from kd.search.pysr import PySRPlugin

    saved = _pysr_saved_payload()
    lc = SearchLifecycle()
    lc.restore(saved)
    assert lc.restored is True

    subject = PySRPlugin()
    subject.state = saved
    lc.prepare()
    subject.prepare(_pure_mock_components())

    assert lc.state is LifecycleState.PREPARED
    assert subject.best_expression == _SENTINEL_EXPRESSION


@pytest.mark.unit
def test_fresh_transition_resets_best_across_prepare() -> None:
    from kd.search.pysr import PySRPlugin

    lc = SearchLifecycle()
    lc.restore({})
    assert lc.restored is False

    subject = PySRPlugin()
    subject.prepare(_pure_mock_components())
    lc.prepare()

    assert lc.state is LifecycleState.PREPARED
    assert subject.best_expression == ""
