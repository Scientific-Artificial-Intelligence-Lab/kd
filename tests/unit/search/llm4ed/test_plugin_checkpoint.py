
from __future__ import annotations

import pytest

import kd.search.llm4ed.plugin as plugin_mod
from kd.llm import BudgetedProvider, LLMProvider
from kd.search.llm4ed import FakeLlm4edBackend
from kd.search.llm4ed.config import Llm4edConfig
from kd.search.llm4ed.plugin import Llm4edPlugin
from tests.unit.search.llm4ed._checkpoint_helpers import (
    RecordingProvider,
    build_fake_transport,
    canonical_state,
    pickle_roundtrip,
)
from tests.unit.search.llm4ed._plugin_helpers import (
    components_for,
    make_config,
    prepared,
    run,
    two_mode_dataset,
)


def _cfg(**overrides: object) -> Llm4edConfig:
    base: dict[str, object] = {
        "seed": 0,
        "pool_size": 5,
        "samples_per_epoch": 4,
        "max_llm_calls_per_propose": 6,
    }
    base.update(overrides)
    return make_config(**base)


def _driver() -> RecordingProvider:
    return RecordingProvider(FakeLlm4edBackend())


def _straight(
    rounds: int, config: Llm4edConfig
) -> tuple[Llm4edPlugin, RecordingProvider]:
    driver = _driver()
    plugin, _ = prepared(config=config, provider=driver, dataset=two_mode_dataset())
    run(plugin, rounds, n=4)
    return plugin, driver


def _restore(
    config: Llm4edConfig,
    payload: dict[str, object],
    provider: LLMProvider | None,
) -> Llm4edPlugin:
    plugin = Llm4edPlugin(config, provider=provider)
    plugin.state = payload
    plugin.prepare(components_for(two_mode_dataset()))
    return plugin







def test_checkpoint_resume_equivalence() -> None:



    config = _cfg(stop_threshold=0.9)
    straight5, _ = _straight(5, config)
    final5 = canonical_state(straight5)

    straight2, _ = _straight(2, config)
    saved2 = canonical_state(straight2)



    assert saved2 != final5

    source, _ = _straight(2, config)
    payload = pickle_roundtrip(source.state)
    resumed = _restore(config, payload, _driver())



    assert canonical_state(resumed) == saved2

    run(resumed, 3, n=4)




    assert canonical_state(resumed) == final5
    assert resumed.best_score == straight5.best_score
    assert resumed.best_expression == straight5.best_expression



    assert straight5.is_done is True
    assert resumed.is_done == straight5.is_done








def test_ledger_survives_checkpoint_structurally() -> None:






    config = _cfg(pool_size=2)
    source, _ = _straight(3, config)
    payload = pickle_roundtrip(source.state)

    heap_scores = {item["score"] for item in payload["elite_pool"]["items"]}
    ledger = payload["elite_pool"]["scores"]
    evicted = [score for score in ledger if score not in heap_scores]

    assert evicted, "scenario must produce evicted-but-ledgered scores"

    resumed = _restore(config, payload, _driver())

    restored = resumed.state["elite_pool"]
    restored_heap = {item["score"] for item in restored["items"]}



    for score in evicted:
        assert score in restored["scores"]
        assert score not in restored_heap







def test_call_counts_continue_across_resume() -> None:


    config = _cfg()
    straight5, _ = _straight(5, config)
    total = straight5.state["call_counts"]

    source, _ = _straight(2, config)
    payload = pickle_roundtrip(source.state)
    resumed = _restore(config, payload, _driver())
    run(resumed, 3, n=4)

    assert resumed.state["call_counts"] == total


def test_selfbuilt_budget_continues_whole_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:




    monkeypatch.setattr(plugin_mod, "OpenAICompatProvider", build_fake_transport)
    config = _cfg(base_url="https://api.example/v1", max_llm_calls_per_run=99)

    source = Llm4edPlugin(config, provider=None)
    source.prepare(components_for(two_mode_dataset()))
    run(source, 2, n=4)
    checkpointed_calls = source.state["call_counts"]
    payload = pickle_roundtrip(source.state)

    resumed = _restore(config, payload, None)



    assert isinstance(resumed._provider, BudgetedProvider)
    assert resumed._provider.calls_made == checkpointed_calls
    assert resumed._provider._max_calls == config.max_llm_calls_per_run


def test_injected_provider_budget_is_injector_owned() -> None:



    config = _cfg()
    source, _ = _straight(2, config)
    payload = pickle_roundtrip(source.state)

    fresh_budget = BudgetedProvider(FakeLlm4edBackend(), max_calls=99)
    resumed = _restore(config, payload, fresh_budget)

    assert resumed._provider is fresh_budget
    assert fresh_budget.calls_made == 0


def test_fresh_second_prepare_resets_selfbuilt_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:







    monkeypatch.setattr(plugin_mod, "OpenAICompatProvider", build_fake_transport)
    config = _cfg(base_url="https://api.example/v1", max_llm_calls_per_run=99)

    plugin = Llm4edPlugin(config, provider=None)
    plugin.prepare(components_for(two_mode_dataset()))
    first_provider = plugin._provider
    run(plugin, 2, n=4)
    spent = plugin._provider.calls_made
    assert spent > 0, "run spent no budget -> the reset assertion would be vacuous"


    plugin.prepare(components_for(two_mode_dataset()))

    assert isinstance(plugin._provider, BudgetedProvider)
    assert plugin._provider is not first_provider
    assert plugin._provider.calls_made == 0
    assert plugin._provider._max_calls == config.max_llm_calls_per_run








def test_seed_continues_bit_identical_across_resume() -> None:
    config = _cfg()
    straight5, driver_straight = _straight(5, config)

    source, driver_source = _straight(2, config)
    split = len(driver_source.requests)
    payload = pickle_roundtrip(source.state)

    driver_resumed = _driver()
    resumed = _restore(config, payload, driver_resumed)
    run(resumed, 3, n=4)



    assert driver_source.seeds + driver_resumed.seeds == driver_straight.seeds
    assert driver_resumed.seeds == driver_straight.seeds[split:]











def test_organize_pool_snapshot_survives_checkpoint() -> None:
    config = _cfg()
    _straight5, driver_straight = _straight(5, config)

    source, driver_source = _straight(2, config)
    split = len(driver_source.requests)

    snapshot = [(item.score, item.expression) for item in source._organize_pool]
    live = [(item.score, item.expression) for item in source._pool.get_top_samples()]



    assert snapshot and snapshot != live

    payload = pickle_roundtrip(source.state)
    driver_resumed = _driver()
    resumed = _restore(config, payload, driver_resumed)



    restored_snapshot = [
        (item.score, item.expression) for item in resumed._organize_pool
    ]
    assert restored_snapshot == snapshot




    run(resumed, 3, n=4)
    assert driver_resumed.prompts[0] == driver_straight.prompts[split]
