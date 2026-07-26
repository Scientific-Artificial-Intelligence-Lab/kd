
from __future__ import annotations

import inspect
import math

import pytest

from kd.core.evaluator import EvaluationResult
from kd.core.platform.requirements import DerivativeReqs
from kd.search.llm4ed import plugin as plugin_mod
from kd.search.llm4ed import viz as llm4ed_viz
from kd.search.llm4ed.config import Llm4edConfig
from kd.search.llm4ed.plugin import STATE_KEYS, Llm4edPlugin
from kd.search.result import invalid_evaluation_result
from tests.unit.search.llm4ed._plugin_helpers import (
    ALL_INVALID,
    GOOD,
    MIX,
    SPREAD,
    SPREAD3,
    FakeProvider,
    make_config,
    prepared,
    reward_of,
    run,
    two_mode_dataset,
)


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
    "runner_batch_size",
    "derivative_requirements",
    "is_done",
    "score_kind",
    "score_direction",
    "config_cls",
    "one_shot",
)







def test_score_contract_is_llm4ed_sparse_reward_max() -> None:

    assert Llm4edPlugin.score_kind == "LLM4ED sparse reward"
    assert Llm4edPlugin.score_direction == "max"


def test_facade_wiring_classvars() -> None:
    assert Llm4edPlugin.config_cls is Llm4edConfig
    assert Llm4edPlugin.one_shot is False


def test_runner_batch_size_equals_samples_per_epoch() -> None:



    plugin = Llm4edPlugin(make_config(samples_per_epoch=13))
    assert plugin.runner_batch_size == 13


def test_plugin_exposes_full_search_algorithm_surface() -> None:


    plugin = Llm4edPlugin(make_config())
    for member in _PROTOCOL_MEMBERS:
        inspect.getattr_static(plugin, member)


def test_derivative_requirements_declares_fd_order_three() -> None:
    reqs = Llm4edPlugin(make_config()).derivative_requirements
    assert reqs == DerivativeReqs(
        provider_kind="finite_diff",
        max_atomic_order=3,
        lhs_order=1,
        needs_surrogate=False,
    )


def test_config_property_is_json_safe_and_algorithm_tagged() -> None:
    config = make_config(seed=7)
    plugin = Llm4edPlugin(config)
    published = plugin.config
    assert published["algorithm"] == "llm4ed"
    assert published["seed"] == 7

    for value in published.values():
        assert isinstance(value, (str, int, float, type(None)))


def test_state_keys_enumerate_the_resume_critical_components() -> None:
    assert {
        "population",
        "elite_pool",
        "phase_counter",
        "rng_state",
        "llm_seed_counter",
        "call_counts",
        "invalid_counts",
        "best",
    } == STATE_KEYS


def test_logged_metrics_are_exactly_the_six_whitelisted_fields() -> None:

    assert plugin_mod._LOGGED_METRICS == (
        "pool_best",
        "pool_median",
        "pool_worst",
        "n_invalid",
        "n_llm_calls",
        "n_valid",
    )







def test_propose_returns_bounded_batch_of_valid_candidates() -> None:
    plugin, _ = prepared(provider=FakeProvider(MIX))
    candidates = plugin.propose(4)
    assert 0 < len(candidates) <= 4

    assert "sin(u)" not in candidates
    assert "u_xxxx" not in candidates


def test_post_init_propose_returns_top_n_and_stops_early() -> None:





    dataset = two_mode_dataset()
    provider = FakeProvider(SPREAD)
    config = make_config(
        init_num=4, samples_per_epoch=3, max_llm_calls_per_propose=12
    )
    plugin, _ = prepared(config=config, provider=provider, dataset=dataset)
    run(plugin, 1)

    before = len(provider.requests)
    batch = plugin.propose(3)
    n_calls = len(provider.requests) - before


    survivors = ["u + u_xx", "u", "u_xx", "x"]
    ranked = sorted(survivors, key=lambda e: reward_of(e, dataset) or 0.0, reverse=True)
    assert len(batch) == 3
    assert set(batch) == set(ranked[:3])
    assert ranked[3] not in batch
    assert len(batch) == len(set(batch))
    assert 0 < n_calls < 12


def test_resampling_does_not_emit_duplicate_candidates() -> None:





    plugin, _ = prepared(
        config=make_config(samples_per_epoch=3, max_llm_calls_per_propose=6),
        provider=FakeProvider(MIX),
    )
    batch = plugin.propose(3)
    assert len(batch) == len(set(batch))
    assert 0 < len(batch) <= 2


def test_init_prompt_requests_init_num_but_returns_n() -> None:






    dataset = two_mode_dataset()
    provider = FakeProvider(SPREAD)
    config = make_config(init_num=6, samples_per_epoch=3, max_llm_calls_per_propose=12)
    plugin, _ = prepared(config=config, provider=provider, dataset=dataset)
    batch = plugin.propose(3)

    assert "generate 6 diverse equations" in provider.prompts[0]

    survivors = ["u + u_xx", "u", "u_xx", "x"]
    ranked = sorted(survivors, key=lambda e: reward_of(e, dataset) or 0.0, reverse=True)
    assert len(batch) == 3
    assert set(batch) == set(ranked[:3])
    assert ranked[3] not in batch
    assert len(batch) == len(set(batch))


def test_evaluate_is_one_to_one_and_order_preserving() -> None:

    plugin, _ = prepared(provider=FakeProvider(MIX))
    candidates = plugin.propose(4)
    results = plugin.evaluate(candidates)
    assert len(results) == len(candidates)
    for candidate, result in zip(candidates, results, strict=True):
        assert result.expression == candidate

        assert result.is_valid
        assert result.score is not None and result.score > 0.5


def test_update_pools_valid_candidates_and_advances_best() -> None:

    plugin, _ = prepared(provider=FakeProvider(GOOD))
    run(plugin, 1)
    assert plugin.best_score > 0.5
    assert plugin.best_expression != ""


def test_update_logs_exactly_the_whitelisted_metrics_with_correct_values() -> None:





    dataset = two_mode_dataset()
    provider = FakeProvider(SPREAD3)
    config = make_config(init_num=3, samples_per_epoch=3, max_llm_calls_per_propose=8)
    plugin, components = prepared(config=config, provider=provider, dataset=dataset)
    recorder = components.recorder
    assert recorder is not None

    run(plugin, 1)

    rewards = sorted(
        reward_of(e, dataset) or 0.0 for e in ("u + u_xx", "u", "u_xx")
    )
    assert recorder.keys() == set(plugin_mod._LOGGED_METRICS)
    assert recorder.get("pool_best")[-1] == pytest.approx(rewards[2])
    assert recorder.get("pool_median")[-1] == pytest.approx(rewards[1])
    assert recorder.get("pool_worst")[-1] == pytest.approx(rewards[0])
    assert rewards[0] != rewards[1] != rewards[2]
    assert recorder.get("n_valid")[-1] == 3
    assert recorder.get("n_invalid")[-1] >= 1
    assert recorder.get("n_llm_calls")[-1] == len(provider.requests)


def test_empty_pool_round_records_gaps_not_measured_zeros() -> None:
    rounds = 2
    plugin, components = prepared(provider=FakeProvider(ALL_INVALID))
    recorder = components.recorder
    assert recorder is not None

    run(plugin, rounds)

    for metric in ("pool_best", "pool_median", "pool_worst"):
        series = recorder.get(metric)
        assert len(series) == rounds, metric
        assert all(math.isnan(value) for value in series), metric
        assert 0.0 not in series, metric
        assert recorder.to_dict()[metric] == [None] * rounds, metric
    spread = llm4ed_viz.get_data("pool_reward_spread", recorder)
    assert spread["y"]["pool_best"] == [None] * rounds







def test_all_invalid_batch_is_a_legal_noop() -> None:

    plugin, _ = prepared(provider=FakeProvider(ALL_INVALID))
    candidates = plugin.propose(4)
    assert candidates == []
    assert plugin.evaluate([]) == []
    plugin.update(plugin.evaluate([]))
    assert plugin.best_score == 0.0


def test_update_skips_invalid_results() -> None:

    plugin, _ = prepared(provider=FakeProvider(GOOD))
    invalid = invalid_evaluation_result("boom", score=None, expression="u_xx")
    plugin.update([invalid])
    assert plugin.best_score == 0.0


def test_best_score_sentinel_is_zero_not_neg_inf() -> None:

    plugin, _ = prepared(provider=FakeProvider(GOOD))
    assert plugin.best_score == 0.0
    assert math.isfinite(plugin.best_score)


def test_build_final_result_with_no_candidates_is_invalid_with_sentinel() -> None:


    plugin, _ = prepared(provider=FakeProvider(ALL_INVALID))
    run(plugin, 1)
    result = plugin.build_final_result()
    assert isinstance(result, EvaluationResult)
    assert result.is_valid is False
    assert result.score == 0.0
    assert result.invalid_reason == "no_candidate"


def test_known_abnormal_coefficient_rejection_is_structural() -> None:
    from kd.search.llm4ed.plugin import _invalid_reason_for_score
    from kd.search.llm4ed.score import ERROR_ABNORMAL_COEF

    assert _invalid_reason_for_score(ERROR_ABNORMAL_COEF) == "structural_reject"







def test_build_final_result_and_target_share_domain() -> None:
    plugin, _ = prepared(provider=FakeProvider(GOOD))
    run(plugin, 1)
    final = plugin.build_final_result()
    target = plugin.build_result_target()
    assert isinstance(final, EvaluationResult)
    assert final.residuals is not None
    assert final.residuals.shape == target.shape







def test_is_done_false_when_best_below_threshold() -> None:

    plugin, _ = prepared(
        config=make_config(stop_threshold=0.995), provider=FakeProvider(GOOD)
    )
    run(plugin, 1)
    assert 0.0 < plugin.best_score < 0.995
    assert plugin.is_done is False


def test_is_done_true_when_best_meets_threshold() -> None:
    plugin, _ = prepared(
        config=make_config(stop_threshold=0.9), provider=FakeProvider(GOOD)
    )
    run(plugin, 1)
    assert plugin.best_score >= 0.9
    assert plugin.is_done is True


def test_is_done_matches_threshold_comparison_exactly() -> None:


    threshold = 0.7
    plugin, _ = prepared(
        config=make_config(stop_threshold=threshold), provider=FakeProvider(GOOD)
    )
    run(plugin, 2)
    assert plugin.is_done == (plugin.best_score >= threshold)


def test_is_done_false_with_empty_pool() -> None:

    plugin, _ = prepared(provider=FakeProvider(ALL_INVALID))
    run(plugin, 1)
    assert plugin.best_score == 0.0
    assert plugin.is_done is False


def test_is_done_is_inclusive_at_the_threshold() -> None:




    warmup, _ = prepared(
        config=make_config(stop_threshold=0.995), provider=FakeProvider(GOOD)
    )
    run(warmup, 1)
    achieved = warmup.best_score
    assert achieved > 0.0

    plugin, _ = prepared(
        config=make_config(stop_threshold=achieved), provider=FakeProvider(GOOD)
    )
    run(plugin, 1)
    assert plugin.best_score == achieved
    assert plugin.is_done is True







def test_empty_state_payload_clears_pending_restore() -> None:


    plugin = Llm4edPlugin(make_config())
    plugin.state = {}
    assert plugin._restore_pending is False
    assert plugin._pending_state is None


def test_nonempty_state_payload_stages_a_restore() -> None:
    plugin = Llm4edPlugin(make_config())
    plugin.state = {"best": {"best_reward": 0.6, "best_expression": "u_xx"}}
    assert plugin._restore_pending is True
    assert plugin._pending_state == {
        "best": {"best_reward": 0.6, "best_expression": "u_xx"}
    }
