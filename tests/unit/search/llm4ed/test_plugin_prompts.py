
from __future__ import annotations

from kd.search.llm4ed.prompts import OPTIMIZE, classify_prompt
from tests.unit.search.llm4ed._plugin_helpers import (
    ALL_INVALID,
    DIRTY,
    SPREAD,
    FakeProvider,
    make_config,
    prepared,
    res,
    reward_of,
    run,
    two_mode_dataset,
)




INIT_SIGNATURE = "Now randomly generate"
OPTIMIZE_SIGNATURE = "Below are some previous equations and their scores"
EVOLUTION_SIGNATURE = "please follow the instructions step-by-step"


def test_first_round_uses_the_initialization_template() -> None:
    fake = FakeProvider(DIRTY)
    plugin, _ = prepared(provider=fake)
    plugin.propose(4)
    assert fake.prompts, "propose issued no LLM calls"
    first = fake.prompts[0]
    assert INIT_SIGNATURE in first

    assert OPTIMIZE_SIGNATURE not in first
    assert EVOLUTION_SIGNATURE not in first


def test_dual_strategy_uses_both_optimize_and_evolution_templates() -> None:


    fake = FakeProvider(DIRTY)
    plugin, _ = prepared(provider=fake)
    run(plugin, 5)
    prompts = fake.prompts
    assert any(INIT_SIGNATURE in p for p in prompts)
    assert any(OPTIMIZE_SIGNATURE in p for p in prompts)
    assert any(EVOLUTION_SIGNATURE in p for p in prompts)


def test_first_non_init_round_is_evolution() -> None:



    provider = FakeProvider(SPREAD)
    plugin, _ = prepared(
        config=make_config(init_num=4, samples_per_epoch=3),
        provider=provider,
        dataset=two_mode_dataset(),
    )
    run(plugin, 2)
    non_init = [p for p in provider.prompts if INIT_SIGNATURE not in p]
    assert non_init, "no post-init prompt was emitted"
    assert EVOLUTION_SIGNATURE in non_init[0]
    assert OPTIMIZE_SIGNATURE not in non_init[0]


def test_optimize_prompt_history_carries_pool_scores() -> None:






    dataset = two_mode_dataset()
    provider = FakeProvider(SPREAD)
    plugin, _ = prepared(
        config=make_config(init_num=4, samples_per_epoch=3),
        provider=provider,
        dataset=dataset,
    )
    best = reward_of("u + u_xx", dataset)
    assert best is not None
    run(plugin, 3)

    optimize_prompts = [p for p in provider.prompts if OPTIMIZE_SIGNATURE in p]
    assert optimize_prompts, "no optimize-phase prompt was emitted"
    op = optimize_prompts[0]
    assert "score:" in op
    assert str(best) in op


def test_phase_advances_after_a_starved_round() -> None:





    def starve_evolution(request: object, _index: int) -> str:
        if EVOLUTION_SIGNATURE in request.prompt:
            return ALL_INVALID
        return SPREAD

    provider = FakeProvider(starve_evolution)
    plugin, _ = prepared(
        config=make_config(
            init_num=4, samples_per_epoch=3, max_llm_calls_per_propose=4
        ),
        provider=provider,
        dataset=two_mode_dataset(),
    )
    run(plugin, 3)
    prompts = provider.prompts
    evo_index = next(
        (i for i, p in enumerate(prompts) if EVOLUTION_SIGNATURE in p), None
    )
    assert evo_index is not None, "no evolution round ran"

    assert any(OPTIMIZE_SIGNATURE in p for p in prompts[evo_index:])


def test_optimize_prompt_history_is_label_stripped() -> None:



    fake = FakeProvider(DIRTY)
    plugin, _ = prepared(provider=fake)
    run(plugin, 5)

    optimize_prompts = [p for p in fake.prompts if OPTIMIZE_SIGNATURE in p]
    assert optimize_prompts, "no optimize-phase prompt was emitted (vacuous test)"
    for prompt in optimize_prompts:
        assert "<select>" not in prompt
        assert "<cross>" not in prompt

        assert classify_prompt(prompt) == OPTIMIZE


def test_organize_shows_prepush_pool_evicted_member_still_displayed() -> None:
















    dataset = two_mode_dataset()
    config = make_config(
        pool_size=2, samples_per_epoch=1, init_num=1, max_llm_calls_per_propose=1
    )
    sequence = [
        res("x"),
        res("u_xx"),
        res("u"),
        res("u + u_xx"),
        ALL_INVALID,
        ALL_INVALID,
        ALL_INVALID,
    ]
    provider = FakeProvider(sequence)
    plugin, _ = prepared(config=config, provider=provider, dataset=dataset)
    run(plugin, 7)

    prompts = provider.prompts
    assert len(prompts) == 7
    evicted_score = reward_of("u_xx", dataset)
    assert evicted_score is not None






    r4 = prompts[4]
    assert OPTIMIZE_SIGNATURE in r4
    assert plugin.best_score == reward_of("u + u_xx", dataset)
    assert str(evicted_score) in r4



    r6 = prompts[6]
    assert OPTIMIZE_SIGNATURE in r6
    assert str(evicted_score) not in r6
