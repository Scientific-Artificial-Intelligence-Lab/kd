
from __future__ import annotations

import itertools

import pytest
import torch

from kd.search.discover.config import DiscoverConfig
from kd.search.discover.plugin import (
    BASELINE_STATE_KEY,
    DISCOVERPlugin,
)
from tests.unit.search import _resume_conformance_helpers as helpers

pytestmark = pytest.mark.unit

_TRANSIENT_CYCLES = 5


def _donor_payload_with_evolved_ewma(config: DiscoverConfig) -> dict:
    donor = DISCOVERPlugin(config)
    donor.prepare(helpers.discover_components(helpers.DiscoverCountingEvaluator()))
    for _ in range(3):
        candidates = donor.propose(donor.runner_batch_size)
        donor.update(donor.evaluate(candidates))
    payload = donor.state
    payload["engine_state"]["best_expression"] = helpers.DISCOVER_CHAMPION_IR
    payload["engine_state"]["best_reward"] = 0.85
    return payload


def test_ewma_resume_reprices_champion_with_live_alpha() -> None:
    config_a = DiscoverConfig(reward_alpha=0.01, baseline="ewma_R")
    config_b = DiscoverConfig(reward_alpha=0.05, baseline="ewma_R")
    spec = helpers.DiscoverStubSpec()
    expected_a = helpers.discover_expected_reprice(config_a, spec)
    expected_b = helpers.discover_expected_reprice(config_b, spec)
    assert expected_a != expected_b

    payload = _donor_payload_with_evolved_ewma(config_a)
    subject = helpers.discover_restore_then_prepare(
        config_b, payload, helpers.DiscoverCountingEvaluator(spec)
    )
    assert subject.best_score == expected_b
    assert subject.best_score != expected_a


def test_ewma_resume_transient_decays_geometrically() -> None:
    config_a = DiscoverConfig(reward_alpha=0.01, baseline="ewma_R")
    config_b = DiscoverConfig(reward_alpha=0.05, baseline="ewma_R")
    payload = _donor_payload_with_evolved_ewma(config_a)


    donor_ewma = float(payload["engine_state"][BASELINE_STATE_KEY]["ewma_reward"])
    assert payload["engine_state"][BASELINE_STATE_KEY]["n_updates"] > 0

    subject = helpers.discover_restore_then_prepare(
        config_b, payload, helpers.DiscoverCountingEvaluator()
    )


    restored = subject._engine._baseline_state
    assert restored.ewma_reward == donor_ewma, "resume did not carry the old EWMA"
    assert restored.n_updates > 0, "restored n_updates must dodge the first-update snap"

    strategy = subject._engine._strategy
    assert strategy.baseline == "ewma_R"
    gamma = strategy.gamma



    target = 0.1 if restored.ewma_reward > 0.5 else 0.9
    rewards = torch.full((8,), float(target))

    state = restored
    ewmas = [state.ewma_reward]
    for _ in range(_TRANSIENT_CYCLES):
        _baseline_value, state = strategy._compute_baseline(rewards, 0.0, state)
        ewmas.append(state.ewma_reward)

    gaps = [abs(value - target) for value in ewmas]


    assert gaps[0] > 0.1


    for earlier, later in itertools.pairwise(gaps):
        assert later < earlier



    bound = gamma**_TRANSIENT_CYCLES * gaps[0]
    assert gaps[-1] <= bound * (1 + 1e-5)
    assert gaps[-1] == pytest.approx(bound, rel=1e-5)
