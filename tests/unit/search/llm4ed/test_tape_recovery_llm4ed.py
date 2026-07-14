
from __future__ import annotations

from pathlib import Path

import pytest

from kd.api import Model
from kd.llm import BudgetedProvider, TapeReplayProvider

from ._plugin_helpers import heat_dataset, make_config

pytestmark = pytest.mark.unit

_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "llm4ed"
    / "heat_uxx_recovery.jsonl"
)


def test_fixture_exists() -> None:
    assert _FIXTURE.is_file(), f"missing llm4ed replay tape: {_FIXTURE}"


def test_tape_replay_recovers_u_xx() -> None:
    model = Model(
        algorithm="llm4ed",
        generations=2,
        config=make_config(),
        provider=TapeReplayProvider(path=_FIXTURE),
    )
    model.fit(heat_dataset())

    assert model.best_expr_ == "u_xx"
    assert model.best_score_ == pytest.approx(0.989, abs=5e-3)
    assert model.result_.iterations == 2


def test_tape_replay_is_deterministic_across_runs() -> None:
    results = []
    for _ in range(2):
        model = Model(
            algorithm="llm4ed",
            generations=2,
            config=make_config(),
            provider=TapeReplayProvider(path=_FIXTURE),
        )
        model.fit(heat_dataset())
        results.append((model.best_expr_, model.best_score_))
    assert results[0] == results[1]


def test_budget_wrapped_replay_recovers_u_xx() -> None:
    model = Model(
        algorithm="llm4ed",
        generations=2,
        config=make_config(),
        provider=BudgetedProvider(
            TapeReplayProvider(path=_FIXTURE), max_calls=200
        ),
    )
    model.fit(heat_dataset())

    assert model.best_expr_ == "u_xx"
