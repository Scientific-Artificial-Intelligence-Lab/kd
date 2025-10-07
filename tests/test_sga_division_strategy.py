"""Unit tests for internal SGA division handling switches."""

import importlib

import pytest


@pytest.fixture(autouse=True)
def reset_strategy_override():
    module = importlib.import_module('kd.model.sga.sgapde.pde')
    module._set_division_strategy_for_tests(None)
    yield
    module._set_division_strategy_for_tests(None)


def test_division_strategy_override_and_env(monkeypatch):
    module = importlib.import_module('kd.model.sga.sgapde.pde')

    # Default should be guard when nothing is set.
    monkeypatch.delenv('KD_SGA_DIVIDE_MODE', raising=False)
    module._set_division_strategy_for_tests(None)
    assert module._resolve_division_strategy() == module._DIVISION_STRATEGY_GUARD

    # Explicit override wins over env.
    module._set_division_strategy_for_tests('legacy')
    assert module._resolve_division_strategy() == module._DIVISION_STRATEGY_LEGACY

    # Clearing override falls back to environment variable.
    module._set_division_strategy_for_tests(None)
    monkeypatch.setenv('KD_SGA_DIVIDE_MODE', 'legacy')
    assert module._resolve_division_strategy() == module._DIVISION_STRATEGY_LEGACY

    # Unknown env value defaults to guard.
    monkeypatch.setenv('KD_SGA_DIVIDE_MODE', 'unknown-mode')
    assert module._resolve_division_strategy() == module._DIVISION_STRATEGY_GUARD

