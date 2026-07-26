
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

import kd.search.discover.golden.runner as golden_runner
from kd.core import safety_counters
from kd.core.safety import safe_div

_GOLDEN_LOGGER = "kd.search.discover.golden.runner"


@pytest.fixture
def counters_on() -> Iterator[None]:
    was_enabled = safety_counters.counters_enabled()
    safety_counters.enable_counters()
    safety_counters.reset()
    yield
    safety_counters.reset()
    if not was_enabled:
        safety_counters.disable_counters()


def _stub_burgers(seed: int, *, data_path: Path | None = None) -> tuple[Any, Any]:
    safe_div(torch.tensor([1.0]), torch.tensor([0.0]))
    return ("stub-result", {"seed": seed})


@pytest.mark.unit
def test_run_golden_emits_labelled_summary_and_file_record(
    counters_on: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(golden_runner, "run_burgers_mode1", _stub_burgers)
    sink = tmp_path / "safety-counters.jsonl"
    monkeypatch.setenv(safety_counters._ENV_FILE, str(sink))


    safe_div(torch.tensor([1.0]), torch.tensor([0.0]))

    with caplog.at_level(logging.INFO, logger=_GOLDEN_LOGGER):
        outcome = golden_runner.run_golden("burgers", "mode1", 42)

    assert outcome == ("stub-result", {"seed": 42})

    messages = [
        r.getMessage() for r in caplog.records if "[SAFETY-COUNTERS]" in r.getMessage()
    ]
    assert len(messages) == 1
    assert messages[0].endswith("label=golden:burgers:mode1:seed=42")

    lines = sink.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["label"] == "golden:burgers:mode1:seed=42"


    assert record["div_calls"] == 1
    assert record["div_guard_fired"] == 1


    assert safety_counters.snapshot().div_calls == 0


@pytest.mark.unit
def test_run_golden_disabled_emits_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(golden_runner, "run_burgers_mode1", _stub_burgers)
    sink = tmp_path / "safety-counters.jsonl"
    monkeypatch.setenv(safety_counters._ENV_FILE, str(sink))
    was_enabled = safety_counters.counters_enabled()
    safety_counters.disable_counters()
    try:
        with caplog.at_level(logging.INFO, logger=_GOLDEN_LOGGER):
            golden_runner.run_golden("burgers", "mode1", 7)
        assert all(
            "[SAFETY-COUNTERS]" not in r.getMessage() for r in caplog.records
        )
        assert not sink.exists()
    finally:
        if was_enabled:
            safety_counters.enable_counters()
