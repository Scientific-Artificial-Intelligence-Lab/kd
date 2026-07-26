
from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from kd.core import safety_counters
from kd.core.evaluator import EvaluationResult
from kd.core.safety import safe_div
from kd.search.protocol import PlatformComponents
from kd.search.runner import ExperimentRunner
from tests.unit.search._runner_mocks import RecordingAlgorithm

_SUMMARY_TOKEN = "[SAFETY-COUNTERS]"


@pytest.fixture
def counters_on() -> Iterator[None]:
    was_enabled = safety_counters.counters_enabled()
    safety_counters.enable_counters()
    safety_counters.reset()
    yield
    safety_counters.reset()
    if not was_enabled:
        safety_counters.disable_counters()


def _summary_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if _SUMMARY_TOKEN in r.getMessage()]


def _emitted_div_calls(message: str) -> int:
    match = re.search(r"div_calls=(\d+)", message)
    assert match is not None, f"no div_calls token in: {message!r}"
    return int(match.group(1))


@pytest.mark.unit
def test_run_emits_safety_summary_when_enabled(
    counters_on: None,
    mock_components: PlatformComponents,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = ExperimentRunner(algorithm=RecordingAlgorithm(), max_iterations=1)
    with caplog.at_level(logging.INFO, logger="kd.search.runner"):
        runner.run(mock_components)

    records = _summary_records(caplog)
    assert len(records) == 1
    message = records[0].getMessage()

    assert _SUMMARY_TOKEN in message
    assert "div_calls=" in message
    assert "log_guard_fired=" in message


    assert safety_counters.snapshot() == safety_counters.SafetyCounterSnapshot(
        div_calls=0,
        div_guard_fired=0,
        exp_calls=0,
        exp_guard_fired=0,
        log_calls=0,
        log_guard_fired=0,
        div_fired_elems=0,
        div_total_elems=0,
        exp_fired_elems=0,
        exp_total_elems=0,
        log_fired_elems=0,
        log_total_elems=0,
    )


@pytest.mark.unit
def test_run_entry_reset_excludes_prerun_calls(
    counters_on: None,
    mock_components: PlatformComponents,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = ExperimentRunner(algorithm=RecordingAlgorithm(), max_iterations=1)

    with caplog.at_level(logging.INFO, logger="kd.search.runner"):
        runner.run(mock_components)
        first = _emitted_div_calls(_summary_records(caplog)[-1].getMessage())


        for _ in range(5):
            safe_div(torch.tensor([1.0]), torch.tensor([0.0]))

        caplog.clear()
        runner.run(mock_components)
        second = _emitted_div_calls(_summary_records(caplog)[-1].getMessage())


    assert first == second


class _FinalResultDividingAlgorithm(RecordingAlgorithm):

    def build_final_result(self) -> EvaluationResult:
        safe_div(torch.tensor([1.0]), torch.tensor([2.0]))
        return super().build_final_result()


@pytest.mark.unit
def test_summary_includes_final_result_evaluation(
    counters_on: None,
    mock_components: PlatformComponents,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = ExperimentRunner(
        algorithm=_FinalResultDividingAlgorithm(), max_iterations=1
    )
    with caplog.at_level(logging.INFO, logger="kd.search.runner"):
        runner.run(mock_components)

    records = _summary_records(caplog)
    assert len(records) == 1



    assert _emitted_div_calls(records[0].getMessage()) >= 1


@pytest.mark.unit
def test_summary_line_carries_runner_label(
    counters_on: None,
    mock_components: PlatformComponents,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = ExperimentRunner(algorithm=RecordingAlgorithm(), max_iterations=1)
    with caplog.at_level(logging.INFO, logger="kd.search.runner"):
        runner.run(mock_components)

    message = _summary_records(caplog)[0].getMessage()
    assert message.endswith("label=runner:RecordingAlgorithm")


@pytest.mark.unit
def test_run_appends_jsonl_sink_when_file_env_set(
    counters_on: None,
    mock_components: PlatformComponents,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sink = tmp_path / "safety-counters.jsonl"
    monkeypatch.setenv(safety_counters._ENV_FILE, str(sink))
    runner = ExperimentRunner(algorithm=RecordingAlgorithm(), max_iterations=1)

    runner.run(mock_components)
    runner.run(mock_components)

    lines = sink.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert record["label"] == "runner:RecordingAlgorithm"
        assert "div_calls" in record and "log_guard_fired" in record


@pytest.mark.unit
def test_run_emits_no_summary_when_disabled(
    mock_components: PlatformComponents,
    caplog: pytest.LogCaptureFixture,
) -> None:
    was_enabled = safety_counters.counters_enabled()
    safety_counters.disable_counters()
    try:
        runner = ExperimentRunner(algorithm=RecordingAlgorithm(), max_iterations=1)
        with caplog.at_level(logging.INFO, logger="kd.search.runner"):
            runner.run(mock_components)
        assert _summary_records(caplog) == []
    finally:
        if was_enabled:
            safety_counters.enable_counters()
