
from __future__ import annotations

import errno
import sys

import pytest

from kd.api import _ProgressPrinter
from kd.search.protocol import PlatformComponents
from kd.search.runner import ExperimentRunner
from tests.unit.search._runner_mocks import ExplodingAlgorithm, StatefulAlgorithm


class _BrokenStdout:

    def __init__(self) -> None:
        self.write_attempts = 0

    def write(self, _text: str) -> int:
        self.write_attempts += 1
        raise BrokenPipeError(errno.EPIPE, "Broken pipe")

    def flush(self) -> None:
        pass


class _UnprintableAlgorithm(StatefulAlgorithm):

    def __init__(self) -> None:
        super().__init__()
        self.score_kind_reads = 0

    @property
    def score_kind(self) -> str:
        self.score_kind_reads += 1
        if self.score_kind_reads == 1:
            raise OSError(errno.EIO, "score label unavailable")
        return "mse"


@pytest.mark.unit
def test_dead_stdout_does_not_abort_the_run(
    mock_components: PlatformComponents, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = _BrokenStdout()
    monkeypatch.setattr(sys, "stdout", broken)
    runner = ExperimentRunner(
        algorithm=StatefulAlgorithm(),
        max_iterations=3,
        callbacks=[_ProgressPrinter(total_generations=3)],
    )

    result = runner.run(mock_components)

    assert result.iterations == 3



    assert broken.write_attempts == 1


@pytest.mark.unit
def test_render_failure_still_aborts_the_run(
    mock_components: PlatformComponents,
) -> None:
    runner = ExperimentRunner(
        algorithm=_UnprintableAlgorithm(),
        max_iterations=3,
        callbacks=[_ProgressPrinter(total_generations=3)],
    )

    with pytest.raises(OSError, match="score label unavailable"):
        runner.run(mock_components)


@pytest.mark.unit
def test_completed_run_prints_the_done_line(
    mock_components: PlatformComponents, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = ExperimentRunner(
        algorithm=StatefulAlgorithm(),
        max_iterations=2,
        callbacks=[_ProgressPrinter(total_generations=2)],
    )

    runner.run(mock_components)

    assert "[kd] Done." in capsys.readouterr().out


@pytest.mark.unit
def test_crashed_run_prints_no_done_line(
    mock_components: PlatformComponents, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = ExperimentRunner(
        algorithm=ExplodingAlgorithm(explode_at=1),
        max_iterations=3,
        callbacks=[_ProgressPrinter(total_generations=3)],
    )

    with pytest.raises(RuntimeError, match="exploded"):
        runner.run(mock_components)

    out = capsys.readouterr().out
    assert "[kd] Generation" in out
    assert "Done." not in out
