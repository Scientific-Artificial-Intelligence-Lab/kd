
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kd.search.callbacks import CheckpointCallback
from kd.search.checkpoint_manifest import (
    FINAL_STATUS_COMPLETED,
    FINAL_STATUS_CRASHED,
    KIND_FINAL,
    load_checkpoint_manifest,
)
from kd.search.protocol import PlatformComponents
from kd.search.runner import ExperimentRunner

from ._runner_mocks import (
    ExplodingAlgorithm,
    RecordingCallback,
    StatefulAlgorithm,
)


def _final_status(directory: Path) -> str | None:
    finals = [e for e in load_checkpoint_manifest(directory) if e.kind == KIND_FINAL]
    assert len(finals) == 1, f"expected exactly one final entry, got {len(finals)}"
    return finals[0].final_status


class _RaisingEndCallback:

    def __init__(self) -> None:
        self.end_called = False

    @property
    def should_stop(self) -> bool:
        return False

    def on_experiment_start(self, algorithm: Any) -> None:
        pass

    def on_iteration_start(self, iteration: int, algorithm: Any) -> None:
        pass

    def on_iteration_end(
        self, iteration: int, algorithm: Any, candidates: list[str], results: list[Any]
    ) -> None:
        pass

    def on_experiment_end(self, algorithm: Any) -> None:
        self.end_called = True
        raise RuntimeError("callback finalize boom")


class TestFinalizeStatus:

    @pytest.mark.unit
    def test_row1_normal_completion_is_completed(
        self, mock_components: PlatformComponents, tmp_path: Path
    ) -> None:
        cb = CheckpointCallback(directory=tmp_path, every_n=1)
        runner = ExperimentRunner(
            algorithm=StatefulAlgorithm(), max_iterations=2, callbacks=[cb]
        )
        runner.run(mock_components)
        assert _final_status(tmp_path) == FINAL_STATUS_COMPLETED

    @pytest.mark.unit
    def test_row2_early_stop_is_completed(
        self, mock_components: PlatformComponents, tmp_path: Path
    ) -> None:
        cb = CheckpointCallback(directory=tmp_path, every_n=1)
        stopper = RecordingCallback(stop_at_iteration=0)
        runner = ExperimentRunner(
            algorithm=StatefulAlgorithm(),
            max_iterations=5,
            callbacks=[cb, stopper],
        )
        runner.run(mock_components)
        assert _final_status(tmp_path) == FINAL_STATUS_COMPLETED

    @pytest.mark.unit
    def test_row3_mid_iteration_crash_is_crashed(
        self, mock_components: PlatformComponents, tmp_path: Path
    ) -> None:
        cb = CheckpointCallback(directory=tmp_path, every_n=1)
        runner = ExperimentRunner(
            algorithm=ExplodingAlgorithm(explode_at=1),
            max_iterations=3,
            callbacks=[cb],
        )
        with pytest.raises(RuntimeError, match="exploded"):
            runner.run(mock_components)
        assert _final_status(tmp_path) == FINAL_STATUS_CRASHED

    @pytest.mark.unit
    def test_row4_start_raise_propagates_no_final_entry(
        self, mock_components: PlatformComponents, tmp_path: Path
    ) -> None:
        from kd.search.checkpoint_manifest import CheckpointManifestError

        (tmp_path / "preexisting.txt").write_text("occupied")
        cb = CheckpointCallback(directory=tmp_path, every_n=1)
        runner = ExperimentRunner(
            algorithm=StatefulAlgorithm(), max_iterations=2, callbacks=[cb]
        )
        with pytest.raises(CheckpointManifestError, match="is not empty"):
            runner.run(mock_components)
        assert not (tmp_path / "checkpoint_final.pt").exists()

    @pytest.mark.unit
    def test_row5_thirdparty_callback_gets_plain_end_on_both_paths(
        self, mock_components: PlatformComponents, tmp_path: Path
    ) -> None:

        recorder_ok = RecordingCallback()
        cb_ok = CheckpointCallback(directory=tmp_path / "ok", every_n=1)
        ExperimentRunner(
            algorithm=StatefulAlgorithm(),
            max_iterations=1,
            callbacks=[cb_ok, recorder_ok],
        ).run(mock_components)
        assert "experiment_end" in recorder_ok.events


        recorder_crash = RecordingCallback()
        cb_crash = CheckpointCallback(directory=tmp_path / "crash", every_n=1)
        with pytest.raises(RuntimeError, match="exploded"):
            ExperimentRunner(
                algorithm=ExplodingAlgorithm(explode_at=0),
                max_iterations=2,
                callbacks=[cb_crash, recorder_crash],
            ).run(mock_components)
        assert "experiment_end" in recorder_crash.events

    @pytest.mark.unit
    def test_row6_other_callback_crash_does_not_flip_status(
        self, mock_components: PlatformComponents, tmp_path: Path
    ) -> None:
        raiser = _RaisingEndCallback()
        cb = CheckpointCallback(directory=tmp_path, every_n=1)


        runner = ExperimentRunner(
            algorithm=StatefulAlgorithm(),
            max_iterations=1,
            callbacks=[raiser, cb],
        )
        runner.run(mock_components)
        assert raiser.end_called
        assert _final_status(tmp_path) == FINAL_STATUS_COMPLETED

    @pytest.mark.unit
    def test_finalize_failure_is_visible_on_the_result(
        self, mock_components: PlatformComponents
    ) -> None:
        raiser = _RaisingEndCallback()
        later = RecordingCallback()
        runner = ExperimentRunner(
            algorithm=StatefulAlgorithm(),
            max_iterations=1,
            callbacks=[raiser, later],
        )

        result = runner.run(mock_components)

        assert "experiment_end" in later.events
        assert len(result.finalize_failures) == 1
        failure = result.finalize_failures[0]
        assert "_RaisingEndCallback" in failure
        assert "on_experiment_end" in failure
        assert "finalize boom" in failure

    @pytest.mark.unit
    def test_clean_run_reports_no_finalize_failures(
        self, mock_components: PlatformComponents
    ) -> None:
        runner = ExperimentRunner(
            algorithm=StatefulAlgorithm(),
            max_iterations=1,
            callbacks=[RecordingCallback()],
        )

        assert runner.run(mock_components).finalize_failures == ()

    @pytest.mark.unit
    def test_row7_retry_inside_active_except_is_completed(
        self, mock_components: PlatformComponents, tmp_path: Path
    ) -> None:
        cb = CheckpointCallback(directory=tmp_path, every_n=1)
        runner = ExperimentRunner(
            algorithm=StatefulAlgorithm(), max_iterations=1, callbacks=[cb]
        )
        try:
            raise ValueError("ambient exception being handled")
        except ValueError:
            runner.run(mock_components)
        assert _final_status(tmp_path) == FINAL_STATUS_COMPLETED
