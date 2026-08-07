
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from kd.search.callbacks import CHECKPOINT_VERSION
from kd.search.lifecycle import LifecycleState
from kd.search.protocol import PlatformComponents
from kd.search.runner import ExperimentRunner
from tests.unit.search._runner_mocks import RecordingAlgorithm


class _FailFirstPrepareAlgorithm(RecordingAlgorithm):

    def __init__(self) -> None:
        super().__init__()
        self._prepare_calls = 0

    def prepare(self, components: PlatformComponents) -> None:
        self._prepare_calls += 1
        if self._prepare_calls == 1:
            raise RuntimeError("prepare failed (first attempt)")
        super().prepare(components)


def _mock_components() -> PlatformComponents:
    return PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=MagicMock(training_result=None),
        registry=MagicMock(),
    )


def _write_nonempty_checkpoint(path: Path) -> None:
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "iteration": 0,
            "algorithm_state": {"best_expression": "u_x"},
        },
        path,
    )


@pytest.mark.unit
def test_restore_arm_survives_failed_prepare(tmp_path: Path) -> None:
    algo = _FailFirstPrepareAlgorithm()
    runner = ExperimentRunner(algorithm=algo, max_iterations=0)
    ckpt = tmp_path / "restore.pt"
    _write_nonempty_checkpoint(ckpt)
    runner.load_checkpoint(ckpt)

    with pytest.raises(RuntimeError, match="prepare failed"):
        runner.run(_mock_components())


    runner.run(_mock_components())

    assert runner.lifecycle is not None
    assert runner.lifecycle.restored is True, (
        "restore arm was cleared by the failed prepare; retry mis-classified FRESH"
    )
    assert runner.lifecycle.state is LifecycleState.DONE


@pytest.mark.unit
def test_fresh_run_after_failed_prepare_is_not_restore(tmp_path: Path) -> None:
    algo = _FailFirstPrepareAlgorithm()
    runner = ExperimentRunner(algorithm=algo, max_iterations=0)

    with pytest.raises(RuntimeError, match="prepare failed"):
        runner.run(_mock_components())

    runner.run(_mock_components())

    assert runner.lifecycle is not None
    assert runner.lifecycle.restored is False
    assert runner.lifecycle.state is LifecycleState.DONE
