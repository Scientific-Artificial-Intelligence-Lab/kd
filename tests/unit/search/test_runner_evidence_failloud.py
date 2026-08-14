
from __future__ import annotations

from pathlib import Path

import pytest

from kd.search.callbacks import CheckpointCallback
from kd.search.checkpoint_manifest import (
    CheckpointManifestEntry,
    CheckpointManifestError,
    CheckpointManifestWriter,
)
from kd.search.iteration_events import IterationEvent, IterationEventEmitter
from kd.search.protocol import PlatformComponents
from kd.search.runner import ExperimentRunner

from ._runner_mocks import StatefulAlgorithm


@pytest.mark.unit
def test_checkpoint_ledger_failure_aborts_the_run(
    mock_components: PlatformComponents,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def boom(self: CheckpointManifestWriter, entry: CheckpointManifestEntry) -> None:
        raise CheckpointManifestError("ledger append failed")

    monkeypatch.setattr(CheckpointManifestWriter, "append", boom)
    runner = ExperimentRunner(
        algorithm=StatefulAlgorithm(),
        max_iterations=3,
        callbacks=[CheckpointCallback(directory=tmp_path, every_n=1)],
    )

    with pytest.raises(CheckpointManifestError, match="ledger append"):
        runner.run(mock_components)






    assert (tmp_path / "checkpoint_000000.pt").exists()


@pytest.mark.unit
def test_event_consumer_failure_aborts_the_run(
    mock_components: PlatformComponents, tmp_path: Path
) -> None:

    def boom(_event: IterationEvent) -> None:
        raise RuntimeError("controller consumer failed")

    emitter = IterationEventEmitter(on_event=boom, jsonl_path=tmp_path / "events.jsonl")
    runner = ExperimentRunner(
        algorithm=StatefulAlgorithm(), max_iterations=3, callbacks=[emitter]
    )

    with pytest.raises(RuntimeError, match="consumer failed"):
        runner.run(mock_components)
