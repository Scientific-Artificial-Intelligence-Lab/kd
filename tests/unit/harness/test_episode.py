
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kd.harness.episode import (
    STATUS_COMPLETED,
    STATUS_NO_RECORD,
    STATUS_RAISED,
    EpisodeOutcome,
    run_episode,
)
from kd.harness.recording import RecordingOptions
from kd.search.iteration_events import IterationEventEmitter
from kd.search.result import RunManifest
from kd.search.run_dir import create_run_dir, finalize_run_dir

from ._helpers import make_entry, make_plan, make_record


_DATASET: Any = object()


class _FakeModel:

    def __init__(self, *, run_record: Any, fit_error: BaseException | None = None):
        self._run_record = run_record
        self._fit_error = fit_error
        self.fit_calls: list[Any] = []

    def fit(self, dataset: Any) -> _FakeModel:
        self.fit_calls.append(dataset)
        if self._fit_error is not None:
            raise self._fit_error
        return self

    @property
    def result_(self) -> Any:
        return type(
            "_Result", (), {"run_record": self._run_record, "manifest": None}
        )()


class _SpyFactory:

    def __init__(self, *, run_record: Any = None, fit_error: BaseException | None = None):
        self.calls: list[dict[str, Any]] = []
        self._run_record = run_record
        self._fit_error = fit_error
        self.model: _FakeModel | None = None

    def __call__(self, **kwargs: Any) -> _FakeModel:
        self.calls.append(kwargs)
        self.model = _FakeModel(run_record=self._run_record, fit_error=self._fit_error)
        return self.model


def test_completed_outcome_carries_record() -> None:
    record = make_record(instrument="sga", seed=0)
    factory = _SpyFactory(run_record=record)
    entry = make_entry("sga", seed=0)

    outcome = run_episode(
        entry=entry, entry_index=3, dataset=_DATASET, model_factory=factory
    )

    assert outcome.status == STATUS_COMPLETED
    assert outcome.record is record
    assert outcome.error_type is None
    assert outcome.error_message is None
    assert outcome.entry_index == 3
    assert outcome.entry is entry
    assert outcome.wallclock_seconds >= 0.0


def test_factory_kwargs_forwarding_via_spy() -> None:

    record = make_record(instrument="pysindy", seed=7)
    factory = _SpyFactory(run_record=record)
    entry = make_entry("pysindy", seed=7, threshold=0.1)

    run_episode(entry=entry, entry_index=0, dataset=_DATASET, model_factory=factory)

    assert len(factory.calls) == 1
    call = factory.calls[0]
    assert call["algorithm"] == "pysindy"
    assert call["seed"] == 7
    assert call["verbose"] is False

    assert call["threshold"] == 0.1

    assert factory.model is not None
    assert factory.model.fit_calls == [_DATASET]


def test_factory_raise_becomes_raised_outcome() -> None:
    def factory(**_kwargs: Any) -> Any:
        raise ValueError("bad instrument wiring")

    entry = make_entry("sga", seed=1)
    outcome = run_episode(
        entry=entry, entry_index=2, dataset=_DATASET, model_factory=factory
    )

    assert outcome.status == STATUS_RAISED
    assert outcome.record is None
    assert outcome.error_type == "ValueError"
    assert outcome.error_message is not None
    assert "bad instrument wiring" in outcome.error_message


def test_fit_raise_becomes_raised_outcome() -> None:
    factory = _SpyFactory(fit_error=RuntimeError("solver diverged"))
    entry = make_entry("sga", seed=1)

    outcome = run_episode(
        entry=entry, entry_index=0, dataset=_DATASET, model_factory=factory
    )

    assert outcome.status == STATUS_RAISED
    assert outcome.error_type == "RuntimeError"
    assert outcome.error_message == "solver diverged"


def test_error_message_truncated_to_2000_chars() -> None:
    factory = _SpyFactory(fit_error=RuntimeError("x" * 5000))
    entry = make_entry("sga", seed=0)

    outcome = run_episode(
        entry=entry, entry_index=0, dataset=_DATASET, model_factory=factory
    )

    assert outcome.error_message is not None
    assert len(outcome.error_message) == 2000


def test_none_run_record_becomes_no_record() -> None:
    factory = _SpyFactory(run_record=None)
    entry = make_entry("sga", seed=0)

    outcome = run_episode(
        entry=entry, entry_index=0, dataset=_DATASET, model_factory=factory
    )

    assert outcome.status == STATUS_NO_RECORD
    assert outcome.record is None
    assert outcome.error_type is None
    assert outcome.error_message is None


def test_keyboard_interrupt_propagates_from_factory() -> None:
    def factory(**_kwargs: Any) -> Any:
        raise KeyboardInterrupt

    entry = make_entry("sga", seed=0)
    with pytest.raises(KeyboardInterrupt):
        run_episode(
            entry=entry, entry_index=0, dataset=_DATASET, model_factory=factory
        )


def test_keyboard_interrupt_propagates_from_fit() -> None:
    factory = _SpyFactory(fit_error=KeyboardInterrupt())
    entry = make_entry("sga", seed=0)
    with pytest.raises(KeyboardInterrupt):
        run_episode(
            entry=entry, entry_index=0, dataset=_DATASET, model_factory=factory
        )


def test_model_kwargs_deep_copied_so_factory_cannot_corrupt_plan() -> None:


    entry = make_entry("sga", seed=0, coeffs=[1.0, 2.0])
    plan = make_plan([entry])
    hash_before = plan.plan_hash()

    def mutating_factory(**kwargs: Any) -> _FakeModel:
        kwargs["coeffs"].append(999.0)
        return _FakeModel(run_record=make_record("sga", seed=0))

    run_episode(
        entry=entry, entry_index=0, dataset=_DATASET, model_factory=mutating_factory
    )

    assert entry.model_kwargs["coeffs"] == [1.0, 2.0]
    assert plan.plan_hash() == hash_before








class _StrictNoDeviceFactory:

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, algorithm: str, seed: int, verbose: bool) -> _FakeModel:
        self.calls.append({"algorithm": algorithm, "seed": seed, "verbose": verbose})
        return _FakeModel(run_record=None)


def test_episode_device_none_not_forwarded() -> None:
    factory = _StrictNoDeviceFactory()
    run_episode(
        entry=make_entry("sga", seed=0),
        entry_index=0,
        dataset=_DATASET,
        model_factory=factory,
        device=None,
    )
    assert "device" not in factory.calls[0]


def test_episode_forwards_device_when_set() -> None:
    factory = _SpyFactory(run_record=None)
    run_episode(
        entry=make_entry("sga", seed=0),
        entry_index=0,
        dataset=_DATASET,
        model_factory=factory,
        device="cpu",
    )
    assert factory.calls[0].get("device") == "cpu"







class _Recorder:
    def to_dict(self) -> dict[str, Any]:
        return {"_best_score": [1.0]}


class _RecordingModel:

    def __init__(
        self,
        *,
        run_record: Any,
        manifest: Any = None,
        fit_error: BaseException | None = None,
    ):
        self._run_record = run_record
        self._manifest = manifest
        self._fit_error = fit_error
        self.fit_kwargs: list[dict[str, Any]] = []

    def fit(self, dataset: Any, **kwargs: Any) -> _RecordingModel:
        self.fit_kwargs.append(kwargs)
        if self._fit_error is not None:
            raise self._fit_error
        return self

    @property
    def result_(self) -> Any:
        return type(
            "_Result",
            (),
            {
                "run_record": self._run_record,
                "manifest": self._manifest,
                "recorder": _Recorder(),
            },
        )()


class _RecordingFactory:
    def __init__(self, **model_kwargs: Any):
        self.calls: list[dict[str, Any]] = []
        self._model_kwargs = model_kwargs
        self.model: _RecordingModel | None = None

    def __call__(self, **kwargs: Any) -> _RecordingModel:
        self.calls.append(kwargs)
        self.model = _RecordingModel(**self._model_kwargs)
        return self.model


def test_recording_requires_run_dir() -> None:
    with pytest.raises(ValueError, match="run_dir"):
        run_episode(
            entry=make_entry("sga", seed=0),
            entry_index=0,
            dataset=_DATASET,
            model_factory=_RecordingFactory(run_record=None),
            recording=RecordingOptions(),
        )


def test_run_dir_wires_checkpoints_events_phases(tmp_path: Path) -> None:
    record = make_record(instrument="sga", seed=0)
    factory = _RecordingFactory(run_record=record)
    run_dir = tmp_path / "runs" / "entry-0000"
    outcome = run_episode(
        entry=make_entry("sga", seed=0),
        entry_index=0,
        dataset=_DATASET,
        model_factory=factory,
        run_dir=run_dir,
        recording=RecordingOptions(events_every_n=3, checkpoint_every=5),
        record_ref="../../records/entry-0000.json",
    )
    kwargs = factory.calls[0]
    assert kwargs["checkpoint_dir"] == run_dir / "checkpoints"
    assert kwargs["checkpoint_every"] == 5
    assert kwargs["phases_path"] == run_dir / "phases.jsonl"
    emitters = [
        cb for cb in kwargs["callbacks"] if isinstance(cb, IterationEventEmitter)
    ]
    assert len(emitters) == 1
    assert outcome.status == STATUS_COMPLETED
    assert outcome.run_dir == run_dir
    assert outcome.run_id is not None and outcome.run_id.startswith("sga-")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["record_ref"] == {
        "path": "../../records/entry-0000.json",
        "record_hash": record.record_hash,
    }
    assert not (run_dir / "record.json").exists()
    assert (run_dir / "recorder.json").is_file()


def test_recording_checkpoint_none_skips_checkpoint_kwargs(
    tmp_path: Path,
) -> None:
    factory = _RecordingFactory(run_record=None)
    run_episode(
        entry=make_entry("sga", seed=0),
        entry_index=0,
        dataset=_DATASET,
        model_factory=factory,
        run_dir=tmp_path / "rd",
        recording=RecordingOptions(checkpoint_every=None, phases=False),
    )
    kwargs = factory.calls[0]
    assert "checkpoint_dir" not in kwargs
    assert "phases_path" not in kwargs


def test_raised_episode_still_seals_run_dir(tmp_path: Path) -> None:
    factory = _RecordingFactory(run_record=None, fit_error=RuntimeError("boom"))
    outcome = run_episode(
        entry=make_entry("sga", seed=0),
        entry_index=0,
        dataset=_DATASET,
        model_factory=factory,
        run_dir=tmp_path / "rd",
    )
    assert outcome.status == STATUS_RAISED
    manifest = json.loads((tmp_path / "rd" / "manifest.json").read_text())
    assert manifest["status"] == "raised"
    assert manifest["lineage"] is None
    assert not (tmp_path / "rd" / "recorder.json").exists()


def test_resume_from_forwarded_and_lineage_sealed(tmp_path: Path) -> None:
    lineage = {
        "resume_from": "/prior/checkpoints/checkpoint_final.pt",
        "source_run_id": "sga-prior",
        "source_config_hash": None,
        "source_final_status": "completed",
        "source_iteration": 4,
    }
    manifest_obj = RunManifest(
        dataset_cache_fingerprint="sha256:dataset",
        kd_version="0.1.0",
        seed=0,
        resumed=True,
        resume_source=lineage,
    )
    factory = _RecordingFactory(
        run_record=make_record(instrument="sga", seed=0), manifest=manifest_obj
    )
    run_episode(
        entry=make_entry("sga", seed=0),
        entry_index=0,
        dataset=_DATASET,
        model_factory=factory,
        run_dir=tmp_path / "rd",
        resume_from="/prior/checkpoints/checkpoint_final.pt",
    )
    assert factory.model is not None
    assert factory.model.fit_kwargs == [
        {"resume_from": "/prior/checkpoints/checkpoint_final.pt"}
    ]
    sealed = json.loads((tmp_path / "rd" / "manifest.json").read_text())
    assert sealed["lineage"] == lineage








@pytest.mark.parametrize("fit_error", [None, RuntimeError("boom")])
def test_persist_outcome_runs_before_the_run_dir_is_sealed(
    tmp_path: Path, fit_error: BaseException | None
) -> None:
    record = None if fit_error is not None else make_record("sga", seed=0)
    factory = _RecordingFactory(run_record=record, fit_error=fit_error)
    run_dir = tmp_path / "rd"
    sealed_at_hook_time: list[bool] = []
    persisted: list[EpisodeOutcome] = []

    def persist(outcome: EpisodeOutcome) -> None:
        sealed_at_hook_time.append((run_dir / "manifest.json").exists())
        persisted.append(outcome)

    outcome = run_episode(
        entry=make_entry("sga", seed=0),
        entry_index=0,
        dataset=_DATASET,
        model_factory=factory,
        run_dir=run_dir,
        record_ref="../../records/entry-0000.json",
        persist_outcome=persist,
    )

    assert sealed_at_hook_time == [False]
    assert persisted == [outcome]
    assert (run_dir / "manifest.json").is_file()


@pytest.mark.parametrize("fit_error", [None, RuntimeError("boom")])
def test_failing_persist_outcome_leaves_the_run_dir_unsealed(
    tmp_path: Path, fit_error: BaseException | None
) -> None:
    record = None if fit_error is not None else make_record("sga", seed=0)
    factory = _RecordingFactory(run_record=record, fit_error=fit_error)
    run_dir = tmp_path / "rd"

    def persist(outcome: EpisodeOutcome) -> None:
        raise OSError("no space left on device")

    with pytest.raises(OSError, match="no space left"):
        run_episode(
            entry=make_entry("sga", seed=0),
            entry_index=0,
            dataset=_DATASET,
            model_factory=factory,
            run_dir=run_dir,
            record_ref="../../records/entry-0000.json",
            persist_outcome=persist,
        )

    assert not (run_dir / "manifest.json").exists()







def test_raised_resumed_episode_carries_the_attempted_lineage(
    tmp_path: Path,
) -> None:
    prior = create_run_dir(tmp_path / "prior")
    finalize_run_dir(
        prior, None, run_id="sga-prior", instrument="sga", status="completed"
    )
    checkpoint = prior.checkpoints / "checkpoint_final.pt"
    factory = _RecordingFactory(run_record=None, fit_error=RuntimeError("boom"))

    outcome = run_episode(
        entry=make_entry("sga", seed=0),
        entry_index=0,
        dataset=_DATASET,
        model_factory=factory,
        run_dir=tmp_path / "rd",
        resume_from=checkpoint,
    )

    expected = {
        "resume_from": str(checkpoint),
        "source_run_id": "sga-prior",
        "source_config_hash": None,
        "source_final_status": None,
        "source_iteration": None,
    }
    assert outcome.status == STATUS_RAISED
    assert outcome.lineage == expected
    sealed = json.loads((tmp_path / "rd" / "manifest.json").read_text())
    assert sealed["lineage"] == expected


def test_raised_resume_outside_a_run_dir_has_no_source_run_id(
    tmp_path: Path,
) -> None:
    factory = _RecordingFactory(run_record=None, fit_error=RuntimeError("boom"))

    outcome = run_episode(
        entry=make_entry("sga", seed=0),
        entry_index=0,
        dataset=_DATASET,
        model_factory=factory,
        run_dir=tmp_path / "rd",
        resume_from=tmp_path / "legacy" / "checkpoint_final.pt",
    )

    assert outcome.lineage is not None
    assert outcome.lineage["source_run_id"] is None
