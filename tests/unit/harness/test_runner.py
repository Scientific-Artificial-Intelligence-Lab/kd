
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kd.harness.episode import (
    STATUS_COMPLETED,
    STATUS_NO_RECORD,
    STATUS_RAISED,
)
from kd.harness.plan import ExperimentPlan
from kd.harness.runner import run_plan
from kd.harness.store import EvidenceStore

from ._helpers import make_entry, make_plan, make_record


_DATASET: Any = object()
_DATASETS: dict[str, Any] = {"burgers_tiny": _DATASET}


class _ScriptedFactory:

    def __init__(self, behaviors: dict[int, Any]):
        self._behaviors = behaviors
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        behavior = self._behaviors[kwargs["seed"]]
        return _ScriptedModel(behavior)


class _ScriptedModel:
    def __init__(self, behavior: Any):
        self._behavior = behavior

    def fit(self, dataset: Any) -> "_ScriptedModel":
        if isinstance(self._behavior, BaseException):
            raise self._behavior
        return self

    @property
    def result_(self) -> Any:
        record = None if isinstance(self._behavior, BaseException) else self._behavior
        return type("_Result", (), {"run_record": record})()


def _single_plan(instrument: str = "sga", seed: int = 0) -> ExperimentPlan:
    return make_plan([make_entry(instrument, seed=seed)], name="one")


def test_completed_run_stores_record_and_attempt(tmp_path: Path) -> None:
    record = make_record(instrument="sga", seed=0)
    factory = _ScriptedFactory({0: record})
    plan = _single_plan("sga", seed=0)
    root = tmp_path / "store"

    result = run_plan(
        plan, datasets=_DATASETS, store_root=root, model_factory=factory
    )

    assert result.store_root == root
    assert len(result.outcomes) == 1
    assert result.outcomes[0].status == STATUS_COMPLETED


    loaded = EvidenceStore.load(root)
    assert set(loaded.records) == {0}
    assert loaded.records[0].record_hash == record.record_hash
    assert len(loaded.attempts) == 1
    attempt = loaded.attempts[0]
    assert attempt["status"] == STATUS_COMPLETED
    assert attempt["instrument"] == "sga"
    assert attempt["seed"] == 0


def test_raised_entry_does_not_abort_batch(tmp_path: Path) -> None:

    good = make_record(instrument="sga", seed=1)
    factory = _ScriptedFactory({0: RuntimeError("boom"), 1: good})
    plan = make_plan(
        [make_entry("sga", seed=0), make_entry("sga", seed=1)], name="mixed"
    )
    root = tmp_path / "store"

    result = run_plan(
        plan, datasets=_DATASETS, store_root=root, model_factory=factory
    )

    assert [o.status for o in result.outcomes] == [STATUS_RAISED, STATUS_COMPLETED]
    assert result.outcomes[0].error_type == "RuntimeError"
    assert result.outcomes[0].error_message == "boom"

    loaded = EvidenceStore.load(root)

    assert len(loaded.attempts) == 2
    assert set(loaded.records) == {1}
    assert [a["status"] for a in loaded.attempts] == [
        STATUS_RAISED,
        STATUS_COMPLETED,
    ]


def test_none_run_record_is_no_record(tmp_path: Path) -> None:
    factory = _ScriptedFactory({0: None})
    plan = _single_plan("sga", seed=0)
    root = tmp_path / "store"

    result = run_plan(
        plan, datasets=_DATASETS, store_root=root, model_factory=factory
    )

    assert result.outcomes[0].status == STATUS_NO_RECORD
    loaded = EvidenceStore.load(root)
    assert loaded.records == {}
    assert loaded.attempts[0]["status"] == STATUS_NO_RECORD


def test_mixed_batch_store_matches_outcomes(tmp_path: Path) -> None:
    r0 = make_record(instrument="sga", seed=0)
    r2 = make_record(instrument="sga", seed=2)
    factory = _ScriptedFactory({0: r0, 1: RuntimeError("x"), 2: r2})
    plan = make_plan(
        [
            make_entry("sga", seed=0),
            make_entry("sga", seed=1),
            make_entry("sga", seed=2),
        ],
        name="triple",
    )
    root = tmp_path / "store"

    result = run_plan(
        plan, datasets=_DATASETS, store_root=root, model_factory=factory
    )

    assert [o.status for o in result.outcomes] == [
        STATUS_COMPLETED,
        STATUS_RAISED,
        STATUS_COMPLETED,
    ]
    loaded = EvidenceStore.load(root)

    assert set(loaded.records) == {0, 2}
    assert loaded.records[0].record_hash == r0.record_hash
    assert loaded.records[2].record_hash == r2.record_hash

    assert [a["entry_index"] for a in loaded.attempts] == [0, 1, 2]
    assert [a["status"] for a in loaded.attempts] == [
        STATUS_COMPLETED,
        STATUS_RAISED,
        STATUS_COMPLETED,
    ]


def test_preflight_collects_all_violations(tmp_path: Path) -> None:


    plan = make_plan(
        [
            make_entry("not_a_real_instrument", dataset_ref="burgers_tiny", seed=0),
            make_entry("sga", dataset_ref="missing_dataset", seed=0),
        ],
        name="bad",
    )
    factory = _ScriptedFactory({})
    root = tmp_path / "store"

    with pytest.raises(ValueError, match="pre-flight") as excinfo:
        run_plan(plan, datasets=_DATASETS, store_root=root, model_factory=factory)

    message = str(excinfo.value)
    assert "not_a_real_instrument" in message
    assert "missing_dataset" in message

    assert not root.exists()
    assert factory.calls == []


def test_keyboard_interrupt_propagates_from_loop(tmp_path: Path) -> None:
    factory = _ScriptedFactory({0: KeyboardInterrupt()})
    plan = _single_plan("sga", seed=0)
    root = tmp_path / "store"

    with pytest.raises(KeyboardInterrupt):
        run_plan(plan, datasets=_DATASETS, store_root=root, model_factory=factory)








class _StrictNoDeviceFactory:

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, algorithm: str, seed: int, verbose: bool) -> Any:
        self.calls.append({"algorithm": algorithm, "seed": seed, "verbose": verbose})
        return _ScriptedModel(None)


def test_run_plan_device_none_does_not_forward_device(tmp_path: Path) -> None:
    factory = _StrictNoDeviceFactory()
    run_plan(
        _single_plan("sga", seed=0),
        datasets=_DATASETS,
        store_root=tmp_path / "store",
        model_factory=factory,
        device=None,
    )
    assert "device" not in factory.calls[0]


def test_run_plan_forwards_device_when_set(tmp_path: Path) -> None:
    record = make_record(instrument="sga", seed=0)
    factory = _ScriptedFactory({0: record})
    run_plan(
        _single_plan("sga", seed=0),
        datasets=_DATASETS,
        store_root=tmp_path / "store",
        model_factory=factory,
        device="cpu",
    )
    assert factory.calls[0].get("device") == "cpu"
