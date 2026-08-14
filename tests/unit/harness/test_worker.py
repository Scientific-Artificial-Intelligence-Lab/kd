
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kd.harness.dispatch import (
    DispatchDatasetSpec,
    DispatchManifestError,
    DispatchRecording,
    build_dispatch_manifest,
    write_dispatch_manifest,
)
from kd.harness.plan import ExperimentPlan, PlanEntry
from kd.harness.recording import RecordingOptions
from kd.harness.store import EvidenceStore
from kd.harness.worker import run_worker

from ._helpers import make_record, make_tiny_dataset



_LOADER_CALLS: dict[str, int] = {}


def counting_loader(*, ref: str) -> Any:
    _LOADER_CALLS[ref] = _LOADER_CALLS.get(ref, 0) + 1
    return make_tiny_dataset(name=ref)


class _StubModel:
    def __init__(self, record: Any) -> None:
        self._record = record

    def fit(self, dataset: Any) -> _StubModel:
        return self

    @property
    def result_(self) -> Any:
        return type(
            "_Result", (), {"run_record": self._record, "manifest": None}
        )()


class _StubFactory:

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> _StubModel:
        self.calls.append(kwargs)
        record = make_record(instrument=kwargs["algorithm"], seed=kwargs["seed"])
        return _StubModel(record)


_LOADER = "tests.unit.harness.test_worker.counting_loader"


def _plan() -> ExperimentPlan:

    return ExperimentPlan(
        name="workerplan",
        entries=(
            PlanEntry(instrument="sga", dataset_ref="a", seed=0, model_kwargs={}),
            PlanEntry(instrument="sga", dataset_ref="a", seed=1, model_kwargs={}),
            PlanEntry(instrument="sga", dataset_ref="b", seed=2, model_kwargs={}),
        ),
    )


def _write_batch(tmp_path: Path) -> Path:
    plan = _plan()
    specs = {
        "a": DispatchDatasetSpec(loader=_LOADER, kwargs={"ref": "a"}),
        "b": DispatchDatasetSpec(loader=_LOADER, kwargs={"ref": "b"}),
    }
    manifest = build_dispatch_manifest(plan, dataset_specs=specs, n_workers=2)
    return Path(write_dispatch_manifest(manifest, tmp_path / "batch"))


def test_worker_projects_subplan_and_returns_shard_root(tmp_path: Path) -> None:
    _LOADER_CALLS.clear()
    dispatch_path = _write_batch(tmp_path)
    plan = _plan()

    shard_root = run_worker(
        dispatch_path, "shard-00", model_factory=_StubFactory()
    )

    assert Path(shard_root) == tmp_path / "batch" / "shards" / "shard-00"
    loaded = EvidenceStore.load(shard_root)

    assert loaded.plan.name == "workerplan::shard-00"
    assert loaded.plan.entries == (plan.entries[0], plan.entries[1])
    assert set(loaded.records) == {0, 1}


def test_worker_loads_only_its_own_shard_refs(tmp_path: Path) -> None:
    _LOADER_CALLS.clear()
    dispatch_path = _write_batch(tmp_path)

    run_worker(dispatch_path, "shard-00", model_factory=_StubFactory())



    assert _LOADER_CALLS == {"a": 1}


def test_worker_unknown_shard_id_raises(tmp_path: Path) -> None:
    _LOADER_CALLS.clear()
    dispatch_path = _write_batch(tmp_path)
    with pytest.raises(DispatchManifestError, match="shard"):
        run_worker(dispatch_path, "shard-99", model_factory=_StubFactory())







class _RichStubModel(_StubModel):

    @property
    def result_(self) -> Any:
        class _Recorder:
            def to_dict(self) -> dict[str, Any]:
                return {"_best_score": [1.0]}

        return type(
            "_Result",
            (),
            {"run_record": self._record, "manifest": None, "recorder": _Recorder()},
        )()


class _RichStubFactory(_StubFactory):
    def __call__(self, **kwargs: Any) -> _RichStubModel:
        self.calls.append(kwargs)
        record = make_record(instrument=kwargs["algorithm"], seed=kwargs["seed"])
        return _RichStubModel(record)


def test_worker_recording_writes_catalog_with_global_indices(
    tmp_path: Path,
) -> None:
    _LOADER_CALLS.clear()
    plan = _plan()
    specs = {
        "a": DispatchDatasetSpec(loader=_LOADER, kwargs={"ref": "a"}),
        "b": DispatchDatasetSpec(loader=_LOADER, kwargs={"ref": "b"}),
    }
    manifest = build_dispatch_manifest(
        plan,
        dataset_specs=specs,
        n_workers=2,
        recording=DispatchRecording(
            options=RecordingOptions(checkpoint_every=None, phases=False),
            catalog="catalog.jsonl",
        ),
    )
    dispatch_path = Path(write_dispatch_manifest(manifest, tmp_path / "batch"))

    shard_root = run_worker(dispatch_path, "shard-01", model_factory=_RichStubFactory())

    assert (shard_root / "runs" / "entry-0000" / "manifest.json").is_file()
    catalog = dispatch_path.parent / "catalog.jsonl"
    rows = [json.loads(line) for line in catalog.read_text().splitlines()]
    assert [row["entry_index"] for row in rows] == [2]
    assert rows[0]["seed"] == 2
    assert rows[0]["plan_hash"] == manifest.plan_hash
    assert rows[0]["plan_hash"] != EvidenceStore.load(shard_root).plan_hash
    run_dir = (catalog.parent / rows[0]["run_dir"]).resolve()
    assert run_dir == (shard_root / "runs" / "entry-0000").resolve()
