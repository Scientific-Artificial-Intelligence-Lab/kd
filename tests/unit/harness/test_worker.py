
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from kd.harness.dispatch import (
    DispatchDatasetSpec,
    DispatchManifestError,
    build_dispatch_manifest,
    write_dispatch_manifest,
)
from kd.harness.worker import run_worker

from kd.harness.plan import ExperimentPlan, PlanEntry
from kd.harness.store import EvidenceStore

from ._helpers import make_record



_LOADER_CALLS: dict[str, int] = {}


def counting_loader(*, ref: str) -> Any:
    _LOADER_CALLS[ref] = _LOADER_CALLS.get(ref, 0) + 1
    return object()


class _StubModel:
    def __init__(self, record: Any) -> None:
        self._record = record

    def fit(self, dataset: Any) -> _StubModel:
        return self

    @property
    def result_(self) -> Any:
        return type("_Result", (), {"run_record": self._record})()


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
