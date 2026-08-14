
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kd.harness.episode import (
    STATUS_COMPLETED,
    STATUS_NO_RECORD,
    STATUS_RAISED,
)
from kd.harness.plan import ExperimentPlan
from kd.harness.recording import RecordingOptions
from kd.harness.runner import run_plan
from kd.harness.store import EvidenceStore
from kd.search.result import RunManifest
from kd.search.run_dir import create_run_dir, finalize_run_dir

from ._helpers import make_entry, make_plan, make_record, make_tiny_dataset

_DATASETS = {"burgers_tiny": make_tiny_dataset()}


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

    def fit(self, dataset: Any) -> _ScriptedModel:
        if isinstance(self._behavior, BaseException):
            raise self._behavior
        return self

    @property
    def result_(self) -> Any:
        record = None if isinstance(self._behavior, BaseException) else self._behavior
        return type("_Result", (), {"run_record": record, "manifest": None})()


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


def test_run_plan_writes_report_md(tmp_path: Path) -> None:
    record = make_record(instrument="sga", seed=0)
    factory = _ScriptedFactory({0: record})
    result = run_plan(
        _single_plan("sga", seed=0),
        datasets=_DATASETS,
        store_root=tmp_path / "store",
        model_factory=factory,
    )
    report = result.store_root / "report.md"
    assert report.is_file()
    assert "1 entries / 1 completed" in report.read_text()

    EvidenceStore.load(result.store_root)







class _RichScriptedModel:

    def __init__(self, behavior: Any, manifest: Any = None):
        self._behavior = behavior
        self._manifest = manifest
        self.fit_kwargs: list[dict[str, Any]] = []

    def fit(self, dataset: Any, **kwargs: Any) -> _RichScriptedModel:
        self.fit_kwargs.append(kwargs)
        if isinstance(self._behavior, BaseException):
            raise self._behavior
        return self

    @property
    def result_(self) -> Any:
        record = None if isinstance(self._behavior, BaseException) else self._behavior

        class _Recorder:
            def to_dict(self) -> dict[str, Any]:
                return {"_best_score": [1.0]}

        return type(
            "_Result",
            (),
            {
                "run_record": record,
                "manifest": self._manifest,
                "recorder": _Recorder(),
            },
        )()


class _RichScriptedFactory:
    def __init__(
        self, behaviors: dict[int, Any], manifests: dict[int, Any] | None = None
    ):
        self._behaviors = behaviors
        self._manifests = manifests or {}
        self.calls: list[dict[str, Any]] = []
        self.models: list[_RichScriptedModel] = []

    def __call__(self, **kwargs: Any) -> _RichScriptedModel:
        self.calls.append(kwargs)
        seed = kwargs["seed"]
        model = _RichScriptedModel(
            self._behaviors[seed], manifest=self._manifests.get(seed)
        )
        self.models.append(model)
        return model


def test_recording_run_plan_catalog_and_global_indices(tmp_path: Path) -> None:
    record = make_record(instrument="sga", seed=1)
    plan = make_plan(
        [
            make_entry("sga", seed=1),
            make_entry("sga", seed=2),
        ],
        name="shardlike",
    )
    factory = _RichScriptedFactory({1: record, 2: RuntimeError("boom")})
    catalog = tmp_path / "batch" / "catalog.jsonl"
    result = run_plan(
        plan,
        datasets=_DATASETS,
        store_root=tmp_path / "batch" / "shards" / "shard-00",
        model_factory=factory,
        recording=RecordingOptions(checkpoint_every=None, phases=False),
        catalog_path=catalog,
        entry_indices=(5, 7),
        resume_from={7: "/prior/checkpoint_final.pt"},
    )

    for local in (0, 1):
        run_dir = result.store_root / "runs" / f"entry-{local:04d}"
        assert (run_dir / "manifest.json").is_file()

    assert factory.models[0].fit_kwargs == [{}]
    assert factory.models[1].fit_kwargs == [
        {"resume_from": "/prior/checkpoint_final.pt"}
    ]
    rows = [json.loads(line) for line in catalog.read_text().splitlines()]
    assert [row["entry_index"] for row in rows] == [5, 7]
    assert rows[0]["status"] == "completed"
    assert rows[0]["record_hash"] == record.record_hash

    resolved = (catalog.parent / rows[0]["record_path"]).resolve()
    assert resolved == (
        result.store_root / "records" / "entry-0000.json"
    ).resolve()
    assert resolved.is_file()
    assert rows[1]["status"] == "raised"
    assert rows[1]["resume_from"] == "/prior/checkpoint_final.pt"
    assert rows[1]["plan_hash"] == rows[0]["plan_hash"]

    EvidenceStore.load(result.store_root)


def test_run_dir_creation_failure_skips_catalog_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_create_run_dir(path: Path) -> None:
        raise OSError("run dir unavailable")

    monkeypatch.setattr("kd.harness.episode.create_run_dir", fail_create_run_dir)
    catalog = tmp_path / "catalog.jsonl"
    result = run_plan(
        _single_plan(),
        datasets=_DATASETS,
        store_root=tmp_path / "store",
        model_factory=_RichScriptedFactory({0: None}),
        recording=RecordingOptions(checkpoint_every=None, phases=False),
        catalog_path=catalog,
    )

    assert result.outcomes[0].status == STATUS_RAISED
    assert not catalog.exists()
    loaded = EvidenceStore.load(result.store_root)
    assert loaded.attempts[0]["status"] == STATUS_RAISED


def test_catalog_rows_carry_the_parent_run_id_of_a_resumed_run(
    tmp_path: Path,
) -> None:
    lineage = {
        "resume_from": "/prior/checkpoints/checkpoint_final.pt",
        "source_run_id": "sga-prior",
        "source_config_hash": None,
        "source_final_status": "completed",
        "source_iteration": 4,
    }
    plan = make_plan(
        [make_entry("sga", seed=1), make_entry("sga", seed=2)], name="lineage"
    )
    factory = _RichScriptedFactory(
        {1: make_record(instrument="sga", seed=1), 2: make_record("sga", seed=2)},
        manifests={
            2: RunManifest(
                dataset_cache_fingerprint="sha256:dataset",
                kd_version="0.1.0",
                seed=2,
                resumed=True,
                resume_source=lineage,
            )
        },
    )
    catalog = tmp_path / "catalog.jsonl"
    run_plan(
        plan,
        datasets=_DATASETS,
        store_root=tmp_path / "store",
        model_factory=factory,
        recording=RecordingOptions(checkpoint_every=None, phases=False),
        catalog_path=catalog,
        resume_from={1: "/prior/checkpoints/checkpoint_final.pt"},
    )

    rows = [json.loads(line) for line in catalog.read_text().splitlines()]
    assert rows[0]["parent_run_id"] is None
    assert rows[1]["parent_run_id"] == "sga-prior"
    assert rows[1]["resume_from"] == "/prior/checkpoints/checkpoint_final.pt"


def test_catalog_row_of_a_raised_resume_carries_the_attempted_parent(
    tmp_path: Path,
) -> None:
    prior = create_run_dir(tmp_path / "prior")
    finalize_run_dir(
        prior, None, run_id="sga-prior", instrument="sga", status="completed"
    )
    checkpoint = prior.checkpoints / "checkpoint_final.pt"
    catalog = tmp_path / "catalog.jsonl"
    run_plan(
        _single_plan("sga", seed=0),
        datasets=_DATASETS,
        store_root=tmp_path / "store",
        model_factory=_RichScriptedFactory({0: RuntimeError("boom")}),
        recording=RecordingOptions(checkpoint_every=None, phases=False),
        catalog_path=catalog,
        resume_from={0: checkpoint},
    )

    rows = [json.loads(line) for line in catalog.read_text().splitlines()]
    assert rows[0]["status"] == STATUS_RAISED
    assert rows[0]["parent_run_id"] == "sga-prior"


def test_store_write_failure_leaves_the_run_dir_unsealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    def fail_add_outcome(self: Any, outcome: Any) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(EvidenceStore, "add_outcome", fail_add_outcome)
    store_root = tmp_path / "store"
    with pytest.raises(OSError, match="no space left"):
        run_plan(
            _single_plan("sga", seed=0),
            datasets=_DATASETS,
            store_root=store_root,
            model_factory=_RichScriptedFactory(
                {0: make_record(instrument="sga", seed=0)}
            ),
            recording=RecordingOptions(checkpoint_every=None, phases=False),
        )

    assert not (store_root / "runs" / "entry-0000" / "manifest.json").exists()


def test_catalog_requires_recording(tmp_path: Path) -> None:
    factory = _RichScriptedFactory({0: make_record(instrument="sga", seed=0)})
    with pytest.raises(ValueError, match="catalog_path requires recording"):
        run_plan(
            _single_plan("sga", seed=0),
            datasets=_DATASETS,
            store_root=tmp_path / "store",
            model_factory=factory,
            catalog_path=tmp_path / "catalog.jsonl",
        )


def test_entry_indices_length_mismatch_rejected(tmp_path: Path) -> None:
    factory = _RichScriptedFactory({0: make_record(instrument="sga", seed=0)})
    with pytest.raises(ValueError, match="entry_indices"):
        run_plan(
            _single_plan("sga", seed=0),
            datasets=_DATASETS,
            store_root=tmp_path / "store",
            model_factory=factory,
            entry_indices=(0, 1),
        )
