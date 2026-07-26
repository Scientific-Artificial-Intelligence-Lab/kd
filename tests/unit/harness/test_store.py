
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kd.harness.plan import PLAN_HASH_SCHEME
from kd.harness.store import (
    EvidenceStore,
    EvidenceStoreError,
    environment_fingerprint,
)

from ._helpers import (
    StubOutcome,
    completed_outcome,
    make_entry,
    make_plan,
    make_record,
    raised_outcome,
)


def _one_entry_plan() -> tuple:
    entry = make_entry("sga", seed=0)
    return make_plan([entry]), entry


def test_environment_fingerprint_shape() -> None:


    import re

    import torch

    env = environment_fingerprint()
    assert set(env) == {
        "kd_version",
        "python",
        "platform",
        "git_sha",
        "torch_version",
    }
    assert all(isinstance(v, str) and v for v in env.values())

    assert re.fullmatch(r"[0-9a-f]{40}", env["git_sha"]) or env["git_sha"] == "unknown"
    assert env["torch_version"] == torch.__version__


def test_git_sha_degrades_to_unknown_on_called_process_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    def _boom(*_a: object, **_k: object) -> bytes:
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(subprocess, "check_output", _boom)
    env = environment_fingerprint()
    assert env["git_sha"] == "unknown"


def test_git_sha_degrades_to_unknown_when_git_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    def _boom(*_a: object, **_k: object) -> bytes:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "check_output", _boom)
    env = environment_fingerprint()
    assert env["git_sha"] == "unknown"


def test_git_sha_unknown_when_module_is_untracked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:



    import subprocess

    real_check_output = subprocess.check_output

    def _untracked(cmd: list[str], **kwargs: object) -> bytes:
        if "ls-files" in cmd:
            raise subprocess.CalledProcessError(1, cmd)
        return real_check_output(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", _untracked)
    env = environment_fingerprint()
    assert env["git_sha"] == "unknown"


def test_create_fresh_dir_and_index_written(tmp_path: Path) -> None:
    plan, _ = _one_entry_plan()
    root = tmp_path / "store"
    store = EvidenceStore.create(root, plan=plan, env=environment_fingerprint())

    assert store.root == root
    index = json.loads((root / "index.json").read_text())
    assert index["index_schema_version"] == 1
    assert index["plan_hash"] == plan.plan_hash()
    assert index["plan_hash_scheme"] == PLAN_HASH_SCHEME
    assert index["attempts"] == []
    assert index["records"] == {}
    assert (root / "records").is_dir()


def test_create_refuses_non_empty_root(tmp_path: Path) -> None:
    plan, _ = _one_entry_plan()
    root = tmp_path / "store"
    root.mkdir()
    (root / "stray.txt").write_text("occupied")

    with pytest.raises(EvidenceStoreError, match="not empty"):
        EvidenceStore.create(root, plan=plan, env=environment_fingerprint())


def test_create_accepts_existing_empty_dir(tmp_path: Path) -> None:
    plan, _ = _one_entry_plan()
    root = tmp_path / "store"
    root.mkdir()
    EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    assert (root / "index.json").is_file()


def test_add_outcome_writes_record_and_attempt(tmp_path: Path) -> None:
    plan, entry = _one_entry_plan()
    store = EvidenceStore.create(
        tmp_path / "s", plan=plan, env=environment_fingerprint()
    )
    record = make_record("sga", seed=0)
    store.add_outcome(completed_outcome(0, entry, record))

    assert set(store.records) == {0}
    assert store.attempts[0]["status"] == "completed"
    assert store.attempts[0]["instrument"] == "sga"
    assert (store.root / "records" / "entry-0000.json").is_file()


def test_add_outcome_raised_appends_attempt_without_record(tmp_path: Path) -> None:
    plan, entry = _one_entry_plan()
    store = EvidenceStore.create(
        tmp_path / "s", plan=plan, env=environment_fingerprint()
    )
    store.add_outcome(raised_outcome(0, entry))

    assert store.records == {}
    assert store.attempts[0]["status"] == "raised"
    assert store.attempts[0]["error_type"] == "RuntimeError"
    assert not (store.root / "records" / "entry-0000.json").exists()


def test_add_outcome_rejects_duplicate_entry(tmp_path: Path) -> None:


    plan, entry = _one_entry_plan()
    store = EvidenceStore.create(
        tmp_path / "s", plan=plan, env=environment_fingerprint()
    )
    store.add_outcome(completed_outcome(0, entry, make_record("sga", seed=0)))

    with pytest.raises(EvidenceStoreError, match="duplicate"):
        store.add_outcome(completed_outcome(0, entry, make_record("sga", seed=0)))


def test_index_rewritten_after_every_outcome(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=s) for s in range(3)]
    plan = make_plan(entries)
    root = tmp_path / "s"
    store = EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    for i, entry in enumerate(entries):
        store.add_outcome(completed_outcome(i, entry, make_record("sga", seed=i)))
        index = json.loads((root / "index.json").read_text())

        assert len(index["attempts"]) == i + 1
        assert len(index["records"]) == i + 1


def test_round_trip_create_add_load(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=0), make_entry("pysindy", seed=1)]
    plan = make_plan(entries)
    root = tmp_path / "s"
    store = EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    store.add_outcome(completed_outcome(0, entries[0], make_record("sga", seed=0)))
    store.add_outcome(
        completed_outcome(1, entries[1], make_record("pysindy", seed=1))
    )

    loaded = EvidenceStore.load(root)
    assert loaded.plan_hash == plan.plan_hash()
    assert loaded.plan.to_dict() == plan.to_dict()
    assert set(loaded.records) == {0, 1}
    assert loaded.records[0].evidence.instrument == "sga"
    assert loaded.records[1].evidence.seed == 1
    assert len(loaded.attempts) == 2


def test_load_missing_index_raises(tmp_path: Path) -> None:
    with pytest.raises(EvidenceStoreError, match="index"):
        EvidenceStore.load(tmp_path / "nope")


def test_load_missing_record_file_raises(tmp_path: Path) -> None:
    plan, entry = _one_entry_plan()
    root = tmp_path / "s"
    store = EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    store.add_outcome(completed_outcome(0, entry, make_record("sga", seed=0)))
    (root / "records" / "entry-0000.json").unlink()

    with pytest.raises(EvidenceStoreError, match="missing record"):
        EvidenceStore.load(root)


def test_load_tamper_flipped_byte_in_record_raises(tmp_path: Path) -> None:
    plan, entry = _one_entry_plan()
    root = tmp_path / "s"
    store = EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    store.add_outcome(
        completed_outcome(0, entry, make_record("sga", seed=0, expression="u_xx"))
    )
    record_file = root / "records" / "entry-0000.json"

    record_file.write_text(record_file.read_text().replace("u_xx", "u_yy"))

    with pytest.raises(EvidenceStoreError):
        EvidenceStore.load(root)


def test_load_index_record_hash_mismatch_raises(tmp_path: Path) -> None:
    plan, entry = _one_entry_plan()
    root = tmp_path / "s"
    store = EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    store.add_outcome(completed_outcome(0, entry, make_record("sga", seed=0)))

    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    index["records"]["0"]["record_hash"] = "0" * 64
    index_path.write_text(json.dumps(index))

    with pytest.raises(EvidenceStoreError, match="record_hash"):
        EvidenceStore.load(root)


def test_load_plan_hash_mismatch_raises(tmp_path: Path) -> None:
    plan, entry = _one_entry_plan()
    root = tmp_path / "s"
    store = EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    store.add_outcome(completed_outcome(0, entry, make_record("sga", seed=0)))

    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    index["plan_hash"] = "deadbeef" * 8
    index_path.write_text(json.dumps(index))

    with pytest.raises(EvidenceStoreError, match="plan_hash"):
        EvidenceStore.load(root)


def test_load_unknown_index_key_raises(tmp_path: Path) -> None:
    plan, _ = _one_entry_plan()
    root = tmp_path / "s"
    EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    index["surprise"] = 1
    index_path.write_text(json.dumps(index))

    with pytest.raises(EvidenceStoreError, match="malformed"):
        EvidenceStore.load(root)


def test_load_invalid_json_index_raises(tmp_path: Path) -> None:
    plan, _ = _one_entry_plan()
    root = tmp_path / "s"
    EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    (root / "index.json").write_text("{ this is not valid json")

    with pytest.raises(EvidenceStoreError, match="not valid JSON"):
        EvidenceStore.load(root)


def test_load_non_integer_record_key_raises(tmp_path: Path) -> None:
    plan, entry = _one_entry_plan()
    root = tmp_path / "s"
    store = EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    store.add_outcome(completed_outcome(0, entry, make_record("sga", seed=0)))



    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    index["records"] = {"not-an-int": index["records"]["0"]}
    index_path.write_text(json.dumps(index))

    with pytest.raises(EvidenceStoreError, match="non-integer record index key"):
        EvidenceStore.load(root)





def _two_entry_store(tmp_path: Path, instruments: tuple[str, str]) -> Path:
    entries = [make_entry(instruments[0], seed=0), make_entry(instruments[1], seed=1)]
    plan = make_plan(entries)
    root = tmp_path / "s"
    store = EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    store.add_outcome(
        completed_outcome(0, entries[0], make_record(instruments[0], seed=0))
    )
    store.add_outcome(
        completed_outcome(1, entries[1], make_record(instruments[1], seed=1))
    )
    return root


def test_load_out_of_range_record_key_raises(tmp_path: Path) -> None:
    plan, entry = _one_entry_plan()
    root = tmp_path / "s"
    store = EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    store.add_outcome(completed_outcome(0, entry, make_record("sga", seed=0)))

    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    index["records"]["5"] = {
        "path": "records/entry-0005.json",
        "record_hash": "0" * 64,
    }
    index_path.write_text(json.dumps(index))

    with pytest.raises(EvidenceStoreError, match="out of range"):
        EvidenceStore.load(root)


def test_load_non_canonical_record_path_raises(tmp_path: Path) -> None:
    plan, entry = _one_entry_plan()
    root = tmp_path / "s"
    store = EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    store.add_outcome(completed_outcome(0, entry, make_record("sga", seed=0)))

    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    index["records"]["0"]["path"] = "records/entry-9999.json"
    index_path.write_text(json.dumps(index))

    with pytest.raises(EvidenceStoreError, match="non-canonical"):
        EvidenceStore.load(root)


def test_load_swapped_record_metadata_raises(tmp_path: Path) -> None:


    root = _two_entry_store(tmp_path, ("sga", "sga"))
    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    index["records"]["0"], index["records"]["1"] = (
        index["records"]["1"],
        index["records"]["0"],
    )
    index_path.write_text(json.dumps(index))

    with pytest.raises(EvidenceStoreError, match="non-canonical"):
        EvidenceStore.load(root)


def test_load_record_plan_instrument_binding_raises(tmp_path: Path) -> None:


    root = _two_entry_store(tmp_path, ("sga", "pysindy"))
    records_dir = root / "records"
    f0, f1 = records_dir / "entry-0000.json", records_dir / "entry-0001.json"
    c0, c1 = f0.read_text(), f1.read_text()
    f0.write_text(c1)
    f1.write_text(c0)

    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    index["records"]["0"]["record_hash"], index["records"]["1"]["record_hash"] = (
        index["records"]["1"]["record_hash"],
        index["records"]["0"]["record_hash"],
    )
    index_path.write_text(json.dumps(index))

    with pytest.raises(EvidenceStoreError, match="instrument mismatch"):
        EvidenceStore.load(root)





def test_load_rejects_unknown_plan_hash_scheme(tmp_path: Path) -> None:
    plan, _ = _one_entry_plan()
    root = tmp_path / "s"
    EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    index["plan_hash_scheme"] = "kd-plan-v2"
    index_path.write_text(json.dumps(index))

    with pytest.raises(EvidenceStoreError, match="plan_hash_scheme"):
        EvidenceStore.load(root)





def test_add_outcome_leaves_no_tmp_record_file(tmp_path: Path) -> None:
    plan, entry = _one_entry_plan()
    root = tmp_path / "s"
    store = EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    store.add_outcome(completed_outcome(0, entry, make_record("sga", seed=0)))

    records_dir = root / "records"
    assert (records_dir / "entry-0000.json").is_file()

    assert not list(records_dir.glob("*.tmp"))


def test_load_rejects_orphan_record_file(tmp_path: Path) -> None:
    plan, entry = _one_entry_plan()
    root = tmp_path / "s"
    store = EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    store.add_outcome(completed_outcome(0, entry, make_record("sga", seed=0)))

    (root / "records" / "entry-0001.json").write_text("{}")

    with pytest.raises(EvidenceStoreError, match="orphan"):
        EvidenceStore.load(root)





def _writable_store(tmp_path: Path, n_entries: int = 1) -> tuple:
    entries = [make_entry("sga", seed=s) for s in range(n_entries)]
    plan = make_plan(entries)
    store = EvidenceStore.create(
        tmp_path / "s", plan=plan, env=environment_fingerprint()
    )
    return store, entries


def test_add_outcome_rejects_unknown_status(tmp_path: Path) -> None:
    store, entries = _writable_store(tmp_path)
    bad = StubOutcome(
        entry_index=0, entry=entries[0], status="bogus", wallclock_seconds=1.0
    )
    with pytest.raises(EvidenceStoreError, match="unrecognized"):
        store.add_outcome(bad)


def test_add_outcome_rejects_completed_without_record(tmp_path: Path) -> None:
    store, entries = _writable_store(tmp_path)
    bad = StubOutcome(
        entry_index=0,
        entry=entries[0],
        status="completed",
        record=None,
        wallclock_seconds=1.0,
    )
    with pytest.raises(EvidenceStoreError, match="completed iff"):
        store.add_outcome(bad)


def test_add_outcome_rejects_noncompleted_with_record(tmp_path: Path) -> None:
    store, entries = _writable_store(tmp_path)
    bad = StubOutcome(
        entry_index=0,
        entry=entries[0],
        status="raised",
        record=make_record("sga", seed=0),
        error_type="ValueError",
        wallclock_seconds=1.0,
    )
    with pytest.raises(EvidenceStoreError, match="completed iff"):
        store.add_outcome(bad)


def test_add_outcome_rejects_raised_without_error_type(tmp_path: Path) -> None:
    store, entries = _writable_store(tmp_path)
    bad = StubOutcome(
        entry_index=0,
        entry=entries[0],
        status="raised",
        error_type=None,
        wallclock_seconds=1.0,
    )
    with pytest.raises(EvidenceStoreError, match="error_type"):
        store.add_outcome(bad)


def test_add_outcome_rejects_bool_entry_index(tmp_path: Path) -> None:
    store, entries = _writable_store(tmp_path)
    bad = StubOutcome(
        entry_index=True,
        entry=entries[0],
        status="raised",
        error_type="ValueError",
        wallclock_seconds=1.0,
    )
    with pytest.raises(EvidenceStoreError, match="bool rejected"):
        store.add_outcome(bad)


def test_add_outcome_rejects_out_of_range_entry_index(tmp_path: Path) -> None:
    store, entries = _writable_store(tmp_path)
    bad = StubOutcome(
        entry_index=5,
        entry=entries[0],
        status="raised",
        error_type="ValueError",
        wallclock_seconds=1.0,
    )
    with pytest.raises(EvidenceStoreError, match="out of range"):
        store.add_outcome(bad)


def test_add_outcome_rejects_negative_wallclock(tmp_path: Path) -> None:
    store, entries = _writable_store(tmp_path)
    bad = StubOutcome(
        entry_index=0,
        entry=entries[0],
        status="raised",
        error_type="ValueError",
        wallclock_seconds=-1.0,
    )
    with pytest.raises(EvidenceStoreError, match="wallclock_seconds"):
        store.add_outcome(bad)


def test_add_outcome_rejects_nonfinite_wallclock(tmp_path: Path) -> None:
    store, entries = _writable_store(tmp_path)
    bad = StubOutcome(
        entry_index=0,
        entry=entries[0],
        status="raised",
        error_type="ValueError",
        wallclock_seconds=float("inf"),
    )
    with pytest.raises(EvidenceStoreError, match="finite"):
        store.add_outcome(bad)


def test_load_returns_read_only_store(tmp_path: Path) -> None:
    plan, entry = _one_entry_plan()
    root = tmp_path / "s"
    store = EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    store.add_outcome(completed_outcome(0, entry, make_record("sga", seed=0)))

    loaded = EvidenceStore.load(root)
    with pytest.raises(EvidenceStoreError, match="read-only"):
        loaded.add_outcome(raised_outcome(0, entry))


def test_add_outcome_rejects_non_str_error_message(tmp_path: Path) -> None:
    store, entries = _writable_store(tmp_path)
    bad = StubOutcome(
        entry_index=0,
        entry=entries[0],
        status="raised",
        error_type="ValueError",
        error_message=b"bytes",
        wallclock_seconds=1.0,
    )
    with pytest.raises(EvidenceStoreError, match="error_message"):
        store.add_outcome(bad)


def test_add_outcome_rejects_non_str_error_type(tmp_path: Path) -> None:
    store, entries = _writable_store(tmp_path)
    bad = StubOutcome(
        entry_index=0,
        entry=entries[0],
        status="raised",
        error_type=123,
        wallclock_seconds=1.0,
    )
    with pytest.raises(EvidenceStoreError, match="error_type"):
        store.add_outcome(bad)





def test_add_outcome_binds_record_seed_to_plan(tmp_path: Path) -> None:


    store, entries = _writable_store(tmp_path)
    with pytest.raises(EvidenceStoreError, match="seed mismatch"):
        store.add_outcome(
            completed_outcome(0, entries[0], make_record("sga", seed=99))
        )


def test_add_outcome_binds_record_instrument_to_plan(tmp_path: Path) -> None:
    store, entries = _writable_store(tmp_path)
    with pytest.raises(EvidenceStoreError, match="instrument mismatch"):
        store.add_outcome(
            completed_outcome(0, entries[0], make_record("pysindy", seed=0))
        )





def test_load_non_canonical_record_key_raises(tmp_path: Path) -> None:
    plan, entry = _one_entry_plan()
    root = tmp_path / "s"
    store = EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    store.add_outcome(completed_outcome(0, entry, make_record("sga", seed=0)))

    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    index["records"] = {"01": index["records"]["0"]}
    index_path.write_text(json.dumps(index))

    with pytest.raises(EvidenceStoreError, match="non-canonical record index key"):
        EvidenceStore.load(root)





def test_load_dataset_identity_mismatch_raises(tmp_path: Path) -> None:

    entries = [
        make_entry("sga", dataset_ref="burgers_tiny", seed=0),
        make_entry("sga", dataset_ref="burgers_tiny", seed=1),
    ]
    plan = make_plan(entries)
    root = tmp_path / "s"
    store = EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    store.add_outcome(
        completed_outcome(
            0, entries[0], make_record("sga", seed=0, dataset_cache_fingerprint="fp:A")
        )
    )
    store.add_outcome(
        completed_outcome(
            1, entries[1], make_record("sga", seed=1, dataset_cache_fingerprint="fp:B")
        )
    )

    with pytest.raises(EvidenceStoreError, match="dataset identity mismatch"):
        EvidenceStore.load(root)





def test_load_attempt_status_flip_to_completed_raises(tmp_path: Path) -> None:
    plan, entry = _one_entry_plan()
    root = tmp_path / "s"
    store = EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    store.add_outcome(raised_outcome(0, entry))

    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    index["attempts"][0]["status"] = "completed"
    index_path.write_text(json.dumps(index))

    with pytest.raises(EvidenceStoreError, match="incoherence"):
        EvidenceStore.load(root)


def test_load_completed_attempt_missing_record_raises(tmp_path: Path) -> None:
    plan, entry = _one_entry_plan()
    root = tmp_path / "s"
    store = EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    store.add_outcome(completed_outcome(0, entry, make_record("sga", seed=0)))


    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    del index["records"]["0"]
    index_path.write_text(json.dumps(index))
    (root / "records" / "entry-0000.json").unlink()

    with pytest.raises(EvidenceStoreError, match="incoherence"):
        EvidenceStore.load(root)


def test_load_duplicate_attempt_raises(tmp_path: Path) -> None:
    plan, entry = _one_entry_plan()
    root = tmp_path / "s"
    store = EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    store.add_outcome(completed_outcome(0, entry, make_record("sga", seed=0)))

    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    index["attempts"].append(dict(index["attempts"][0]))
    index_path.write_text(json.dumps(index))

    with pytest.raises(EvidenceStoreError, match="duplicate attempt"):
        EvidenceStore.load(root)


def test_load_attempt_plan_instrument_mismatch_raises(tmp_path: Path) -> None:
    plan, entry = _one_entry_plan()
    root = tmp_path / "s"
    store = EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    store.add_outcome(completed_outcome(0, entry, make_record("sga", seed=0)))

    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    index["attempts"][0]["instrument"] = "pysindy"
    index_path.write_text(json.dumps(index))

    with pytest.raises(EvidenceStoreError, match="instrument mismatch"):
        EvidenceStore.load(root)


def test_load_non_dict_attempt_raises(tmp_path: Path) -> None:


    plan, entry = _one_entry_plan()
    root = tmp_path / "s"
    store = EvidenceStore.create(root, plan=plan, env=environment_fingerprint())
    store.add_outcome(completed_outcome(0, entry, make_record("sga", seed=0)))

    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    index["attempts"] = [42]
    index_path.write_text(json.dumps(index))

    with pytest.raises(EvidenceStoreError, match="must be a JSON object"):
        EvidenceStore.load(root)





def test_evidence_store_error_exported_from_package() -> None:
    import kd.harness

    assert "EvidenceStoreError" in kd.harness.__all__
    assert kd.harness.EvidenceStoreError is EvidenceStoreError
