
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kd.search.checkpoint_manifest import (
    CKPTMAN_SCHEMA_VERSION,
    FINAL_STATUS_COMPLETED,
    KIND_FINAL,
    KIND_PERIODIC,
    MANIFEST_FILENAME,
    CheckpointManifestEntry,
    CheckpointManifestError,
    CheckpointManifestWriter,
    load_checkpoint_manifest,
)

_VALID_HASH = "a" * 64


def _entry(**overrides: object) -> CheckpointManifestEntry:
    base: dict[str, object] = {
        "filename": "checkpoint_000000.pt",
        "kind": KIND_PERIODIC,
        "final_status": None,
        "iteration": 0,
        "best_score": 0.5,
        "best_expression": "u_t = -u",
        "created_at": "2026-07-22T00:00:00+00:00",
        "algorithm": "sga",
        "seed": 7,
        "config_hash": _VALID_HASH,
        "library_fingerprint": None,
        "kd_version": "0.4.0",
    }
    base.update(overrides)
    return CheckpointManifestEntry(**base)


def _final(**overrides: object) -> CheckpointManifestEntry:
    return _entry(
        **{
            "filename": "checkpoint_final.pt",
            "kind": KIND_FINAL,
            "final_status": FINAL_STATUS_COMPLETED,
            **overrides,
        }
    )


def _valid_dir(directory: Path) -> CheckpointManifestWriter:
    writer = CheckpointManifestWriter.create(directory)
    (directory / "checkpoint_000000.pt").write_text("periodic")
    writer.append(_entry())
    (directory / "checkpoint_final.pt").write_text("final")
    writer.append(_final())
    return writer


def _write_raw_manifest(directory: Path, payload: object) -> None:
    (directory / MANIFEST_FILENAME).write_text(json.dumps(payload))







def test_load_returns_entries_in_append_order(tmp_path: Path) -> None:
    _valid_dir(tmp_path)
    entries = load_checkpoint_manifest(tmp_path)
    assert [e.filename for e in entries] == [
        "checkpoint_000000.pt",
        "checkpoint_final.pt",
    ]


def test_load_zero_entry_ledger(tmp_path: Path) -> None:
    CheckpointManifestWriter.create(tmp_path)
    assert load_checkpoint_manifest(tmp_path) == ()


def test_load_accepts_str_directory(tmp_path: Path) -> None:
    _valid_dir(tmp_path)
    assert len(load_checkpoint_manifest(str(tmp_path))) == 2







def test_load_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(CheckpointManifestError, match="not a directory"):
        load_checkpoint_manifest(tmp_path / "nope")


def test_load_path_is_file(tmp_path: Path) -> None:
    afile = tmp_path / "afile"
    afile.write_text("x")
    with pytest.raises(CheckpointManifestError, match="not a directory"):
        load_checkpoint_manifest(afile)


def test_load_dir_without_manifest(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    with pytest.raises(CheckpointManifestError, match="no checkpoint manifest"):
        load_checkpoint_manifest(tmp_path / "sub")







def test_load_garbage_json(tmp_path: Path) -> None:
    (tmp_path / MANIFEST_FILENAME).write_text("{not json")
    with pytest.raises(CheckpointManifestError, match="invalid JSON"):
        load_checkpoint_manifest(tmp_path)


def test_load_invalid_utf8(tmp_path: Path) -> None:
    (tmp_path / MANIFEST_FILENAME).write_bytes(b"\xff\xfe{}")
    with pytest.raises(CheckpointManifestError, match="invalid JSON"):
        load_checkpoint_manifest(tmp_path)


def test_load_top_level_list(tmp_path: Path) -> None:
    _write_raw_manifest(tmp_path, [1, 2, 3])
    with pytest.raises(CheckpointManifestError, match="manifest payload"):
        load_checkpoint_manifest(tmp_path)







def _header(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "scheme": "kd-ckptman-v1",
        "schema_version": CKPTMAN_SCHEMA_VERSION,
        "entries": [],
        "lineage": None,
    }
    base.update(overrides)
    return base


def test_load_header_extra_key(tmp_path: Path) -> None:
    _write_raw_manifest(tmp_path, {**_header(), "extra": 1})
    with pytest.raises(CheckpointManifestError, match="Unknown"):
        load_checkpoint_manifest(tmp_path)


def test_load_header_missing_key(tmp_path: Path) -> None:
    payload = _header()
    del payload["entries"]
    _write_raw_manifest(tmp_path, payload)
    with pytest.raises(CheckpointManifestError, match="Missing required"):
        load_checkpoint_manifest(tmp_path)


def test_load_wrong_scheme(tmp_path: Path) -> None:
    _write_raw_manifest(tmp_path, _header(scheme="kd-ckptman-v2"))
    with pytest.raises(CheckpointManifestError, match="scheme"):
        load_checkpoint_manifest(tmp_path)


def test_load_wrong_schema_version(tmp_path: Path) -> None:
    _write_raw_manifest(tmp_path, _header(schema_version=3))
    with pytest.raises(CheckpointManifestError, match="schema_version"):
        load_checkpoint_manifest(tmp_path)


def test_load_v1_header_without_lineage(tmp_path: Path) -> None:
    payload = _header(schema_version=1)
    del payload["lineage"]
    _write_raw_manifest(tmp_path, payload)
    assert load_checkpoint_manifest(tmp_path) == ()


def test_load_v2_rejects_malformed_lineage(tmp_path: Path) -> None:
    _write_raw_manifest(tmp_path, _header(lineage={"resume_from": ""}))
    with pytest.raises(CheckpointManifestError, match="lineage"):
        load_checkpoint_manifest(tmp_path)


def test_load_bool_schema_version(tmp_path: Path) -> None:
    _write_raw_manifest(tmp_path, _header(schema_version=True))
    with pytest.raises(CheckpointManifestError, match="schema_version"):
        load_checkpoint_manifest(tmp_path)


def test_load_entries_not_list(tmp_path: Path) -> None:
    _write_raw_manifest(tmp_path, _header(entries={"a": 1}))
    with pytest.raises(CheckpointManifestError, match="entries"):
        load_checkpoint_manifest(tmp_path)


def test_load_entry_field_violation_carries_index(tmp_path: Path) -> None:
    bad = _entry().to_dict()
    bad["iteration"] = -5
    _write_raw_manifest(tmp_path, _header(entries=[bad]))
    (tmp_path / "checkpoint_000000.pt").write_text("x")
    with pytest.raises(CheckpointManifestError, match="iteration"):
        load_checkpoint_manifest(tmp_path)







def test_load_ledger_names_deleted_file(tmp_path: Path) -> None:
    _valid_dir(tmp_path)
    (tmp_path / "checkpoint_000000.pt").unlink()
    with pytest.raises(CheckpointManifestError, match="missing checkpoint file"):
        load_checkpoint_manifest(tmp_path)







def test_load_orphan_pt(tmp_path: Path) -> None:
    _valid_dir(tmp_path)
    (tmp_path / "extra.pt").write_text("stray")
    with pytest.raises(CheckpointManifestError, match="orphan"):
        load_checkpoint_manifest(tmp_path)


def test_load_orphan_pt_tmp(tmp_path: Path) -> None:
    _valid_dir(tmp_path)
    (tmp_path / "checkpoint_000003.pt.tmp").write_text("torn")
    with pytest.raises(CheckpointManifestError, match="orphan"):
        load_checkpoint_manifest(tmp_path)


def test_load_orphan_manifest_tmp(tmp_path: Path) -> None:
    _valid_dir(tmp_path)
    (tmp_path / f"{MANIFEST_FILENAME}.tmp").write_text("torn")
    with pytest.raises(CheckpointManifestError, match="orphan"):
        load_checkpoint_manifest(tmp_path)


def test_load_ignores_subdirectory(tmp_path: Path) -> None:
    _valid_dir(tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "run.log").write_text("noise")
    entries = load_checkpoint_manifest(tmp_path)
    assert len(entries) == 2







def test_load_duplicate_filename(tmp_path: Path) -> None:
    dup = _entry().to_dict()
    _write_raw_manifest(tmp_path, _header(entries=[dup, dict(dup)]))
    (tmp_path / "checkpoint_000000.pt").write_text("x")
    with pytest.raises(CheckpointManifestError, match="duplicate"):
        load_checkpoint_manifest(tmp_path)


def test_load_two_final_entries(tmp_path: Path) -> None:
    f1 = _final().to_dict()
    f2 = _final(filename="checkpoint_final2.pt").to_dict()
    _write_raw_manifest(tmp_path, _header(entries=[f1, f2]))
    (tmp_path / "checkpoint_final.pt").write_text("x")
    (tmp_path / "checkpoint_final2.pt").write_text("x")
    with pytest.raises(CheckpointManifestError, match="final entry"):
        load_checkpoint_manifest(tmp_path)
