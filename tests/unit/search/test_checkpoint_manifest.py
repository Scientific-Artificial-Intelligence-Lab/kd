
from __future__ import annotations

import dataclasses
import json
import math
from datetime import datetime

import pytest

from kd.search.checkpoint_manifest import (
    CKPTMAN_CONFIG_HASH_SCHEME,
    CKPTMAN_SCHEMA_VERSION,
    CKPTMAN_SCHEME,
    FINAL_STATUS_COMPLETED,
    FINAL_STATUS_CRASHED,
    KIND_FINAL,
    KIND_PERIODIC,
    MANIFEST_FILENAME,
    CheckpointManifestEntry,
    CheckpointManifestError,
    CheckpointManifestWriter,
    build_manifest_entry,
    config_hash_of_snapshot,
    load_checkpoint_manifest,
)

_VALID_HASH = "a" * 64


def _periodic_entry(**overrides: object) -> CheckpointManifestEntry:
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


def _final_entry(**overrides: object) -> CheckpointManifestEntry:
    return _periodic_entry(
        **{
            "filename": "checkpoint_final.pt",
            "kind": KIND_FINAL,
            "final_status": FINAL_STATUS_COMPLETED,
            **overrides,
        }
    )


def _payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "version": 1,
        "iteration": 3,
        "algorithm_state": {"x": 1},
        "best_score": 0.25,
        "best_expression": "u_t = u_xx",
        "algorithm": "sga",
        "config": {"algorithm": "sga", "seed": 11},
        "config_canon_scheme": "kd-config-v1",
    }
    base.update(overrides)
    return base







def test_scheme_constants() -> None:
    assert CKPTMAN_SCHEME == "kd-ckptman-v1"

    assert CKPTMAN_SCHEMA_VERSION == 2
    assert MANIFEST_FILENAME == "manifest.json"
    assert CKPTMAN_CONFIG_HASH_SCHEME == "kd-confighash-v1"


def test_frozen_key_table_equals_dataclass_fields() -> None:
    from kd.search._checkpoint_manifest_verify import _CKPTMAN_V1_FIELDS

    assert set(_CKPTMAN_V1_FIELDS) == {
        f.name for f in dataclasses.fields(CheckpointManifestEntry)
    }







def test_round_trip_periodic() -> None:
    entry = _periodic_entry()
    assert CheckpointManifestEntry.from_dict(entry.to_dict()) == entry


def test_round_trip_final() -> None:
    entry = _final_entry(final_status=FINAL_STATUS_CRASHED)
    assert CheckpointManifestEntry.from_dict(entry.to_dict()) == entry


def test_to_dict_key_order_is_frozen_table() -> None:
    from kd.search._checkpoint_manifest_verify import _CKPTMAN_V1_FIELDS

    assert tuple(_periodic_entry().to_dict()) == _CKPTMAN_V1_FIELDS







def test_decode_rejects_non_dict_payload() -> None:
    with pytest.raises(CheckpointManifestError, match="dict"):
        CheckpointManifestEntry.from_dict(["not", "a", "dict"])


def test_decode_rejects_unknown_key() -> None:
    data = _periodic_entry().to_dict()
    data["surprise"] = 1
    with pytest.raises(CheckpointManifestError, match="Unknown"):
        CheckpointManifestEntry.from_dict(data)


def test_decode_rejects_missing_key() -> None:
    data = _periodic_entry().to_dict()
    del data["seed"]
    with pytest.raises(CheckpointManifestError, match="Missing required"):
        CheckpointManifestEntry.from_dict(data)


def test_decode_rejects_bad_kind() -> None:
    with pytest.raises(CheckpointManifestError, match="kind"):
        _periodic_entry(kind="bogus")


def test_decode_rejects_unhashable_kind() -> None:
    with pytest.raises(CheckpointManifestError, match="kind"):
        _periodic_entry(kind=["periodic"])


def test_decode_rejects_unhashable_final_status() -> None:
    with pytest.raises(CheckpointManifestError, match="final_status"):
        _final_entry(final_status=["completed"])


def test_decode_rejects_final_without_status() -> None:
    with pytest.raises(CheckpointManifestError, match="final_status"):
        _final_entry(final_status=None)


def test_decode_rejects_periodic_with_status() -> None:
    with pytest.raises(CheckpointManifestError, match="final_status"):
        _periodic_entry(final_status=FINAL_STATUS_COMPLETED)


def test_decode_rejects_wrong_final_status_vocab() -> None:
    with pytest.raises(CheckpointManifestError, match="final_status"):
        _final_entry(final_status="raised")


def test_decode_rejects_bool_iteration() -> None:
    with pytest.raises(CheckpointManifestError, match="iteration"):
        _periodic_entry(iteration=True)


def test_decode_rejects_negative_iteration() -> None:
    with pytest.raises(CheckpointManifestError, match="iteration"):
        _periodic_entry(iteration=-1)


def test_decode_rejects_non_finite_best_score() -> None:
    with pytest.raises(CheckpointManifestError, match="best_score"):
        _periodic_entry(best_score=float("inf"))


def test_decode_rejects_bool_best_score() -> None:
    with pytest.raises(CheckpointManifestError, match="best_score"):
        _periodic_entry(best_score=True)


def test_decode_rejects_empty_best_expression() -> None:
    with pytest.raises(CheckpointManifestError, match="best_expression"):
        _periodic_entry(best_expression="")


def test_decode_rejects_bool_seed() -> None:
    with pytest.raises(CheckpointManifestError, match="seed"):
        _periodic_entry(seed=True)


@pytest.mark.parametrize(
    "bad_hash",
    ["a" * 63, "a" * 65, "A" * 64, "g" * 64],
)
def test_decode_rejects_bad_config_hash(bad_hash: str) -> None:
    with pytest.raises(CheckpointManifestError, match="config_hash"):
        _periodic_entry(config_hash=bad_hash)


def test_decode_rejects_empty_kd_version() -> None:
    with pytest.raises(CheckpointManifestError, match="kd_version"):
        _periodic_entry(kd_version="")


@pytest.mark.parametrize(
    "bad_name",
    ["sub/checkpoint.pt", "../escape.pt", "a/b.pt"],
)
def test_decode_rejects_filename_with_separator(bad_name: str) -> None:
    with pytest.raises(CheckpointManifestError, match="filename"):
        _periodic_entry(filename=bad_name)


def test_decode_rejects_manifest_filename() -> None:
    with pytest.raises(CheckpointManifestError, match="filename"):
        _periodic_entry(filename=MANIFEST_FILENAME)


def test_decode_rejects_tmp_filename() -> None:
    with pytest.raises(CheckpointManifestError, match="filename"):
        _periodic_entry(filename="checkpoint_000000.pt.tmp")


def test_decode_rejects_dotdot_filename() -> None:


    with pytest.raises(CheckpointManifestError, match="filename"):
        _periodic_entry(filename="..")







def test_builder_degrades_inf_score_to_none() -> None:
    entry = build_manifest_entry(
        _payload(best_score=float("inf")),
        filename="checkpoint_000003.pt",
        kind=KIND_PERIODIC,
        final_status=None,
    )
    assert entry.best_score is None


def test_builder_degrades_nan_score_to_none() -> None:
    entry = build_manifest_entry(
        _payload(best_score=float("nan")),
        filename="checkpoint_000003.pt",
        kind=KIND_PERIODIC,
        final_status=None,
    )
    assert entry.best_score is None


def test_builder_accepts_finite_int_score_as_float() -> None:
    entry = build_manifest_entry(
        _payload(best_score=3),
        filename="checkpoint_000003.pt",
        kind=KIND_PERIODIC,
        final_status=None,
    )
    assert entry.best_score == 3.0
    assert isinstance(entry.best_score, float)


def test_builder_empty_best_expression_to_none() -> None:
    entry = build_manifest_entry(
        _payload(best_expression=""),
        filename="checkpoint_000003.pt",
        kind=KIND_PERIODIC,
        final_status=None,
    )
    assert entry.best_expression is None


def test_builder_degraded_writer_all_none() -> None:
    entry = build_manifest_entry(
        _payload(algorithm=None, config=None),
        filename="checkpoint_000003.pt",
        kind=KIND_PERIODIC,
        final_status=None,
    )
    assert entry.algorithm is None
    assert entry.seed is None
    assert entry.config_hash is None
    assert entry.library_fingerprint is None


def test_builder_seed_zero_preserved() -> None:
    entry = build_manifest_entry(
        _payload(config={"algorithm": "sga", "seed": 0}),
        filename="checkpoint_000003.pt",
        kind=KIND_PERIODIC,
        final_status=None,
    )
    assert entry.seed == 0


def test_builder_library_fingerprint_absent_is_none() -> None:
    entry = build_manifest_entry(
        _payload(config={"algorithm": "pysr", "seed": 1}),
        filename="checkpoint_000003.pt",
        kind=KIND_PERIODIC,
        final_status=None,
    )
    assert entry.library_fingerprint is None


def test_builder_library_fingerprint_carried() -> None:
    entry = build_manifest_entry(
        _payload(
            config={"algorithm": "pysr", "seed": 1, "library_fingerprint": "pysr@1"}
        ),
        filename="checkpoint_000003.pt",
        kind=KIND_PERIODIC,
        final_status=None,
    )
    assert entry.library_fingerprint == "pysr@1"


def test_builder_final_status_wired() -> None:
    entry = build_manifest_entry(
        _payload(),
        filename="checkpoint_final.pt",
        kind=KIND_FINAL,
        final_status=FINAL_STATUS_CRASHED,
    )
    assert entry.kind == KIND_FINAL
    assert entry.final_status == FINAL_STATUS_CRASHED







def test_created_at_is_utc_iso8601() -> None:
    entry = build_manifest_entry(
        _payload(),
        filename="checkpoint_000003.pt",
        kind=KIND_PERIODIC,
        final_status=None,
    )
    parsed = datetime.fromisoformat(entry.created_at)
    assert parsed.utcoffset() is not None
    assert parsed.utcoffset().total_seconds() == 0










_GOLDEN_SNAPSHOT = {"algorithm": "sga", "seed": 7}
_GOLDEN_HASH = "636ab5e3e5bdbf635c087cbb7725db63c06df6fd64782e75f70cfed093a410e2"


def test_config_hash_is_64_hex() -> None:
    digest = config_hash_of_snapshot(_GOLDEN_SNAPSHOT)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_config_hash_golden_pin() -> None:
    assert config_hash_of_snapshot(_GOLDEN_SNAPSHOT) == _GOLDEN_HASH


def test_config_hash_pairwise_distinct() -> None:
    a = config_hash_of_snapshot({"algorithm": "sga", "seed": 7})
    b = config_hash_of_snapshot({"algorithm": "sga", "seed": 8})
    assert a != b


def test_config_hash_key_order_invariant() -> None:
    a = config_hash_of_snapshot({"algorithm": "sga", "seed": 7})
    b = config_hash_of_snapshot({"seed": 7, "algorithm": "sga"})
    assert a == b


def test_builder_config_hash_matches_direct_hasher() -> None:
    payload = _payload()
    entry = build_manifest_entry(
        payload,
        filename="checkpoint_000003.pt",
        kind=KIND_PERIODIC,
        final_status=None,
    )
    assert entry.config_hash == config_hash_of_snapshot(payload["config"])


def test_builder_config_hash_matches_real_payload_snapshot() -> None:
    from kd.search.checkpoint_payload import build_checkpoint_payload
    from kd.search.sga import SGAConfig, SGAPlugin

    payload = build_checkpoint_payload(2, SGAPlugin(SGAConfig()))
    entry = build_manifest_entry(
        payload,
        filename="checkpoint_000002.pt",
        kind=KIND_PERIODIC,
        final_status=None,
    )
    assert entry.config_hash == config_hash_of_snapshot(payload["config"])







def test_full_manifest_round_trip(tmp_path: object) -> None:
    from pathlib import Path

    directory = Path(tmp_path)
    writer = CheckpointManifestWriter.create(directory)
    writer.append(_periodic_entry())
    writer.append(_final_entry())

    raw = json.loads((directory / MANIFEST_FILENAME).read_text())
    assert set(raw) == {"scheme", "schema_version", "entries", "lineage"}
    assert raw["lineage"] is None
    assert raw["scheme"] == CKPTMAN_SCHEME
    assert raw["schema_version"] == CKPTMAN_SCHEMA_VERSION

    assert [e["filename"] for e in raw["entries"]] == [
        "checkpoint_000000.pt",
        "checkpoint_final.pt",
    ]
    decoded = [CheckpointManifestEntry.from_dict(e) for e in raw["entries"]]
    assert decoded == [_periodic_entry(), _final_entry()]







def test_writer_create_seals_lineage_into_header(tmp_path: object) -> None:
    from pathlib import Path

    directory = Path(tmp_path) / "ckpt"
    lineage = {
        "resume_from": "/runs/a/checkpoints/checkpoint_final.pt",
        "source_run_id": None,
        "source_config_hash": "0" * 64,
        "source_final_status": "completed",
        "source_iteration": 3,
    }
    CheckpointManifestWriter.create(directory, lineage=lineage)
    raw = json.loads((directory / MANIFEST_FILENAME).read_text())
    assert raw["lineage"] == lineage
    assert load_checkpoint_manifest(directory) == ()


def test_writer_create_rejects_malformed_lineage(tmp_path: object) -> None:
    from pathlib import Path

    directory = Path(tmp_path) / "ckpt"
    with pytest.raises(CheckpointManifestError, match="lineage"):
        CheckpointManifestWriter.create(directory, lineage={"resume_from": ""})


def test_writer_create_fresh_dir_writes_empty_ledger(tmp_path: object) -> None:
    from pathlib import Path

    directory = Path(tmp_path) / "ckpt"
    writer = CheckpointManifestWriter.create(directory)
    manifest = directory / MANIFEST_FILENAME
    assert manifest.is_file()
    raw = json.loads(manifest.read_text())
    assert raw["entries"] == []
    assert writer.entries == ()


def test_writer_create_accepts_existing_empty_dir(tmp_path: object) -> None:
    from pathlib import Path

    directory = Path(tmp_path) / "empty"
    directory.mkdir()
    writer = CheckpointManifestWriter.create(directory)
    assert (directory / MANIFEST_FILENAME).is_file()
    assert writer.entries == ()


def test_writer_create_refuses_non_empty_dir(tmp_path: object) -> None:
    from pathlib import Path

    directory = Path(tmp_path) / "full"
    directory.mkdir()
    (directory / "leftover.pt").write_text("x")
    with pytest.raises(CheckpointManifestError, match="is not empty"):
        CheckpointManifestWriter.create(directory)


def test_writer_create_refuses_file_as_path(tmp_path: object) -> None:
    from pathlib import Path

    path = Path(tmp_path) / "afile"
    path.write_text("x")
    with pytest.raises(CheckpointManifestError, match="not a directory"):
        CheckpointManifestWriter.create(path)


def test_writer_append_refuses_duplicate_filename(tmp_path: object) -> None:
    from pathlib import Path

    writer = CheckpointManifestWriter.create(Path(tmp_path))
    writer.append(_periodic_entry())
    with pytest.raises(CheckpointManifestError, match="already listed"):
        writer.append(_periodic_entry())


def test_writer_append_refuses_second_final(tmp_path: object) -> None:
    from pathlib import Path

    writer = CheckpointManifestWriter.create(Path(tmp_path))
    writer.append(_final_entry())
    with pytest.raises(CheckpointManifestError, match="final entry"):
        writer.append(_final_entry(filename="checkpoint_final2.pt"))


def test_writer_append_leaves_no_tmp_residue(tmp_path: object) -> None:
    from pathlib import Path

    directory = Path(tmp_path)
    writer = CheckpointManifestWriter.create(directory)
    writer.append(_periodic_entry())
    assert not list(directory.glob("*.tmp"))


def test_writer_entries_property_reflects_append_order(tmp_path: object) -> None:
    from pathlib import Path

    writer = CheckpointManifestWriter.create(Path(tmp_path))
    e0 = _periodic_entry(filename="checkpoint_000000.pt", iteration=0)
    e1 = _periodic_entry(filename="checkpoint_000001.pt", iteration=1)
    writer.append(e0)
    writer.append(e1)
    assert writer.entries == (e0, e1)







def test_builder_degrades_non_json_safe_overlay_to_none() -> None:
    payload = _payload(config={"algorithm": "sga", "seed": 1, "bad": object()})
    entry = build_manifest_entry(
        payload,
        filename="checkpoint_000003.pt",
        kind=KIND_PERIODIC,
        final_status=None,
    )
    assert entry.config_hash is None


def test_builder_degrades_non_finite_overlay_to_none() -> None:
    payload = _payload(config={"algorithm": "sga", "seed": 1, "bad": math.inf})
    entry = build_manifest_entry(
        payload,
        filename="checkpoint_000003.pt",
        kind=KIND_PERIODIC,
        final_status=None,
    )
    assert entry.config_hash is None


def test_raw_hasher_raises_on_non_json_safe_input() -> None:
    with pytest.raises((TypeError, ValueError)):
        config_hash_of_snapshot({"algorithm": "sga", "bad": object()})


def test_raw_hasher_raises_on_non_finite_input() -> None:
    with pytest.raises((TypeError, ValueError)):
        config_hash_of_snapshot({"algorithm": "sga", "bad": math.inf})
