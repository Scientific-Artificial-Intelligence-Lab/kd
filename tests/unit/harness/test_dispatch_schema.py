
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from kd.harness.dispatch import (
    _DISPATCH_V1_DATASET_KEYS,
    _DISPATCH_V1_KEYS,
    _DISPATCH_V1_RESOURCES_KEYS,
    _DISPATCH_V1_SHARD_KEYS,
    DISPATCH_ARTIFACT_TAG,
    DISPATCH_SCHEMA_VERSION,
    DispatchDatasetSpec,
    DispatchManifest,
    DispatchManifestError,
    DispatchRecording,
    DispatchResources,
    ShardSpec,
    read_dispatch_manifest,
    write_dispatch_manifest,
)
from kd.harness.plan import ExperimentPlan, PlanEntry
from kd.harness.recording import RecordingOptions






def _plan() -> ExperimentPlan:
    return ExperimentPlan(
        name="dispatch-codec",
        entries=(
            PlanEntry(
                instrument="sga",
                dataset_ref="burgers_tiny",
                seed=0,
                model_kwargs={"generations": 2},
            ),
            PlanEntry(
                instrument="discover",
                dataset_ref="kdv_tiny",
                seed=1,
                model_kwargs={},
            ),
        ),
    )


def _valid_payload(plan: ExperimentPlan) -> dict[str, Any]:
    return {
        "artifact": "kd-dispatch-v1",
        "dispatch_schema_version": 1,
        "plan": plan.to_dict(),
        "plan_hash": plan.plan_hash(),
        "shards": [
            {
                "shard_id": "shard-00",
                "entry_indices": [0],
                "cuda_visible_devices": "",
                "device": None,
                "heavy": False,
                "timeout_seconds": None,
                "memory_max_gb": None,
            },
            {
                "shard_id": "shard-01",
                "entry_indices": [1],
                "cuda_visible_devices": "0",
                "device": "cuda",
                "heavy": True,
                "timeout_seconds": 10.0,
                "memory_max_gb": None,
            },
        ],
        "datasets": {
            "burgers_tiny": {"loader": "kd.generate_burgers_data", "kwargs": {}},
            "kdv_tiny": {"loader": "kd.load_kdv", "kwargs": {}},
        },
        "resources": {"max_concurrent_heavy": 1, "grace_seconds": 30.0},
    }


def _manifest(
    plan: ExperimentPlan, recording: DispatchRecording | None = None
) -> DispatchManifest:
    return DispatchManifest(
        plan=plan,
        plan_hash=plan.plan_hash(),
        recording=recording,
        shards=(
            ShardSpec(
                shard_id="shard-00",
                entry_indices=(0,),
                cuda_visible_devices="",
                device=None,
                heavy=False,
                timeout_seconds=None,
                memory_max_gb=None,
            ),
            ShardSpec(
                shard_id="shard-01",
                entry_indices=(1,),
                cuda_visible_devices="0",
                device="cuda",
                heavy=True,
                timeout_seconds=10.0,
                memory_max_gb=None,
            ),
        ),
        datasets={
            "burgers_tiny": DispatchDatasetSpec(loader="kd.generate_burgers_data", kwargs={}),
            "kdv_tiny": DispatchDatasetSpec(loader="kd.load_kdv", kwargs={}),
        },
        resources=DispatchResources(max_concurrent_heavy=1, grace_seconds=30.0),
    )


def _write_payload(tmp_path: Path, payload: Any) -> Path:
    path = tmp_path / "dispatch.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _with_recording_resume(
    payload: dict[str, Any], resume_from: dict[str, str]
) -> dict[str, Any]:
    payload["dispatch_schema_version"] = 2
    payload["recording"] = {
        "events_every_n": 1,
        "checkpoint_every": None,
        "checkpoint_keep_last": None,
        "phases": True,
        "catalog": None,
        "resume_from": resume_from,
    }
    return payload





def test_tag_and_version_constants_are_literal() -> None:
    assert DISPATCH_ARTIFACT_TAG == "kd-dispatch-v1"

    assert DISPATCH_SCHEMA_VERSION == 2





def test_frozen_key_tables_are_the_literal_scheme() -> None:
    assert frozenset(
        {
            "artifact",
            "dispatch_schema_version",
            "plan",
            "plan_hash",
            "shards",
            "datasets",
            "resources",
        }
    ) == _DISPATCH_V1_KEYS
    assert frozenset(
        {
            "shard_id",
            "entry_indices",
            "cuda_visible_devices",
            "device",
            "heavy",
            "timeout_seconds",
            "memory_max_gb",
        }
    ) == _DISPATCH_V1_SHARD_KEYS
    assert frozenset({"loader", "kwargs"}) == _DISPATCH_V1_DATASET_KEYS
    assert frozenset(
        {"max_concurrent_heavy", "grace_seconds"}
    ) == _DISPATCH_V1_RESOURCES_KEYS





def test_serialized_tree_key_sets_match_frozen_tables(tmp_path: Path) -> None:
    written = write_dispatch_manifest(_manifest(_plan()), tmp_path / "batch")
    payload = json.loads(Path(written).read_text(encoding="utf-8"))

    assert set(payload) == set(_DISPATCH_V1_KEYS) | {"recording"}
    assert payload["recording"] is None
    for shard in payload["shards"]:
        assert set(shard) == set(_DISPATCH_V1_SHARD_KEYS)
    for spec in payload["datasets"].values():
        assert set(spec) == set(_DISPATCH_V1_DATASET_KEYS)
    assert set(payload["resources"]) == set(_DISPATCH_V1_RESOURCES_KEYS)





def test_write_read_roundtrip_is_equal(tmp_path: Path) -> None:
    manifest = _manifest(_plan())
    written = write_dispatch_manifest(manifest, tmp_path / "batch")
    assert Path(written).name == "dispatch.json"
    assert read_dispatch_manifest(written) == manifest


def test_write_creates_shards_and_logs_dirs(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    write_dispatch_manifest(_manifest(_plan()), batch)
    assert (batch / "shards").is_dir()
    assert (batch / "logs").is_dir()





def test_manifest_carries_no_abs_path_and_no_timestamp(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    written = write_dispatch_manifest(_manifest(_plan()), batch)
    text = Path(written).read_text(encoding="utf-8")


    assert str(batch) not in text
    assert str(batch.resolve()) not in text

    assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", text) is None
    assert "NaN" not in text
    assert "Infinity" not in text





def test_read_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_dispatch_manifest(tmp_path / "does_not_exist.json")


def test_read_rejects_non_json(tmp_path: Path) -> None:
    path = tmp_path / "dispatch.json"
    path.write_text("not json {", encoding="utf-8")
    with pytest.raises(DispatchManifestError):
        read_dispatch_manifest(path)


def test_read_rejects_non_object_top_level(tmp_path: Path) -> None:
    path = tmp_path / "dispatch.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(DispatchManifestError):
        read_dispatch_manifest(path)


def test_read_rejects_wrong_artifact_tag(tmp_path: Path) -> None:
    payload = _valid_payload(_plan())
    payload["artifact"] = "kd-dispatch-v2"
    with pytest.raises(DispatchManifestError, match="artifact"):
        read_dispatch_manifest(_write_payload(tmp_path, payload))


def test_read_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    payload = _valid_payload(_plan())
    payload["extra"] = 1
    with pytest.raises(DispatchManifestError, match="Unknown"):
        read_dispatch_manifest(_write_payload(tmp_path, payload))


def test_read_rejects_missing_top_level_key(tmp_path: Path) -> None:
    payload = _valid_payload(_plan())
    del payload["resources"]
    with pytest.raises(DispatchManifestError, match="Missing"):
        read_dispatch_manifest(_write_payload(tmp_path, payload))


def test_read_rejects_unknown_shard_key(tmp_path: Path) -> None:
    payload = _valid_payload(_plan())
    payload["shards"][0]["extra"] = 1
    with pytest.raises(DispatchManifestError, match="Unknown"):
        read_dispatch_manifest(_write_payload(tmp_path, payload))


def test_read_rejects_missing_shard_key(tmp_path: Path) -> None:
    payload = _valid_payload(_plan())
    del payload["shards"][0]["heavy"]
    with pytest.raises(DispatchManifestError, match="Missing"):
        read_dispatch_manifest(_write_payload(tmp_path, payload))


def test_read_rejects_unknown_dataset_key(tmp_path: Path) -> None:
    payload = _valid_payload(_plan())
    payload["datasets"]["burgers_tiny"]["extra"] = 1
    with pytest.raises(DispatchManifestError, match="Unknown"):
        read_dispatch_manifest(_write_payload(tmp_path, payload))


def test_read_rejects_unknown_resources_key(tmp_path: Path) -> None:
    payload = _valid_payload(_plan())
    payload["resources"]["extra"] = 1
    with pytest.raises(DispatchManifestError, match="Unknown"):
        read_dispatch_manifest(_write_payload(tmp_path, payload))


@pytest.mark.parametrize("version", [True, 3, "1", 0, None])
def test_read_rejects_bad_schema_version(tmp_path: Path, version: object) -> None:
    payload = _valid_payload(_plan())
    payload["dispatch_schema_version"] = version
    with pytest.raises(DispatchManifestError, match="dispatch_schema_version"):
        read_dispatch_manifest(_write_payload(tmp_path, payload))


def test_read_rejects_nan_token(tmp_path: Path) -> None:


    payload = _valid_payload(_plan())
    path = tmp_path / "dispatch.json"
    raw = json.dumps(payload)
    raw = raw.replace('"grace_seconds": 30.0', '"grace_seconds": NaN')
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(DispatchManifestError):
        read_dispatch_manifest(path)


def test_read_rejects_plan_hash_mismatch(tmp_path: Path) -> None:
    payload = _valid_payload(_plan())
    payload["plan_hash"] = "0" * 64
    with pytest.raises(DispatchManifestError, match="plan_hash"):
        read_dispatch_manifest(_write_payload(tmp_path, payload))


def test_read_rejects_empty_plan_hash(tmp_path: Path) -> None:
    payload = _valid_payload(_plan())
    payload["plan_hash"] = ""
    with pytest.raises(DispatchManifestError, match="plan_hash"):
        read_dispatch_manifest(_write_payload(tmp_path, payload))


def test_read_rejects_datasets_missing_a_ref(tmp_path: Path) -> None:
    payload = _valid_payload(_plan())
    del payload["datasets"]["kdv_tiny"]
    with pytest.raises(DispatchManifestError, match="dataset"):
        read_dispatch_manifest(_write_payload(tmp_path, payload))


def test_read_rejects_datasets_extra_ref(tmp_path: Path) -> None:
    payload = _valid_payload(_plan())
    payload["datasets"]["ghost_ref"] = {"loader": "kd.load_kdv", "kwargs": {}}
    with pytest.raises(DispatchManifestError, match="dataset"):
        read_dispatch_manifest(_write_payload(tmp_path, payload))


def test_read_rejects_overlapping_entry_indices(tmp_path: Path) -> None:

    payload = _valid_payload(_plan())
    payload["shards"][1]["entry_indices"] = [0]
    with pytest.raises(DispatchManifestError, match="entry_indices"):
        read_dispatch_manifest(_write_payload(tmp_path, payload))


def test_read_rejects_entry_index_out_of_range(tmp_path: Path) -> None:
    payload = _valid_payload(_plan())
    payload["shards"][1]["entry_indices"] = [99]
    with pytest.raises(DispatchManifestError):
        read_dispatch_manifest(_write_payload(tmp_path, payload))


def test_read_rejects_non_strictly_increasing_entry_indices(tmp_path: Path) -> None:

    plan = ExperimentPlan(
        name="three",
        entries=tuple(
            PlanEntry(instrument="sga", dataset_ref=ref, seed=seed, model_kwargs={})
            for seed, ref in enumerate(("burgers_tiny", "kdv_tiny", "kdv_tiny"))
        ),
    )
    payload = {
        "artifact": "kd-dispatch-v1",
        "dispatch_schema_version": 1,
        "plan": plan.to_dict(),
        "plan_hash": plan.plan_hash(),
        "shards": [
            {
                "shard_id": "shard-00",
                "entry_indices": [2, 1, 0],
                "cuda_visible_devices": "",
                "device": None,
                "heavy": False,
                "timeout_seconds": None,
                "memory_max_gb": None,
            }
        ],
        "datasets": {
            "burgers_tiny": {"loader": "kd.generate_burgers_data", "kwargs": {}},
            "kdv_tiny": {"loader": "kd.load_kdv", "kwargs": {}},
        },
        "resources": {"max_concurrent_heavy": 1, "grace_seconds": 30.0},
    }
    with pytest.raises(DispatchManifestError, match="entry_indices"):
        read_dispatch_manifest(_write_payload(tmp_path, payload))


def test_read_rejects_bool_in_entry_indices(tmp_path: Path) -> None:
    payload = _valid_payload(_plan())
    payload["shards"][0]["entry_indices"] = [True]
    with pytest.raises(DispatchManifestError, match="entry_indices"):
        read_dispatch_manifest(_write_payload(tmp_path, payload))


def test_read_rejects_duplicate_shard_id(tmp_path: Path) -> None:
    payload = _valid_payload(_plan())
    payload["shards"][1]["shard_id"] = "shard-00"
    with pytest.raises(DispatchManifestError, match="shard_id"):
        read_dispatch_manifest(_write_payload(tmp_path, payload))


def test_read_rejects_malformed_shard_id(tmp_path: Path) -> None:
    payload = _valid_payload(_plan())
    payload["shards"][0]["shard_id"] = "worker0"
    with pytest.raises(DispatchManifestError, match="shard_id"):
        read_dispatch_manifest(_write_payload(tmp_path, payload))


def test_read_rejects_negative_timeout(tmp_path: Path) -> None:
    payload = _valid_payload(_plan())
    payload["shards"][0]["timeout_seconds"] = -1.0
    with pytest.raises(DispatchManifestError, match="timeout"):
        read_dispatch_manifest(_write_payload(tmp_path, payload))


def test_read_rejects_zero_max_concurrent_heavy(tmp_path: Path) -> None:
    payload = _valid_payload(_plan())
    payload["resources"]["max_concurrent_heavy"] = 0
    with pytest.raises(DispatchManifestError, match="max_concurrent_heavy"):
        read_dispatch_manifest(_write_payload(tmp_path, payload))





def test_dataclass_construction_reruns_validation() -> None:


    with pytest.raises(DispatchManifestError, match="shard_id"):
        ShardSpec(
            shard_id="nope",
            entry_indices=(0,),
            cuda_visible_devices="",
            device=None,
            heavy=False,
            timeout_seconds=None,
            memory_max_gb=None,
        )





def test_v1_payload_decodes_with_recording_none(tmp_path: Path) -> None:
    manifest = read_dispatch_manifest(
        _write_payload(tmp_path, _valid_payload(_plan()))
    )
    assert manifest.recording is None


def test_recording_block_roundtrips(tmp_path: Path) -> None:
    recording = DispatchRecording(
        options=RecordingOptions(events_every_n=2, checkpoint_every=5),
        catalog="catalog.jsonl",
        resume_from={1: "../prior/shards/shard-00/runs/entry-0001/checkpoints/x.pt"},
    )
    manifest = _manifest(_plan(), recording=recording)
    written = write_dispatch_manifest(manifest, tmp_path / "batch")
    reread = read_dispatch_manifest(written)
    assert reread == manifest
    assert reread.recording is not None
    assert reread.recording.resume_from == recording.resume_from


def test_recording_rejects_absolute_catalog_and_bad_resume_index() -> None:
    with pytest.raises(DispatchManifestError, match="catalog"):
        DispatchRecording(
            options=RecordingOptions(), catalog="/abs/catalog.jsonl"
        )
    plan = _plan()
    with pytest.raises(DispatchManifestError, match="resume_from"):
        _manifest(
            plan,
            recording=DispatchRecording(
                options=RecordingOptions(), resume_from={99: "x.pt"}
            ),
        )


def test_recording_rejects_resume_index_outside_shard_coverage() -> None:
    plan = _plan()
    base = _manifest(plan)
    with pytest.raises(DispatchManifestError, match="resume_from.*1"):
        DispatchManifest(
            plan=plan,
            plan_hash=plan.plan_hash(),
            shards=(base.shards[0],),
            datasets=base.datasets,
            resources=base.resources,
            recording=DispatchRecording(
                options=RecordingOptions(), resume_from={1: "x.pt"}
            ),
        )


def test_recording_rejects_malformed_resume_key(tmp_path: Path) -> None:
    payload = _with_recording_resume(_valid_payload(_plan()), {"+1": "x.pt"})
    with pytest.raises(DispatchManifestError, match="resume_from"):
        read_dispatch_manifest(_write_payload(tmp_path, payload))


def test_recording_rejects_aliasing_resume_keys(tmp_path: Path) -> None:
    payload = _with_recording_resume(
        _valid_payload(_plan()), {"1": "a.pt", "01": "b.pt"}
    )
    with pytest.raises(DispatchManifestError, match="duplicate"):
        read_dispatch_manifest(_write_payload(tmp_path, payload))
