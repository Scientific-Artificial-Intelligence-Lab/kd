
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from kd.harness.dispatch_log import (
    _DISPATCH_LOG_V1_KEYS,
    _DISPATCH_LOG_V1_WORKER_ENV_KEYS,
    _DISPATCH_LOG_V1_WORKER_KEYS,
    _KILL_REASON_VOCABULARY,
    DISPATCH_LOG_ARTIFACT_TAG,
    DISPATCH_LOG_SCHEMA_VERSION,
    DispatchLog,
    DispatchLogError,
    WorkerLogRow,
    read_dispatch_log,
    write_dispatch_log,
)

_ENV6 = {
    "CUDA_VISIBLE_DEVICES": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
}


def _row(shard_id: str, **overrides: Any) -> WorkerLogRow:
    kwargs: dict[str, Any] = {
        "shard_id": shard_id,
        "pid": 4242,
        "exit_code": 0,
        "kill_reason": None,
        "sigkill_used": False,
        "started_at": "2026-07-22T00:00:00+00:00",
        "finished_at": "2026-07-22T00:00:01+00:00",
        "wallclock_seconds": 1.0,
        "timeout_seconds": None,
        "env": dict(_ENV6),
        "heavy": False,
    }
    kwargs.update(overrides)
    return WorkerLogRow(**kwargs)


def _log(**overrides: Any) -> DispatchLog:
    kwargs: dict[str, Any] = {
        "plan_hash": "a" * 64,
        "workers": (_row("shard-00"), _row("shard-01", heavy=True)),
        "lost_entries": (),
        "heavy_peak": 1,
        "started_at": "2026-07-22T00:00:00+00:00",
        "finished_at": "2026-07-22T00:00:02+00:00",
    }
    kwargs.update(overrides)
    return DispatchLog(**kwargs)


def _valid_payload() -> dict[str, Any]:
    return {
        "artifact": "kd-dispatch-log-v1",
        "dispatch_log_schema_version": 1,
        "plan_hash": "a" * 64,
        "workers": [
            {
                "shard_id": "shard-00",
                "pid": 4242,
                "exit_code": 0,
                "kill_reason": None,
                "sigkill_used": False,
                "started_at": "2026-07-22T00:00:00+00:00",
                "finished_at": "2026-07-22T00:00:01+00:00",
                "wallclock_seconds": 1.0,
                "timeout_seconds": None,
                "env": dict(_ENV6),
                "heavy": False,
            }
        ],
        "lost_entries": [],
        "heavy_peak": 1,
        "started_at": "2026-07-22T00:00:00+00:00",
        "finished_at": "2026-07-22T00:00:02+00:00",
    }


def _write_payload(tmp_path: Path, payload: Any) -> Path:
    path = tmp_path / "dispatch-log.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path





def test_tag_version_and_frozen_tables() -> None:
    assert DISPATCH_LOG_ARTIFACT_TAG == "kd-dispatch-log-v1"
    assert DISPATCH_LOG_SCHEMA_VERSION == 1
    assert frozenset(
        {
            "artifact",
            "dispatch_log_schema_version",
            "plan_hash",
            "workers",
            "lost_entries",
            "heavy_peak",
            "started_at",
            "finished_at",
        }
    ) == _DISPATCH_LOG_V1_KEYS
    assert frozenset(
        {
            "shard_id",
            "pid",
            "exit_code",
            "kill_reason",
            "sigkill_used",
            "started_at",
            "finished_at",
            "wallclock_seconds",
            "timeout_seconds",
            "env",
            "heavy",
        }
    ) == _DISPATCH_LOG_V1_WORKER_KEYS
    assert frozenset(
        {
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "PYTORCH_NVML_BASED_CUDA_CHECK",
        }
    ) == _DISPATCH_LOG_V1_WORKER_ENV_KEYS
    assert frozenset(
        {"timeout", "spawn_failed", "dispatcher_interrupted"}
    ) == _KILL_REASON_VOCABULARY





def test_serialized_tree_key_sets_match_frozen_tables(tmp_path: Path) -> None:
    written = write_dispatch_log(_log(), tmp_path / "dispatch-log.json")
    payload = json.loads(Path(written).read_text(encoding="utf-8"))
    assert set(payload) == set(_DISPATCH_LOG_V1_KEYS)
    assert payload["workers"], "fixture must carry >=1 worker row"
    for worker in payload["workers"]:
        assert set(worker) == set(_DISPATCH_LOG_V1_WORKER_KEYS)
        assert set(worker["env"]) == set(_DISPATCH_LOG_V1_WORKER_ENV_KEYS)





def test_write_read_roundtrip_is_equal(tmp_path: Path) -> None:
    log = _log()
    written = write_dispatch_log(log, tmp_path / "dispatch-log.json")
    assert read_dispatch_log(written) == log





def test_worker_rows_serialized_in_numeric_shard_order(tmp_path: Path) -> None:


    scrambled = (
        _row("shard-100"),
        _row("shard-09"),
        _row("shard-99"),
        _row("shard-10"),
    )
    written = write_dispatch_log(
        _log(workers=scrambled), tmp_path / "dispatch-log.json"
    )
    payload = json.loads(Path(written).read_text(encoding="utf-8"))
    order = [int(w["shard_id"].split("-")[1]) for w in payload["workers"]]
    assert order == [9, 10, 99, 100]


def test_lost_entries_serialized_ascending(tmp_path: Path) -> None:
    written = write_dispatch_log(
        _log(lost_entries=(5, 1, 3)), tmp_path / "dispatch-log.json"
    )
    payload = json.loads(Path(written).read_text(encoding="utf-8"))
    assert payload["lost_entries"] == [1, 3, 5]





def test_read_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_dispatch_log(tmp_path / "nope.json")


def test_read_rejects_wrong_tag(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["artifact"] = "kd-dispatch-log-v2"
    with pytest.raises(DispatchLogError, match="artifact"):
        read_dispatch_log(_write_payload(tmp_path, payload))


def test_read_rejects_unknown_top_key(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["extra"] = 1
    with pytest.raises(DispatchLogError, match="Unknown"):
        read_dispatch_log(_write_payload(tmp_path, payload))


def test_read_rejects_missing_top_key(tmp_path: Path) -> None:
    payload = _valid_payload()
    del payload["heavy_peak"]
    with pytest.raises(DispatchLogError, match="Missing"):
        read_dispatch_log(_write_payload(tmp_path, payload))


def test_read_rejects_unknown_worker_key(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["workers"][0]["extra"] = 1
    with pytest.raises(DispatchLogError, match="Unknown"):
        read_dispatch_log(_write_payload(tmp_path, payload))


def test_read_rejects_missing_worker_key(tmp_path: Path) -> None:
    payload = _valid_payload()
    del payload["workers"][0]["heavy"]
    with pytest.raises(DispatchLogError, match="Missing"):
        read_dispatch_log(_write_payload(tmp_path, payload))


def test_read_rejects_wrong_env_subface(tmp_path: Path) -> None:
    payload = _valid_payload()
    del payload["workers"][0]["env"]["MKL_NUM_THREADS"]
    with pytest.raises(DispatchLogError):
        read_dispatch_log(_write_payload(tmp_path, payload))


def test_read_rejects_extra_env_key(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["workers"][0]["env"]["SECRET"] = "leak"
    with pytest.raises(DispatchLogError):
        read_dispatch_log(_write_payload(tmp_path, payload))


def test_read_rejects_kill_reason_outside_vocabulary(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["workers"][0]["kill_reason"] = "murdered"
    with pytest.raises(DispatchLogError, match="kill_reason"):
        read_dispatch_log(_write_payload(tmp_path, payload))


def test_read_accepts_each_vocabulary_kill_reason(tmp_path: Path) -> None:
    for reason in ("timeout", "spawn_failed", "dispatcher_interrupted"):
        payload = _valid_payload()
        payload["workers"][0]["kill_reason"] = reason
        log = read_dispatch_log(_write_payload(tmp_path, payload))
        assert log.workers[0].kill_reason == reason


def test_read_rejects_nan_token(tmp_path: Path) -> None:
    payload = _valid_payload()
    raw = json.dumps(payload).replace(
        '"wallclock_seconds": 1.0', '"wallclock_seconds": NaN'
    )
    path = tmp_path / "dispatch-log.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(DispatchLogError):
        read_dispatch_log(path)





def test_read_rejects_negative_wallclock(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["workers"][0]["wallclock_seconds"] = -1.0
    with pytest.raises(DispatchLogError, match="wallclock_seconds"):
        read_dispatch_log(_write_payload(tmp_path, payload))


def test_read_rejects_negative_timeout(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["workers"][0]["timeout_seconds"] = -5.0
    with pytest.raises(DispatchLogError, match="timeout_seconds"):
        read_dispatch_log(_write_payload(tmp_path, payload))


def test_read_rejects_bool_pid(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["workers"][0]["pid"] = True
    with pytest.raises(DispatchLogError, match="pid"):
        read_dispatch_log(_write_payload(tmp_path, payload))


def test_read_rejects_non_bool_sigkill_used(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["workers"][0]["sigkill_used"] = 1
    with pytest.raises(DispatchLogError, match="sigkill_used"):
        read_dispatch_log(_write_payload(tmp_path, payload))


def test_read_rejects_non_bool_heavy(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["workers"][0]["heavy"] = "yes"
    with pytest.raises(DispatchLogError, match="heavy"):
        read_dispatch_log(_write_payload(tmp_path, payload))


def test_read_rejects_malformed_shard_id(tmp_path: Path) -> None:

    payload = _valid_payload()
    payload["workers"][0]["shard_id"] = "bogus"
    with pytest.raises(DispatchLogError, match="shard_id"):
        read_dispatch_log(_write_payload(tmp_path, payload))


def test_read_rejects_duplicate_shard_ids(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["workers"].append(dict(payload["workers"][0]))
    with pytest.raises(DispatchLogError, match="duplicate"):
        read_dispatch_log(_write_payload(tmp_path, payload))


def test_read_rejects_non_str_plan_hash(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["plan_hash"] = 12345
    with pytest.raises(DispatchLogError, match="plan_hash"):
        read_dispatch_log(_write_payload(tmp_path, payload))
