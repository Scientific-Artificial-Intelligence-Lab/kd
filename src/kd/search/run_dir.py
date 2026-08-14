
from __future__ import annotations

import hashlib
import json
import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from kd.core.jsonsafe import JSON_INDENT_SPACES
from kd.search._record_schema import validate_lineage
from kd.search.result import ExperimentResult

__all__ = [
    "CHECKPOINTS_DIRNAME",
    "EVENTS_FILENAME",
    "PHASES_FILENAME",
    "RECORDER_FILENAME",
    "RECORD_FILENAME",
    "RUNDIR_MANIFEST_FILENAME",
    "RUNDIR_SCHEME",
    "RUNDIR_SCHEMA_VERSION",
    "RunDirPaths",
    "create_run_dir",
    "finalize_run_dir",
    "new_run_id",
    "run_id_of_run_dir",
]

RUNDIR_SCHEME: Final[str] = "kd-rundir-v1"
RUNDIR_SCHEMA_VERSION: Final[int] = 1

RUNDIR_MANIFEST_FILENAME: Final[str] = "manifest.json"
RECORD_FILENAME: Final[str] = "record.json"
EVENTS_FILENAME: Final[str] = "events.jsonl"
PHASES_FILENAME: Final[str] = "phases.jsonl"
RECORDER_FILENAME: Final[str] = "recorder.json"
CHECKPOINTS_DIRNAME: Final[str] = "checkpoints"





RUNDIR_STATUS_VOCABULARY: Final[frozenset[str]] = frozenset(
    {"completed", "raised", "no_record"}
)




_RUNDIR_V1_FIELDS: Final[tuple[str, ...]] = (
    "scheme",
    "schema_version",
    "run_id",
    "created_at",
    "instrument",
    "dataset_cache_fingerprint",
    "seed",
    "config_hash",
    "status",
    "artifacts",
    "record_ref",
    "checkpoints_dir",
    "lineage",
)





@dataclass(frozen=True)
class RunDirPaths:

    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / RUNDIR_MANIFEST_FILENAME

    @property
    def record(self) -> Path:
        return self.root / RECORD_FILENAME

    @property
    def events(self) -> Path:
        return self.root / EVENTS_FILENAME

    @property
    def phases(self) -> Path:
        return self.root / PHASES_FILENAME

    @property
    def recorder(self) -> Path:
        return self.root / RECORDER_FILENAME

    @property
    def checkpoints(self) -> Path:
        return self.root / CHECKPOINTS_DIRNAME


def new_run_id(instrument: str) -> str:
    if not instrument or not isinstance(instrument, str):
        raise ValueError(f"instrument must be a non-empty str; got {instrument!r}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{instrument}-{stamp}-{secrets.token_hex(3)}"


def create_run_dir(root: Path | str) -> RunDirPaths:
    root = Path(root)
    if root.exists():
        if not root.is_dir():
            raise ValueError(f"run directory path is not a directory: {root}")
        if any(root.iterdir()):
            raise ValueError(
                f"run directory is not empty (one run per directory): {root}"
            )
    root.mkdir(parents=True, exist_ok=True)
    return RunDirPaths(root=root)


def _sha256_of_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _atomic_json_dump(payload: Any, path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=JSON_INDENT_SPACES,
            allow_nan=False,
            sort_keys=True,
        )
    os.replace(tmp_path, path)


def finalize_run_dir(
    paths: RunDirPaths,
    result: ExperimentResult | None,
    *,
    run_id: str,
    instrument: str,
    status: str,
    lineage: Mapping[str, Any] | None = None,
    record_ref: str | None = None,
) -> dict[str, Any]:
    if status not in RUNDIR_STATUS_VOCABULARY:
        raise ValueError(
            f"status must be one of {sorted(RUNDIR_STATUS_VOCABULARY)!r}; "
            f"got {status!r}"
        )
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"run_id must be a non-empty str; got {run_id!r}")
    if not isinstance(instrument, str) or not instrument:
        raise ValueError(f"instrument must be a non-empty str; got {instrument!r}")
    lineage_payload = validate_lineage(lineage, error_cls=ValueError)

    record_ref_payload: dict[str, Any] | None = None
    if result is not None:
        _atomic_json_dump(result.recorder.to_dict(), paths.recorder)
        if record_ref is not None:
            if result.run_record is None:
                raise ValueError(
                    "record_ref given but result carries no run_record to "
                    "reference"
                )
            record_ref_payload = {
                "path": record_ref,
                "record_hash": result.run_record.record_hash,
            }
        elif result.run_record is not None:
            result.run_record.save(paths.record)

    artifacts = {
        name: _sha256_of_file(paths.root / name)
        if (paths.root / name).is_file()
        else None
        for name in (
            RECORD_FILENAME,
            EVENTS_FILENAME,
            PHASES_FILENAME,
            RECORDER_FILENAME,
        )
    }

    manifest = result.manifest if result is not None else None
    run_record = result.run_record if result is not None else None
    payload: dict[str, Any] = {
        "scheme": RUNDIR_SCHEME,
        "schema_version": RUNDIR_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "instrument": instrument,
        "dataset_cache_fingerprint": (
            manifest.dataset_cache_fingerprint if manifest is not None else None
        ),
        "seed": manifest.seed if manifest is not None else None,
        "config_hash": (
            run_record.run_spec_hash if run_record is not None else None
        ),
        "status": status,
        "artifacts": artifacts,
        "record_ref": record_ref_payload,
        "checkpoints_dir": (
            CHECKPOINTS_DIRNAME if paths.checkpoints.is_dir() else None
        ),
        "lineage": lineage_payload,
    }
    assert set(payload) == set(_RUNDIR_V1_FIELDS)
    _atomic_json_dump(payload, paths.manifest)
    return payload


def run_id_of_run_dir(directory: Path | str) -> str | None:
    manifest_path = Path(directory) / RUNDIR_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    if not isinstance(data, dict) or data.get("scheme") != RUNDIR_SCHEME:
        return None
    run_id = data.get("run_id")
    return run_id if isinstance(run_id, str) and run_id else None
