
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Final

from kd.core.jsonsafe import finite_or_none
from kd.search.records import RunRecord
from kd.search.result import ExperimentResult

__all__ = [
    "BEST_EXPRESSION_MAX_CHARS",
    "CATALOG_FILENAME",
    "DEFAULT_RUNS_ROOT",
    "RUNCAT_SCHEME",
    "RUNCAT_SCHEMA_VERSION",
    "append_catalog_row",
    "catalog_row_from_record",
    "catalog_row_from_result",
]

RUNCAT_SCHEME: Final[str] = "kd-runcat-v1"
RUNCAT_SCHEMA_VERSION: Final[int] = 1

CATALOG_FILENAME: Final[str] = "catalog.jsonl"



DEFAULT_RUNS_ROOT: Final[Path] = Path("runs")



BEST_EXPRESSION_MAX_CHARS: Final[int] = 200




_RUNCAT_V1_FIELDS: Final[tuple[str, ...]] = (
    "scheme",
    "schema_version",
    "run_id",
    "created_at",
    "kd_version",
    "instrument",
    "dataset_name",
    "dataset_cache_fingerprint",
    "seed",
    "config_hash",
    "status",
    "best_score",
    "score_kind",
    "score_direction",
    "nmse",
    "best_expression",
    "run_dir",
    "record_path",
    "record_hash",
    "parent_run_id",
    "resume_from",
    "plan_hash",
    "entry_index",
)
_RUNCAT_V1_FIELD_SET: Final[frozenset[str]] = frozenset(_RUNCAT_V1_FIELDS)


def catalog_row_from_result(
    result: ExperimentResult | None,
    *,
    run_id: str,
    created_at: str,
    instrument: str,
    status: str,
    run_dir: str,
    dataset_name: str | None = None,
    dataset_cache_fingerprint: str | None = None,
    seed: int | None = None,
    record_path: str | None = None,
    parent_run_id: str | None = None,
    resume_from: str | None = None,
    plan_hash: str | None = None,
    entry_index: int | None = None,
) -> dict[str, Any]:
    from kd import __version__ as kd_version

    best_score: float | None = None
    score_kind: str | None = None
    score_direction: str | None = None
    nmse: float | None = None
    best_expression: str | None = None
    config_hash: str | None = None
    record_hash: str | None = None
    if result is not None:
        best_score = finite_or_none(result.best_score)
        score_kind = result.score_kind
        score_direction = result.score_direction
        nmse = finite_or_none(result.final_eval.nmse)
        if result.best_expression:
            best_expression = result.best_expression[:BEST_EXPRESSION_MAX_CHARS]
        if result.manifest is not None:
            dataset_cache_fingerprint = result.manifest.dataset_cache_fingerprint
            seed = result.manifest.seed
        dataset_name = result.dataset_name
        if result.run_record is not None:
            config_hash = result.run_record.run_spec_hash
            record_hash = result.run_record.record_hash

    return {
        "scheme": RUNCAT_SCHEME,
        "schema_version": RUNCAT_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created_at,
        "kd_version": kd_version,
        "instrument": instrument,
        "dataset_name": dataset_name,
        "dataset_cache_fingerprint": dataset_cache_fingerprint,
        "seed": seed,
        "config_hash": config_hash,
        "status": status,
        "best_score": best_score,
        "score_kind": score_kind,
        "score_direction": score_direction,
        "nmse": nmse,
        "best_expression": best_expression,
        "run_dir": run_dir,
        "record_path": record_path,
        "record_hash": record_hash,
        "parent_run_id": parent_run_id,
        "resume_from": resume_from,
        "plan_hash": plan_hash,
        "entry_index": entry_index,
    }


def catalog_row_from_record(
    record: RunRecord | None,
    *,
    run_id: str,
    created_at: str,
    instrument: str,
    status: str,
    run_dir: str,
    dataset_name: str | None = None,
    dataset_cache_fingerprint: str | None = None,
    seed: int | None = None,
    record_path: str | None = None,
    parent_run_id: str | None = None,
    resume_from: str | None = None,
    plan_hash: str | None = None,
    entry_index: int | None = None,
) -> dict[str, Any]:
    from kd import __version__ as kd_version

    best_score: float | None = None
    score_kind: str | None = None
    score_direction: str | None = None
    nmse: float | None = None
    best_expression: str | None = None
    config_hash: str | None = None
    record_hash: str | None = None
    if record is not None:
        evidence = record.evidence
        best_score = finite_or_none(evidence.score)
        score_kind = evidence.score_kind
        score_direction = evidence.score_direction
        nmse = finite_or_none(evidence.nmse)
        if evidence.expression:
            best_expression = evidence.expression[:BEST_EXPRESSION_MAX_CHARS]
        dataset_name = evidence.dataset_name
        dataset_cache_fingerprint = evidence.dataset_cache_fingerprint
        seed = evidence.seed
        config_hash = record.run_spec_hash
        record_hash = record.record_hash

    return {
        "scheme": RUNCAT_SCHEME,
        "schema_version": RUNCAT_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created_at,
        "kd_version": kd_version,
        "instrument": instrument,
        "dataset_name": dataset_name,
        "dataset_cache_fingerprint": dataset_cache_fingerprint,
        "seed": seed,
        "config_hash": config_hash,
        "status": status,
        "best_score": best_score,
        "score_kind": score_kind,
        "score_direction": score_direction,
        "nmse": nmse,
        "best_expression": best_expression,
        "run_dir": run_dir,
        "record_path": record_path,
        "record_hash": record_hash,
        "parent_run_id": parent_run_id,
        "resume_from": resume_from,
        "plan_hash": plan_hash,
        "entry_index": entry_index,
    }


def append_catalog_row(catalog_path: Path | str, row: dict[str, Any]) -> None:
    keys = set(row)
    if keys != _RUNCAT_V1_FIELD_SET:
        unknown = sorted(keys - _RUNCAT_V1_FIELD_SET)
        missing = sorted(_RUNCAT_V1_FIELD_SET - keys)
        raise ValueError(
            f"catalog row key drift: unknown {unknown!r}, missing {missing!r}"
        )
    for field in ("seed", "entry_index"):
        value = row[field]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError(
                f"catalog row {field} must be an int or None; got {value!r}"
            )
    for field in ("best_score", "nmse"):
        value = row[field]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(
                f"catalog row {field} must be finite or None; got {value!r}"
            )

    line = (
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")

    catalog_path = Path(catalog_path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        catalog_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, mode=0o644
    )
    try:
        written = os.write(fd, line)
        if written != len(line):
            raise OSError(
                f"partial catalog append ({written}/{len(line)} bytes) to "
                f"{catalog_path}"
            )
    finally:
        os.close(fd)
