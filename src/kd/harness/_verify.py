
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from kd.harness.plan import PLAN_HASH_SCHEME, ExperimentPlan, PlanEntry
from kd.search.records import RunRecord

INDEX_SCHEMA_VERSION: Final[int] = 1
INDEX_FILENAME: Final[str] = "index.json"
RECORDS_DIRNAME: Final[str] = "records"




STATUS_COMPLETED: Final[str] = "completed"
STATUS_RAISED: Final[str] = "raised"
STATUS_VOCABULARY: Final[frozenset[str]] = frozenset(
    {STATUS_COMPLETED, STATUS_RAISED, "no_record"}
)

_INDEX_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "index_schema_version",
        "plan",
        "plan_hash",
        "plan_hash_scheme",
        "env",
        "attempts",
        "records",
    }
)


_ATTEMPT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "entry_index",
        "instrument",
        "dataset_ref",
        "seed",
        "status",
        "error_type",
        "error_message",
        "wallclock_seconds",
    }
)


class EvidenceStoreError(Exception):
    pass


def record_relpath(entry_index: int) -> str:
    return f"{RECORDS_DIRNAME}/entry-{entry_index:04d}.json"


def verify_record_plan_binding(
    record: RunRecord, plan_entry: PlanEntry, entry_index: int
) -> None:
    if record.evidence.instrument != plan_entry.instrument:
        raise EvidenceStoreError(
            f"record/plan instrument mismatch for entry {entry_index}: "
            f"record {record.evidence.instrument!r} != plan "
            f"{plan_entry.instrument!r}"
        )
    if record.evidence.seed != plan_entry.seed:
        raise EvidenceStoreError(
            f"record/plan seed mismatch for entry {entry_index}: record "
            f"{record.evidence.seed!r} != plan {plan_entry.seed!r}"
        )


@dataclass(frozen=True, kw_only=True)
class VerifiedIndex:

    plan: ExperimentPlan
    plan_hash: str
    env: dict[str, str]
    attempts: list[dict[str, Any]]
    records: dict[int, RunRecord]
    records_index: dict[str, dict[str, str]]


def _decode_index(root: Path) -> dict[str, Any]:
    index_path = root / INDEX_FILENAME
    if not index_path.is_file():
        raise EvidenceStoreError(f"no evidence index at {index_path}")
    with index_path.open(encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise EvidenceStoreError(
                f"evidence index is not valid JSON ({index_path}): {exc}"
            ) from exc
    if not isinstance(data, dict):
        raise EvidenceStoreError("evidence index must be a JSON object")

    version = data.get("index_schema_version")
    if type(version) is not int or version != INDEX_SCHEMA_VERSION:
        raise EvidenceStoreError(
            f"unsupported index_schema_version: got {version!r}; "
            f"supported: {[INDEX_SCHEMA_VERSION]!r}"
        )
    actual = frozenset(data)
    if actual != _INDEX_FIELDS:
        unknown = sorted(actual - _INDEX_FIELDS)
        missing = sorted(_INDEX_FIELDS - actual)
        raise EvidenceStoreError(
            f"malformed evidence index (unknown={unknown!r}, missing={missing!r})"
        )




    stored_scheme = data["plan_hash_scheme"]
    if stored_scheme != PLAN_HASH_SCHEME:
        raise EvidenceStoreError(
            f"unsupported plan_hash_scheme: got {stored_scheme!r}; "
            f"expected {PLAN_HASH_SCHEME!r}"
        )
    return data


def _decode_plan(data: dict[str, Any]) -> ExperimentPlan:
    plan_payload = data["plan"]
    if not isinstance(plan_payload, dict):
        raise EvidenceStoreError("evidence index plan must be a JSON object")
    try:
        plan = ExperimentPlan.from_dict(plan_payload)
    except Exception as exc:
        raise EvidenceStoreError(f"undecodable plan in index: {exc}") from exc

    stored_plan_hash = data["plan_hash"]
    recomputed = plan.plan_hash()
    if stored_plan_hash != recomputed:
        raise EvidenceStoreError(
            f"plan_hash mismatch: index {stored_plan_hash!r} != recomputed "
            f"{recomputed!r}"
        )
    return plan


def _verify_records(
    root: Path, plan: ExperimentPlan, records_index: Any
) -> tuple[dict[int, RunRecord], dict[str, dict[str, str]]]:
    if not isinstance(records_index, dict):
        raise EvidenceStoreError("evidence index records must be a JSON object")

    n_entries = len(plan.entries)
    records: dict[int, RunRecord] = {}
    clean_index: dict[str, dict[str, str]] = {}
    for key, meta in records_index.items():



        if (
            not isinstance(meta, dict)
            or not isinstance(meta.get("path"), str)
            or not isinstance(meta.get("record_hash"), str)
        ):
            raise EvidenceStoreError(
                f"malformed record index entry for {key!r}: {meta!r}"
            )
        try:
            entry_index = int(key)
        except (TypeError, ValueError) as exc:
            raise EvidenceStoreError(
                f"non-integer record index key {key!r}"
            ) from exc


        if str(entry_index) != key:
            raise EvidenceStoreError(
                f"non-canonical record index key {key!r} "
                f"(expected {str(entry_index)!r})"
            )
        if not 0 <= entry_index < n_entries:
            raise EvidenceStoreError(
                f"record index key {key!r} out of range for plan with "
                f"{n_entries} entries"
            )
        canonical_path = record_relpath(entry_index)
        if meta["path"] != canonical_path:
            raise EvidenceStoreError(
                f"non-canonical record path for entry {entry_index}: "
                f"{meta['path']!r} != {canonical_path!r}"
            )
        record_path = root / meta["path"]
        if not record_path.is_file():
            raise EvidenceStoreError(f"missing record file: {record_path}")
        try:
            record = RunRecord.load(record_path)
        except Exception as exc:
            raise EvidenceStoreError(
                f"record failed to load/verify ({record_path}): {exc}"
            ) from exc
        if record.record_hash != meta["record_hash"]:
            raise EvidenceStoreError(
                f"record_hash mismatch for entry {key!r}: index "
                f"{meta['record_hash']!r} != loaded {record.record_hash!r}"
            )
        verify_record_plan_binding(record, plan.entries[entry_index], entry_index)
        records[entry_index] = record
        clean_index[key] = {
            "path": meta["path"],
            "record_hash": meta["record_hash"],
        }

    _reject_orphan_records(root, clean_index)
    return records, clean_index


def _reject_orphan_records(
    root: Path, clean_index: dict[str, dict[str, str]]
) -> None:
    records_dir = root / RECORDS_DIRNAME
    if not records_dir.is_dir():
        return
    referenced = {Path(meta["path"]).name for meta in clean_index.values()}
    for child in sorted(records_dir.iterdir()):
        if child.is_file() and child.name not in referenced:
            raise EvidenceStoreError(f"unreferenced record file (orphan): {child}")


def _verify_data_identity(
    plan: ExperimentPlan, records: dict[int, RunRecord]
) -> None:
    by_ref: dict[str, str] = {}
    for entry_index, record in records.items():
        ref = plan.entries[entry_index].dataset_ref
        fingerprint = record.evidence.dataset_cache_fingerprint
        if ref in by_ref and by_ref[ref] != fingerprint:
            raise EvidenceStoreError(
                f"dataset identity mismatch for dataset_ref {ref!r}: "
                f"{by_ref[ref]!r} != {fingerprint!r}"
            )
        by_ref.setdefault(ref, fingerprint)


def _verify_attempts(
    plan: ExperimentPlan, attempts: Any, records: dict[int, RunRecord]
) -> list[dict[str, Any]]:
    if not isinstance(attempts, list):
        raise EvidenceStoreError("evidence index attempts must be a JSON array")

    n_entries = len(plan.entries)
    seen: set[int] = set()
    verified: list[dict[str, Any]] = []
    for position, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            raise EvidenceStoreError(
                f"attempt[{position}] must be a JSON object; "
                f"got {type(attempt).__name__}"
            )
        actual = frozenset(attempt)
        if actual != _ATTEMPT_FIELDS:
            unknown = sorted(actual - _ATTEMPT_FIELDS)
            missing = sorted(_ATTEMPT_FIELDS - actual)
            raise EvidenceStoreError(
                f"malformed attempt[{position}] (unknown={unknown!r}, "
                f"missing={missing!r})"
            )

        entry_index = attempt["entry_index"]
        if isinstance(entry_index, bool) or not isinstance(entry_index, int):
            raise EvidenceStoreError(
                f"attempt[{position}].entry_index must be an int (bool "
                f"rejected); got {entry_index!r}"
            )
        if not 0 <= entry_index < n_entries:
            raise EvidenceStoreError(
                f"attempt[{position}].entry_index {entry_index} out of range "
                f"for plan with {n_entries} entries"
            )
        if entry_index in seen:
            raise EvidenceStoreError(f"duplicate attempt for entry {entry_index}")
        seen.add(entry_index)

        status = attempt["status"]
        if status not in STATUS_VOCABULARY:
            raise EvidenceStoreError(
                f"attempt[{position}].status {status!r} not in "
                f"{sorted(STATUS_VOCABULARY)!r}"
            )
        for field in ("error_type", "error_message"):
            value = attempt[field]
            if value is not None and not isinstance(value, str):
                raise EvidenceStoreError(
                    f"attempt[{position}].{field} must be str or null; "
                    f"got {value!r}"
                )
        wallclock = attempt["wallclock_seconds"]
        if isinstance(wallclock, bool) or not isinstance(wallclock, (int, float)):
            raise EvidenceStoreError(
                f"attempt[{position}].wallclock_seconds must be a real number; "
                f"got {wallclock!r}"
            )
        if not math.isfinite(wallclock) or wallclock < 0:
            raise EvidenceStoreError(
                f"attempt[{position}].wallclock_seconds must be finite and "
                f"non-negative; got {wallclock!r}"
            )

        plan_entry = plan.entries[entry_index]
        for field, plan_value in (
            ("instrument", plan_entry.instrument),
            ("dataset_ref", plan_entry.dataset_ref),
            ("seed", plan_entry.seed),
        ):
            if attempt[field] != plan_value:
                raise EvidenceStoreError(
                    f"attempt[{position}] {field} mismatch for entry "
                    f"{entry_index}: {attempt[field]!r} != plan {plan_value!r}"
                )

        has_record = entry_index in records
        if (status == STATUS_COMPLETED) != has_record:
            raise EvidenceStoreError(
                f"attempt[{position}] completed/record incoherence for entry "
                f"{entry_index}: status={status!r}, record "
                f"{'present' if has_record else 'absent'}"
            )
        if status == STATUS_RAISED and attempt["error_type"] is None:
            raise EvidenceStoreError(
                f"attempt[{position}] status 'raised' requires error_type "
                f"(entry {entry_index})"
            )
        verified.append(dict(attempt))

    completed_indices = {
        a["entry_index"] for a in verified if a["status"] == STATUS_COMPLETED
    }
    record_indices = set(records)
    if completed_indices != record_indices:
        raise EvidenceStoreError(
            f"records/attempts index mismatch: completed attempts "
            f"{sorted(completed_indices)!r} != records {sorted(record_indices)!r}"
        )
    return verified


def load_verified_index(root: Path) -> VerifiedIndex:
    data = _decode_index(root)
    plan = _decode_plan(data)
    records, clean_index = _verify_records(root, plan, data["records"])
    _verify_data_identity(plan, records)
    attempts = _verify_attempts(plan, data["attempts"], records)

    env = data["env"]
    if not isinstance(env, dict):
        raise EvidenceStoreError("evidence index env must be a JSON object")

    return VerifiedIndex(
        plan=plan,
        plan_hash=data["plan_hash"],
        env=env,
        attempts=attempts,
        records=records,
        records_index=clean_index,
    )
