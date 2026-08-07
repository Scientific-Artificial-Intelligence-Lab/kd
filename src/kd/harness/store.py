
from __future__ import annotations

import json
import logging
import math
import os
import platform
import subprocess
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import kd
from kd.core.jsonsafe import JSON_INDENT_SPACES
from kd.harness._verify import (
    INDEX_FILENAME,
    INDEX_SCHEMA_VERSION,
    RECORDS_DIRNAME,
    STATUS_COMPLETED,
    STATUS_RAISED,
    STATUS_VOCABULARY,
    EvidenceStoreError,
    load_verified_index,
    record_relpath,
    verify_record_plan_binding,
)
from kd.harness.plan import PLAN_HASH_SCHEME, ExperimentPlan
from kd.search.records import RunRecord

logger = logging.getLogger(__name__)


def environment_fingerprint() -> dict[str, str]:
    return {
        "kd_version": kd.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": _git_sha(),
        "torch_version": _torch_version(),
    }


def _git_sha() -> str:
    module_file = Path(__file__).resolve()
    cwd = module_file.parent
    try:
        subprocess.check_output(
            ["git", "ls-files", "--error-unmatch", module_file.name],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
        )
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"
    return output.decode("utf-8").strip()


def _torch_version() -> str:
    try:
        import torch

        return str(torch.__version__)
    except Exception:
        return "unknown"


class EvidenceStore:

    def __init__(
        self,
        *,
        root: Path,
        plan: ExperimentPlan,
        plan_hash: str,
        env: dict[str, str],
        attempts: list[dict[str, Any]],
        records: dict[int, RunRecord],
        records_index: dict[str, dict[str, str]],
        read_only: bool,
    ) -> None:




        self._root = root
        self._plan = plan
        self._plan_hash = plan_hash
        self._env = dict(env)
        self._attempts = attempts
        self._records = records
        self._records_index = records_index
        self._read_only = read_only



    @property
    def root(self) -> Path:
        return self._root

    @property
    def plan(self) -> ExperimentPlan:
        return self._plan

    @property
    def plan_hash(self) -> str:
        return self._plan_hash

    @property
    def env(self) -> dict[str, str]:
        return dict(self._env)

    @property
    def attempts(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(attempt) for attempt in self._attempts)

    @property
    def records(self) -> MappingProxyType[int, RunRecord]:
        return MappingProxyType(self._records)



    @classmethod
    def create(
        cls,
        root: Path,
        *,
        plan: ExperimentPlan,
        env: dict[str, str],
    ) -> EvidenceStore:
        root = Path(root)
        if root.exists():
            if not root.is_dir():
                raise EvidenceStoreError(
                    f"EvidenceStore root is not a directory: {root}"
                )
            if any(root.iterdir()):
                raise EvidenceStoreError(
                    f"EvidenceStore root is not empty (append/resume is not "
                    f"supported): {root}"
                )
        root.mkdir(parents=True, exist_ok=True)
        (root / RECORDS_DIRNAME).mkdir(exist_ok=True)
        store = cls(
            root=root,
            plan=plan,
            plan_hash=plan.plan_hash(),
            env=dict(env),
            attempts=[],
            records={},
            records_index={},
            read_only=False,
        )
        store._persist_index()
        return store



    def _validate_new_outcome(self, outcome: Any) -> int:
        entry_index = outcome.entry_index
        if isinstance(entry_index, bool) or not isinstance(entry_index, int):
            raise EvidenceStoreError(
                f"outcome.entry_index must be an int (bool rejected); "
                f"got {entry_index!r}"
            )
        n_entries = len(self._plan.entries)
        if not 0 <= entry_index < n_entries:
            raise EvidenceStoreError(
                f"outcome.entry_index {entry_index} out of range for plan with "
                f"{n_entries} entries"
            )
        if any(a["entry_index"] == entry_index for a in self._attempts):
            raise EvidenceStoreError(
                f"duplicate outcome for entry {entry_index} (one attempt per "
                f"entry in H1; append/resume is future work)"
            )

        status = outcome.status
        if status not in STATUS_VOCABULARY:
            raise EvidenceStoreError(
                f"unrecognized outcome.status {status!r}; expected one of "
                f"{sorted(STATUS_VOCABULARY)!r}"
            )
        if (status == STATUS_COMPLETED) != (outcome.record is not None):
            raise EvidenceStoreError(
                f"status/record incoherent (completed iff a record is present): "
                f"status={status!r}, record is "
                f"{'present' if outcome.record is not None else 'None'}"
            )
        if status == STATUS_RAISED and outcome.error_type is None:
            raise EvidenceStoreError(
                "status 'raised' requires a non-null error_type"
            )




        for field in ("error_type", "error_message"):
            value = getattr(outcome, field)
            if value is not None and not isinstance(value, str):
                raise EvidenceStoreError(
                    f"outcome.{field} must be str or null; got {value!r}"
                )

        wallclock = outcome.wallclock_seconds
        if isinstance(wallclock, bool) or not isinstance(wallclock, (int, float)):
            raise EvidenceStoreError(
                f"outcome.wallclock_seconds must be a real number; "
                f"got {wallclock!r}"
            )
        if not math.isfinite(wallclock) or wallclock < 0:
            raise EvidenceStoreError(
                f"outcome.wallclock_seconds must be finite and non-negative; "
                f"got {wallclock!r}"
            )


        return cast(int, entry_index)

    def add_outcome(self, outcome: Any) -> None:
        if self._read_only:
            raise EvidenceStoreError(
                "cannot add_outcome to a read-only (loaded) store; "
                "resume/append is H2+ future work"
            )
        entry_index = self._validate_new_outcome(outcome)
        entry = outcome.entry
        record: RunRecord | None = outcome.record

        if record is not None:



            verify_record_plan_binding(
                record, self._plan.entries[entry_index], entry_index
            )
            relpath = record_relpath(entry_index)
            record_path = self._root / relpath





            if record_path.exists():
                raise EvidenceStoreError(
                    f"refusing to overwrite existing record file for entry "
                    f"{entry_index}: {record_path}"
                )



            tmp_path = record_path.with_name(f"{record_path.name}.tmp")
            record.save(tmp_path)
            os.replace(tmp_path, record_path)
            self._records[entry_index] = record
            self._records_index[str(entry_index)] = {
                "path": relpath,
                "record_hash": record.record_hash,
            }

        self._attempts.append(
            {
                "entry_index": entry_index,
                "instrument": entry.instrument,
                "dataset_ref": entry.dataset_ref,
                "seed": entry.seed,
                "status": outcome.status,
                "error_type": outcome.error_type,
                "error_message": outcome.error_message,
                "wallclock_seconds": outcome.wallclock_seconds,
            }
        )
        self._persist_index()
        logger.info(
            "evidence-store: recorded entry %d status=%s (%d attempts, %d records)",
            entry_index,
            outcome.status,
            len(self._attempts),
            len(self._records),
        )



    @classmethod
    def load(cls, root: Path) -> EvidenceStore:
        root = Path(root)
        verified = load_verified_index(root)
        return cls(
            root=root,
            plan=verified.plan,
            plan_hash=verified.plan_hash,
            env=verified.env,
            attempts=verified.attempts,
            records=verified.records,
            records_index=verified.records_index,
            read_only=True,
        )



    def _index_payload(self) -> dict[str, Any]:
        return {
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "plan": self._plan.to_dict(),
            "plan_hash": self._plan_hash,
            "plan_hash_scheme": PLAN_HASH_SCHEME,
            "env": dict(self._env),
            "attempts": [dict(attempt) for attempt in self._attempts],
            "records": {
                key: dict(meta) for key, meta in self._records_index.items()
            },
        }

    def _persist_index(self) -> None:



        index_path = self._root / INDEX_FILENAME
        tmp_path = self._root / f"{INDEX_FILENAME}.tmp"
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(
                self._index_payload(),
                handle,
                indent=JSON_INDENT_SPACES,
                allow_nan=False,
                sort_keys=True,
            )
        os.replace(tmp_path, index_path)




__all__ = [
    "EvidenceStore",
    "EvidenceStoreError",
    "environment_fingerprint",
]
