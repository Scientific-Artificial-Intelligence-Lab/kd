
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kd.harness._verify import (
    agreed_env_fields,
    verify_data_identity,
    verify_single_source_attempts,
)
from kd.harness.plan import ExperimentPlan, PlanEntry
from kd.harness.store import EvidenceStore, EvidenceStoreError
from kd.search.records import RunRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class _ReplayOutcome:

    entry_index: int
    entry: PlanEntry
    status: str
    record: RunRecord | None
    error_type: str | None
    error_message: str | None
    wallclock_seconds: float


def _source_rows(
    paths: Sequence[Path | str], stores: Sequence[EvidenceStore]
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    offset = 0
    for path, store in zip(paths, stores, strict=True):
        n_entries = len(store.plan.entries)
        rows.append(
            {
                "root": str(Path(path).resolve()),
                "plan_name": store.plan.name,
                "plan_hash": store.plan_hash,
                "entry_offset": offset,
                "n_entries": n_entries,
                "env": store.env,
            }
        )
        offset += n_entries
    return tuple(rows)


def _shifted_outcomes(
    plan: ExperimentPlan,
    stores: Sequence[EvidenceStore],
    rows: Sequence[dict[str, Any]],
) -> dict[int, _ReplayOutcome]:
    outcomes: dict[int, _ReplayOutcome] = {}
    for store, row in zip(stores, rows, strict=True):
        for attempt in store.attempts:
            local = attempt["entry_index"]
            index = row["entry_offset"] + local
            outcomes[index] = _ReplayOutcome(
                entry_index=index,
                entry=plan.entries[index],
                status=attempt["status"],
                record=store.records.get(local),
                error_type=attempt["error_type"],
                error_message=attempt["error_message"],
                wallclock_seconds=attempt["wallclock_seconds"],
            )
    return outcomes


def merge_stores(
    sources: Sequence[Path | str],
    *,
    root: Path | str,
    name: str,
    version_note: str | None = None,
) -> EvidenceStore:
    if not sources:
        raise EvidenceStoreError("merge_stores needs at least one source store")
    stores = [EvidenceStore.load(Path(path)) for path in sources]
    for path, store in zip(sources, stores, strict=True):
        if store.sources:
            raise EvidenceStoreError(
                f"source {path} is a merged store; merge its original stores instead"
            )
    plan = ExperimentPlan(
        name=name,
        entries=tuple(entry for store in stores for entry in store.plan.entries),
    )
    rows = _source_rows(sources, stores)
    outcomes = _shifted_outcomes(plan, stores, rows)
    verify_single_source_attempts(plan, rows, outcomes)
    verify_data_identity(
        plan,
        {i: o.record for i, o in outcomes.items() if o.record is not None},
    )
    merged = EvidenceStore.create(
        Path(root),
        plan=plan,
        env=agreed_env_fields([store.env for store in stores]),
        sources=rows,
        version_note=version_note,
    )
    for index in sorted(outcomes):
        merged.add_outcome(outcomes[index])

    reloaded = EvidenceStore.load(Path(root))
    logger.info(
        "merge_stores: %d sources, %d attempts (%d records) into %s",
        len(stores),
        len(outcomes),
        len(reloaded.records),
        root,
    )

    from kd.harness.report import write_store_report

    write_store_report(reloaded)
    return reloaded


__all__ = ["merge_stores"]
