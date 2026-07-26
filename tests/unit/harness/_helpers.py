
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kd.harness.plan import ExperimentPlan, PlanEntry
from kd.harness.store import EvidenceStore, environment_fingerprint
from kd.search.records import (
    EVIDENCE_HASH_SCHEME,
    RECORD_HASH_SCHEME,
    RUN_RECORD_SCHEMA_VERSION,
    EvidenceRecord,
    RunCost,
    RunRecord,
    seal_record_hash,
)
from kd.search.run_spec import RUN_SPEC_HASH_SCHEME, RunSpec


def make_record(
    instrument: str = "sga",
    *,
    seed: int = 0,
    expression: str = "u_xx",
    dataset_name: str = "burgers_tiny",
    dataset_cache_fingerprint: str = "sha256:dataset",
    catalog_fit: dict[str, Any] | None = None,
    is_valid: bool = True,
    invalid_reason: str | None = None,
    score_kind: str = "AIC",
    score_direction: str = "min",
    headline_coefficient_source: str = "native",
    score: float | None = 1.5,
    nmse: float | None = None,
    r2: float | None = None,
    config: dict[str, Any] | None = None,
) -> RunRecord:
    evidence = EvidenceRecord(
        instrument=instrument,
        dataset_name=dataset_name,
        dataset_cache_fingerprint=dataset_cache_fingerprint,
        seed=seed,
        is_valid=is_valid,
        expression=expression,
        score_kind=score_kind,
        score_direction=score_direction,
        headline_coefficient_source=headline_coefficient_source,
        score=score,
        nmse=nmse,
        r2=r2,
        catalog_fit=catalog_fit,
        invalid_reason=invalid_reason,
    )
    run_spec_config: dict[str, Any] = {"algorithm": instrument}
    if config is not None:
        run_spec_config.update(config)
    run_spec = RunSpec(
        kd_version="0.4.0",
        config=run_spec_config,
        dataset_cache_fingerprint=dataset_cache_fingerprint,
    )
    return seal_record_hash(
        RunRecord(
            schema_version=RUN_RECORD_SCHEMA_VERSION,
            evidence_hash_scheme=EVIDENCE_HASH_SCHEME,
            created_at="2026-07-21T00:00:00+00:00",
            cost=RunCost(
                wallclock_seconds=1.0,
                search_seconds=1.0,
                boundary_results=0,
                boundary_invalid_results=0,
            ),
            evidence=evidence,
            evidence_hash=evidence.content_hash(),
            run_spec=run_spec,
            run_spec_hash=run_spec.run_spec_hash,
            run_spec_hash_scheme=RUN_SPEC_HASH_SCHEME,
            record_hash="",
            record_hash_scheme=RECORD_HASH_SCHEME,
        )
    )


def make_entry(
    instrument: str = "sga",
    *,
    dataset_ref: str = "burgers_tiny",
    seed: int = 0,
    **model_kwargs: Any,
) -> PlanEntry:
    return PlanEntry(
        instrument=instrument,
        dataset_ref=dataset_ref,
        seed=seed,
        model_kwargs=dict(model_kwargs),
    )


def make_plan(entries: list[PlanEntry], *, name: str = "test_plan") -> ExperimentPlan:
    return ExperimentPlan(name=name, entries=tuple(entries))


@dataclass(frozen=True, kw_only=True)
class StubOutcome:

    entry_index: int
    entry: PlanEntry
    status: str
    record: RunRecord | None = None
    error_type: str | None = None
    error_message: str | None = None
    wallclock_seconds: float = 0.0


def completed_outcome(
    entry_index: int, entry: PlanEntry, record: RunRecord
) -> StubOutcome:
    return StubOutcome(
        entry_index=entry_index,
        entry=entry,
        status="completed",
        record=record,
        wallclock_seconds=1.0,
    )


def raised_outcome(
    entry_index: int,
    entry: PlanEntry,
    *,
    error_type: str = "RuntimeError",
    error_message: str = "boom",
) -> StubOutcome:
    return StubOutcome(
        entry_index=entry_index,
        entry=entry,
        status="raised",
        record=None,
        error_type=error_type,
        error_message=error_message,
        wallclock_seconds=0.5,
    )


def no_record_outcome(entry_index: int, entry: PlanEntry) -> StubOutcome:
    return StubOutcome(
        entry_index=entry_index,
        entry=entry,
        status="no_record",
        record=None,
        error_type=None,
        error_message=None,
        wallclock_seconds=0.5,
    )


def attempt_dict(
    entry_index: int,
    *,
    instrument: str = "sga",
    dataset_ref: str = "burgers_tiny",
    seed: int = 0,
    status: str = "completed",
    error_type: str | None = None,
    error_message: str | None = None,
    wallclock_seconds: float = 1.0,
) -> dict[str, Any]:
    return {
        "entry_index": entry_index,
        "instrument": instrument,
        "dataset_ref": dataset_ref,
        "seed": seed,
        "status": status,
        "error_type": error_type,
        "error_message": error_message,
        "wallclock_seconds": wallclock_seconds,
    }


_DEFAULT_ENV = {"kd_version": "0.4.0", "python": "3.12.0", "platform": "linux"}


def build_sealed_store(
    root: Path,
    *,
    entries: list[PlanEntry],
    records: dict[int, RunRecord] | None = None,
    raised: dict[int, tuple[str, str]] | None = None,
    no_record: list[int] | None = None,
    env: dict[str, str] | None = None,
    name: str = "consensus_plan",
) -> EvidenceStore:
    records = records or {}
    raised = raised or {}
    no_record = no_record or []
    plan = make_plan(list(entries), name=name)
    store = EvidenceStore.create(root, plan=plan, env=dict(env or _DEFAULT_ENV))
    for index, entry in enumerate(entries):
        if index in records:
            store.add_outcome(completed_outcome(index, entry, records[index]))
        elif index in raised:
            error_type, error_message = raised[index]
            store.add_outcome(
                raised_outcome(
                    index,
                    entry,
                    error_type=error_type,
                    error_message=error_message,
                )
            )
        elif index in no_record:
            store.add_outcome(no_record_outcome(index, entry))
    return EvidenceStore.load(root)


def fresh_env() -> dict[str, str]:
    return environment_fingerprint()
