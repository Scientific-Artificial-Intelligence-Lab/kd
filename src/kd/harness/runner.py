
from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import kd
from kd.api import Model
from kd.harness._verify import record_relpath
from kd.harness.episode import EpisodeOutcome, run_episode
from kd.harness.plan import ExperimentPlan
from kd.harness.recording import RecordingOptions
from kd.harness.store import EvidenceStore, environment_fingerprint
from kd.search.run_catalog import append_catalog_row, catalog_row_from_record

if TYPE_CHECKING:
    from kd.data.schema import PDEDataset

logger = logging.getLogger(__name__)





RUN_DIRS_DIRNAME = "runs"


@dataclass(frozen=True, kw_only=True)
class PlanRunResult:

    store_root: Path
    outcomes: tuple[EpisodeOutcome, ...]


def _supported_instruments() -> set[str]:
    return {schema["algorithm"] for schema in kd.instrument_schemas()}


def _preflight(plan: ExperimentPlan, datasets: Mapping[str, PDEDataset]) -> None:
    supported = _supported_instruments()
    violations: list[str] = []
    for index, entry in enumerate(plan.entries):
        if entry.instrument not in supported:
            violations.append(
                f"entry[{index}]: unknown instrument {entry.instrument!r} "
                f"(supported: {sorted(supported)!r})"
            )
        if entry.dataset_ref not in datasets:
            violations.append(
                f"entry[{index}]: dataset_ref {entry.dataset_ref!r} not in "
                f"provided datasets (available: {sorted(datasets)!r})"
            )
    if violations:
        joined = "\n ".join(violations)
        raise ValueError(
            f"plan pre-flight failed ({len(violations)} violation(s)):\n {joined}"
        )


def _catalog_row_for_outcome(
    outcome: EpisodeOutcome,
    *,
    dataset: PDEDataset,
    catalog_path: Path,
    run_dir: Path,
    run_id: str,
    plan_hash: str,
    global_index: int,
    resume_source: str | None,
) -> dict[str, Any]:
    record_path: str | None = None
    if outcome.record is not None:
        record_file = run_dir.parent.parent / record_relpath(outcome.entry_index)
        record_path = os.path.relpath(record_file, catalog_path.parent)


    dataset_name = dataset.name
    fingerprint = None
    if outcome.record is None:
        from kd.data.schema import compute_dataset_fingerprint

        fingerprint = compute_dataset_fingerprint(dataset)


    parent_run_id = (
        None if outcome.lineage is None else outcome.lineage["source_run_id"]
    )
    return catalog_row_from_record(
        outcome.record,
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        instrument=outcome.entry.instrument,
        status=outcome.status,
        run_dir=os.path.relpath(run_dir, catalog_path.parent),
        dataset_name=dataset_name,
        dataset_cache_fingerprint=fingerprint,
        seed=outcome.entry.seed,
        record_path=record_path,
        parent_run_id=parent_run_id,
        resume_from=resume_source,
        plan_hash=plan_hash,
        entry_index=global_index,
    )


def run_plan(
    plan: ExperimentPlan,
    *,
    datasets: Mapping[str, PDEDataset],
    store_root: Path,
    model_factory: Callable[..., Any] = Model,
    device: str | None = None,
    recording: RecordingOptions | None = None,
    catalog_path: Path | None = None,
    entry_indices: Sequence[int] | None = None,
    resume_from: Mapping[int, Path | str] | None = None,
    plan_hash: str | None = None,
) -> PlanRunResult:
    _preflight(plan, datasets)
    if catalog_path is not None and recording is None:
        raise ValueError(
            "catalog_path requires recording (rows point at episode run dirs)"
        )
    if entry_indices is not None and len(entry_indices) != len(plan.entries):
        raise ValueError(
            f"entry_indices length {len(entry_indices)} != plan length "
            f"{len(plan.entries)}"
        )

    store = EvidenceStore.create(
        Path(store_root),
        plan=plan,
        env=environment_fingerprint(),
    )
    catalog_plan_hash = store.plan_hash if plan_hash is None else plan_hash
    logger.info(
        "run_plan: executing %d entries of plan %r into %s",
        len(plan.entries),
        plan.name,
        store.root,
    )

    outcomes: list[EpisodeOutcome] = []
    for entry_index, entry in enumerate(plan.entries):
        global_index = (
            entry_indices[entry_index] if entry_indices is not None else entry_index
        )
        episode_kwargs: dict[str, Any] = {}
        if recording is not None:
            episode_kwargs["run_dir"] = (
                store.root / RUN_DIRS_DIRNAME / f"entry-{entry_index:04d}"
            )
            episode_kwargs["recording"] = recording
            episode_kwargs["record_ref"] = f"../../{record_relpath(entry_index)}"
        resume_path = (
            resume_from.get(global_index) if resume_from is not None else None
        )






        outcome = run_episode(
            entry=entry,
            entry_index=entry_index,
            dataset=datasets[entry.dataset_ref],
            model_factory=model_factory,
            device=device,
            resume_from=resume_path,


            persist_outcome=store.add_outcome,
            **episode_kwargs,
        )
        outcomes.append(outcome)
        if catalog_path is not None:
            if outcome.run_dir is None or outcome.run_id is None:



                continue
            append_catalog_row(
                catalog_path,
                _catalog_row_for_outcome(
                    outcome,
                    dataset=datasets[entry.dataset_ref],
                    catalog_path=catalog_path,
                    run_dir=outcome.run_dir,
                    run_id=outcome.run_id,
                    plan_hash=catalog_plan_hash,
                    global_index=global_index,
                    resume_source=(
                        str(resume_path) if resume_path is not None else None
                    ),
                ),
            )



    from kd.harness.report import write_store_report

    write_store_report(store)
    return PlanRunResult(store_root=store.root, outcomes=tuple(outcomes))


__all__ = [
    "PlanRunResult",
    "run_plan",
]
