
from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import torch

import kd
from kd.api import Model
from kd.core.jsonsafe import JSON_INDENT_SPACES
from kd.data.regression import TabularDataset
from kd.data.tabular_bridge import dataset_from_tabular
from kd.harness._verify import record_relpath
from kd.harness.episode import (
    STATUS_COMPLETED,
    STATUS_RAISED,
    EpisodeOutcome,
    run_episode,
)
from kd.harness.jobs import JobResult, run_jobs
from kd.harness.plan import ExperimentPlan, PlanEntry
from kd.harness.recording import RecordingOptions
from kd.harness.schedule import (
    PINNED_ENVIRONMENT,
    Job,
    ResourceLimits,
    child_environment,
)
from kd.harness.store import EvidenceStore, environment_fingerprint
from kd.harness.supervise import (
    ProcessResult,
    ProcessSupervisor,
    check_memory_cap_available,
)
from kd.search.records import RunRecord
from kd.search.run_catalog import append_catalog_row, catalog_row_from_record

if TYPE_CHECKING:
    from kd.data.schema import PDEDataset

logger = logging.getLogger(__name__)





RUN_DIRS_DIRNAME = "runs"





JOBS_DIRNAME: Final[str] = "jobs"

JOB_FILENAME: Final[str] = "job.json"

OUTCOME_FILENAME: Final[str] = "outcome.json"

STAGED_RECORD_FILENAME: Final[str] = "record.json"


PROCESS_FILENAME: Final[str] = "process.json"

WORKER_LOG_FILENAME: Final[str] = "worker.log"


PROCESS_ENDED_ERROR: Final[str] = "EpisodeProcessEnded"

_JOB_DEVICES: Final = (None, "cpu", "cuda")


@dataclass(frozen=True, kw_only=True)
class PlanRunResult:

    store_root: Path
    outcomes: tuple[EpisodeOutcome, ...]


@dataclass(frozen=True, kw_only=True)
class ParallelOptions:

    limits: ResourceLimits
    device_by_instrument: Mapping[str, str] = field(default_factory=dict)
    memory_estimate_gb: Mapping[str, float] | float = 0.0
    blas_threads: int = 1
    python_hash_seed: int | None = None
    hard_seconds: float | None = None
    grace_seconds: float = 30.0
    memory_max_gb: float | None = None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=JSON_INDENT_SPACES, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def stage_outcome(job_dir: Path, outcome: EpisodeOutcome) -> None:
    if outcome.record is not None:
        outcome.record.save(job_dir / STAGED_RECORD_FILENAME)
    _write_json_atomic(
        job_dir / OUTCOME_FILENAME,
        {
            "entry_index": outcome.entry_index,
            "status": outcome.status,
            "error_type": outcome.error_type,
            "error_message": outcome.error_message,
            "error_traceback": outcome.error_traceback,
            "wallclock_seconds": outcome.wallclock_seconds,
            "run_id": outcome.run_id,
            "run_dir": None if outcome.run_dir is None else str(outcome.run_dir),
            "lineage": outcome.lineage,
        },
    )


def _collected_outcome(
    job_dir: Path, entry_index: int, entry: PlanEntry, process: ProcessResult
) -> EpisodeOutcome:
    path = job_dir / OUTCOME_FILENAME
    if not path.is_file():
        return EpisodeOutcome(
            entry_index=entry_index,
            entry=entry,
            status=STATUS_RAISED,
            record=None,
            error_type=PROCESS_ENDED_ERROR,
            error_message=(
                "the entry's process ended without an outcome: "
                f"ended_by={process.ended_by.value}, "
                f"stop_reason={process.stop_reason}, "
                f"returncode={process.returncode}, "
                f"peak_rss_bytes={process.peak_rss_bytes}"
            ),
            wallclock_seconds=process.wallclock_seconds,
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return EpisodeOutcome(
        entry_index=data["entry_index"],
        entry=entry,
        status=data["status"],
        record=(
            RunRecord.load(job_dir / STAGED_RECORD_FILENAME)
            if data["status"] == STATUS_COMPLETED
            else None
        ),
        error_type=data["error_type"],
        error_message=data["error_message"],
        error_traceback=data["error_traceback"],
        wallclock_seconds=data["wallclock_seconds"],
        run_id=data["run_id"],
        run_dir=None if data["run_dir"] is None else Path(data["run_dir"]),
        lineage=data["lineage"],
    )


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
        plan_hash=plan_hash,
        entry_index=outcome.entry_index,
    )


def _parallel_devices(
    plan: ExperimentPlan,
    *,
    model_factory: Callable[..., Any],
    device: str | None,
    parallel: ParallelOptions,
) -> list[str | None]:
    if model_factory is not Model:
        raise ValueError(
            "parallel runs build the facade in each child process, so "
            "model_factory cannot be injected (a callable does not cross the "
            "process boundary)"
        )
    local = sorted(
        {
            entry.instrument
            for entry in plan.entries
            if kd.extension_source("algorithm", entry.instrument) == "process"
        }
    )
    if local:
        raise ValueError(
            f"instruments {local} are registered in this process only; a "
            "parallel child imports kd without them. Register them from an "
            "installed kd.extensions entry-point hook, or run the plan serially"
        )
    devices = [
        parallel.device_by_instrument.get(entry.instrument, device)
        for entry in plan.entries
    ]

    unknown = sorted({str(d) for d in devices if d not in _JOB_DEVICES})
    if unknown:
        raise ValueError(
            f"parallel entry devices must be None, 'cpu' or 'cuda'; got {unknown}"
        )
    if parallel.memory_max_gb is not None:
        check_memory_cap_available()
    return devices


class _ParallelBatch:

    def __init__(
        self,
        plan: ExperimentPlan,
        *,
        store: EvidenceStore,
        datasets: Mapping[str, PDEDataset],
        recording: RecordingOptions | None,
        catalog_path: Path | None,
        parallel: ParallelOptions,
    ) -> None:
        self._plan = plan
        self._store = store
        self._datasets = datasets
        self._recording = recording
        self._catalog_path = catalog_path
        self._parallel = parallel
        self._root = store.root / JOBS_DIRNAME
        self._environments: dict[str, dict[str, str | None]] = {}
        self.outcomes: dict[int, EpisodeOutcome] = {}

    def write_jobs(self, devices: list[str | None]) -> list[Job]:
        datasets_dir = self._root / "datasets"
        datasets_dir.mkdir(parents=True)
        files: dict[str, Path] = {}
        jobs = []
        for index, (entry, device) in enumerate(
            zip(self._plan.entries, devices, strict=True)
        ):
            if entry.dataset_ref not in files:

                path = (datasets_dir / f"dataset-{len(files):02d}.pt").resolve()
                torch.save(self._datasets[entry.dataset_ref], path)
                files[entry.dataset_ref] = path
            key = f"entry-{index:04d}"
            (self._root / key).mkdir()
            self._write_job(key, index, entry, device, files[entry.dataset_ref])
            estimate = self._parallel.memory_estimate_gb
            jobs.append(
                Job(
                    key=key,
                    device="cuda" if device == "cuda" else "cpu",
                    memory_estimate_gb=(
                        estimate
                        if isinstance(estimate, float | int)
                        else estimate.get(entry.instrument, 0.0)
                    ),
                    payload=index,
                )
            )
        return jobs

    def _write_job(
        self,
        key: str,
        index: int,
        entry: PlanEntry,
        device: str | None,
        dataset_file: Path,
    ) -> None:
        recording = self._recording
        _write_json_atomic(
            self._root / key / JOB_FILENAME,
            {
                "entry": entry.to_dict(),
                "entry_index": index,
                "dataset_file": str(dataset_file),
                "device": device,

                "run_dir": (
                    None
                    if recording is None
                    else str((self._store.root / RUN_DIRS_DIRNAME / key).resolve())
                ),
                "record_ref": (
                    None if recording is None else f"../../{record_relpath(index)}"
                ),
                "recording": None if recording is None else asdict(recording),
            },
        )

    def launch(self, job: Job, gpu: int | None) -> ProcessSupervisor:
        env = child_environment(
            os.environ,
            blas_threads=self._parallel.blas_threads,
            python_hash_seed=self._parallel.python_hash_seed,
            gpu=gpu,
        )
        pinned = {name: env.get(name) for name in PINNED_ENVIRONMENT}
        if gpu is not None:
            pinned["CUDA_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"]
        self._environments[job.key] = pinned
        job_dir = self._root / job.key
        with (job_dir / WORKER_LOG_FILENAME).open("ab") as log:
            return ProcessSupervisor(
                [
                    sys.executable,
                    "-m",
                    "kd.harness._entry_worker",
                    str((job_dir / JOB_FILENAME).resolve()),
                ],
                env=env,
                hard_seconds=self._parallel.hard_seconds,
                grace_seconds=self._parallel.grace_seconds,
                stdout=log,
                stderr=log,
                memory_max_gb=self._parallel.memory_max_gb,
            )

    def finished(self, result: JobResult) -> None:
        index: int = result.job.payload
        entry = self._plan.entries[index]
        job_dir = self._root / result.job.key
        outcome = _collected_outcome(job_dir, index, entry, result.process)
        self._store.add_outcome(outcome)
        self.outcomes[index] = outcome
        _write_json_atomic(
            job_dir / PROCESS_FILENAME,
            {
                **asdict(result.process),
                "ended_by": result.process.ended_by.value,
                "gpu": result.gpu,
                "environment": self._environments[result.job.key],
            },
        )
        if (
            self._catalog_path is not None
            and outcome.run_dir is not None
            and outcome.run_id is not None
        ):
            append_catalog_row(
                self._catalog_path,
                _catalog_row_for_outcome(
                    outcome,
                    dataset=self._datasets[entry.dataset_ref],
                    catalog_path=self._catalog_path,
                    run_dir=outcome.run_dir,
                    run_id=outcome.run_id,
                    plan_hash=self._store.plan_hash,
                ),
            )


def _run_serial(
    plan: ExperimentPlan,
    *,
    store: EvidenceStore,
    datasets: Mapping[str, PDEDataset],
    model_factory: Callable[..., Any],
    device: str | None,
    recording: RecordingOptions | None,
    catalog_path: Path | None,
) -> list[EpisodeOutcome]:
    outcomes: list[EpisodeOutcome] = []
    for entry_index, entry in enumerate(plan.entries):
        episode_kwargs: dict[str, Any] = {}
        if recording is not None:
            episode_kwargs["run_dir"] = (
                store.root / RUN_DIRS_DIRNAME / f"entry-{entry_index:04d}"
            )
            episode_kwargs["recording"] = recording
            episode_kwargs["record_ref"] = f"../../{record_relpath(entry_index)}"






        outcome = run_episode(
            entry=entry,
            entry_index=entry_index,
            dataset=datasets[entry.dataset_ref],
            model_factory=model_factory,
            device=device,


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
                    plan_hash=store.plan_hash,
                ),
            )
    return outcomes


def run_plan(
    plan: ExperimentPlan,
    *,
    datasets: Mapping[str, PDEDataset | TabularDataset],
    store_root: Path,
    model_factory: Callable[..., Any] = Model,
    device: str | None = None,
    recording: RecordingOptions | None = None,
    catalog_path: Path | None = None,
    parallel: ParallelOptions | None = None,
) -> PlanRunResult:







    normalized: dict[str, PDEDataset] = {
        ref: dataset_from_tabular(ds) if isinstance(ds, TabularDataset) else ds
        for ref, ds in datasets.items()
    }
    _preflight(plan, normalized)
    if catalog_path is not None and recording is None:
        raise ValueError(
            "catalog_path requires recording (rows point at episode run dirs)"
        )


    devices = (
        []
        if parallel is None
        else _parallel_devices(
            plan, model_factory=model_factory, device=device, parallel=parallel
        )
    )

    store = EvidenceStore.create(
        Path(store_root),
        plan=plan,
        env=environment_fingerprint(),
    )
    logger.info(
        "run_plan: executing %d entries of plan %r into %s",
        len(plan.entries),
        plan.name,
        store.root,
    )

    if parallel is None:
        outcomes = _run_serial(
            plan,
            store=store,
            datasets=normalized,
            model_factory=model_factory,
            device=device,
            recording=recording,
            catalog_path=catalog_path,
        )
    else:
        batch = _ParallelBatch(
            plan,
            store=store,
            datasets=normalized,
            recording=recording,
            catalog_path=catalog_path,
            parallel=parallel,
        )
        run_jobs(
            batch.write_jobs(devices),
            limits=parallel.limits,
            launch=batch.launch,
            finished=batch.finished,
        )

        outcomes = [batch.outcomes[i] for i in range(len(plan.entries))]



    from kd.harness.report import write_store_report

    write_store_report(store)
    return PlanRunResult(store_root=store.root, outcomes=tuple(outcomes))


__all__ = [
    "ParallelOptions",
    "PlanRunResult",
    "run_plan",
]
