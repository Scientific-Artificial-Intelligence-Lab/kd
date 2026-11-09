
from __future__ import annotations

import json
import logging
import signal
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any

import psutil

import kd
from kd.harness import (
    REPORT_SIDECAR_DIRNAME,
    RESULT_SIDECAR_FILENAME,
    EpisodeOutcome,
    PlanEntry,
    RecordingOptions,
    environment_fingerprint,
    run_episode,
)
from kd.search import (
    CATALOG_FILENAME,
    RunDirPaths,
    append_catalog_row,
    catalog_row_from_record,
)
from kdui import jobstore
from kdui.jobstore import JobContext
from kdui.models import (
    FitRequest,
    FitResult,
    JobError,
    JobOutcome,
    Lineage,
    StoredJob,
    VerificationCheck,
)
from kdui.ops.inspect import load_dataset
from kdui.ops.preprocess import apply_preprocessing
from kdui.ops.verify import check_run
from kdui.settings import Settings

logger = logging.getLogger(__name__)

ENVIRONMENT_FILE = "environment.json"
VERIFICATION_FILE = "verification.json"
ARTIFACT_ERRORS_FILE = "errors.json"



_MIN_BUDGET_SECONDS = 0.001


class _StopFile:

    def __init__(self, job_dir: Path) -> None:
        self._job_dir = job_dir

    def on_experiment_start(self, algorithm: Any) -> None:
        pass

    def on_iteration_start(self, iteration: int, algorithm: Any) -> None:
        pass

    def on_iteration_end(
        self, iteration: int, algorithm: Any, candidates: Any, results: Any
    ) -> None:
        pass

    def on_experiment_end(self, algorithm: Any) -> None:
        pass

    @property
    def should_stop(self) -> bool:
        return jobstore.stop_requested(self._job_dir)


def _interrupt(signum: int, frame: FrameType | None) -> None:
    raise kd.SearchInterrupted("SIGTERM received: the fit was asked to end")


def _soft_budget(max_seconds: float | None) -> float | None:
    if max_seconds is None:
        return None
    spent = time.time() - psutil.Process().create_time()
    return max(_MIN_BUDGET_SECONDS, max_seconds - spent)


def _resume_checkpoint(job: JobContext, lineage: Lineage | None) -> Path | None:
    if lineage is None or lineage.resumed_from_run_id is None:
        return None
    parent = RunDirPaths(root=job.workspace.run_dir(lineage.resumed_from_run_id))
    return kd.resolve_checkpoint(parent.checkpoints, lineage.resume_iteration).path


def _raised_outcome(episode: EpisodeOutcome) -> JobOutcome:
    if episode.status == "raised":
        assert episode.error_type is not None and episode.error_message is not None
        error = JobError(error_type=episode.error_type, message=episode.error_message)
    else:
        error = JobError(
            error_type="NoRecord", message="The fit ended without a sealed record."
        )
    return JobOutcome(
        at=datetime.now(UTC), status="raised", kd_run_id=episode.run_id, error=error
    )


def _catalog_row(
    request: FitRequest, job: JobContext, episode: EpisodeOutcome
) -> dict[str, Any]:
    assert episode.run_id is not None
    lineage = episode.lineage
    resume_from = None
    if lineage is not None:
        resume_from = (
            Path(lineage["resume_from"]).relative_to(job.workspace.runs).as_posix()
        )
    return catalog_row_from_record(
        episode.record,
        run_id=episode.run_id,
        created_at=datetime.now(UTC).isoformat(),
        instrument=request.algorithm,
        status=episode.status,
        run_dir=job.job_id,
        seed=request.seeds[0],
        parent_run_id=None if lineage is None else lineage["source_run_id"],
        resume_from=resume_from,
    )


def _after_seal(job: JobContext, steps: dict[str, Callable[[], object]]) -> None:
    errors = []
    for name, step in steps.items():
        try:
            step()
        except Exception as error:


            logger.exception("after the seal: %s failed", name)
            errors.append(
                {
                    "step": name,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
    if errors:
        job.artifacts_dir.mkdir(parents=True, exist_ok=True)
        jobstore.write_once(
            job.artifacts_dir / ARTIFACT_ERRORS_FILE,
            json.dumps(errors, ensure_ascii=False, indent=2),
        )


def run_fit(request: FitRequest, job: JobContext) -> FitResult:
    stored: StoredJob = jobstore.read_request(job.job_dir)
    (seed,) = request.seeds
    signal.signal(signal.SIGTERM, _interrupt)
    loaded = load_dataset(
        job.workspace, request.dataset_id, with_reference=stored.reference_in_report
    )
    loaded_fingerprint = kd.compute_dataset_fingerprint(loaded)
    dataset = apply_preprocessing(loaded, request.preprocessing)

    built: list[kd.Model] = []

    def keeping_factory(**kwargs: Any) -> kd.Model:
        model = kd.Model(**kwargs)
        built.append(model)
        return model

    resume_from = _resume_checkpoint(job, stored.lineage)
    episode = run_episode(
        entry=PlanEntry(
            instrument=request.algorithm,
            dataset_ref=request.dataset_id,
            seed=seed,
            model_kwargs={
                name: value for name, value in request.params.items() if name != "seed"
            },
        ),
        entry_index=0,
        dataset=dataset,
        model_factory=keeping_factory,
        device=request.device,
        run_dir=job.run_dir,
        resume_from=resume_from,
        reseed=stored.lineage is not None and stored.lineage.reseed,
        sketch=(
            None
            if request.sketch is None
            else kd.load_sketch(request.sketch, dataset=dataset)
        ),
        max_seconds=_soft_budget(request.max_seconds),
        recording=RecordingOptions(
            checkpoint_every=(
                Settings.checkpoint_every
                if request.checkpoint_every is None
                else request.checkpoint_every
            ),
            checkpoint_keep_last=Settings.checkpoint_keep_last,
        ),
        extra_callbacks=[_StopFile(job.job_dir)],
    )
    if episode.record is None:
        outcome = _raised_outcome(episode)
        jobstore.write_outcome(job.job_dir, outcome)
        assert outcome.error is not None
        logger.error("the fit raised:\n%s", episode.error_traceback)
        raise RuntimeError(f"{outcome.error.error_type}: {outcome.error.message}")
    jobstore.write_outcome(
        job.job_dir,
        JobOutcome(
            at=datetime.now(UTC),
            status="completed",
            kd_stop_reason=episode.record.cost.stop_reason,
            kd_run_id=episode.run_id,
        ),
    )
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    model = built[0]
    run_dir = job.run_dir

    def write_environment() -> None:
        jobstore.write_once(
            run_dir / ENVIRONMENT_FILE,
            json.dumps(
                {
                    "scheme": "kdui-environment-v1",
                    "environment": environment_fingerprint(),
                    "dataset_id": request.dataset_id,
                    "loaded_fingerprint": loaded_fingerprint,
                    "preprocessing": request.preprocessing.model_dump(mode="json"),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

    def write_verification() -> None:
        checks = check_run(
            run_dir,
            dataset,
            [VerificationCheck.ABLATION, VerificationCheck.BLOCK_DISPERSION],
            run_id=job.job_id,
            artifacts_dir=job.artifacts_dir,
        )
        jobstore.write_once(run_dir / VERIFICATION_FILE, checks.model_dump_json())

    _after_seal(
        job,
        {
            "result": lambda: model.result_.save(run_dir / RESULT_SIDECAR_FILENAME),
            "environment": write_environment,
            "catalog": lambda: append_catalog_row(
                job.workspace.runs / CATALOG_FILENAME,
                _catalog_row(request, job, episode),
            ),
            "report": lambda: model.report(
                run_dir / REPORT_SIDECAR_DIRNAME,
                include_ground_truth=stored.reference_in_report,
            ),
            "verification": write_verification,
        },
    )
    return FitResult(run_id=job.job_id)
