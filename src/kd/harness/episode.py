
from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from kd.api import Model
from kd.harness.plan import PlanEntry
from kd.harness.recording import RecordingOptions
from kd.search.iteration_events import IterationEventEmitter
from kd.search.run_dir import (
    CHECKPOINTS_DIRNAME,
    RunDirPaths,
    create_run_dir,
    finalize_run_dir,
    new_run_id,
    run_id_of_run_dir,
)

if TYPE_CHECKING:
    from kd.data.schema import PDEDataset
    from kd.search.records import RunRecord
    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)



STATUS_COMPLETED: Final[str] = "completed"
STATUS_RAISED: Final[str] = "raised"
STATUS_NO_RECORD: Final[str] = "no_record"

_ERROR_MESSAGE_MAX_CHARS: Final[int] = 2000


@dataclass(frozen=True, kw_only=True)
class EpisodeOutcome:

    entry_index: int
    entry: PlanEntry
    status: str
    record: RunRecord | None
    error_type: str | None
    error_message: str | None
    wallclock_seconds: float



    run_id: str | None = None
    run_dir: Path | None = None





    lineage: dict[str, Any] | None = None


def _recording_kwargs(
    paths: RunDirPaths, options: RecordingOptions
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "callbacks": [
            IterationEventEmitter(
                jsonl_path=paths.events,
                every_n_iterations=options.events_every_n,
            )
        ]
    }
    if options.checkpoint_every is not None:
        kwargs["checkpoint_dir"] = paths.checkpoints
        kwargs["checkpoint_every"] = options.checkpoint_every
        if options.checkpoint_keep_last is not None:
            kwargs["checkpoint_keep_last"] = options.checkpoint_keep_last
    if options.phases:
        kwargs["phases_path"] = paths.phases
    return kwargs


def _attempted_lineage(resume_from: Path | str) -> dict[str, Any]:
    parent = Path(resume_from).resolve().parent
    return {
        "resume_from": str(resume_from),
        "source_run_id": (
            run_id_of_run_dir(parent.parent)
            if parent.name == CHECKPOINTS_DIRNAME
            else None
        ),
        "source_config_hash": None,
        "source_final_status": None,
        "source_iteration": None,
    }


def _finalize_recording(
    paths: RunDirPaths,
    result: ExperimentResult | None,
    *,
    run_id: str,
    entry: PlanEntry,
    status: str,
    lineage: dict[str, Any] | None,
    record_ref: str | None,
) -> None:
    finalize_run_dir(
        paths,
        result,
        run_id=run_id,
        instrument=entry.instrument,
        status=status,
        lineage=lineage,
        record_ref=(
            record_ref
            if result is not None and result.run_record is not None
            else None
        ),
    )


def run_episode(
    *,
    entry: PlanEntry,
    entry_index: int,
    dataset: PDEDataset,
    model_factory: Callable[..., Any] = Model,
    device: str | None = None,
    run_dir: Path | None = None,
    resume_from: Path | str | None = None,
    recording: RecordingOptions | None = None,
    record_ref: str | None = None,
    persist_outcome: Callable[[EpisodeOutcome], None] | None = None,
) -> EpisodeOutcome:
    if run_dir is None and (recording is not None or record_ref is not None):
        raise ValueError(
            "recording / record_ref require run_dir (they configure the run "
            "directory an episode writes)"
        )
    start = time.perf_counter()
    paths: RunDirPaths | None = None
    run_id: str | None = None
    try:
        options = recording if recording is not None else RecordingOptions()


        model_kwargs = copy.deepcopy(entry.model_kwargs)



        sibling_kwargs: dict[str, Any] = {}
        if device is not None:
            sibling_kwargs["device"] = device
        if run_dir is not None:
            paths = create_run_dir(run_dir)
            run_id = new_run_id(entry.instrument)
            sibling_kwargs.update(_recording_kwargs(paths, options))
        model = model_factory(
            algorithm=entry.instrument,
            seed=entry.seed,
            verbose=False,
            **model_kwargs,
            **sibling_kwargs,
        )
        if resume_from is not None:
            model.fit(dataset, resume_from=resume_from)
        else:
            model.fit(dataset)
        result: ExperimentResult | None = model.result_
        record: RunRecord | None = model.result_.run_record
    except Exception as exc:
        elapsed = time.perf_counter() - start
        error_type = type(exc).__qualname__
        error_message = str(exc)[:_ERROR_MESSAGE_MAX_CHARS]
        logger.info(
            "episode %d (%s/%s seed=%d) raised %s after %.3fs",
            entry_index,
            entry.instrument,
            entry.dataset_ref,
            entry.seed,
            error_type,
            elapsed,
        )
        outcome = EpisodeOutcome(
            entry_index=entry_index,
            entry=entry,
            status=STATUS_RAISED,
            record=None,
            error_type=error_type,
            error_message=error_message,
            wallclock_seconds=elapsed,
            run_id=run_id,
            run_dir=None if paths is None else paths.root,

            lineage=(
                None if resume_from is None else _attempted_lineage(resume_from)
            ),
        )
        if persist_outcome is not None:
            persist_outcome(outcome)
        if paths is not None and run_id is not None:





            _finalize_recording(
                paths,
                None,
                run_id=run_id,
                entry=entry,
                status=STATUS_RAISED,
                lineage=outcome.lineage,
                record_ref=None,
            )
        return outcome

    elapsed = time.perf_counter() - start
    status = STATUS_NO_RECORD if record is None else STATUS_COMPLETED
    if record is None:
        logger.info(
            "episode %d (%s/%s seed=%d) produced no run_record after %.3fs",
            entry_index,
            entry.instrument,
            entry.dataset_ref,
            entry.seed,
            elapsed,
        )
    else:
        logger.info(
            "episode %d (%s/%s seed=%d) completed in %.3fs",
            entry_index,
            entry.instrument,
            entry.dataset_ref,
            entry.seed,
            elapsed,
        )
    outcome = EpisodeOutcome(
        entry_index=entry_index,
        entry=entry,
        status=status,
        record=record,
        error_type=None,
        error_message=None,
        wallclock_seconds=elapsed,
        run_id=run_id,
        run_dir=None if paths is None else paths.root,


        lineage=(
            result.manifest.resume_source
            if result is not None and result.manifest is not None
            else None
        ),
    )
    if persist_outcome is not None:
        persist_outcome(outcome)
    if paths is not None and run_id is not None:


        _finalize_recording(
            paths,
            result,
            run_id=run_id,
            entry=entry,
            status=status,
            lineage=outcome.lineage,
            record_ref=record_ref,
        )
    return outcome


__all__ = [
    "STATUS_COMPLETED",
    "STATUS_NO_RECORD",
    "STATUS_RAISED",
    "EpisodeOutcome",
    "run_episode",
]
