
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter
from pydantic import TypeAdapter, ValidationError

from kdui import jobstore
from kdui.api import ApiError, SchedulerDep, SettingsDep
from kdui.events import CursorError, read_records, read_text
from kdui.models import (
    ErrorCode,
    EventsPage,
    JobAccepted,
    JobDetail,
    JobKind,
    JobList,
    JobQueue,
    JobRequest,
    JobResult,
    JobState,
    LogPage,
    ProgressEvent,
)
from kdui.scheduler import Scheduler, UnknownJobError, is_job_id
from kdui.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["jobs"])

_EVENT = TypeAdapter[ProgressEvent](ProgressEvent)


def _not_found(job_id: str) -> ApiError:
    return ApiError(404, ErrorCode.NOT_FOUND, f"no job {job_id!r}")


def _bad_cursor(error: CursorError) -> ApiError:
    return ApiError(400, ErrorCode.INVALID_REQUEST, str(error), field="cursor")


def _job_dir(settings: Settings, job_id: str) -> Path:
    job_dir = settings.workspace.job_dir(job_id)
    if not is_job_id(job_id) or not (job_dir / jobstore.REQUEST_FILE).exists():
        raise _not_found(job_id)
    return job_dir


def _detail(scheduler: Scheduler, job_id: str) -> JobDetail:
    try:
        return scheduler.detail(job_id)
    except UnknownJobError:
        raise _not_found(job_id) from None


@router.post("/jobs", response_model=JobAccepted, status_code=202)
def submit_job(body: JobRequest, scheduler: SchedulerDep) -> JobAccepted:
    return JobAccepted(jobs=scheduler.submit(body))


@router.get("/jobs", response_model=JobList)
def list_jobs(
    scheduler: SchedulerDep,
    state: JobState | None = None,
    kind: JobKind | None = None,
    queue: JobQueue | None = None,
    limit: int = 50,
    cursor: str | None = None,
) -> JobList:
    jobs = scheduler.summaries(state=state, kind=kind, queue=queue)
    if cursor is not None:
        ids = [job.job_id for job in jobs]
        if cursor not in ids:
            raise ApiError(
                400, ErrorCode.INVALID_REQUEST, "unknown page cursor", field="cursor"
            )
        jobs = jobs[ids.index(cursor) + 1:]
    page = jobs[: max(1, limit)]
    more = len(jobs) > len(page)
    return JobList(jobs=page, next_cursor=page[-1].job_id if more else None)


@router.get("/jobs/{job_id}", response_model=JobDetail)
def read_job(job_id: str, scheduler: SchedulerDep) -> JobDetail:
    return _detail(scheduler, job_id)


@router.get("/jobs/{job_id}/result", response_model=JobResult)
def read_job_result(job_id: str, settings: SettingsDep) -> JobResult:
    job_dir = _job_dir(settings, job_id)
    if not (job_dir / jobstore.RESULT_FILE).exists():
        raise ApiError(409, ErrorCode.NOT_READY, f"job {job_id!r} has no result yet")
    return jobstore.read_result(job_dir)


@router.get("/jobs/{job_id}/events", response_model=EventsPage)
def read_job_events(
    job_id: str, settings: SettingsDep, cursor: str | None = None
) -> EventsPage:
    path = _job_dir(settings, job_id) / jobstore.PROGRESS_FILE
    try:
        page = read_records(path, cursor, max_bytes=settings.page_max_bytes)
    except CursorError as error:
        raise _bad_cursor(error) from None
    events: list[ProgressEvent] = []
    skipped = page.skipped
    for record in page.records:
        try:
            events.append(_EVENT.validate_python(record))
        except ValidationError:
            skipped += 1
    if skipped:
        logger.warning(
            "job %s: skipped %d unreadable progress line(s)", job_id, skipped
        )
    return EventsPage(
        events=events,
        next_cursor=page.next_cursor,
        has_more=page.has_more,
        reset=page.reset,
        skipped=skipped,
    )


@router.get("/jobs/{job_id}/logs", response_model=LogPage)
def read_job_logs(
    job_id: str, settings: SettingsDep, cursor: str | None = None
) -> LogPage:
    path = _job_dir(settings, job_id) / jobstore.LOG_FILE
    try:
        page = read_text(path, cursor, max_bytes=settings.page_max_bytes)
    except CursorError as error:
        raise _bad_cursor(error) from None
    return LogPage(
        text=page.text,
        next_cursor=page.next_cursor,
        has_more=page.has_more,
        reset=page.reset,
    )


@router.post("/jobs/{job_id}/stop", response_model=JobDetail, status_code=202)
def stop_job(job_id: str, scheduler: SchedulerDep) -> JobDetail:
    try:
        scheduler.stop(job_id)
    except UnknownJobError:
        raise _not_found(job_id) from None
    return _detail(scheduler, job_id)


@router.post("/jobs/{job_id}/kill", response_model=JobDetail, status_code=202)
def kill_job(job_id: str, scheduler: SchedulerDep) -> JobDetail:
    try:
        scheduler.kill(job_id)
    except UnknownJobError:
        raise _not_found(job_id) from None
    return _detail(scheduler, job_id)
