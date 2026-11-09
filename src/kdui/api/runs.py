
from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import threading
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from fastapi import APIRouter
from pydantic import BaseModel, TypeAdapter

from kdui import kdproc
from kdui.api import ApiError, SchedulerDep, SettingsDep
from kdui.models import (
    TERMINAL_STATES,
    CandidateList,
    CandidatesRequest,
    CheckpointList,
    CheckpointsRequest,
    ErrorCode,
    FitRequest,
    HistoryImportRequest,
    HistoryScanRequest,
    HistorySource,
    HistorySourceList,
    JobAccepted,
    JobKind,
    JobResult,
    JobState,
    Lineage,
    ResumeRequest,
    RunDetail,
    RunList,
    RunListRequest,
    RunOrigin,
    RunProjectionRequest,
    StoredJob,
    VerificationList,
    VerifyResult,
)
from kdui.scheduler import Scheduler
from kdui.settings import Settings, Workspace

logger = logging.getLogger(__name__)

router = APIRouter(tags=["runs"])

ResponseT = TypeVar("ResponseT", bound=BaseModel)

RUNDIR_SCHEME = "kd-rundir-v1"
MANIFEST_FILE = "manifest.json"
LIVE_FILES = ("events.jsonl", "phases.jsonl")
VERIFICATION_FILE = "verification.json"
IMPORTED_PREFIX = "i-"

_JOB_RESULT: TypeAdapter[JobResult] = TypeAdapter(JobResult)


class UnknownRunError(LookupError):
    pass


class ManifestError(Exception):

    def __init__(self, directory: Path, reason: str) -> None:
        super().__init__(
            f"The folder {directory} has a {MANIFEST_FILE} that is not valid "
            f"JSON ({reason}). Repair or remove that file to read the run in it."
        )
        self.directory = directory
        self.reason = reason


@dataclass(frozen=True)
class RunDirs:

    found: list[Path]
    unreadable: list[ManifestError]







def read_manifest(directory: Path) -> dict[str, Any]:
    try:
        manifest: dict[str, Any] = json.loads(
            (directory / MANIFEST_FILE).read_text(encoding="utf-8")
        )
    except ValueError as error:
        raise ManifestError(directory, str(error)) from error
    return manifest


def is_run_dir(directory: Path, files: list[str] | set[str]) -> bool:
    if MANIFEST_FILE in files:
        return bool(read_manifest(directory).get("scheme") == RUNDIR_SCHEME)
    return any(name in files for name in LIVE_FILES)


def record_path(run_dir: Path, manifest: dict[str, Any]) -> Path:
    reference = manifest.get("record_ref")
    return run_dir / (reference["path"] if reference is not None else "record.json")


def discover_run_dirs(root: Path) -> RunDirs:
    found: list[Path] = []
    unreadable: list[ManifestError] = []
    for current, directories, files in os.walk(root):
        directory = Path(current)
        try:
            run = is_run_dir(directory, files)
        except ManifestError as error:
            unreadable.append(error)
            directories.clear()
            continue
        if run:
            directories.clear()
            found.append(directory)
        else:
            directories.sort()
    return RunDirs(found, unreadable)


def imported_run_id(source_id: str, root: Path, run_dir: Path) -> str:
    relative = run_dir.relative_to(root).as_posix()
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:10]
    return f"{IMPORTED_PREFIX}{source_id.rsplit('-', 1)[-1]}-{digest}"


def ui_runs(workspace: Workspace) -> list[RunProjectionRequest]:
    return [
        _ui_run(workspace, directory.name)
        for directory in sorted(workspace.runs.iterdir())
        if directory.is_dir()
    ]


def source_runs(
    source: HistorySource,
) -> tuple[list[RunProjectionRequest], list[ManifestError]]:
    root = Path(source.path)
    walked = discover_run_dirs(root)
    runs = [
        RunProjectionRequest(
            run_id=imported_run_id(source.source_id, root, run_dir),
            origin=RunOrigin.IMPORTED,
            run_dir=str(run_dir),
            job_dir=None,
            source_id=source.source_id,
        )
        for run_dir in walked.found
    ]
    return runs, walked.unreadable


def read_sources(workspace: Workspace) -> list[HistorySource]:
    sources = [
        HistorySource.model_validate_json(path.read_text(encoding="utf-8"))
        for path in workspace.history.glob("*.json")
    ]
    return sorted(sources, key=lambda source: source.imported_at)


def resolve_run(workspace: Workspace, run_id: str) -> RunProjectionRequest:
    if run_id.startswith(IMPORTED_PREFIX):
        source_part = run_id.removeprefix(IMPORTED_PREFIX).split("-", 1)[0]
        for source in read_sources(workspace):
            if source.source_id.rsplit("-", 1)[-1] != source_part:
                continue
            runs, unreadable = source_runs(source)
            for run in runs:
                if run.run_id == run_id:
                    return run
            for error in unreadable:
                root = Path(source.path)
                if imported_run_id(source.source_id, root, error.directory) == run_id:
                    raise error
        raise UnknownRunError(run_id)
    run_dir = workspace.run_dir(run_id)
    if run_id in {"", ".", ".."} or run_dir.parent != workspace.runs:
        raise UnknownRunError(run_id)
    if not run_dir.is_dir():
        raise UnknownRunError(run_id)
    return _ui_run(workspace, run_id)


def _ui_run(workspace: Workspace, job_id: str) -> RunProjectionRequest:
    job_dir = workspace.job_dir(job_id)
    return RunProjectionRequest(
        run_id=job_id,
        origin=RunOrigin.UI,
        run_dir=str(workspace.run_dir(job_id)),
        job_dir=str(job_dir) if job_dir.is_dir() else None,
        source_id=None,
    )


def write_once(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    temporary.write_text(text, encoding="utf-8")
    if path.exists():
        temporary.unlink()
        raise FileExistsError(path)
    os.replace(temporary, path)







def kd_call(
    settings: Settings,
    operation: str,
    request: BaseModel,
    response_type: type[ResponseT],
) -> ResponseT:
    try:
        return kdproc.call(
            operation,
            request,
            response_type,
            timeout_seconds=settings.kdproc_timeout_seconds,
        )
    except kdproc.KdCallError as error:
        raise ApiError(422, ErrorCode.KD_ERROR, error.message) from error


_CACHE_ENTRIES = 64
_cache: OrderedDict[tuple[str, str], tuple[tuple[object, ...], BaseModel]] = (
    OrderedDict()
)
_cache_lock = threading.Lock()


def _file_entry(path: str) -> tuple[str, int, int, int, int] | None:
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return None
    return (path, stat.st_ino, stat.st_ctime_ns, stat.st_mtime_ns, stat.st_size)


def _files_stamp(roots: Iterable[Path]) -> tuple[object, ...]:
    files: list[tuple[str, int, int, int, int]] = []
    records: list[tuple[str, object]] = []
    for root in roots:
        for current, _, names in os.walk(root):
            for name in names:


                entry = _file_entry(os.path.join(current, name))
                if entry is not None:
                    files.append(entry)
        for run_dir in discover_run_dirs(root).found:
            if (run_dir / MANIFEST_FILE).is_file():
                record = str(record_path(run_dir, read_manifest(run_dir)))
                records.append((record, _file_entry(record)))
    return (tuple(sorted(files)), tuple(records))


def _cached_kd_call(
    settings: Settings,
    operation: str,
    request: BaseModel,
    response_type: type[ResponseT],
    roots: Iterable[Path],
) -> ResponseT:
    key = (operation, request.model_dump_json())
    stamp = _files_stamp(roots)
    with _cache_lock:
        kept = _cache.get(key)
        if kept is not None and kept[0] == stamp:
            _cache.move_to_end(key)
            answer = kept[1]
            assert isinstance(answer, response_type)
            return answer
    answer = kd_call(settings, operation, request, response_type)
    with _cache_lock:
        _cache[key] = (stamp, answer)
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_ENTRIES:
            _cache.popitem(last=False)
    return answer


def _run_roots(run: RunProjectionRequest) -> list[Path]:
    return [Path(run.run_dir)] + ([] if run.job_dir is None else [Path(run.job_dir)])


def _run(settings: Settings, run_id: str) -> RunProjectionRequest:
    try:
        return resolve_run(settings.workspace, run_id)
    except UnknownRunError as error:
        raise ApiError(404, ErrorCode.NOT_FOUND, f"no run {run_id}") from error


def _stored_job(run: RunProjectionRequest) -> StoredJob:
    if run.job_dir is None:
        raise ApiError(
            409,
            ErrorCode.CONFLICT,
            f"run {run.run_id} was started outside the UI: the UI has no request "
            "and no registered dataset for it",
        )
    text = (Path(run.job_dir) / "request.json").read_text(encoding="utf-8")
    return StoredJob.model_validate_json(text)


def _live(scheduler: Scheduler, run: RunProjectionRequest) -> bool:
    return (
        run.job_dir is not None
        and scheduler.summary(run.run_id).state not in TERMINAL_STATES
    )


def _projection(
    settings: Settings,
    scheduler: Scheduler,
    run: RunProjectionRequest,
    operation: str,
    request: BaseModel,
    response_type: type[ResponseT],
) -> ResponseT:
    if _live(scheduler, run):
        return kd_call(settings, operation, request, response_type)
    return _cached_kd_call(settings, operation, request, response_type, _run_roots(run))


def _checkpoints(
    settings: Settings, scheduler: Scheduler, run: RunProjectionRequest
) -> CheckpointList:
    if _live(scheduler, run):
        return CheckpointList(
            run_id=run.run_id,
            resumable=False,
            reason="the fit is still going; it can be resumed once it has ended",
            unit=None,
            checkpoints=[],
            resume_safe_params=[],
        )
    request = CheckpointsRequest(run_id=run.run_id, run_dir=run.run_dir)
    return _cached_kd_call(
        settings,
        "result.list_checkpoints",
        request,
        CheckpointList,
        [Path(run.run_dir)],
    )







@router.get("/runs", response_model=RunList)
def list_runs(
    settings: SettingsDep,
    scheduler: SchedulerDep,
    dataset_id: str | None = None,
    algorithm: str | None = None,
    origin: RunOrigin | None = None,
    limit: int = 50,
    cursor: str | None = None,
) -> RunList:
    workspace = settings.workspace
    sources = read_sources(workspace)
    request = RunListRequest(workspace=str(workspace.root), sources=sources)
    if scheduler.summaries(kind=JobKind.FIT, state=JobState.RUNNING):
        listing = kd_call(settings, "history.list_runs", request, RunList)
    else:
        roots = [workspace.runs, workspace.jobs]
        roots += [Path(source.path) for source in sources]
        listing = _cached_kd_call(
            settings, "history.list_runs", request, RunList, roots
        )
    runs = [
        run
        for run in listing.runs
        if (dataset_id is None or run.dataset_id == dataset_id)
        and (algorithm is None or run.algorithm == algorithm)
        and (origin is None or run.origin == origin)
    ]
    oldest = datetime.min.replace(tzinfo=UTC)
    runs.sort(key=lambda run: (run.created_at or oldest, run.run_id), reverse=True)
    start = int(cursor) if cursor is not None else 0
    page = runs[start: start + limit]
    more = start + limit < len(runs)
    return RunList(
        runs=page,
        next_cursor=str(start + limit) if more else None,
        warnings=listing.warnings,
    )


@router.get("/runs/{run_id}", response_model=RunDetail)
def read_run(run_id: str, settings: SettingsDep, scheduler: SchedulerDep) -> RunDetail:
    run = _run(settings, run_id)
    return _projection(settings, scheduler, run, "result.project_run", run, RunDetail)


@router.get("/runs/{run_id}/candidates", response_model=CandidateList)
def read_run_candidates(
    run_id: str,
    settings: SettingsDep,
    scheduler: SchedulerDep,
    iteration: int | None = None,
) -> CandidateList:
    run = _run(settings, run_id)
    request = CandidatesRequest(run_dir=run.run_dir, iteration=iteration)
    return _projection(
        settings,
        scheduler,
        run,
        "result.project_candidates",
        request,
        CandidateList,
    )


@router.get("/runs/{run_id}/checkpoints", response_model=CheckpointList)
def read_run_checkpoints(
    run_id: str, settings: SettingsDep, scheduler: SchedulerDep
) -> CheckpointList:
    return _checkpoints(settings, scheduler, _run(settings, run_id))


@router.get("/runs/{run_id}/verifications", response_model=VerificationList)
def read_run_verifications(
    run_id: str, settings: SettingsDep, scheduler: SchedulerDep
) -> VerificationList:
    run = _run(settings, run_id)
    verifications: list[VerifyResult] = []
    for job in scheduler.summaries(kind=JobKind.VERIFY, state=JobState.COMPLETED):
        if job.run_id != run.run_id:
            continue
        path = settings.workspace.job_dir(job.job_id) / "result.json"
        result = _JOB_RESULT.validate_json(path.read_text(encoding="utf-8"))
        assert isinstance(result, VerifyResult)
        verifications.append(result)
    by_fit = Path(run.run_dir) / VERIFICATION_FILE
    if by_fit.is_file():
        verifications.append(
            VerifyResult.model_validate_json(by_fit.read_text(encoding="utf-8"))
        )
    return VerificationList(verifications=verifications)


@router.post("/runs/{run_id}/resume", response_model=JobAccepted, status_code=202)
def resume_run(
    run_id: str, body: ResumeRequest, settings: SettingsDep, scheduler: SchedulerDep
) -> JobAccepted:
    run = _run(settings, run_id)
    stored = _stored_job(run)
    original = stored.request
    assert isinstance(original, FitRequest)
    if body.seed is not None and not body.reseed:
        raise ApiError(
            422, ErrorCode.INVALID_REQUEST, "a new seed needs reseed", field="seed"
        )
    checkpoints = _checkpoints(settings, scheduler, run)
    if not checkpoints.resumable:
        raise ApiError(409, ErrorCode.CONFLICT, checkpoints.reason or "not resumable")
    available = [checkpoint.iteration for checkpoint in checkpoints.checkpoints]
    iteration = max(available) if body.iteration is None else body.iteration
    if iteration not in available:
        raise ApiError(
            409,
            ErrorCode.CONFLICT,
            f"no resumable iteration {iteration}; the available iterations are "
            f"{available}",
            field="iteration",
        )
    changed = sorted(
        name
        for name, value in body.params.items()
        if original.params.get(name) != value
    )
    fixed = [name for name in changed if name not in checkpoints.resume_safe_params]
    if fixed:
        raise ApiError(
            409,
            ErrorCode.CONFLICT,
            f"kd keeps {', '.join(fixed)} fixed on resume; only "
            f"{', '.join(checkpoints.resume_safe_params) or 'nothing'} may change. "
            "Clone the run to change the others.",
            field="params",
        )
    request = original.model_copy(
        update={
            "params": {**original.params, **body.params},
            "seeds": [original.seeds[0] if body.seed is None else body.seed],
            "max_seconds": body.max_seconds,

            "hard_seconds": body.hard_seconds,
            "memory_cap_bytes": body.memory_cap_bytes,
        }
    )
    lineage = Lineage(
        resumed_from_run_id=run.run_id, resume_iteration=iteration, reseed=body.reseed
    )
    return JobAccepted(jobs=scheduler.submit(request, lineage=lineage))


@router.post("/runs/{run_id}/clone", response_model=JobAccepted, status_code=202)
def clone_run(
    run_id: str, body: FitRequest, settings: SettingsDep, scheduler: SchedulerDep
) -> JobAccepted:
    run = _run(settings, run_id)
    lineage = Lineage(cloned_from_run_id=run.run_id)
    return JobAccepted(jobs=scheduler.submit(body, lineage=lineage))


@router.get("/history/sources", response_model=HistorySourceList)
def list_history_sources(settings: SettingsDep) -> HistorySourceList:
    return HistorySourceList(sources=read_sources(settings.workspace))


@router.post("/history/sources", response_model=HistorySource, status_code=201)
def add_history_source(
    body: HistoryImportRequest, settings: SettingsDep
) -> HistorySource:
    path = Path(body.path).expanduser().resolve()
    if not path.is_dir():
        raise ApiError(
            404, ErrorCode.NOT_FOUND, f"no directory at {path}", field="path"
        )
    for source in read_sources(settings.workspace):
        if Path(source.path) == path:
            return source
    now = datetime.now(UTC).replace(microsecond=0)
    source_id = f"h-{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
    request = HistoryScanRequest(source_id=source_id, path=str(path), imported_at=now)
    source = kd_call(settings, "history.scan_history", request, HistorySource)
    write_once(
        settings.workspace.history / f"{source_id}.json",
        source.model_dump_json(indent=1),
    )
    return source
