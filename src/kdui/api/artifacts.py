
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Response
from fastapi.responses import FileResponse

from kdui.api import ApiError, SchedulerDep, SettingsDep
from kdui.api.runs import LIVE_FILES, UnknownRunError, resolve_run
from kdui.models import (
    TERMINAL_STATES,
    Artifact,
    ArtifactKind,
    ArtifactList,
    ErrorCode,
)
from kdui.scheduler import Scheduler, UnknownJobError
from kdui.settings import Settings

router = APIRouter(tags=["artifacts"])

Owner = Literal["job", "run"]

MEDIA_TYPES = {
    ".html": "text/html",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".pdf": "application/pdf",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".csv": "text/csv",
    ".log": "text/plain",
    ".txt": "text/plain",
}
REPORT_SANDBOX = (
    "sandbox allow-scripts; default-src 'none'; script-src 'unsafe-inline';"
    " style-src 'unsafe-inline'; img-src data:; font-src data:"
)

_RECORD_FILES = frozenset(
    {"record.json", "recorder.json", "manifest.json", *LIVE_FILES}
)
_SKIPPED_DIRS = frozenset({"checkpoints"})


@dataclass(frozen=True)
class _Listed:
    artifact: Artifact
    path: Path


def artifact_id(owner: Owner, owner_id: str, relative: str) -> str:
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
    return f"a.{owner}.{owner_id}.{digest}"


def _kind(owner: Owner, relative: str) -> ArtifactKind:
    name = relative.rsplit("/", 1)[-1]
    suffix = Path(name).suffix.lower()
    if suffix == ".html":
        return ArtifactKind.REPORT
    if suffix in {".svg", ".png", ".pdf"}:
        return ArtifactKind.FIGURE
    if suffix == ".log":
        return ArtifactKind.LOG
    if owner == "run" and relative in _RECORD_FILES:
        return ArtifactKind.RECORD
    return ArtifactKind.DATA


def _files(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and path.relative_to(root).parts[0] not in _SKIPPED_DIRS
    )


def _listing(
    owner: Owner, owner_id: str, root: Path, relatives: list[str]
) -> list[_Listed]:
    listed = []
    for relative in relatives:
        path = root / relative
        artifact = Artifact(
            artifact_id=artifact_id(owner, owner_id, relative),
            owner=owner,
            owner_id=owner_id,
            name=relative,
            kind=_kind(owner, relative),
            media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
            size_bytes=path.stat().st_size,
        )
        listed.append(_Listed(artifact, path))
    return listed


def _job_ended(scheduler: Scheduler, job_id: str) -> bool:
    try:
        return scheduler.summary(job_id).state in TERMINAL_STATES
    except UnknownJobError as error:
        raise ApiError(404, ErrorCode.NOT_FOUND, f"no job {job_id}") from error


def _job_files(settings: Settings, scheduler: Scheduler, job_id: str) -> list[_Listed]:
    ended = _job_ended(scheduler, job_id)
    job_dir = settings.workspace.job_dir(job_id)
    relatives = (
        [f"artifacts/{relative}" for relative in _files(job_dir / "artifacts")]
        if ended
        else []
    )
    if (job_dir / "worker.log").is_file():
        relatives.append("worker.log")
    if ended and (job_dir / "result.json").is_file():
        relatives.append("result.json")
    return _listing("job", job_id, job_dir, relatives)


def _run_files(settings: Settings, scheduler: Scheduler, run_id: str) -> list[_Listed]:
    try:
        run = resolve_run(settings.workspace, run_id)
    except UnknownRunError as error:
        raise ApiError(404, ErrorCode.NOT_FOUND, f"no run {run_id}") from error
    root = Path(run.run_dir)
    relatives = _files(root)
    if run.job_dir is not None and not _job_ended(scheduler, run_id):
        relatives = [relative for relative in relatives if relative in LIVE_FILES]
    return _listing("run", run.run_id, root, relatives)


@router.get("/jobs/{job_id}/artifacts", response_model=ArtifactList)
def list_job_artifacts(
    job_id: str, settings: SettingsDep, scheduler: SchedulerDep
) -> ArtifactList:
    listed = _job_files(settings, scheduler, job_id)
    return ArtifactList(artifacts=[entry.artifact for entry in listed])


@router.get("/runs/{run_id}/artifacts", response_model=ArtifactList)
def list_run_artifacts(
    run_id: str, settings: SettingsDep, scheduler: SchedulerDep
) -> ArtifactList:
    listed = _run_files(settings, scheduler, run_id)
    return ArtifactList(artifacts=[entry.artifact for entry in listed])


@router.get("/artifacts/{artifact_id}", response_class=Response)
def read_artifact(
    artifact_id: str, settings: SettingsDep, scheduler: SchedulerDep
) -> Response:
    parts = artifact_id.split(".")
    if len(parts) != 4 or parts[0] != "a" or parts[1] not in {"job", "run"}:
        raise ApiError(404, ErrorCode.NOT_FOUND, f"no artifact {artifact_id}")
    owner_id = parts[2]
    listed = (
        _job_files(settings, scheduler, owner_id)
        if parts[1] == "job"
        else _run_files(settings, scheduler, owner_id)
    )
    for entry in listed:
        if entry.artifact.artifact_id == artifact_id:
            headers = (
                {"Content-Security-Policy": REPORT_SANDBOX}
                if entry.artifact.kind is ArtifactKind.REPORT
                else None
            )
            return FileResponse(
                entry.path,
                media_type=entry.artifact.media_type,
                headers=headers,
                content_disposition_type="inline",
                filename=entry.path.name,
            )
    raise ApiError(404, ErrorCode.NOT_FOUND, f"no artifact {artifact_id}")
