
from __future__ import annotations

import logging
import os
import secrets
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
from pydantic import TypeAdapter, ValidationError

from kdui.models import (
    IterationUpdate,
    JobKind,
    JobOutcome,
    JobPhase,
    JobResult,
    JobState,
    LaunchRecord,
    Marker,
    PhaseChange,
    ProcessIdentity,
    ProgressEvent,
    StateRecord,
    StopReason,
    StoredJob,
    SupervisorStarted,
    WorkerEnded,
)
from kdui.settings import Workspace

if sys.platform == "win32":
    import winreg

logger = logging.getLogger(__name__)

REQUEST_FILE = "request.json"
CANCELLED_FILE = "cancelled.json"
STOP_FILE = "stop.json"
KILL_FILE = "kill.json"
INTERRUPT_FILE = "interrupt.json"
LAUNCH_FILE = "launch.json"
STATE_FILE = "state.jsonl"
PROGRESS_FILE = "progress.jsonl"
LOG_FILE = "worker.log"
SUPERVISOR_LOG_FILE = "supervisor.log"
OUTCOME_FILE = "outcome.json"
RESULT_FILE = "result.json"
ARTIFACTS_DIR = "artifacts"

_BOOT_ID_FILE = Path("/proc/sys/kernel/random/boot_id")
_BOOT_ID_KEY = (
    r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management"
    r"\PrefetchParameters"
)
_START_FOLLOWS_CLOCK = sys.platform == "linux"
_CREATE_TIME_TOLERANCE_SECONDS = 2.0

_STATE_RECORD: TypeAdapter[StateRecord] = TypeAdapter(StateRecord)
_PROGRESS_EVENT: TypeAdapter[ProgressEvent] = TypeAdapter(ProgressEvent)
_JOB_RESULT: TypeAdapter[JobResult] = TypeAdapter(JobResult)



_SEALED_FIT: dict[str, tuple[JobState, StopReason]] = {
    "iterations_exhausted": (JobState.COMPLETED, StopReason.ITERATIONS_EXHAUSTED),
    "algorithm_done": (JobState.COMPLETED, StopReason.ALGORITHM_DONE),
    "early_stopping": (JobState.COMPLETED, StopReason.EARLY_STOPPING),
    "callback": (JobState.COMPLETED, StopReason.USER_STOP),
    "wall_clock_budget": (JobState.BUDGET_STOPPED, StopReason.WALL_CLOCK_BUDGET),
    "wall_clock_budget_before_search": (
        JobState.BUDGET_STOPPED,
        StopReason.WALL_CLOCK_BUDGET_BEFORE_SEARCH,
    ),
    "evaluation_budget": (JobState.BUDGET_STOPPED, StopReason.EVALUATION_BUDGET),
}


_ENDED_BY_CAUSE: dict[str, tuple[JobState, StopReason]] = {
    "requested": (JobState.CANCELLED, StopReason.KILLED),
    "timeout": (JobState.FAILED, StopReason.HARD_TIMEOUT),
    "memory_limit": (JobState.FAILED, StopReason.MEMORY_LIMIT),
    "system_kill": (JobState.FAILED, StopReason.SYSTEM_KILLED),
}


@dataclass(frozen=True)
class JobContext:

    workspace: Workspace
    job_id: str

    @property
    def job_dir(self) -> Path:
        return self.workspace.job_dir(self.job_id)

    @property
    def run_dir(self) -> Path:
        return self.workspace.run_dir(self.job_id)

    @property
    def artifacts_dir(self) -> Path:
        return self.job_dir / ARTIFACTS_DIR


@dataclass(frozen=True)
class JobFiles:

    job: StoredJob
    cancelled: Marker | None
    stop: Marker | None
    kill: Marker | None
    launch: LaunchRecord | None
    state: tuple[StateRecord, ...]
    last_phase: PhaseChange | None
    last_iteration: IterationUpdate | None
    best_equation_latex: str | None
    outcome: JobOutcome | None
    has_result: bool


@dataclass(frozen=True)
class DerivedState:

    state: JobState
    phase: JobPhase | None
    stop_reason: StopReason | None


def new_job_id(now: datetime) -> str:
    stamp = now.astimezone(UTC).strftime("%Y%m%d-%H%M%S")
    return f"j-{stamp}-{secrets.token_hex(3)}"


def write_once(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"{path} is written once and exists already")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.rename(temporary, path)


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line + "\n")


def complete_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    data = path.read_bytes()
    complete = data[: data.rfind(b"\n") + 1]
    return complete.decode("utf-8", errors="replace").splitlines()


def _parsed_lines(path: Path, adapter: TypeAdapter[Any]) -> list[Any]:
    records = []
    skipped = 0
    for line in complete_lines(path):
        try:
            records.append(adapter.validate_json(line))
        except ValidationError:
            skipped += 1
    if skipped:
        logger.warning("%s: %d unreadable line(s) skipped", path, skipped)
    return records


def _read_marker(path: Path) -> Marker | None:
    if not path.exists():
        return None
    return Marker.model_validate_json(path.read_text(encoding="utf-8"))


def _write_marker(path: Path, at: datetime) -> None:
    if not path.exists():
        write_once(path, Marker(at=at).model_dump_json())


def create_job(workspace: Workspace, job: StoredJob) -> Path:
    job_dir = workspace.job_dir(job.job_id)
    job_dir.mkdir(parents=True)
    write_once(job_dir / REQUEST_FILE, job.model_dump_json())
    return job_dir


def list_job_ids(workspace: Workspace) -> list[str]:
    if not workspace.jobs.exists():
        return []
    return sorted(
        entry.name
        for entry in workspace.jobs.iterdir()
        if (entry / REQUEST_FILE).exists()
    )


def read_request(job_dir: Path) -> StoredJob:
    return StoredJob.model_validate_json(
        (job_dir / REQUEST_FILE).read_text(encoding="utf-8")
    )


def read_job(job_dir: Path) -> JobFiles:
    launch_file = job_dir / LAUNCH_FILE
    outcome_file = job_dir / OUTCOME_FILE
    last_phase: PhaseChange | None = None
    last_iteration: IterationUpdate | None = None
    best_equation_latex: str | None = None
    for event in _parsed_lines(job_dir / PROGRESS_FILE, _PROGRESS_EVENT):
        if isinstance(event, PhaseChange):
            last_phase = event
        else:
            last_iteration = event
            if event.best_equation_latex is not None:
                best_equation_latex = event.best_equation_latex
    return JobFiles(
        job=read_request(job_dir),
        cancelled=_read_marker(job_dir / CANCELLED_FILE),
        stop=_read_marker(job_dir / STOP_FILE),
        kill=_read_marker(job_dir / KILL_FILE),
        launch=(
            LaunchRecord.model_validate_json(launch_file.read_text(encoding="utf-8"))
            if launch_file.exists()
            else None
        ),
        state=tuple(_parsed_lines(job_dir / STATE_FILE, _STATE_RECORD)),
        last_phase=last_phase,
        last_iteration=last_iteration,
        best_equation_latex=best_equation_latex,
        outcome=(
            JobOutcome.model_validate_json(outcome_file.read_text(encoding="utf-8"))
            if outcome_file.exists()
            else None
        ),
        has_result=(job_dir / RESULT_FILE).exists(),
    )


def write_cancelled(job_dir: Path, at: datetime) -> None:
    _write_marker(job_dir / CANCELLED_FILE, at)


def write_stop(job_dir: Path, at: datetime) -> None:
    _write_marker(job_dir / STOP_FILE, at)


def write_kill(job_dir: Path, at: datetime) -> None:
    _write_marker(job_dir / KILL_FILE, at)


def write_launch(job_dir: Path, launch: LaunchRecord) -> None:
    write_once(job_dir / LAUNCH_FILE, launch.model_dump_json())


def stop_requested(job_dir: Path) -> bool:
    return (job_dir / STOP_FILE).exists()


def kill_requested(job_dir: Path) -> bool:
    return (job_dir / KILL_FILE).exists()


def write_interrupt(job_dir: Path, at: datetime) -> None:
    _write_marker(job_dir / INTERRUPT_FILE, at)


def interrupt_requested(job_dir: Path) -> bool:
    return (job_dir / INTERRUPT_FILE).exists()


def append_state(job_dir: Path, record: StateRecord) -> None:
    _append_line(job_dir / STATE_FILE, record.model_dump_json())


def append_progress(job_dir: Path, event: ProgressEvent) -> None:
    _append_line(job_dir / PROGRESS_FILE, event.model_dump_json())


def write_outcome(job_dir: Path, outcome: JobOutcome) -> None:
    write_once(job_dir / OUTCOME_FILE, outcome.model_dump_json())


def write_result(job_dir: Path, result: JobResult) -> None:
    write_once(job_dir / RESULT_FILE, result.model_dump_json())


def read_result(job_dir: Path) -> JobResult:
    return _JOB_RESULT.validate_json(
        (job_dir / RESULT_FILE).read_text(encoding="utf-8")
    )


def process_identity(pid: int) -> ProcessIdentity:
    return ProcessIdentity(pid=pid, create_time=psutil.Process(pid).create_time())


def process_alive(identity: ProcessIdentity) -> bool:
    try:
        process = psutil.Process(identity.pid)
        started = process.create_time()
        status = process.status()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    return (
        abs(started - identity.create_time) <= _CREATE_TIME_TOLERANCE_SECONDS
        and status != psutil.STATUS_ZOMBIE
    )


@dataclass(frozen=True)
class Boot:

    boot_id: str
    boot_time: float


def current_boot() -> Boot:
    return Boot(boot_id=_boot_id(), boot_time=psutil.boot_time())


def _boot_id() -> str:
    if sys.platform == "win32":
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _BOOT_ID_KEY) as key:
            return str(winreg.QueryValueEx(key, "BootId")[0])
    return _BOOT_ID_FILE.read_text(encoding="ascii").strip()


def clock_shift(files: JobFiles, boot: Boot) -> float | None:
    for record in files.state:
        if isinstance(record, SupervisorStarted):
            if record.boot_id != boot.boot_id:
                return None
            return boot.boot_time - record.boot_time if _START_FOLLOWS_CLOCK else 0.0
    return 0.0


def shifted(identity: ProcessIdentity, shift: float) -> ProcessIdentity:
    return ProcessIdentity(pid=identity.pid, create_time=identity.create_time + shift)


def derive_state(
    files: JobFiles,
    *,
    alive: Callable[[ProcessIdentity], bool],
    boot: Boot,
    now: datetime,
    launch_grace_seconds: float,
) -> DerivedState:
    if files.launch is None:
        if files.cancelled is not None:
            return DerivedState(
                JobState.CANCELLED, None, StopReason.CANCELLED_WHILE_QUEUED
            )
        return DerivedState(JobState.QUEUED, None, None)

    supervisor: SupervisorStarted | None = None
    ended: WorkerEnded | None = None
    processes: list[ProcessIdentity] = []
    for record in files.state:
        if isinstance(record, WorkerEnded):
            ended = record
        else:
            processes.append(record.process)
            if isinstance(record, SupervisorStarted):
                supervisor = record

    if ended is None:
        starting = (
            supervisor is None
            and (now - files.launch.at).total_seconds() < launch_grace_seconds
        )
        shift = clock_shift(files, boot)
        if starting or (
            shift is not None
            and any(alive(shifted(process, shift)) for process in processes)
        ):
            phase = (
                JobPhase.STARTING
                if files.last_phase is None
                else files.last_phase.phase
            )
            return DerivedState(JobState.RUNNING, phase, None)
        if files.outcome is None:
            return DerivedState(
                JobState.INTERRUPTED,
                None,
                StopReason.MACHINE_RESTARTED
                if shift is None
                else StopReason.PROCESS_LOST,
            )

    outcome = files.outcome
    if outcome is not None and outcome.status == "completed":
        if files.job.kind is not JobKind.FIT:
            return DerivedState(JobState.COMPLETED, None, StopReason.FINISHED)
        assert outcome.kd_stop_reason is not None
        state, reason = _SEALED_FIT[outcome.kd_stop_reason]
        return DerivedState(state, None, reason)
    if ended is not None and ended.cause is not None:
        state, reason = _ENDED_BY_CAUSE[ended.cause]
        return DerivedState(state, None, reason)
    return DerivedState(
        JobState.FAILED,
        None,
        StopReason.TERMINATED if outcome is None else StopReason.ERROR,
    )
