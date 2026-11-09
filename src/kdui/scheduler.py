
from __future__ import annotations

import logging
import os
import re
import socket
import subprocess
import sys
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

import psutil

from kdui import jobstore
from kdui.jobstore import Boot, DerivedState, JobFiles
from kdui.models import (
    QUEUE_OF_KIND,
    CapabilitySnapshot,
    FitRequest,
    JobDetail,
    JobKind,
    JobLimits,
    JobProgress,
    JobQueue,
    JobRequest,
    JobState,
    JobSummary,
    LaunchRecord,
    Lineage,
    MemoryBudget,
    MemoryCapKind,
    PriorKnowledge,
    ProcessIdentity,
    ProcessUsage,
    QueueStatus,
    StoredJob,
    SupervisorStarted,
    VerifyRequest,
    WaitingCode,
    WaitingReason,
    WorkerEnded,
    WorkerStarted,
)
from kdui.settings import GIB, Settings

logger = logging.getLogger(__name__)

LAUNCH_GRACE_SECONDS = 30.0

SUPERVISOR_ARGV: tuple[str, ...] = (sys.executable, "-m", "kdui.supervisor")

SAMPLED_CAP_WARNING = (
    "The kernel cannot enforce a memory cap on this machine. Each job's cap is"
    " enforced by sampling its memory, so a job can pass its cap briefly before"
    " it is ended."
)
ATTACHED_WARNING = (
    "Windows did not let the job's supervisor leave the terminal's job:"
    " closing the terminal that runs kd-ui may end running jobs."
)
LOW_HEADROOM_SYSTEM_WARNING = (
    "Memory headroom is low now: a fit with the smallest default cap does not"
    " fit beside what is running. Jobs still start, one after another."
)

_R = TypeVar("_R", SupervisorStarted, WorkerStarted, WorkerEnded)

_JOB_ID = re.compile(r"^j-\d{8}-\d{6}-[0-9a-f]{6}$")

DEFAULT_CAP_MARGIN_BYTES = GIB


def is_job_id(text: str) -> bool:
    return _JOB_ID.match(text) is not None


class UnknownJobError(KeyError):
    pass


@dataclass(frozen=True)
class _Job:
    files: JobFiles
    derived: DerivedState
    shift: float | None

    @property
    def stored(self) -> StoredJob:
        return self.files.job

    @property
    def job_id(self) -> str:
        return self.files.job.job_id

    @property
    def queue(self) -> JobQueue:
        return self.files.job.queue

    @property
    def state(self) -> JobState:
        return self.derived.state


@dataclass(frozen=True)
class _Memory:

    total: int
    available: int
    reserve: int
    allowance: int
    running_cap: int
    unused_running: int
    any_running: bool

    @property
    def admissible(self) -> int:
        return self.available - self.unused_running - self.reserve

    def fits(self, cap: int) -> bool:
        return cap + self.allowance <= self.admissible


def _gib(value: int) -> str:
    return f"{value / GIB:.1f} GiB"


def _process(job: _Job, identity: ProcessIdentity) -> psutil.Process | None:
    if job.shift is None or not jobstore.process_alive(
        jobstore.shifted(identity, job.shift)
    ):
        return None
    try:
        return psutil.Process(identity.pid)
    except psutil.NoSuchProcess:
        return None


def _tree_rss(process: psutil.Process) -> int:
    total = 0
    try:
        members = [process, *process.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    for member in members:
        try:
            total += member.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return total


def _end_tree(process: psutil.Process) -> None:
    try:
        members = [*process.children(recursive=True), process]
    except psutil.NoSuchProcess:
        return
    for member in members:
        try:
            member.kill()
        except psutil.NoSuchProcess:
            continue


def _record(files: JobFiles, kind: type[_R]) -> _R | None:
    for record in reversed(files.state):
        if isinstance(record, kind):
            return record
    return None


def _start_detached(
    argv: list[str], cwd: Path, breakaway: bool
) -> subprocess.Popen[bytes]:
    env = {**os.environ, "PYTHONUTF8": "1"}
    devnull = subprocess.DEVNULL
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        if breakaway:
            flags |= subprocess.CREATE_BREAKAWAY_FROM_JOB
        return subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=devnull,
            stdout=devnull,
            stderr=devnull,
            creationflags=flags,
        )
    return subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdin=devnull,
        stdout=devnull,
        stderr=devnull,
        start_new_session=True,
    )


def _windows_breakaway_allowed(cwd: Path) -> bool:
    try:
        _start_detached([sys.executable, "-c", ""], cwd, breakaway=True).wait()
    except OSError:
        return False
    return True


class Scheduler:

    def __init__(
        self, settings: Settings, *, supervisor_argv: Sequence[str] = SUPERVISOR_ARGV
    ) -> None:
        self.settings = settings
        self.supervisor_argv = tuple(supervisor_argv)
        self._lock = threading.RLock()
        self._children: list[subprocess.Popen[bytes]] = []
        self._breakaway: bool | None = None



    def _job_dir(self, job_id: str) -> Path:
        job_dir = self.settings.workspace.job_dir(job_id)
        if not is_job_id(job_id) or not (job_dir / jobstore.REQUEST_FILE).exists():
            raise UnknownJobError(job_id)
        return job_dir

    def _derive(self, files: JobFiles, boot: Boot) -> _Job:
        derived = jobstore.derive_state(
            files,
            alive=jobstore.process_alive,
            boot=boot,
            now=datetime.now(UTC),
            launch_grace_seconds=LAUNCH_GRACE_SECONDS,
        )
        return _Job(files, derived, jobstore.clock_shift(files, boot))

    def _read(self, job_id: str) -> _Job:
        files = jobstore.read_job(self._job_dir(job_id))
        return self._derive(files, jobstore.current_boot())

    def _read_all(self) -> list[_Job]:
        workspace = self.settings.workspace
        boot = jobstore.current_boot()
        jobs = [
            self._derive(jobstore.read_job(workspace.job_dir(job_id)), boot)
            for job_id in jobstore.list_job_ids(workspace)
        ]
        return sorted(jobs, key=lambda job: (job.stored.created_at, job.job_id))

    def _snapshot(self) -> CapabilitySnapshot | None:
        path = self.settings.workspace.capabilities_file
        if not path.exists():
            return None
        return CapabilitySnapshot.model_validate_json(path.read_text(encoding="utf-8"))

    def memory_cap_kind(self) -> MemoryCapKind | None:
        snapshot = self._snapshot()
        return None if snapshot is None else snapshot.environment.memory_cap_kind



    def _used(self, job: _Job) -> int:
        supervisor = _record(job.files, SupervisorStarted)
        worker = _record(job.files, WorkerStarted)
        for record in (supervisor, worker):
            process = None if record is None else _process(job, record.process)
            if process is not None:
                return _tree_rss(process)
        return 0

    def _memory(self, jobs: list[_Job]) -> _Memory:
        settings = self.settings
        virtual = psutil.virtual_memory()
        allowance = settings.supervisor_allowance_bytes
        running = [job for job in jobs if job.state is JobState.RUNNING]
        claims = [job.stored.memory_cap_bytes + allowance for job in running]
        unused = sum(
            max(0, claim - self._used(job))
            for job, claim in zip(running, claims, strict=True)
        )
        return _Memory(
            total=virtual.total,
            available=virtual.available,
            reserve=max(
                settings.reserve_min_bytes,
                int(settings.reserve_fraction * virtual.total),
            ),
            allowance=allowance,
            running_cap=sum(claims),
            unused_running=unused,
            any_running=bool(running),
        )

    def _default_fit_cap(self, memory: _Memory) -> int:
        settings = self.settings
        alone = (
            memory.available
            - memory.reserve
            - memory.allowance
            - DEFAULT_CAP_MARGIN_BYTES
        )
        upper = int(settings.fit_cap_fraction * memory.total)
        return max(settings.fit_cap_floor_bytes, min(alone, upper))

    def _budget(self, memory: _Memory) -> MemoryBudget:
        return MemoryBudget(
            total_bytes=memory.total,
            available_bytes=memory.available,
            reserve_bytes=memory.reserve,
            supervisor_allowance_bytes=memory.allowance,
            running_cap_bytes=memory.running_cap,
            admissible_bytes=max(0, memory.admissible),
            default_fit_cap_bytes=self._default_fit_cap(memory),
            default_light_cap_bytes=self.settings.light_cap_bytes,
        )

    def _low_headroom(self, cap: int, memory: _Memory) -> str:
        return (
            f"Memory headroom was low: the cap of {_gib(cap)} plus"
            f" {_gib(memory.allowance)} for the supervisor is more than the"
            f" {_gib(max(0, memory.admissible))} available beyond the reserve."
            " The job starts anyway; it may end for memory."
        )



    def _slots(self, queue: JobQueue) -> int:
        if queue is JobQueue.FIT:
            return self.settings.fit_slots
        return self.settings.light_slots

    def _plan(
        self, jobs: list[_Job], memory: _Memory
    ) -> dict[str, WaitingReason | None]:
        plan: dict[str, WaitingReason | None] = {}
        for queue in JobQueue:
            slots = self._slots(queue)
            running = sum(
                1
                for job in jobs
                if job.queue is queue and job.state is JobState.RUNNING
            )
            queued = [
                job
                for job in jobs
                if job.queue is queue and job.state is JobState.QUEUED
            ]
            for position, job in enumerate(queued):
                plan[job.job_id] = self._waiting(job, position, slots, running, memory)
        return plan

    def _waiting(
        self, job: _Job, position: int, slots: int, running: int, memory: _Memory
    ) -> WaitingReason | None:
        queue = job.queue.value
        if position > 0:
            return WaitingReason(
                code=WaitingCode.AHEAD,
                message=f"{position} earlier job(s) of the {queue} queue start first.",
            )
        if running >= slots:
            return WaitingReason(
                code=WaitingCode.SLOT,
                message=(
                    f"The {queue} queue runs {slots} job(s) at a time"
                    f" and {running} is running."
                ),
            )
        cap = job.stored.memory_cap_bytes
        if memory.any_running and not memory.fits(cap):
            return WaitingReason(
                code=WaitingCode.MEMORY,
                message=(
                    f"Its cap of {_gib(cap)} plus {_gib(memory.allowance)} for the"
                    " supervisor does not fit beside the running jobs"
                    f" ({_gib(max(0, memory.admissible))} free beyond the reserve)."
                    " It starts when one of them has ended."
                ),
                needed_bytes=cap + memory.allowance,
                admissible_bytes=max(0, memory.admissible),
            )
        return None

    def _launch(self, job: _Job, memory: _Memory) -> None:
        warnings: list[str] = []
        cap = job.stored.memory_cap_bytes
        if not memory.fits(cap):
            warnings.append(self._low_headroom(cap, memory))
        root = self.settings.workspace.root
        if sys.platform == "win32" and self._breakaway is None:
            self._breakaway = _windows_breakaway_allowed(root)
        if self._breakaway is False:
            warnings.append(ATTACHED_WARNING)
        job_dir = self.settings.workspace.job_dir(job.job_id)
        launch = LaunchRecord(
            at=datetime.now(UTC),
            hostname=socket.gethostname(),
            supervisor_allowance_bytes=memory.allowance,
            warnings=warnings,
        )
        jobstore.write_launch(job_dir, launch)
        argv = [*self.supervisor_argv, str(job_dir.resolve())]
        child = _start_detached(argv, root, breakaway=self._breakaway is not False)
        self._children.append(child)
        logger.info("launched job %s (supervisor pid %d)", job.job_id, child.pid)

    def _enforce_orphans(self, jobs: list[_Job]) -> None:
        now = datetime.now(UTC)
        for job in jobs:
            if job.state is not JobState.RUNNING:
                continue
            supervisor = _record(job.files, SupervisorStarted)
            worker = _record(job.files, WorkerStarted)
            if (
                supervisor is None
                or worker is None
                or _process(job, supervisor.process) is not None
            ):
                continue
            process = _process(job, worker.process)
            if process is None:
                continue
            hard = job.stored.hard_seconds
            overdue = hard is not None and (now - worker.at).total_seconds() > (
                hard + self.settings.kill_grace_seconds
            )
            if overdue or job.files.kill is not None:
                logger.warning(
                    "ending worker %d of job %s: its supervisor is gone",
                    worker.process.pid,
                    job.job_id,
                )
                _end_tree(process)

    def _reap(self) -> None:
        self._children = [child for child in self._children if child.poll() is None]

    def tick(self) -> None:
        with self._lock:
            self._reap()
            jobs = self._read_all()
            self._enforce_orphans(jobs)
            while True:
                memory = self._memory(jobs)
                plan = self._plan(jobs, memory)
                ready = [
                    job
                    for job in jobs
                    if job.job_id in plan and plan[job.job_id] is None
                ]
                if not ready:
                    return
                self._launch(ready[0], memory)
                jobs = self._read_all()

    def reconcile(self) -> None:
        with self._lock:
            jobs = self._read_all()
            self._enforce_orphans(jobs)
            counts = Counter(job.state.value for job in jobs)
            logger.info("reconciled %d job(s): %s", len(jobs), dict(counts))



    def _job_warnings(
        self,
        request: JobRequest,
        cap: int,
        memory: _Memory,
        snapshot: CapabilitySnapshot | None,
    ) -> list[str]:
        warnings: list[str] = []
        if isinstance(request, FitRequest) and snapshot is not None:
            for algorithm in snapshot.algorithms:
                if algorithm.algorithm == request.algorithm:
                    warnings.extend(algorithm.warnings)
            if request.schema_digest != snapshot.schema_digest:
                warnings.append(
                    "The form was built from an older list of algorithm parameters;"
                    " kd checks the parameters when the job starts."
                )
        if not memory.any_running and not memory.fits(cap):
            warnings.append(self._low_headroom(cap, memory))
        return warnings

    def _stored(
        self,
        request: JobRequest,
        lineage: Lineage | None,
        memory: _Memory,
        snapshot: CapabilitySnapshot | None,
    ) -> StoredJob:
        settings = self.settings
        now = datetime.now(UTC)
        if isinstance(request, FitRequest):
            cap = request.memory_cap_bytes or self._default_fit_cap(memory)
            if request.hard_seconds is not None:
                hard: float | None = request.hard_seconds
            elif request.max_seconds is not None:
                hard = request.max_seconds + settings.hard_slack_seconds
            else:
                hard = None
            marker = settings.workspace.reference_marker(request.dataset_id)
            prior: PriorKnowledge | None = PriorKnowledge(
                sketch=request.sketch is not None
            )
            reference = marker.exists()
        else:
            cap, hard, prior, reference = (
                settings.light_cap_bytes,
                settings.light_hard_seconds,
                None,
                False,
            )
        return StoredJob(
            job_id=jobstore.new_job_id(now),
            kind=JobKind(request.kind),
            queue=QUEUE_OF_KIND[JobKind(request.kind)],
            created_at=now,
            request=request,
            memory_cap_bytes=cap,
            hard_seconds=hard,
            lineage=lineage,
            prior_knowledge=prior,
            reference_in_report=reference,
            warnings=self._job_warnings(request, cap, memory, snapshot),
        )

    def submit(
        self, request: JobRequest, *, lineage: Lineage | None = None
    ) -> list[JobSummary]:
        with self._lock:
            memory = self._memory(self._read_all())
            snapshot = self._snapshot()
            requests: list[JobRequest] = (
                [request.model_copy(update={"seeds": [seed]}) for seed in request.seeds]
                if isinstance(request, FitRequest)
                else [request]
            )
            job_ids = []
            for one in requests:
                stored = self._stored(one, lineage, memory, snapshot)
                jobstore.create_job(self.settings.workspace, stored)
                job_ids.append(stored.job_id)
            return [self.summary(job_id) for job_id in job_ids]



    def stop(self, job_id: str) -> None:
        with self._lock:
            job = self._read(job_id)
            job_dir = self.settings.workspace.job_dir(job_id)
            if job.state is JobState.QUEUED:
                jobstore.write_cancelled(job_dir, datetime.now(UTC))
            elif job.state is JobState.RUNNING:
                jobstore.write_stop(job_dir, datetime.now(UTC))

    def kill(self, job_id: str) -> None:
        with self._lock:
            job = self._read(job_id)
            job_dir = self.settings.workspace.job_dir(job_id)
            if job.state is JobState.QUEUED:
                jobstore.write_cancelled(job_dir, datetime.now(UTC))
            elif job.state is JobState.RUNNING:
                jobstore.write_kill(job_dir, datetime.now(UTC))
                self._enforce_orphans([self._read(job_id)])



    def _summary(self, job: _Job, waiting: WaitingReason | None) -> JobSummary:
        files, stored = job.files, job.stored
        request = stored.request
        launch = files.launch
        worker = _record(files, WorkerStarted)
        fit = request if isinstance(request, FitRequest) else None
        run_id = request.run_id if isinstance(request, VerifyRequest) else None
        if fit is not None and launch is not None:
            run_id = stored.job_id
        running = job.state is JobState.RUNNING
        process = (
            None if worker is None or not running else _process(job, worker.process)
        )
        return JobSummary(
            job_id=stored.job_id,
            kind=stored.kind,
            queue=stored.queue,
            state=job.state,
            phase=job.derived.phase,
            stop_reason=job.derived.stop_reason,
            created_at=stored.created_at,
            started_at=None if launch is None else launch.at,
            ended_at=self._ended_at(job),
            waiting=waiting if job.state is JobState.QUEUED else None,
            dataset_id=None if fit is None else fit.dataset_id,
            algorithm=None if fit is None else fit.algorithm,
            seed=None if fit is None else fit.seeds[0],
            run_id=run_id,
            progress=self._progress(files),
            limits=JobLimits(
                memory_cap_bytes=stored.memory_cap_bytes,
                memory_cap_kind=None if worker is None else worker.memory_cap_kind,
                max_seconds=None if fit is None else fit.max_seconds,
                hard_seconds=stored.hard_seconds,
            ),
            memory_bytes=None if process is None else _tree_rss(process),
            hostname=None if launch is None else launch.hostname,


            warnings=list(
                dict.fromkeys(
                    [*stored.warnings, *([] if launch is None else launch.warnings)]
                )
            ),
        )

    @staticmethod
    def _ended_at(job: _Job) -> datetime | None:
        files = job.files
        ended = _record(files, WorkerEnded)
        if ended is not None:
            return ended.at
        if job.state is JobState.CANCELLED and files.cancelled is not None:
            return files.cancelled.at
        if job.state is not JobState.RUNNING and files.outcome is not None:
            return files.outcome.at
        return None

    @staticmethod
    def _progress(files: JobFiles) -> JobProgress | None:
        last = files.last_iteration
        if last is None:
            return None
        return JobProgress(
            iteration=last.iteration,
            best_score=last.best_score,
            best_equation_latex=files.best_equation_latex,
            elapsed_seconds=last.elapsed_seconds,
        )

    def _waiting_of(self, job: _Job) -> WaitingReason | None:
        if job.state is not JobState.QUEUED:
            return None
        jobs = self._read_all()
        return self._plan(jobs, self._memory(jobs)).get(job.job_id)

    def summary(self, job_id: str) -> JobSummary:
        job = self._read(job_id)
        return self._summary(job, self._waiting_of(job))

    def detail(self, job_id: str) -> JobDetail:
        job = self._read(job_id)
        files = job.files
        ended = _record(files, WorkerEnded)
        process = None
        if ended is not None:
            process = ProcessUsage(
                returncode=ended.returncode,
                wallclock_seconds=ended.wallclock_seconds,
                peak_memory_bytes=ended.peak_memory_bytes,
                memory_measure=ended.memory_measure,
            )
        return JobDetail(
            **self._summary(job, self._waiting_of(job)).model_dump(),
            request=job.stored.request,
            lineage=job.stored.lineage,
            prior_knowledge=job.stored.prior_knowledge,
            process=process,
            error=None if files.outcome is None else files.outcome.error,
            stop_requested_at=None if files.stop is None else files.stop.at,
            kill_requested_at=None if files.kill is None else files.kill.at,
            result_available=files.has_result,
        )

    def summaries(
        self,
        *,
        state: JobState | None = None,
        kind: JobKind | None = None,
        queue: JobQueue | None = None,
    ) -> list[JobSummary]:
        jobs = self._read_all()
        plan = self._plan(jobs, self._memory(jobs))
        return [
            self._summary(job, plan.get(job.job_id))
            for job in reversed(jobs)
            if (state is None or job.state is state)
            and (kind is None or job.stored.kind is kind)
            and (queue is None or job.queue is queue)
        ]

    def queues(self) -> list[QueueStatus]:
        states = [(job.queue, job.state) for job in self._read_all()]
        return [
            QueueStatus(
                queue=queue,
                slots=self._slots(queue),
                running=states.count((queue, JobState.RUNNING)),
                queued=states.count((queue, JobState.QUEUED)),
            )
            for queue in JobQueue
        ]

    def memory_budget(self) -> MemoryBudget:
        return self._budget(self._memory(self._read_all()))

    def warnings(self) -> list[str]:
        warnings: list[str] = []
        if self.memory_cap_kind() is MemoryCapKind.SAMPLED:
            warnings.append(SAMPLED_CAP_WARNING)
        memory = self._memory(self._read_all())
        if not memory.fits(self.settings.fit_cap_floor_bytes):
            warnings.append(LOW_HEADROOM_SYSTEM_WARNING)
        if self._breakaway is False:
            warnings.append(ATTACHED_WARNING)
        return warnings
