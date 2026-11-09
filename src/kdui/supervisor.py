
from __future__ import annotations

import json
import logging
import os
import signal
import socket
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import IO, Literal

import psutil

import kd
from kd.harness import (
    ProcessEnding,
    ProcessResult,
    ProcessSupervisor,
    check_memory_cap_available,
)
from kd.search import IterationEvent, PhaseEvent, RunDirPaths
from kdui import jobstore
from kdui.models import (
    FitRequest,
    IterationUpdate,
    JobPhase,
    MemoryCapKind,
    MemoryMeasure,
    NativeScore,
    PhaseChange,
    StoredJob,
    SupervisorStarted,
    WorkerEnded,
    WorkerStarted,
)
from kdui.ops.result import expression_latex
from kdui.settings import GIB, Settings, Workspace

logger = logging.getLogger(__name__)

_POLL_SECONDS = 0.5
_FORCED_END_SECONDS = 10.0
_NEAR_CAP_FRACTION = 0.8

_KILLED = -9
_TERMINATED = -15


_PHASE_OF_KD: dict[str, JobPhase] = {
    "fit_started": JobPhase.PREPARING,
    "search_started": JobPhase.SEARCHING,
    "search_ended": JobPhase.REPORTING,
}

_ENDED_BY: dict[ProcessEnding, Literal["exit", "stop", "kill"]] = {
    ProcessEnding.EXIT: "exit",
    ProcessEnding.SIGTERM: "stop",
    ProcessEnding.SIGKILL: "kill",
    ProcessEnding.JOB_TERMINATE: "kill",
}

EndCause = Literal["timeout", "requested", "memory_limit", "system_kill"]


@dataclass(frozen=True)
class WorkerEnd:

    ended_by: Literal["exit", "stop", "kill"]
    cause: EndCause | None
    peak_memory_bytes: int
    memory_measure: MemoryMeasure


def classify_end(
    result: ProcessResult,
    *,
    cap_kind: MemoryCapKind,
    cap_bytes: int,
    sampled_breach: bool,
    sampled_peak_bytes: int,
    interrupted: bool,
) -> WorkerEnd:
    if cap_kind is MemoryCapKind.JOB_OBJECT:
        assert result.peak_committed_bytes is not None
        peak, measure = result.peak_committed_bytes, MemoryMeasure.COMMIT
    else:
        peak = max(result.peak_rss_bytes, sampled_peak_bytes)
        measure = MemoryMeasure.RSS
    uninvited = result.stop_reason is None
    cause: EndCause | None
    if sampled_breach:
        cause = "memory_limit"
    elif result.stop_reason is not None:
        cause = result.stop_reason
    elif result.returncode != 0 and _job_limit_reached(result):


        cause = "memory_limit"
    elif cap_kind is MemoryCapKind.CGROUP and (
        result.returncode == _KILLED and peak >= _NEAR_CAP_FRACTION * cap_bytes
    ):

        cause = "memory_limit"
    elif uninvited and (result.returncode in (_KILLED, _TERMINATED) or interrupted):
        cause = "system_kill"
    else:
        cause = None
    return WorkerEnd(_ENDED_BY[result.ended_by], cause, peak, measure)


def _job_limit_reached(result: ProcessResult) -> bool:
    if result.memory_backend != "windows-job":
        return False
    assert result.memory_limit_bytes is not None
    assert result.peak_committed_bytes is not None
    return result.peak_committed_bytes >= result.memory_limit_bytes


def _tree_rss(pid: int) -> int:
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.NoSuchProcess:

            continue
    return total


class ProgressJournal:

    def __init__(self, job_dir: Path, run_dir: Path, algorithm: str) -> None:
        self._job_dir = job_dir
        self._paths = RunDirPaths(root=run_dir)
        self._algorithm = algorithm
        self._score: tuple[str, Literal["min", "max"]] | None = None
        self._phases_seen = 0
        self._events_offset = 0
        self._last_expression: str | None = None
        self.skipped = 0

    def publish(self) -> None:
        lines = jobstore.complete_lines(self._paths.phases)
        for line in lines[self._phases_seen:]:
            try:
                event = PhaseEvent.from_dict(json.loads(line))
            except ValueError:
                self._skip(self._paths.phases, line)
                continue
            phase = _PHASE_OF_KD.get(event.phase)
            if phase is None:
                continue
            if phase is JobPhase.REPORTING:
                self._publish_events()
            at = datetime.fromisoformat(event.created_at)
            jobstore.append_progress(self._job_dir, PhaseChange(at=at, phase=phase))
        self._phases_seen = len(lines)
        self._publish_events()

    def _skip(self, source: Path, line: str) -> None:
        self.skipped += 1
        logger.warning(
            "%s: a line could not be read and is skipped (%d so far): %.200s",
            source.name,
            self.skipped,
            line,
            exc_info=True,
        )

    def _publish_events(self) -> None:
        if not self._paths.events.exists():
            return
        with self._paths.events.open("rb") as stream:
            stream.seek(self._events_offset)
            data = stream.read()
        complete = data[: data.rfind(b"\n") + 1]
        self._events_offset += len(complete)
        for line in complete.decode("utf-8", errors="replace").splitlines():
            try:
                event = IterationEvent.from_dict(json.loads(line))
            except ValueError:
                self._skip(self._paths.events, line)
                continue
            jobstore.append_progress(self._job_dir, self._iteration_update(event))

    def _equation(self, expression: str | None) -> str | None:
        if expression == self._last_expression:
            return None
        self._last_expression = expression
        if expression is None:
            return None
        try:
            return expression_latex(expression)
        except Exception:


            logger.exception("no LaTeX for the best expression %.200s", expression)
            return None

    def _iteration_update(self, event: IterationEvent) -> IterationUpdate:
        if self._score is None:
            schema = next(
                schema
                for schema in kd.instrument_schemas()
                if schema["algorithm"] == self._algorithm
            )
            self._score = (schema["score_kind"], schema["score_direction"])
        kind, direction = self._score
        diagnostics = event.diagnostics or {}
        return IterationUpdate(
            iteration=event.iteration,
            elapsed_seconds=event.elapsed_seconds,
            n_candidates=event.n_candidates,
            n_invalid=event.n_invalid,
            n_unique_candidates=diagnostics.get("n_unique_candidates"),
            mean_complexity=diagnostics.get("mean_complexity"),
            best_score=NativeScore(
                value=event.best_score, kind=kind, direction=direction
            ),
            best_equation_latex=self._equation(event.best_expression),
        )


def _start_worker(
    job_dir: Path, job: StoredJob, log: IO[bytes]
) -> tuple[ProcessSupervisor, MemoryCapKind]:
    command = [sys.executable, "-m", "kdui.worker", str(job_dir)]
    environment = {**os.environ, "PYTHONUTF8": "1"}
    try:
        check_memory_cap_available()
    except (RuntimeError, OSError) as error:


        logger.warning("no kernel memory cap here, the cap is sampled: %s", error)
        memory_max_gb = None
    else:
        memory_max_gb = round(job.memory_cap_bytes / GIB, 3)
    worker = ProcessSupervisor(
        command,
        env=environment,
        hard_seconds=job.hard_seconds,
        grace_seconds=Settings.kill_grace_seconds,
        stdout=log,
        stderr=log,
        memory_max_gb=memory_max_gb,
    )
    if worker.memory_backend is None:
        return worker, MemoryCapKind.SAMPLED
    if worker.memory_backend == "windows-job":
        return worker, MemoryCapKind.JOB_OBJECT
    return worker, MemoryCapKind.CGROUP


class _WorkerInterrupt:

    def __init__(self, job_dir: Path, hard_seconds: float | None) -> None:
        self._job_dir = job_dir
        self._hard_seconds = hard_seconds
        self._started = time.monotonic()
        self._written = False

    def after_poll(self, polled_at: float, *, forced: bool) -> None:
        expired = (
            self._hard_seconds is not None
            and polled_at - self._started >= self._hard_seconds
        )
        if self._written or not (forced or expired):
            return
        logger.info("the worker is told to end its operation (interrupt.json)")
        jobstore.write_interrupt(self._job_dir, datetime.now(UTC))
        self._written = True


class _EndRequest:

    def __init__(self) -> None:
        self.received = False

    def __call__(self, signum: int, frame: FrameType | None) -> None:
        self.received = True


def _watch(
    job_dir: Path,
    job: StoredJob,
    worker: ProcessSupervisor,
    cap_kind: MemoryCapKind,
    journal: ProgressJournal | None,
    end_request: _EndRequest,
) -> tuple[ProcessResult, bool, int]:
    forced_at: float | None = None
    breached = False
    sampled_peak = 0
    interrupt = (
        _WorkerInterrupt(job_dir, job.hard_seconds) if sys.platform == "win32" else None
    )
    while True:
        polled_at = time.monotonic()
        result = worker.poll()
        if result is not None:
            return result, breached, sampled_peak
        now = time.monotonic()
        if forced_at is None and (
            end_request.received or jobstore.kill_requested(job_dir)
        ):
            logger.info("forced end requested")
            worker.request_stop()
            forced_at = now
        elif forced_at is not None and now - forced_at >= _FORCED_END_SECONDS:
            worker.hurry()
        if interrupt is not None:
            interrupt.after_poll(polled_at, forced=forced_at is not None)
        if cap_kind is MemoryCapKind.SAMPLED and not breached:
            resident = _tree_rss(worker.pid)
            sampled_peak = max(sampled_peak, resident)
            if resident > job.memory_cap_bytes:
                logger.warning(
                    "memory limit reached: %d bytes resident, cap %d bytes",
                    resident,
                    job.memory_cap_bytes,
                )
                breached = True
                worker.request_stop()
                worker.hurry()
        if journal is not None:
            journal.publish()
        time.sleep(_POLL_SECONDS)


def _interrupted(job_dir: Path) -> bool:
    outcome = jobstore.read_job(job_dir).outcome
    return (
        outcome is not None
        and outcome.error is not None
        and outcome.error.error_type == kd.SearchInterrupted.__name__
    )


def supervise(job_dir: Path) -> int:
    job = jobstore.read_request(job_dir)
    end_request = _EndRequest()
    signal.signal(signal.SIGTERM, end_request)
    boot = jobstore.current_boot()
    jobstore.append_state(
        job_dir,
        SupervisorStarted(
            at=datetime.now(UTC),
            process=jobstore.process_identity(os.getpid()),
            boot_id=boot.boot_id,
            boot_time=boot.boot_time,
            hostname=socket.gethostname(),
        ),
    )
    jobstore.append_progress(
        job_dir, PhaseChange(at=datetime.now(UTC), phase=JobPhase.STARTING)
    )
    journal = None
    if isinstance(job.request, FitRequest):
        run_dir = Workspace.of_job_dir(job_dir).run_dir(job.job_id)
        journal = ProgressJournal(job_dir, run_dir, job.request.algorithm)
    with (job_dir / jobstore.LOG_FILE).open("ab") as log:
        worker, cap_kind = _start_worker(job_dir, job, log)
        try:
            jobstore.append_state(
                job_dir,
                WorkerStarted(
                    at=datetime.now(UTC),
                    process=jobstore.process_identity(worker.pid),
                    memory_cap_kind=cap_kind,
                ),
            )
            if journal is None:
                jobstore.append_progress(
                    job_dir, PhaseChange(at=datetime.now(UTC), phase=JobPhase.WORKING)
                )
            result, breached, sampled_peak = _watch(
                job_dir, job, worker, cap_kind, journal, end_request
            )
        except BaseException:
            worker.abort()
            raise
    if journal is not None:
        journal.publish()
        if journal.skipped:
            logger.warning("%d line(s) of kd's event files skipped", journal.skipped)
    end = classify_end(
        result,
        cap_kind=cap_kind,
        cap_bytes=job.memory_cap_bytes,
        sampled_breach=breached,
        sampled_peak_bytes=sampled_peak,
        interrupted=_interrupted(job_dir),
    )
    jobstore.append_state(
        job_dir,
        WorkerEnded(
            at=datetime.now(UTC),
            returncode=result.returncode,
            ended_by=end.ended_by,
            cause=end.cause,
            wallclock_seconds=result.wallclock_seconds,
            peak_memory_bytes=end.peak_memory_bytes,
            memory_measure=end.memory_measure,
        ),
    )
    logger.info(
        "worker ended: status %s, ended by %s, cause %s",
        result.returncode,
        end.ended_by,
        end.cause,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    (argument,) = sys.argv[1:] if argv is None else argv
    job_dir = Path(argument)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(
                job_dir / jobstore.SUPERVISOR_LOG_FILE, encoding="utf-8"
            )
        ],
    )
    try:
        return supervise(job_dir)
    except Exception:


        logger.exception("supervision of %s failed", job_dir)
        raise


if __name__ == "__main__":
    sys.exit(main())
