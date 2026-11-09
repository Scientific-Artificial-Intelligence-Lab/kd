
from __future__ import annotations

import logging
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import Event
from typing import IO, Any, Literal

from kd.harness._supervise_linux import build_capped_command

if sys.platform == "win32":
    from kd.harness import _supervise_windows as _platform
else:
    from kd.harness import _supervise_linux as _platform

logger = logging.getLogger(__name__)


class ProcessEnding(StrEnum):

    EXIT = "exit"
    SIGTERM = "sigterm"
    SIGKILL = "sigkill"
    JOB_TERMINATE = "job_terminate"





_SOFT_ENDING: ProcessEnding | None = (
    None if sys.platform == "win32" else ProcessEnding.SIGTERM
)
_HARD_ENDING: ProcessEnding = (
    ProcessEnding.JOB_TERMINATE if sys.platform == "win32" else ProcessEnding.SIGKILL
)


@dataclass(frozen=True, kw_only=True)
class ProcessResult:

    pid: int
    returncode: int | None
    ended_by: ProcessEnding
    stop_reason: Literal["timeout", "requested"] | None
    sigkill_used: bool
    started_at: str
    term_sent_at: str | None
    kill_sent_at: str | None
    ended_at: str
    wallclock_seconds: float
    peak_rss_bytes: int
    left_descendants: bool
    group_emptied: bool
    memory_backend: Literal["linux-cgroup", "windows-job"] | None
    memory_limit_bytes: int | None
    peak_committed_bytes: int | None
    job_terminated_at: str | None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def check_memory_cap_available() -> None:
    _platform.check_memory_cap_available()


class ProcessSupervisor:

    def __init__(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str],
        hard_seconds: float | None,
        grace_seconds: float,
        cwd: Path | None = None,
        stdout: IO[Any] | None = None,
        stderr: IO[Any] | None = None,
        memory_max_gb: float | None = None,
    ) -> None:
        self._hard_seconds = hard_seconds
        self._grace_seconds = grace_seconds
        self._started = time.monotonic()
        self._started_at = _utc_now()
        self._term_deadline: float | None = None
        self._kill_deadline: float | None = None
        self._sent_at: dict[ProcessEnding, str] = {}
        self._ended_by = ProcessEnding.EXIT
        self._stop_reason: Literal["timeout", "requested"] | None = None
        self._left_descendants = False
        self._peak_rss_bytes = 0
        self._finished: ProcessResult | None = None
        self._tree = _platform.ProcessTree(
            command,
            env=env,
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
            memory_max_gb=memory_max_gb,
        )
        try:
            self._peak_rss_bytes = self._tree.rss_bytes()
        except BaseException:
            self.abort()
            raise

    @property
    def pid(self) -> int:
        return self._tree.pid

    @property
    def memory_backend(self) -> Literal["linux-cgroup", "windows-job"] | None:
        return self._tree.memory_backend

    @property
    def memory_limit_bytes(self) -> int | None:
        return self._tree.memory_limit_bytes

    def poll(self) -> ProcessResult | None:
        if self._finished is not None:
            return self._finished
        returncode = self._tree.leader_returncode()
        alive = returncode is None
        if not alive and not self._tree.members_remain():
            return self._finish(returncode, group_emptied=True)
        self._peak_rss_bytes = max(self._peak_rss_bytes, self._tree.rss_bytes())
        now = time.monotonic()
        if self._term_deadline is None:
            expired = self._hard_seconds is not None and (
                now - self._started >= self._hard_seconds
            )
            if not alive:
                self._left_descendants = True
            elif expired and self._stop_reason is None:
                self._stop_reason = "timeout"
            if not alive or self._stop_reason is not None:
                self._send(now, force=False, alive=alive)
        elif self._kill_deadline is None and now >= self._term_deadline:
            self._send(now, force=True, alive=alive)
        elif self._kill_deadline is not None and now >= self._kill_deadline:
            logger.warning(
                "process group %d remains after %s", self.pid, _HARD_ENDING.name
            )
            return self._finish(returncode, group_emptied=False)
        return None

    def request_stop(self) -> None:
        if self._term_deadline is not None or self._finished is not None:
            return
        returncode = self._tree.leader_returncode()
        alive = returncode is None
        if not alive and not self._tree.members_remain():
            self._finish(returncode, group_emptied=True)
            return
        if alive and self._stop_reason is None:
            self._stop_reason = "requested"
        self._left_descendants = not alive
        self._send(time.monotonic(), force=False, alive=alive)

    def hurry(self) -> None:
        if self._term_deadline is not None:
            self._term_deadline = time.monotonic()

    def _send(self, now: float, *, force: bool, alive: bool) -> None:






        self._record(_HARD_ENDING if force else _SOFT_ENDING, alive=alive)
        try:
            self._tree.stop(force=force)
        except OSError:
            self._set_deadline(now, force=force)
            raise
        self._set_deadline(now, force=force)

    def _record(self, ending: ProcessEnding | None, *, alive: bool) -> None:
        if ending is None:
            return
        if alive:
            self._ended_by = ending
        self._sent_at[ending] = _utc_now()

    def _set_deadline(self, now: float, *, force: bool) -> None:
        if force:
            self._kill_deadline = now + self._grace_seconds
        else:
            self._term_deadline = now + self._grace_seconds

    def _finish(self, returncode: int | None, *, group_emptied: bool) -> ProcessResult:
        kill_sent_at = self._sent_at.get(ProcessEnding.SIGKILL)
        self._finished = ProcessResult(
            pid=self.pid,
            returncode=returncode,
            ended_by=self._ended_by,
            stop_reason=self._stop_reason,
            sigkill_used=kill_sent_at is not None,
            started_at=self._started_at,
            term_sent_at=self._sent_at.get(ProcessEnding.SIGTERM),
            kill_sent_at=kill_sent_at,
            ended_at=_utc_now(),
            wallclock_seconds=time.monotonic() - self._started,
            peak_rss_bytes=self._peak_rss_bytes,
            left_descendants=self._left_descendants,
            group_emptied=group_emptied,
            memory_backend=self._tree.memory_backend,
            memory_limit_bytes=self._tree.memory_limit_bytes,
            peak_committed_bytes=self._tree.peak_committed_bytes(),
            job_terminated_at=self._sent_at.get(ProcessEnding.JOB_TERMINATE),
        )
        self._tree.close()
        return self._finished

    def abort(self) -> None:
        try:
            self._send(
                time.monotonic(),
                force=True,
                alive=self._tree.leader_returncode() is None,
            )

            self._tree.wait_leader(self._grace_seconds)
        finally:
            self._tree.close()

    def wait(
        self, *, poll_seconds: float = 0.5, cancelled: Event | None = None
    ) -> ProcessResult:
        cancellation = Event() if cancelled is None else cancelled
        try:
            while True:
                if cancellation.is_set():
                    raise CancelledError
                result = self.poll()
                if result is not None:
                    return result
                cancellation.wait(poll_seconds)
        except BaseException:
            self.abort()
            raise


def run_supervised(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    hard_seconds: float | None,
    grace_seconds: float,
    cwd: Path | None = None,
    stdout: IO[Any] | None = None,
    stderr: IO[Any] | None = None,
    memory_max_gb: float | None = None,
    poll_seconds: float = 0.5,
) -> ProcessResult:
    worker = ProcessSupervisor(
        command,
        env=env,
        hard_seconds=hard_seconds,
        grace_seconds=grace_seconds,
        cwd=cwd,
        stdout=stdout,
        stderr=stderr,
        memory_max_gb=memory_max_gb,
    )
    return worker.wait(poll_seconds=poll_seconds)


__all__ = [
    "ProcessEnding",
    "ProcessResult",
    "ProcessSupervisor",
    "build_capped_command",
    "check_memory_cap_available",
    "run_supervised",
]
