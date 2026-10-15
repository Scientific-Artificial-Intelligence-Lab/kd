
from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import Event
from typing import IO, Any, Literal

logger = logging.getLogger(__name__)


class ProcessEnding(StrEnum):

    EXIT = "exit"
    SIGTERM = "sigterm"
    SIGKILL = "sigkill"


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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _member_rss(path: Path, pgid: int) -> int:
    try:
        stat = (path / "stat").read_bytes().rsplit(b")", 1)[1].split()
        if int(stat[2]) != pgid:
            return 0
        for line in (path / "status").read_bytes().splitlines():
            if line.startswith(b"VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return 0
    return 0


def _group_rss(pgid: int) -> int:

    proc = Path("/proc")
    total = _member_rss(proc / str(pgid), pgid)
    for path in proc.iterdir():
        if path.name.isdigit() and path.name != str(pgid):
            total += _member_rss(path, pgid)
    return total


def _signal_group(pgid: int, sig: signal.Signals) -> None:

    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, sig)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def build_capped_command(
    command: Sequence[str], memory_max_gb: float | None
) -> list[str]:
    if memory_max_gb is None:
        return list(command)
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "-p",
        f"MemoryMax={memory_max_gb}G",
        "-p",
        "MemorySwapMax=0",
        "--",
        *command,
    ]


def check_memory_cap_available() -> None:
    probe = subprocess.run(
        build_capped_command(["/bin/true"], 1.0),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "a capped systemd user scope cannot be started here: "
            f"{probe.stderr.strip()}"
        )


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
        self._term_sent_at: str | None = None
        self._kill_sent_at: str | None = None
        self._ended_by = ProcessEnding.EXIT
        self._stop_reason: Literal["timeout", "requested"] | None = None
        self._left_descendants = False
        self._peak_rss_bytes = 0
        self._finished: ProcessResult | None = None
        self._proc = subprocess.Popen(
            build_capped_command(command, memory_max_gb),
            env=env,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            self._peak_rss_bytes = _group_rss(self.pid)
        except BaseException:
            self.abort()
            raise

    @property
    def pid(self) -> int:
        return self._proc.pid

    def poll(self) -> ProcessResult | None:
        if self._finished is not None:
            return self._finished
        alive = self._proc.poll() is None
        if not alive and not _group_alive(self.pid):
            return self._finish(group_emptied=True)
        self._peak_rss_bytes = max(self._peak_rss_bytes, _group_rss(self.pid))
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
                self._send(signal.SIGTERM, now, alive=alive)
        elif self._kill_deadline is None and now >= self._term_deadline:
            self._send(signal.SIGKILL, now, alive=alive)
        elif self._kill_deadline is not None and now >= self._kill_deadline:
            logger.warning("process group %d remains after SIGKILL", self.pid)
            return self._finish(group_emptied=False)
        return None

    def request_stop(self) -> None:
        if self._term_deadline is not None or self._finished is not None:
            return
        alive = self._proc.poll() is None
        if not alive and not _group_alive(self.pid):
            self._finish(group_emptied=True)
            return
        if alive and self._stop_reason is None:
            self._stop_reason = "requested"
        self._left_descendants = not alive
        self._send(signal.SIGTERM, time.monotonic(), alive=alive)

    def hurry(self) -> None:
        if self._term_deadline is not None:
            self._term_deadline = time.monotonic()

    def _send(self, sig: signal.Signals, now: float, *, alive: bool) -> None:






        self._record(sig, alive=alive)
        try:
            _signal_group(self.pid, sig)
        except OSError:
            self._set_deadline(sig, now)
            raise
        self._set_deadline(sig, now)

    def _record(self, sig: signal.Signals, *, alive: bool) -> None:
        sent_at = _utc_now()
        if sig == signal.SIGTERM:
            if alive:
                self._ended_by = ProcessEnding.SIGTERM
            self._term_sent_at = sent_at
        else:
            if alive:
                self._ended_by = ProcessEnding.SIGKILL
            self._kill_sent_at = sent_at

    def _set_deadline(self, sig: signal.Signals, now: float) -> None:
        if sig == signal.SIGTERM:
            self._term_deadline = now + self._grace_seconds
        else:
            self._kill_deadline = now + self._grace_seconds

    def _finish(self, *, group_emptied: bool) -> ProcessResult:
        self._finished = ProcessResult(
            pid=self.pid,
            returncode=self._proc.returncode,
            ended_by=self._ended_by,
            stop_reason=self._stop_reason,
            sigkill_used=self._kill_sent_at is not None,
            started_at=self._started_at,
            term_sent_at=self._term_sent_at,
            kill_sent_at=self._kill_sent_at,
            ended_at=_utc_now(),
            wallclock_seconds=time.monotonic() - self._started,
            peak_rss_bytes=self._peak_rss_bytes,
            left_descendants=self._left_descendants,
            group_emptied=group_emptied,
        )
        return self._finished

    def abort(self) -> None:
        self._send(signal.SIGKILL, time.monotonic(), alive=self._proc.poll() is None)

        with contextlib.suppress(subprocess.TimeoutExpired):
            self._proc.wait(timeout=self._grace_seconds)

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
