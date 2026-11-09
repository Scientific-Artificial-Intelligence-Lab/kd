
from __future__ import annotations

import contextlib
import os
import signal
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any, Final, Literal

_GIB: Final = 1024**3


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


class ProcessTree:

    def __init__(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str],
        cwd: Path | None,
        stdout: IO[Any] | None,
        stderr: IO[Any] | None,
        memory_max_gb: float | None,
    ) -> None:
        self.memory_backend: Literal["linux-cgroup"] | None = (
            None if memory_max_gb is None else "linux-cgroup"
        )
        self.memory_limit_bytes: int | None = (
            None if memory_max_gb is None else int(memory_max_gb * _GIB)
        )
        self._proc = subprocess.Popen(
            build_capped_command(command, memory_max_gb),
            env=env,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        self.pid: int = self._proc.pid

    def leader_returncode(self) -> int | None:
        return self._proc.poll()

    def members_remain(self) -> bool:
        try:
            os.killpg(self.pid, 0)
        except ProcessLookupError:
            return False
        return True

    def rss_bytes(self) -> int:

        proc = Path("/proc")
        total = _member_rss(proc / str(self.pid), self.pid)
        for path in proc.iterdir():
            if path.name.isdigit() and path.name != str(self.pid):
                total += _member_rss(path, self.pid)
        return total

    def peak_committed_bytes(self) -> int | None:
        return None

    def stop(self, *, force: bool) -> None:

        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.pid, signal.SIGKILL if force else signal.SIGTERM)

    def wait_leader(self, seconds: float) -> None:
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._proc.wait(timeout=seconds)

    def close(self) -> None:
        pass
