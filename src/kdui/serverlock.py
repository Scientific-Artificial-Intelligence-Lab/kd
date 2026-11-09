
from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

from kdui import jobstore
from kdui.models import ProcessIdentity
from kdui.settings import Workspace

LOCK_FILE = "server.lock"
HOLDER_FILE = "server.json"

if sys.platform == "win32":
    import msvcrt

    def _try_lock(handle: IO[bytes]) -> bool:
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

else:
    import fcntl

    def _try_lock(handle: IO[bytes]) -> bool:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True


class WorkspaceBusyError(RuntimeError):
    pass


class ServerLock:

    def __init__(self, handle: IO[bytes]) -> None:
        self._handle = handle

    def close(self) -> None:
        self._handle.close()


def hold_workspace(workspace: Workspace, *, host: str, port: int) -> ServerLock:
    root = workspace.root
    handle = (root / LOCK_FILE).open("ab")
    if not _try_lock(handle):
        handle.close()
        raise WorkspaceBusyError(_busy_message(root))
    holder = {
        **jobstore.process_identity(os.getpid()).model_dump(),
        "url": f"http://{host}:{port}/",
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    temporary = root / f".{HOLDER_FILE}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(holder), encoding="utf-8")
    os.replace(temporary, root / HOLDER_FILE)
    return ServerLock(handle)


def _busy_message(root: Path) -> str:
    path = root / HOLDER_FILE
    where = ""
    if path.is_file():
        holder = json.loads(path.read_text(encoding="utf-8"))
        identity = ProcessIdentity(pid=holder["pid"], create_time=holder["create_time"])
        if jobstore.process_alive(identity):
            where = (
                f": pid {holder['pid']}, {holder['url']}"
                f" (started {holder['started_at']})"
            )
    return (
        f"Another kd-ui holds the workspace {root}{where}. Open that one, stop it"
        " first, or give this one another --workspace."
    )
