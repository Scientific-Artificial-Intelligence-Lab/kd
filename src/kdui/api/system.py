
from __future__ import annotations

import os
import platform
import socket
import sys
from importlib.metadata import PackageNotFoundError, version

from fastapi import APIRouter

from kdui.api import SchedulerDep, SettingsDep
from kdui.models import SystemInfo

router = APIRouter(tags=["system"])

_DISTRIBUTIONS = ("sail-kd", "torch")


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in _DISTRIBUTIONS:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


@router.get("/system", response_model=SystemInfo)
def read_system(settings: SettingsDep, scheduler: SchedulerDep) -> SystemInfo:
    return SystemInfo(
        hostname=socket.gethostname(),
        platform="windows" if sys.platform == "win32" else sys.platform,
        wsl="microsoft" in platform.release().lower(),
        cpu_count=os.cpu_count() or 1,
        versions=_versions(),
        workspace=str(settings.workspace.root),
        memory=scheduler.memory_budget(),
        memory_cap_kind=scheduler.memory_cap_kind(),
        queues=scheduler.queues(),
        warnings=scheduler.warnings(),
    )
