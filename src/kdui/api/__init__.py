
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from kdui.models import ErrorCode
from kdui.scheduler import Scheduler
from kdui.settings import Settings


class ApiError(Exception):

    def __init__(
        self,
        status_code: int,
        code: ErrorCode,
        message: str,
        *,
        field: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.field = field


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_scheduler(request: Request) -> Scheduler:
    scheduler: Scheduler = request.app.state.scheduler
    return scheduler


SettingsDep = Annotated[Settings, Depends(get_settings)]

SchedulerDep = Annotated[Scheduler, Depends(get_scheduler)]
