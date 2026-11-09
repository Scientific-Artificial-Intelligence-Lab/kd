
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.types import Scope

from kdui.api import (
    ApiError,
    artifacts,
    capabilities,
    datasets,
    jobs,
    runs,
    session,
    system,
)
from kdui.api.runs import ManifestError
from kdui.kdproc import KdProcessError
from kdui.models import API_PREFIX, ErrorCode, ErrorResponse
from kdui.scheduler import Scheduler
from kdui.settings import Settings

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

_CODE_OF_STATUS = {
    401: ErrorCode.UNAUTHORIZED,
    403: ErrorCode.FORBIDDEN,
    404: ErrorCode.NOT_FOUND,
    409: ErrorCode.CONFLICT,
    413: ErrorCode.PAYLOAD_TOO_LARGE,
}

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    status: {"model": ErrorResponse} for status in (401, 403, 404, 409, 422, 504)
}


class _PageFiles(StaticFiles):

    def file_response(
        self,
        full_path: str | os.PathLike[str],
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        response = super().file_response(full_path, stat_result, scope, status_code)
        response.headers["Cache-Control"] = "no-cache"
        return response


def _error(status_code: int, body: ErrorResponse) -> JSONResponse:
    return JSONResponse(body.model_dump(mode="json"), status_code=status_code)


async def _api_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return _error(
        exc.status_code,
        ErrorResponse(code=exc.code, message=exc.message, field=exc.field),
    )


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, HTTPException)
    code = _CODE_OF_STATUS.get(exc.status_code, ErrorCode.INVALID_REQUEST)
    return _error(exc.status_code, ErrorResponse(code=code, message=str(exc.detail)))


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    first = exc.errors()[0]
    location = ".".join(str(part) for part in first["loc"][1:])
    return _error(
        422,
        ErrorResponse(
            code=ErrorCode.INVALID_REQUEST,
            message=str(first["msg"]),
            field=location or None,
        ),
    )


async def _kd_process_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, KdProcessError)
    logger.error("short kd call gave no answer: %s", exc)
    return _error(504, ErrorResponse(code=ErrorCode.KD_UNAVAILABLE, message=str(exc)))


async def _manifest_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ManifestError)
    return _error(422, ErrorResponse(code=ErrorCode.KD_ERROR, message=str(exc)))


async def _drive(scheduler: Scheduler, interval_seconds: float) -> None:
    while True:
        try:
            await asyncio.to_thread(scheduler.tick)
        except Exception:


            logger.exception("scheduler pass failed")
        await asyncio.sleep(interval_seconds)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    scheduler: Scheduler = app.state.scheduler
    await asyncio.to_thread(scheduler.reconcile)
    driver = asyncio.create_task(_drive(scheduler, settings.scheduler_interval_seconds))
    yield
    driver.cancel()


def create_app(settings: Settings) -> FastAPI:
    settings.workspace.create()
    app = FastAPI(
        title="KD",
        version=version("sail-kd"),
        docs_url=None,
        redoc_url=None,
        lifespan=_lifespan,
        responses=_ERROR_RESPONSES,
    )
    app.state.settings = settings
    app.state.scheduler = Scheduler(settings)

    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(HTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(KdProcessError, _kd_process_error)
    app.add_exception_handler(ManifestError, _manifest_error)

    app.include_router(session.router, prefix=API_PREFIX)
    protected = [Depends(session.require_session)]
    for resource in (system, capabilities, datasets, jobs, runs, artifacts):
        app.include_router(resource.router, prefix=API_PREFIX, dependencies=protected)

    app.mount("/", _PageFiles(directory=STATIC_DIR, html=True), name="static")
    return app
