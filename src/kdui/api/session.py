
from __future__ import annotations

import hashlib
import hmac
from importlib.metadata import version
from urllib.parse import urlsplit

from fastapi import APIRouter, Request, Response

from kdui.api import ApiError, SettingsDep
from kdui.models import (
    REQUEST_KEY_HEADER,
    ErrorCode,
    SessionInfo,
    SessionOpened,
    SessionRequest,
)
from kdui.settings import Settings

router = APIRouter(tags=["session"])

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def cookie_name(settings: Settings) -> str:
    return f"kdui_session_{settings.port}"


def cookie_value(token: str) -> str:
    return hmac.new(
        token.encode("utf-8"), b"kdui-session-v1", hashlib.sha256
    ).hexdigest()


def request_key(token: str) -> str:
    return hmac.new(
        token.encode("utf-8"), b"kdui-request-v1", hashlib.sha256
    ).hexdigest()


def _check_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if request.method in _SAFE_METHODS or origin is None:
        return
    if urlsplit(origin).netloc != request.headers.get("host"):
        raise ApiError(
            403, ErrorCode.FORBIDDEN, "Request from another origin was refused."
        )


def require_session(request: Request, settings: SettingsDep) -> None:
    _check_origin(request)
    presented = request.cookies.get(cookie_name(settings), "")
    if not hmac.compare_digest(presented, cookie_value(settings.token)):
        raise ApiError(
            401,
            ErrorCode.UNAUTHORIZED,
            "No valid session. Open the URL that kd-ui printed at startup.",
        )
    if request.method in _SAFE_METHODS:
        return
    key = request.headers.get(REQUEST_KEY_HEADER, "")
    if not hmac.compare_digest(key, request_key(settings.token)):
        raise ApiError(
            401,
            ErrorCode.UNAUTHORIZED,
            "This request lacks the page's key. Open the URL that kd-ui printed"
            " at startup.",
        )


def _session_info(settings: Settings) -> SessionInfo:
    return SessionInfo(
        workspace=str(settings.workspace.root), version=version("sail-kd")
    )


@router.post("/session", response_model=SessionOpened)
def open_session(
    body: SessionRequest, request: Request, response: Response, settings: SettingsDep
) -> SessionOpened:
    _check_origin(request)
    if not hmac.compare_digest(
        body.token.encode("utf-8"), settings.token.encode("utf-8")
    ):
        raise ApiError(401, ErrorCode.UNAUTHORIZED, "The token is not valid.")
    response.set_cookie(
        cookie_name(settings),
        cookie_value(settings.token),
        httponly=True,
        samesite="strict",
        path="/",
    )
    return SessionOpened(
        **_session_info(settings).model_dump(),
        request_key=request_key(settings.token),
    )


@router.get("/session", response_model=SessionInfo)
def read_session(request: Request, settings: SettingsDep) -> SessionInfo:
    require_session(request, settings)
    return _session_info(settings)
