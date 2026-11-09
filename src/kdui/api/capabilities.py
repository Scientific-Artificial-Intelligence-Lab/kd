
from __future__ import annotations

import os
from importlib.metadata import version
from pathlib import Path
from typing import TypeVar

from fastapi import APIRouter
from pydantic import BaseModel

from kdui import kdproc
from kdui.api import ApiError, SettingsDep
from kdui.models import (
    Capabilities,
    CapabilitiesRequest,
    CapabilitySnapshot,
    ErrorCode,
    PreflightCall,
    PreflightRequest,
    PreflightResult,
    PreprocessPreview,
    PreprocessPreviewCall,
    PreprocessPreviewRequest,
)
from kdui.settings import Settings

router = APIRouter(tags=["capabilities"])

ResponseT = TypeVar("ResponseT", bound=BaseModel)


def _call(
    settings: Settings,
    operation: str,
    request: BaseModel,
    response_type: type[ResponseT],
) -> ResponseT:
    try:
        return kdproc.call(
            operation,
            request,
            response_type,
            timeout_seconds=settings.kdproc_timeout_seconds,
        )
    except kdproc.KdCallError as error:
        raise ApiError(422, ErrorCode.KD_ERROR, error.message) from error


def _write_snapshot(path: Path, snapshot: CapabilitySnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(snapshot.model_dump_json(indent=1), encoding="utf-8")
    os.replace(temporary, path)


def _refresh(settings: Settings) -> Capabilities:
    snapshot = _call(
        settings,
        "schemas.project_capabilities",
        CapabilitiesRequest(),
        CapabilitySnapshot,
    )
    _write_snapshot(settings.workspace.capabilities_file, snapshot)
    return Capabilities(snapshot=snapshot)


@router.get("/capabilities", response_model=Capabilities)
def read_capabilities(settings: SettingsDep) -> Capabilities:
    path = settings.workspace.capabilities_file
    if path.exists():
        snapshot = CapabilitySnapshot.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if snapshot.kd_version == version("sail-kd"):
            return Capabilities(snapshot=snapshot)
    return _refresh(settings)


@router.post("/capabilities/refresh", response_model=Capabilities)
def refresh_capabilities(settings: SettingsDep) -> Capabilities:
    return _refresh(settings)


@router.post("/preflight", response_model=PreflightResult)
def check_fit_request(body: PreflightRequest, settings: SettingsDep) -> PreflightResult:
    call = PreflightCall(workspace=str(settings.workspace.root), request=body)
    return _call(settings, "schemas.preflight", call, PreflightResult)


@router.post("/preprocess-preview", response_model=PreprocessPreview)
def preview_preprocessing(
    body: PreprocessPreviewRequest, settings: SettingsDep
) -> PreprocessPreview:
    call = PreprocessPreviewCall(
        workspace=str(settings.workspace.root),
        request=body,
        max_points=settings.preview_max_points,
        max_cells=settings.preview_max_cells,
    )
    return _call(settings, "preprocess.preview_preprocessing", call, PreprocessPreview)
