
from __future__ import annotations

import asyncio
import hashlib
import os
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TypeVar

from fastapi import APIRouter, Request
from pydantic import BaseModel

from kdui import kdproc
from kdui.api import ApiError, SettingsDep
from kdui.models import (
    CatalogListing,
    CatalogRequest,
    CatalogSource,
    DatasetDetail,
    DatasetList,
    DatasetPreview,
    DatasetPreviewRequest,
    DatasetRecord,
    DatasetRegistration,
    DatasetSummary,
    ErrorCode,
    FieldSlice,
    InspectionRequest,
    PathImport,
    ReferenceEquation,
    ReferenceRequest,
    ReferenceToggle,
    SliceRequest,
    SliceSpec,
    SourceInspection,
    Upload,
    UploadRecord,
)
from kdui.settings import Settings, Workspace

router = APIRouter(tags=["datasets"])

ResponseT = TypeVar("ResponseT", bound=BaseModel)

DATASET_FILE = "dataset.json"
PREVIEW_FILE = "preview.json"
UPLOAD_FILE = "upload.json"
_ID_HEX_DIGITS = 16
_CHUNK_BYTES = 1024 * 1024







@router.get("/datasets", response_model=DatasetList)
def list_datasets(settings: SettingsDep) -> DatasetList:
    workspace = settings.workspace
    registered = [
        _summary(workspace, _read_dataset_record(workspace, path.parent.name))
        for path in workspace.datasets.glob(f"*/{DATASET_FILE}")
    ]
    registered.sort(key=lambda summary: summary.created_at, reverse=True)
    return DatasetList(catalog=_catalog(settings).datasets, registered=registered)


@router.post("/datasets", response_model=DatasetDetail, status_code=201)
def register_dataset(body: DatasetRegistration, settings: SettingsDep) -> DatasetDetail:
    workspace = settings.workspace
    dataset_id = "d-" + _digest(
        body.source.model_dump_json() + body.recipe.model_dump_json()
    )
    directory = workspace.dataset_dir(dataset_id)
    if not (directory / DATASET_FILE).exists():
        if isinstance(body.source, CatalogSource):

            label = name = body.source.catalog_id
        else:
            upload = _read_upload_record(workspace, body.source.upload_id)
            label, name = upload.filename, dataset_id
        record = DatasetRecord(
            dataset_id=dataset_id,
            label=body.label or label,
            source=body.source,
            recipe=body.recipe,
            name=name,
            created_at=datetime.now(UTC),
            reference_available=isinstance(body.source, CatalogSource),
        )
        directory.mkdir(parents=True, exist_ok=True)
        _write_once(directory / DATASET_FILE, record.model_dump_json(indent=1))
    return _detail(settings, _read_dataset_record(workspace, dataset_id))


@router.get("/datasets/{dataset_id}", response_model=DatasetDetail)
def read_dataset(dataset_id: str, settings: SettingsDep) -> DatasetDetail:
    return _detail(settings, _read_dataset_record(settings.workspace, dataset_id))


@router.get("/datasets/{dataset_id}/preview", response_model=DatasetPreview)
def read_dataset_preview(dataset_id: str, settings: SettingsDep) -> DatasetPreview:
    workspace = settings.workspace
    _read_dataset_record(workspace, dataset_id)
    path = workspace.dataset_dir(dataset_id) / PREVIEW_FILE
    if path.exists():
        return DatasetPreview.model_validate_json(path.read_text(encoding="utf-8"))
    preview = _call(
        "inspect.preview_dataset",
        DatasetPreviewRequest(workspace=str(workspace.root), dataset_id=dataset_id),
        DatasetPreview,
        settings,
    )
    _write_once(path, preview.model_dump_json(indent=1))
    return preview


@router.post("/datasets/{dataset_id}/slice", response_model=FieldSlice)
def read_field_slice(
    dataset_id: str, body: SliceSpec, settings: SettingsDep
) -> FieldSlice:
    workspace = settings.workspace
    _read_dataset_record(workspace, dataset_id)
    return _call(
        "inspect.slice_field",
        SliceRequest(
            workspace=str(workspace.root),
            dataset_id=dataset_id,
            spec=body,
            max_points=settings.preview_max_points,
            max_cells=settings.preview_max_cells,
        ),
        FieldSlice,
        settings,
    )


@router.post("/datasets/{dataset_id}/reference", response_model=DatasetDetail)
def set_reference_toggle(
    dataset_id: str, body: ReferenceToggle, settings: SettingsDep
) -> DatasetDetail:
    workspace = settings.workspace
    record = _read_dataset_record(workspace, dataset_id)
    marker = workspace.reference_marker(dataset_id)
    if not body.shown:
        marker.unlink(missing_ok=True)
        return _detail(settings, record)
    if not isinstance(record.source, CatalogSource):
        raise ApiError(
            409,
            ErrorCode.CONFLICT,
            "This dataset has no reference equation; only catalog datasets have one.",
            field="shown",
        )
    equation = _reference(settings, record.source)
    marker.touch()
    return _detail(settings, record, equation=equation)







@router.post("/uploads", response_model=Upload, status_code=201)
async def create_upload(
    filename: str, request: Request, settings: SettingsDep
) -> Upload:
    workspace = settings.workspace
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(name).suffix.lower()
    workspace.uploads.mkdir(parents=True, exist_ok=True)
    incoming = workspace.uploads / f".incoming-{uuid.uuid4().hex}"
    hasher = hashlib.sha256()
    size = 0
    try:
        with incoming.open("wb") as handle:
            async for chunk in request.stream():
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise ApiError(
                        413,
                        ErrorCode.PAYLOAD_TOO_LARGE,
                        f"{name} is larger than the upload limit of "
                        f"{settings.max_upload_bytes} bytes. Import it by its path "
                        "on this machine instead (POST /uploads/import); that has "
                        "no size limit.",
                    )
                hasher.update(chunk)
                await asyncio.to_thread(handle.write, chunk)
        sha256 = hasher.hexdigest()
        upload_id = "u-" + _digest(f"{sha256}:{suffix}")
        directory = workspace.upload_dir(upload_id)
        if not (directory / UPLOAD_FILE).exists():
            directory.mkdir(parents=True, exist_ok=True)
            data = directory / f"data{suffix}"
            if not data.exists():
                incoming.rename(data)
            record = UploadRecord(
                upload_id=upload_id,
                filename=name,
                suffix=suffix,
                size_bytes=size,
                sha256=sha256,
                created_at=datetime.now(UTC),
            )
            _write_once(directory / UPLOAD_FILE, record.model_dump_json(indent=1))
    finally:
        incoming.unlink(missing_ok=True)
    return _upload(_read_upload_record(workspace, upload_id))


@router.post("/uploads/import", response_model=Upload, status_code=201)
def import_path(body: PathImport, settings: SettingsDep) -> Upload:
    workspace = settings.workspace
    path = Path(body.path).expanduser().resolve()
    if not path.exists():
        raise ApiError(404, ErrorCode.NOT_FOUND, f"No file at {path}", field="path")
    upload_id = "u-" + _digest(f"path:{path}")
    directory = workspace.upload_dir(upload_id)
    if not (directory / UPLOAD_FILE).exists():
        record = UploadRecord(
            upload_id=upload_id,
            filename=path.name,
            suffix=path.suffix.lower(),
            size_bytes=path.stat().st_size,
            sha256=None,
            path=str(path),
            created_at=datetime.now(UTC),
        )
        directory.mkdir(parents=True, exist_ok=True)
        _write_once(directory / UPLOAD_FILE, record.model_dump_json(indent=1))
    return _upload(_read_upload_record(workspace, upload_id))


@router.get("/uploads/{upload_id}", response_model=Upload)
def read_upload(upload_id: str, settings: SettingsDep) -> Upload:
    return _upload(_read_upload_record(settings.workspace, upload_id))


@router.get("/uploads/{upload_id}/inspection", response_model=SourceInspection)
def inspect_upload(upload_id: str, settings: SettingsDep) -> SourceInspection:
    workspace = settings.workspace
    record = _read_upload_record(workspace, upload_id)
    path = (
        record.path
        if record.path is not None
        else str(workspace.upload_dir(upload_id) / f"data{record.suffix}")
    )
    request = InspectionRequest(path=path)
    return _call("inspect.inspect_source", request, SourceInspection, settings)







def _call(
    operation: str,
    request: BaseModel,
    response_type: type[ResponseT],
    settings: Settings,
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


def _catalog(settings: Settings) -> CatalogListing:
    cache = settings.workspace.cache / f"catalog-{version('sail-kd')}.json"
    if cache.exists():
        return CatalogListing.model_validate_json(cache.read_text(encoding="utf-8"))
    listing = _call("inspect.list_catalog", CatalogRequest(), CatalogListing, settings)
    cache.parent.mkdir(parents=True, exist_ok=True)
    _write_once(cache, listing.model_dump_json(indent=1))
    return listing


def _reference(settings: Settings, source: CatalogSource) -> ReferenceEquation:
    name = f"reference-{source.catalog_id}-{version('sail-kd')}.json"
    cache = settings.workspace.cache / name
    if cache.exists():
        return ReferenceEquation.model_validate_json(cache.read_text(encoding="utf-8"))
    equation = _call(
        "inspect.reference_equation",
        ReferenceRequest(catalog_id=source.catalog_id),
        ReferenceEquation,
        settings,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    _write_once(cache, equation.model_dump_json())
    return equation


def _summary(workspace: Workspace, record: DatasetRecord) -> DatasetSummary:
    preview = workspace.dataset_dir(record.dataset_id) / PREVIEW_FILE
    fingerprint = (
        DatasetPreview.model_validate_json(
            preview.read_text(encoding="utf-8")
        ).fingerprint
        if preview.exists()
        else None
    )
    return DatasetSummary(
        dataset_id=record.dataset_id,
        label=record.label,
        source=record.source,
        created_at=record.created_at,
        fingerprint=fingerprint,
        reference_available=record.reference_available,
        reference_shown=workspace.reference_marker(record.dataset_id).exists(),
    )


def _detail(
    settings: Settings,
    record: DatasetRecord,
    *,
    equation: ReferenceEquation | None = None,
) -> DatasetDetail:
    summary = _summary(settings.workspace, record)
    if summary.reference_shown and equation is None:
        assert isinstance(record.source, CatalogSource)
        equation = _reference(settings, record.source)
    return DatasetDetail(
        **summary.model_dump(),
        recipe=record.recipe,
        reference_equation=equation if summary.reference_shown else None,
    )


def _upload(record: UploadRecord) -> Upload:
    return Upload(
        upload_id=record.upload_id,
        filename=record.filename,
        size_bytes=record.size_bytes,
        sha256=record.sha256,
        path=record.path,
        created_at=record.created_at,
    )


def _read_dataset_record(workspace: Workspace, dataset_id: str) -> DatasetRecord:
    path = workspace.dataset_dir(dataset_id) / DATASET_FILE
    if not path.is_file():
        raise ApiError(404, ErrorCode.NOT_FOUND, f"No dataset {dataset_id}")
    return DatasetRecord.model_validate_json(path.read_text(encoding="utf-8"))


def _read_upload_record(workspace: Workspace, upload_id: str) -> UploadRecord:
    path = workspace.upload_dir(upload_id) / UPLOAD_FILE
    if not path.is_file():
        raise ApiError(404, ErrorCode.NOT_FOUND, f"No upload {upload_id}")
    return UploadRecord.model_validate_json(path.read_text(encoding="utf-8"))


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_ID_HEX_DIGITS]


def _write_once(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    temporary.write_text(text, encoding="utf-8")
    os.rename(temporary, path)
