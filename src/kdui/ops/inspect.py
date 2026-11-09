
from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import kd
from kd import PDEDataset, TabularDataset, TrajectorySet
from kd.search import mode_inputs
from kdui.models import (
    ArrayInfo,
    CatalogDataset,
    CatalogListing,
    CatalogRequest,
    DatasetPreview,
    DatasetPreviewRequest,
    DatasetRecord,
    DatasetReport,
    FieldSlice,
    InspectionRequest,
    ModeApplicability,
    ReferenceEquation,
    ReferenceRequest,
    SamplingInfo,
    SliceRequest,
    SliceSpec,
    SourceInspection,
    UploadRecord,
)
from kdui.settings import Workspace

_SAMPLE_SEED = 0

_HOMOGENEOUS_LHS = "0"

_RECIPE_LOAD_KEYS = (
    "kind",
    "coords",
    "fields",
    "field_axes",
    "select",
    "layout",
    "lhs",
    "periodic",
    "trajectory_id",
    "params",
)







def inspect_source(request: InspectionRequest) -> SourceInspection:
    report = kd.inspect_file(request.path).to_dict()
    return SourceInspection(
        container=report["container"],
        arrays=[ArrayInfo.model_validate(entry) for entry in report["arrays"]],
        layouts=report["layouts"],
    )


def preview_dataset(request: DatasetPreviewRequest) -> DatasetPreview:
    dataset = load_dataset(
        Workspace(Path(request.workspace)), request.dataset_id, with_reference=False
    )
    report = kd.preview_report(dataset).to_dict()
    return DatasetPreview(
        created_at=datetime.now(UTC),
        fingerprint=kd.compute_dataset_fingerprint(dataset),
        report=DatasetReport.model_validate(report),
        applicability=_applicability(report),
        slices=_default_slices(report),
    )


def slice_field(request: SliceRequest) -> FieldSlice:
    dataset = load_dataset(
        Workspace(Path(request.workspace)), request.dataset_id, with_reference=False
    )
    return sample_field(
        dataset,
        request.spec,
        dataset_id=request.dataset_id,
        quantity_latex=_name_latex(request.spec.field),
        max_points=request.max_points,
        max_cells=request.max_cells,
    )


def list_catalog(request: CatalogRequest) -> CatalogListing:
    return CatalogListing(
        datasets=[
            CatalogDataset(
                catalog_id=row["id"],
                topology=row["topology"],
                axes=row["axes"],
                lhs=row["lhs"],
                file_format=row["fmt"],
                tier=row["tier"],
                tags=row["tags"],
            )
            for row in kd.list_datasets_answer_blind()
        ]
    )


def reference_equation(request: ReferenceRequest) -> ReferenceEquation:
    spec = kd.get_dataset(request.catalog_id)
    law = kd.law_from_equation(spec.equation, axes=spec.axes, fields=("u",))
    formatted = kd.format_pde(
        list(law.terms),
        list(law.coefficients),
        lhs=_HOMOGENEOUS_LHS if law.lhs is None else law.lhs,
    )
    return ReferenceEquation(latex=formatted.latex)







def load_dataset(
    workspace: Workspace, dataset_id: str, *, with_reference: bool
) -> PDEDataset | TabularDataset:
    record = DatasetRecord.model_validate_json(
        (workspace.dataset_dir(dataset_id) / "dataset.json").read_text(encoding="utf-8")
    )
    recipe = record.recipe
    loaded: PDEDataset | TabularDataset | TrajectorySet
    if record.source.origin == "catalog":


        options = {
            key: value
            for key in _RECIPE_LOAD_KEYS
            if (value := getattr(recipe, key)) is not None
        }
        loaded = kd.load(record.source.catalog_id, **options)
    else:
        options = {key: getattr(recipe, key) for key in _RECIPE_LOAD_KEYS}
        loaded = kd.load(
            _upload_file(workspace, record.source.upload_id),
            name=record.name,
            **options,
        )
    if isinstance(loaded, TrajectorySet):
        if recipe.target_state is None:
            raise ValueError(
                "This file loads as a set of trajectories: choose the state whose "
                "derivative is regressed (recipe.target_state)."
            )
        loaded = kd.trajectories_to_tabular(
            loaded, recipe.target_state, trim=recipe.trim or 0
        )
    if isinstance(loaded, PDEDataset) and not with_reference:
        loaded.ground_truth = None
    return loaded


def sample_field(
    dataset: PDEDataset | TabularDataset,
    spec: SliceSpec,
    *,
    dataset_id: str,
    quantity_latex: str,
    max_points: int,
    max_cells: int,
) -> FieldSlice:
    if isinstance(dataset, TabularDataset):
        topology = "tabular"
        full_shape = [int(dataset.y.shape[0])]
        fixed: dict[str, int] = {}
        coordinates, values, sampling = _sample_points(
            {axis: _table_column(dataset, axis) for axis in spec.axes},
            _table_column(dataset, spec.field),
            max_points,
        )
    else:
        topology = str(dataset.topology.value)
        field = _numpy(dataset.get_field(spec.field))
        full_shape = list(field.shape)
        if topology == "grid":
            coordinates, values, fixed, sampling = _sample_grid(
                dataset, field, spec, max_points=max_points, max_cells=max_cells
            )
        else:
            fixed = {}
            coordinates, values, sampling = _sample_points(
                {axis: _numpy(dataset.get_coords(axis)) for axis in spec.axes},
                field,
                max_points,
            )
    shown, missing = _finite_or_none(values)
    return FieldSlice(
        dataset_id=dataset_id,
        fingerprint=kd.compute_dataset_fingerprint(dataset),
        field=spec.field,
        quantity_latex=quantity_latex,
        topology=topology,
        axes=list(spec.axes),
        coordinates={
            axis: [float(v) for v in array] for axis, array in coordinates.items()
        },
        values=shown,
        fixed=fixed,
        full_shape=full_shape,
        sampling=sampling,
        missing_count=missing,
    )







def _upload_file(workspace: Workspace, upload_id: str) -> Path:
    directory = workspace.upload_dir(upload_id)
    record = UploadRecord.model_validate_json(
        (directory / "upload.json").read_text(encoding="utf-8")
    )
    if record.path is not None:
        return Path(record.path)
    return directory / f"data{record.suffix}"


def _applicability(report: dict[str, Any]) -> list[ModeApplicability]:
    inputs = mode_inputs(report)
    rows = []
    for schema in kd.instrument_schemas():
        for mode in schema["modes"]:
            verdict = kd.mode_accepts(mode, **inputs)
            rows.append(
                ModeApplicability(
                    algorithm=schema["algorithm"],
                    mode=mode["name"],
                    accepts=verdict.accepts,
                    reason=verdict.reason,
                    params=verdict.params,
                )
            )
    return rows


def _default_slices(report: dict[str, Any]) -> list[SliceSpec]:
    fields = [entry["name"] for entry in report["fields"]]
    if report["topology"] == "tabular":
        target = report["lhs_field"]
        return [
            SliceSpec(field=target, axes=[name]) for name in fields if name != target
        ]
    sizes = {entry["name"]: entry["n"] for entry in report["axes"]}
    axes = list(sizes)
    if report["topology"] != "grid":
        return [SliceSpec(field=name, axes=axes[:2]) for name in fields]
    spatial = [axis for axis in axes if axis != report["lhs_axis"]]
    shown = spatial[:2] if len(spatial) >= 2 else axes[:2]
    fixed = {axis: sizes[axis] // 2 for axis in axes if axis not in shown}
    return [SliceSpec(field=name, axes=shown, fixed=fixed) for name in fields]


def _sample_grid(
    dataset: PDEDataset,
    field: npt.NDArray[Any],
    spec: SliceSpec,
    *,
    max_points: int,
    max_cells: int,
) -> tuple[dict[str, npt.NDArray[Any]], npt.NDArray[Any], dict[str, int], SamplingInfo]:
    assert dataset.axis_order is not None
    order = list(dataset.axis_order)
    fixed = {
        axis: spec.fixed.get(axis, field.shape[order.index(axis)] // 2)
        for axis in order
        if axis not in spec.axes
    }
    index = tuple(slice(None) if axis in spec.axes else fixed[axis] for axis in order)
    cut = field[index]
    kept = [axis for axis in order if axis in spec.axes]
    cut = np.transpose(cut, [kept.index(axis) for axis in spec.axes])
    sizes = list(cut.shape)
    if len(sizes) == 1:
        strides = [max(1, math.ceil(sizes[0] / max_points))]
    else:
        strides = _grid_strides(sizes[0], sizes[1], max_cells)
    thinned = cut[tuple(slice(None, None, step) for step in strides)]
    coordinates = {
        axis: _numpy(dataset.get_coords(axis))[::step]
        for axis, step in zip(spec.axes, strides, strict=True)
    }
    n_total = math.prod(sizes)
    reduced = any(step > 1 for step in strides)
    sampling = SamplingInfo(
        method="stride" if reduced else "none",
        n_total=n_total,
        n_shown=int(thinned.size),
        stride=dict(zip(spec.axes, strides, strict=True)) if reduced else None,
    )
    return coordinates, thinned, fixed, sampling


def _grid_strides(rows: int, columns: int, max_cells: int) -> list[int]:
    steps = [1, 1]
    while math.ceil(rows / steps[0]) * math.ceil(columns / steps[1]) > max_cells:
        if math.ceil(rows / steps[0]) >= math.ceil(columns / steps[1]):
            steps[0] += 1
        else:
            steps[1] += 1
    return steps


def _sample_points(
    coordinates: dict[str, npt.NDArray[Any]],
    values: npt.NDArray[Any],
    max_points: int,
) -> tuple[dict[str, npt.NDArray[Any]], npt.NDArray[Any], SamplingInfo]:
    n_total = int(values.shape[0])
    if n_total <= max_points:
        return (
            coordinates,
            values,
            SamplingInfo(method="none", n_total=n_total, n_shown=n_total, stride=None),
        )
    chosen = np.sort(
        np.random.default_rng(_SAMPLE_SEED).choice(n_total, max_points, replace=False)
    )
    return (
        {axis: array[chosen] for axis, array in coordinates.items()},
        values[chosen],
        SamplingInfo(method="random", n_total=n_total, n_shown=max_points, stride=None),
    )


def _table_column(dataset: TabularDataset, name: str) -> npt.NDArray[Any]:
    if name == dataset.target_name:
        return np.asarray(dataset.y, dtype=float)
    if name not in dataset.var_names:
        raise ValueError(
            f"Column {name!r} is not in this table; columns: "
            f"{[*dataset.var_names, dataset.target_name]}"
        )
    return np.asarray(dataset.X[:, list(dataset.var_names).index(name)], dtype=float)


def _numpy(values: Any) -> npt.NDArray[Any]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=float)


def _finite_or_none(
    values: npt.NDArray[Any],
) -> tuple[list[float | None] | list[list[float | None]], int]:
    finite = np.isfinite(values)
    missing = int(values.size - np.count_nonzero(finite))
    if values.ndim == 1:
        return [
            float(v) if ok else None for v, ok in zip(values, finite, strict=True)
        ], missing
    return [
        [float(v) if ok else None for v, ok in zip(row, row_ok, strict=True)]
        for row, row_ok in zip(values, finite, strict=True)
    ], missing


_SUBSCRIPTED = re.compile(r"([A-Za-z]+)_([A-Za-z0-9]+)")


def _name_latex(name: str) -> str:
    if re.fullmatch(r"[A-Za-z]", name):
        return name
    if match := _SUBSCRIPTED.fullmatch(name):
        return f"{match.group(1)}_{{{match.group(2)}}}"
    return r"\mathrm{" + name.replace("_", r"\_") + "}"
