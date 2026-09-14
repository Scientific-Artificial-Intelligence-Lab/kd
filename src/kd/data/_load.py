
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Sequence
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from kd.data._factory import parse_lhs_spec
from kd.data.catalog import DATASET_CATALOG, get_dataset
from kd.data.containers import ArrayEntry, Inventory, as_axis_vector, read_inventory
from kd.data.layouts import Layout, layout_by_name, recognize
from kd.data.layouts.kd_npz import regularize_axis
from kd.data.schema import PDEDataset
from kd.data.source import DatasetSource


@dataclass(kw_only=True)
class _FileOptions:
    coords: dict[str, str] | None
    fields: dict[str, str] | None
    field_axes: Sequence[str] | None
    select: dict[str, int] | None
    layout: str | None
    loader: Callable[[Path], PDEDataset] | None
    lhs: str
    periodic: Iterable[str] | None
    name: str | None
    dtype: torch.dtype


def load(
    source: str | Path,
    *,
    coords: dict[str, str] | None = None,
    fields: dict[str, str] | None = None,
    field_axes: Sequence[str] | None = None,
    select: dict[str, int] | None = None,
    layout: str | None = None,
    loader: Callable[[Path], PDEDataset] | None = None,
    lhs: str = "u_t",
    periodic: Iterable[str] | None = None,
    name: str | None = None,
    dtype: torch.dtype = torch.float64,
) -> PDEDataset:
    options = _FileOptions(
        coords=coords,
        fields=fields,
        field_axes=field_axes,
        select=select,
        layout=layout,
        loader=loader,
        lhs=lhs,
        periodic=periodic,
        name=name,
        dtype=dtype,
    )
    return _load_source(source, options)


def _load_source(source: str | Path, options: _FileOptions) -> PDEDataset:
    if isinstance(source, str) and source in DATASET_CATALOG:
        _validate_catalog_options(options)
        return get_dataset(source).loader()
    path = Path(source).resolve()
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")
    dataset, container, route, hints = _load_file(path, options)


    result = copy(dataset)
    result.source = DatasetSource(
        path=str(path),
        sha256=_sha256(path),
        container=container,
        layout=route,
        hints=hints,
        select=options.select,
    )
    return result


def _reject_keywords(route: str, **keywords: object) -> None:
    for keyword, value in keywords.items():
        if value is not None:
            raise ValueError(f"{route} takes no {keyword}=")


def _validate_catalog_options(options: _FileOptions) -> None:
    _reject_keywords(
        "catalog dataset",
        coords=options.coords,
        fields=options.fields,
        field_axes=options.field_axes,
        select=options.select,
        layout=options.layout,
        loader=options.loader,
        periodic=options.periodic,
        name=options.name,
    )
    if options.lhs != "u_t":
        raise ValueError("catalog dataset takes no nondefault lhs=")
    if options.dtype != torch.float64:
        raise ValueError("catalog dataset takes no nondefault dtype=")


def _load_file(
    path: Path,
    options: _FileOptions,
) -> tuple[PDEDataset, str, str, dict[str, Any]]:
    if path.suffix.lower() == ".xlsx":
        return (
            _load_xlsx(path, options),
            "xlsx",
            "xlsx",
            {
                "coords": options.coords,
                "fields": options.fields,
            },
        )
    if options.loader is not None:
        return _load_custom(path, options)
    inventory = read_inventory(path)
    if inventory.container == "csv" and all(
        len(entry.shape) == 1 for entry in inventory.entries.values()
    ):
        return _load_csv_longform(inventory, options)
    if options.coords is not None or options.fields is not None:
        if options.coords is None or options.fields is None:
            raise ValueError("coords= and fields= go together")
        _reject_keywords("mapping route", layout=options.layout)
        dataset = _load_mapping(inventory, options)
        return (
            dataset,
            inventory.container,
            "mapping",
            {
                "coords": options.coords,
                "fields": options.fields,
                "field_axes": list(options.field_axes) if options.field_axes else None,
            },
        )
    _reject_keywords("layout route", field_axes=options.field_axes)
    layout = _choose_layout(inventory, options.layout)
    dataset = layout.build(
        inventory,
        select=options.select,
        lhs=options.lhs,
        periodic=options.periodic,
        name=options.name or path.stem,
        dtype=options.dtype,
    )
    return dataset, inventory.container, layout.name, {}


def _load_csv_longform(
    inventory: Inventory,
    options: _FileOptions,
) -> tuple[PDEDataset, str, str, dict[str, Any]]:
    _reject_keywords(
        "long-form CSV",
        field_axes=options.field_axes,
        select=options.select,
        layout=options.layout,
    )
    coords, fields = _csv_mappings(inventory, options)
    columns: dict[str, np.ndarray] = {}
    for key in (*coords.values(), *fields.values()):
        if key not in inventory.entries:
            raise ValueError(
                f"unknown CSV column {key!r}; available: {list(inventory.entries)}"
            )
        entry = inventory.entries[key]
        if not entry.numeric:
            raise ValueError(f"CSV column {key!r} must be numeric")
        values = entry.read()
        if not np.isfinite(values).all():
            raise ValueError(f"CSV column {key!r} must contain only finite values")
        columns[key] = values
    dataset = _csv_points(
        {axis: columns[key] for axis, key in coords.items()},
        {field: columns[key] for field, key in fields.items()},
        options,
        name=options.name or inventory.path.stem,
    )
    return dataset, "csv", "csv-longform", {"coords": coords, "fields": fields}


def _csv_mappings(
    inventory: Inventory,
    options: _FileOptions,
) -> tuple[dict[str, str], dict[str, str]]:
    if options.coords is not None or options.fields is not None:
        if options.coords is None or options.fields is None:
            raise ValueError("coords= and fields= go together")
        coords, fields = options.coords, options.fields
    else:
        field, _, _ = parse_lhs_spec(
            options.lhs,
            fields=inventory.entries,
            coords=inventory.entries,
        )
        fields = {field: field}
        coords = {key: key for key in inventory.entries if key != field}
    if not coords or not fields:
        raise ValueError("CSV coords= and fields= must be non-empty mappings")
    return coords, fields


def _csv_points(
    coords: dict[str, np.ndarray],
    fields: dict[str, np.ndarray],
    options: _FileOptions,
    *,
    name: str,
) -> PDEDataset:
    points = np.column_stack(list(coords.values()))
    unique_axes = {axis: np.unique(values) for axis, values in coords.items()}
    size = math.prod(len(values) for values in unique_axes.values())
    if size == len(points) and len(np.unique(points, axis=0)) == len(points):
        indices = tuple(
            np.searchsorted(unique_axes[axis], values)
            for axis, values in coords.items()
        )
        grid_fields: dict[str, torch.Tensor | np.ndarray] = {}
        shape = tuple(len(values) for values in unique_axes.values())
        for field, values in fields.items():
            grid = np.empty(shape, dtype=values.dtype)
            grid[indices] = values
            grid_fields[field] = grid
        grid_axes: dict[str, torch.Tensor | np.ndarray | Sequence[float]] = {
            axis: regularize_axis(axis, values) if len(values) > 1 else values
            for axis, values in unique_axes.items()
        }
        return PDEDataset.from_arrays(
            grid_axes,
            grid_fields,
            lhs=options.lhs,
            periodic=options.periodic,
            name=name,
            dtype=options.dtype,
        )
    dataset = PDEDataset.from_scatter(
        dict(coords),
        dict(fields),
        lhs=options.lhs,
        name=name,
        dtype=options.dtype,
    )
    if options.periodic is not None:
        assert dataset.axes is not None
        for axis in options.periodic:
            if axis not in dataset.axes:
                raise ValueError(f"periodic references unknown axis {axis!r}")
            dataset.axes[axis].is_periodic = True
    return dataset


def _load_xlsx(path: Path, options: _FileOptions) -> PDEDataset:
    if options.coords is None or options.fields is None:
        raise ValueError("xlsx needs coords= and fields=")
    _reject_keywords(
        "xlsx",
        field_axes=options.field_axes,
        select=options.select,
        layout=options.layout,
        loader=options.loader,
        periodic=options.periodic,
    )
    return PDEDataset.from_xlsx(
        path,
        coords=options.coords,
        fields=options.fields,
        lhs=options.lhs,
        name=options.name or path.stem,
    )


def _load_custom(
    path: Path,
    options: _FileOptions,
) -> tuple[PDEDataset, str, str, dict[str, Any]]:
    _reject_keywords(
        "loader route",
        coords=options.coords,
        fields=options.fields,
        field_axes=options.field_axes,
        select=options.select,
        layout=options.layout,
    )
    assert options.loader is not None
    dataset = options.loader(path)
    if not isinstance(dataset, PDEDataset):
        raise TypeError(f"loader must return PDEDataset, got {type(dataset).__name__}")
    suffix = path.suffix.lower()
    container = "hdf5" if suffix in {".h5", ".hdf5"} else suffix.lstrip(".")
    if suffix == ".mat" and h5py.is_hdf5(path):
        container = "mat73"
    route = f"{options.loader.__module__}.{options.loader.__qualname__}"
    return dataset, container, route, {}


def _load_mapping(inventory: Inventory, options: _FileOptions) -> PDEDataset:
    if inventory.container in {"npy", "csv"}:
        raise ValueError(
            f"{inventory.container} carries no coordinate arrays; "
            "rebuild the grid and use PDEDataset.from_arrays"
        )
    assert options.coords is not None and options.fields is not None
    for key in (*options.coords.values(), *options.fields.values()):
        if key not in inventory.entries:
            raise ValueError(
                f"unknown array key {key!r}; "
                f"inventory entries: {list(inventory.entries)}"
            )
    coords: dict[str, torch.Tensor | np.ndarray | Sequence[float]] = {
        axis: regularize_axis(axis, as_axis_vector(key, inventory.entries[key].read()))
        for axis, key in options.coords.items()
    }
    extra_axes = _mapping_extra_axes(options, inventory)
    fields: dict[str, torch.Tensor | np.ndarray] = {
        field: _mapped_field(
            inventory.entries[key],
            inventory=inventory,
            options=options,
            extra_axes=extra_axes,
        )
        for field, key in options.fields.items()
    }
    return PDEDataset.from_arrays(
        coords=coords,
        fields=fields,
        lhs=options.lhs,
        periodic=options.periodic,
        name=options.name or inventory.path.stem,
        dtype=options.dtype,
    )


def _mapping_extra_axes(options: _FileOptions, inventory: Inventory) -> set[str]:
    assert options.coords is not None
    axes = options.field_axes
    if axes is not None:
        assert options.fields is not None
        for key in options.fields.values():
            ndim = len(inventory.entries[key].shape)
            if len(axes) != ndim:
                raise ValueError(
                    f"field_axes length {len(axes)} does not equal array {key!r} "
                    f"ndim {ndim}"
                )
        if len(axes) != len(set(axes)):
            raise ValueError(f"field_axes contains duplicate axis names: {list(axes)}")
        missing = set(options.coords) - set(axes)
        if missing:
            raise ValueError(
                f"field_axes must include coordinate axes {sorted(missing)}"
            )
    extras = set() if axes is None else set(axes) - set(options.coords)
    selected = {} if options.select is None else options.select
    unknown = set(selected) - extras
    if unknown:
        raise ValueError(
            f"select keys {sorted(unknown)} are not extra axes {sorted(extras)}"
        )
    for axis in sorted(extras):
        if axis not in selected:
            raise ValueError(
                f"extra axis {axis!r} needs an index; pass select={{'{axis}': i}}"
            )
    return extras


def _mapped_field(
    entry: ArrayEntry,
    *,
    inventory: Inventory,
    options: _FileOptions,
    extra_axes: set[str],
) -> np.ndarray:
    if options.field_axes is None:
        return entry.read()
    axes = list(options.field_axes)
    selected = {} if options.select is None else options.select
    if axes and axes[0] in extra_axes and inventory.container in {"hdf5", "mat73"}:
        values = entry.read_index0(selected[axes.pop(0)])
    else:
        values = entry.read()
    indices = tuple(
        selected[axis] if axis in extra_axes else slice(None) for axis in axes
    )
    values = values[indices]
    remaining = [axis for axis in axes if axis not in extra_axes]
    assert options.coords is not None
    return np.transpose(values, [remaining.index(axis) for axis in options.coords])


def _choose_layout(inventory: Inventory, name: str | None) -> Layout:
    if name is not None:
        layout = layout_by_name(name)
        if not layout.matches(inventory):
            raise ValueError(f"layout {name!r} does not match:\n{inventory.describe()}")
        return layout
    matches = recognize(inventory)
    if len(matches) > 1:
        raise ValueError(
            f"matching layouts {[layout.name for layout in matches]}; pass layout="
        )
    if not matches:
        darcy = (
            "\nPDEBench Darcy is a steady dataset with no time axis."
            if {"nu", "tensor"}.issubset(inventory.entries)
            else ""
        )
        raise ValueError(
            f"{inventory.describe()}\n"
            f"no registered layout recognizes this file.{darcy}\n"
            'Name the arrays with kd.load(path, coords={"x": "<key>", "t": "<key>"}, '
            'fields={"u": "<key>"})'
        )
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
