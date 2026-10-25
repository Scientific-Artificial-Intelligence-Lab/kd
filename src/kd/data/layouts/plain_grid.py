
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch

from kd.data.containers import ContainerKind, Inventory, as_axis_vector
from kd.data.layouts.kd_npz import AXIS_ORDER_KEY, regularize_axis
from kd.data.schema import PDEDataset

TIME_KEY: Final[str] = "t"
SPACE_KEYS: Final[tuple[str, ...]] = ("x", "y", "z")


CONTAINERS: Final[frozenset[ContainerKind]] = frozenset({"npz", "mat", "mat73"})



SELF_DESCRIBING_KEYS: Final[tuple[str, ...]] = (AXIS_ORDER_KEY, "format")


@dataclass(frozen=True, kw_only=True)
class _PlainGrid:

    lengths: dict[str, int]
    fields: list[str]
    shape: tuple[int, ...]


def _vector_length(shape: tuple[int, ...]) -> int | None:
    sizes = [size for size in shape if size != 1]
    return sizes[0] if len(sizes) == 1 else None


def _coordinate_lengths(inventory: Inventory) -> dict[str, int] | None:
    entries = inventory.entries
    coords = [key for key in entries if key in (TIME_KEY, *SPACE_KEYS)]
    if TIME_KEY not in coords or len(coords) < 2:
        return None
    lengths: dict[str, int] = {}
    for key in coords:
        length = _vector_length(entries[key].shape)
        if length is None:
            return None
        lengths[key] = length
    return lengths


def _plain_grid(inventory: Inventory) -> _PlainGrid | None:
    entries = inventory.entries
    if inventory.container not in CONTAINERS or any(
        key in entries for key in SELF_DESCRIBING_KEYS
    ):
        return None
    lengths = _coordinate_lengths(inventory)
    if lengths is None:
        return None

    fields = [key for key in entries if key not in lengths and entries[key].shape != ()]
    shapes = {entries[key].shape for key in fields}
    if len(shapes) != 1:
        return None
    (shape,) = shapes
    if not _dimensions_fit(shape, lengths):
        return None
    return _PlainGrid(lengths=lengths, fields=fields, shape=shape)


def _dimensions_fit(shape: tuple[int, ...], lengths: Mapping[str, int]) -> bool:
    return sorted(shape) == sorted(lengths.values())


def _call(coords: Mapping[str, str], fields: Mapping[str, str], axes: str) -> str:
    return (
        f"kd.load(path, coords={json.dumps(dict(coords))}, "
        f"fields={json.dumps(dict(fields))}, field_axes={axes})"
    )


def _tied_calls(
    shape: tuple[int, ...],
    lengths: Mapping[str, int],
    *,
    coords: Mapping[str, str],
    fields: Mapping[str, str],
    field: str,
) -> list[str]:
    if len(lengths) == 2:
        first, second = lengths
        key = fields[field]
        return [
            f"{_call(coords, fields, json.dumps([first, second]))}"
            f" # {key}[i, j] is {field}({first}_i, {second}_j)",
            f"{_call(coords, fields, json.dumps([second, first]))}"
            f" # {key}[i, j] is {field}({first}_j, {second}_i)",
        ]
    choices = []
    for size in shape:
        names = [json.dumps(axis) for axis, length in lengths.items() if length == size]
        choices.append(names[0] if len(names) == 1 else f"<{' or '.join(names)}>")
    return [
        f"{_call(coords, fields, '[' + ', '.join(choices) + ']')}"
        " # each <...> is the coordinate stored along that dimension"
    ]


def read_storage_order(
    shape: tuple[int, ...],
    lengths: Mapping[str, int],
    *,
    inventory: Inventory,
    route: str,
    coords: Mapping[str, str],
    fields: Mapping[str, str],
    field: str,
) -> list[str]:
    described = (
        f"{inventory.describe()}\n{route}: field {json.dumps(field)} "
        f"(array {fields[field]!r}, shape {shape}), "
        f"coordinate lengths {json.dumps(dict(lengths))}"
    )
    if not _dimensions_fit(shape, lengths):
        raise ValueError(
            f"{described}: the dimension lengths are not the coordinate lengths "
            "in any order.\nName every dimension, in storage order, with "
            + _call(coords, fields, "[<coordinate or extra axis of each dimension>]")
            + '; an extra axis also takes select={"<axis>": <index>}'
        )
    if len(set(lengths.values())) != len(lengths):
        calls = _tied_calls(shape, lengths, coords=coords, fields=fields, field=field)
        raise ValueError(
            f"{described}: two coordinates share a length, so the storage order "
            "cannot be read from the dimension lengths.\nState it with "
            + ("one of:\n" if len(calls) > 1 else "")
            + "\n".join(calls)
        )
    by_length = {length: axis for axis, length in lengths.items()}
    return [by_length[size] for size in shape]


def _storage_order(grid: _PlainGrid, inventory: Inventory) -> list[str]:
    return read_storage_order(
        grid.shape,
        grid.lengths,
        inventory=inventory,
        route="plain grid",
        coords={key: key for key in grid.lengths},
        fields={key: key for key in grid.fields},
        field=grid.fields[0],
    )


class PlainGridLayout:

    name = "plain-grid"

    def matches(self, inventory: Inventory) -> bool:
        return _plain_grid(inventory) is not None

    def build(
        self,
        inventory: Inventory,
        *,
        select: dict[str, int] | None,
        lhs: str | None,
        periodic: Iterable[str] | None,
        name: str,
        dtype: torch.dtype,
    ) -> PDEDataset:
        if select is not None:
            raise ValueError(f"{self.name} has no sample axis")
        grid = _plain_grid(inventory)
        assert grid is not None
        entries = inventory.entries
        coords: dict[str, torch.Tensor | np.ndarray | Sequence[float]] = {
            axis: regularize_axis(axis, as_axis_vector(axis, entries[axis].read()))
            for axis in _storage_order(grid, inventory)
        }
        fields: dict[str, torch.Tensor | np.ndarray] = {
            key: entries[key].read() for key in grid.fields
        }
        return PDEDataset.from_arrays(
            coords=coords,
            fields=fields,
            lhs=lhs,
            periodic=periodic,
            name=name,
            dtype=dtype,
        )


PLAIN_GRID = PlainGridLayout()


__all__ = [
    "CONTAINERS",
    "PLAIN_GRID",
    "SELF_DESCRIBING_KEYS",
    "SPACE_KEYS",
    "TIME_KEY",
    "PlainGridLayout",
    "read_storage_order",
]
