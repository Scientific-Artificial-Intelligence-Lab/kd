
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Final

import numpy as np
import torch

from kd.data.containers import Inventory, as_axis_vector
from kd.data.layouts.kd_npz import regularize_axis
from kd.data.schema import PDEDataset

KEY_TENSOR: Final[str] = "tensor"
KEY_X: Final[str] = "x-coordinate"
KEY_Y: Final[str] = "y-coordinate"
KEY_T: Final[str] = "t-coordinate"
KEY_NU: Final[str] = "nu"
CFD_FIELDS: Final[tuple[str, ...]] = ("Vx", "Vy", "Vz", "density", "pressure")


def _n_spatial(inventory: Inventory) -> int:
    return 2 if KEY_Y in inventory.entries else 1


def _has_sample_axis(ndim: int, n_spatial: int) -> bool:
    return ndim == n_spatial + 2


def _pick_sample(select: dict[str, int] | None, n_samples: int) -> int:
    if select is None:
        raise ValueError(f"file has {n_samples} samples; pass select={{'sample': i}}")
    if set(select) != {"sample"}:
        raise ValueError(
            f"select keys must be exactly ['sample'], got {sorted(select)}"
        )
    index = select["sample"]
    if index < 0 or index >= n_samples:
        raise IndexError(f"sample index {index} out of range for {n_samples} samples")
    return index


def _time_axis(t_values: np.ndarray, nt: int) -> np.ndarray:
    if len(t_values) == nt:
        return t_values
    if len(t_values) == nt + 1:
        return t_values[:nt]
    raise ValueError(
        f"t-coordinate has {len(t_values)} entries but the solution has {nt} time steps"
    )


def _present_cfd_fields(inventory: Inventory) -> list[str]:
    return [field for field in CFD_FIELDS if field in inventory.entries]


def _validate_no_sample_selection(select: dict[str, int] | None) -> None:
    if select is not None:
        raise ValueError("file has no sample axis")


def _read_solutions(
    inventory: Inventory,
    field_names: list[str],
    *,
    select: dict[str, int] | None,
    n_spatial: int,
) -> dict[str, np.ndarray]:
    shape = inventory.entries[field_names[0]].shape
    has_sample = _has_sample_axis(len(shape), n_spatial)
    if has_sample:
        sample = _pick_sample(select, shape[0])
        return {
            field: inventory.entries[field].read_index0(sample) for field in field_names
        }
    _validate_no_sample_selection(select)
    return {field: inventory.entries[field].read() for field in field_names}


def _coordinates(
    inventory: Inventory,
    solution_shape: tuple[int, ...],
) -> dict[str, torch.Tensor | np.ndarray | Sequence[float]]:
    nt, nx, *remaining = solution_shape
    x = regularize_axis(KEY_X, as_axis_vector(KEY_X, inventory.entries[KEY_X].read()))
    if len(x) != nx:
        raise ValueError(
            f"x-coordinate has {len(x)} entries but the solution x dimension has {nx}"
        )
    coords: dict[str, torch.Tensor | np.ndarray | Sequence[float]] = {"x": x}
    if remaining:
        ny = remaining[0]
        y = regularize_axis(
            KEY_Y, as_axis_vector(KEY_Y, inventory.entries[KEY_Y].read())
        )
        if len(y) != ny:
            raise ValueError(
                f"y-coordinate has {len(y)} entries but the solution y dimension "
                f"has {ny}"
            )
        coords["y"] = y
    t_values = as_axis_vector(KEY_T, inventory.entries[KEY_T].read())
    coords["t"] = regularize_axis(KEY_T, _time_axis(t_values, nt))
    return coords


def _transpose_solution(values: np.ndarray) -> np.ndarray:
    return np.transpose(values, (*range(1, values.ndim), 0))


class PDEBench1D:

    name = "pdebench-1d"

    def matches(self, inventory: Inventory) -> bool:
        required = {KEY_TENSOR, KEY_X, KEY_T}
        return (
            inventory.container == "hdf5"
            and required.issubset(inventory.entries)
            and KEY_Y not in inventory.entries
            and len(inventory.entries[KEY_TENSOR].shape) in {2, 3}
        )

    def build(
        self,
        inventory: Inventory,
        *,
        select: dict[str, int] | None,
        lhs: str,
        periodic: Iterable[str] | None,
        name: str,
        dtype: torch.dtype,
    ) -> PDEDataset:
        solutions = _read_solutions(inventory, [KEY_TENSOR], select=select, n_spatial=1)
        solution = solutions[KEY_TENSOR]
        coords = _coordinates(inventory, tuple(solution.shape))
        return PDEDataset.from_arrays(
            coords=coords,
            fields={"u": _transpose_solution(solution)},
            lhs=lhs,
            periodic=periodic,
            name=name,
            ground_truth=None,
            dtype=dtype,
        )


class PDEBenchCFD:

    name = "pdebench-cfd"

    def matches(self, inventory: Inventory) -> bool:
        if inventory.container != "hdf5" or KEY_TENSOR in inventory.entries:
            return False
        if KEY_X not in inventory.entries or KEY_T not in inventory.entries:
            return False
        field_names = _present_cfd_fields(inventory)
        if not field_names:
            return False
        shapes = {inventory.entries[field].shape for field in field_names}
        if len(shapes) != 1:
            return False
        ndim = len(inventory.entries[field_names[0]].shape)
        n_spatial = _n_spatial(inventory)
        return ndim in {n_spatial + 1, n_spatial + 2}

    def build(
        self,
        inventory: Inventory,
        *,
        select: dict[str, int] | None,
        lhs: str,
        periodic: Iterable[str] | None,
        name: str,
        dtype: torch.dtype,
    ) -> PDEDataset:
        field_names = _present_cfd_fields(inventory)
        if "Vz" in field_names:
            raise ValueError("3-D CFD is not supported")
        n_spatial = _n_spatial(inventory)
        ndim = len(inventory.entries[field_names[0]].shape)
        if ndim not in {n_spatial + 1, n_spatial + 2}:
            raise ValueError("3-D CFD is not supported")
        solutions = _read_solutions(
            inventory, field_names, select=select, n_spatial=n_spatial
        )
        first = solutions[field_names[0]]
        coords = _coordinates(inventory, tuple(first.shape))
        fields: dict[str, torch.Tensor | np.ndarray] = {
            field: _transpose_solution(solutions[field]) for field in field_names
        }
        return PDEDataset.from_arrays(
            coords=coords,
            fields=fields,
            lhs=lhs,
            periodic=periodic,
            name=name,
            ground_truth=None,
            dtype=dtype,
        )


PDEBENCH_1D = PDEBench1D()
PDEBENCH_CFD = PDEBenchCFD()


__all__ = [
    "CFD_FIELDS",
    "KEY_NU",
    "KEY_T",
    "KEY_TENSOR",
    "KEY_X",
    "KEY_Y",
    "PDEBENCH_1D",
    "PDEBENCH_CFD",
    "PDEBench1D",
    "PDEBenchCFD",
]
