
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import torch

from kd.data.containers import Inventory
from kd.data.schema import PDEDataset

AXIS_ORDER_KEY: Final[str] = "axis_order"
UNIFORM_RTOL: Final[float] = 1e-6
AXIS_ENDPOINT_ATOL: Final[float] = 1e-5


@dataclass(frozen=True, kw_only=True)
class KdNpzArrays:

    axis_order: list[str]
    axes: dict[str, np.ndarray]
    fields: dict[str, np.ndarray]


def decode_axis_order(raw: npt.NDArray[Any]) -> list[str]:
    if raw.ndim != 1:
        raise ValueError(f"axis_order must be 1D, got shape {raw.shape}")
    axis_order: list[str] = []
    for item in raw.tolist():
        if isinstance(item, bytes):
            axis_order.append(item.decode("utf-8"))
        else:
            axis_order.append(str(item))
    if len(axis_order) != len(set(axis_order)):
        raise ValueError(f"axis_order contains duplicate axes: {axis_order}")
    return axis_order


def load_field_array(
    data: np.lib.npyio.NpzFile,
    field_name: str,
) -> npt.NDArray[np.floating[Any]]:
    field = np.asarray(data[field_name])
    if not np.issubdtype(field.dtype, np.floating):
        raise ValueError(f"field '{field_name}' must be floating-point")
    if not np.isfinite(field).all():
        raise ValueError(f"field '{field_name}' must contain only finite values")
    return cast(npt.NDArray[np.floating[Any]], field)


def regularize_axis(
    axis_name: str,
    values: npt.NDArray[Any],
) -> npt.NDArray[np.float64]:
    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError(f"axis '{axis_name}' must be a 1D array with >=2 values")
    if not np.isfinite(axis).all():
        raise ValueError(f"axis '{axis_name}' must contain only finite values")
    diffs = np.diff(axis)
    if np.any(diffs <= 0.0):
        raise ValueError(f"axis '{axis_name}' must be strictly increasing")
    if is_uniform_spacing(diffs):
        return axis
    return reconstruct_uniform_axis(axis_name, axis, diffs)


def is_uniform_spacing(diffs: npt.NDArray[np.float64]) -> bool:
    dx = float(diffs[0])
    return float(diffs.max() - diffs.min()) <= abs(dx) * UNIFORM_RTOL


def reconstruct_uniform_axis(
    axis_name: str,
    axis: npt.NDArray[np.float64],
    diffs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    step = float(np.median(diffs))
    expected = np.linspace(
        axis[0],
        axis[0] + step * axis.size,
        axis.size,
        endpoint=False,
    )
    if (
        abs(float(axis[0] - expected[0])) > AXIS_ENDPOINT_ATOL
        or abs(float(axis[-1] - expected[-1])) > AXIS_ENDPOINT_ATOL
    ):
        raise ValueError(
            f"axis '{axis_name}' is not uniformly spaced and endpoints disagree "
            f"with endpoint=False reconstruction"
        )
    if np.any(np.abs(axis[1:-1] - expected[1:-1]) > AXIS_ENDPOINT_ATOL):
        raise ValueError(
            f"axis '{axis_name}' is not uniformly spaced: interior coordinates "
            f"disagree with endpoint=False reconstruction"
        )
    return expected


def read_kd_npz(path: str | Path, *, field: str | None = None) -> KdNpzArrays:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Data file not found: {resolved}")

    with np.load(resolved, allow_pickle=False) as data:
        axis_order = decode_axis_order(_required_array(data, AXIS_ORDER_KEY))
        required_keys = {AXIS_ORDER_KEY, *axis_order}
        if field is not None:
            required_keys.add(field)
        missing = required_keys - set(data.files)
        if missing:
            raise ValueError(f"{resolved} missing required keys: {sorted(missing)}")

        axes = {
            axis: regularize_axis(axis, _required_array(data, axis))
            for axis in axis_order
        }
        expected_shape = tuple(axes[axis].shape[0] for axis in axis_order)
        field_names = (
            [field]
            if field is not None
            else [
                key
                for key in data.files
                if key != AXIS_ORDER_KEY and key not in axis_order
            ]
        )
        fields: dict[str, np.ndarray] = {}
        for field_name in field_names:
            raw = np.asarray(data[field_name])
            if raw.shape != expected_shape:
                raise ValueError(
                    f"field '{field_name}' shape mismatch: expected {expected_shape}, "
                    f"got {raw.shape}"
                )
            fields[field_name] = load_field_array(data, field_name)

    return KdNpzArrays(axis_order=axis_order, axes=axes, fields=fields)


def _required_array(data: np.lib.npyio.NpzFile, key: str) -> npt.NDArray[Any]:
    if key not in data.files:
        raise ValueError(f"npz file missing required keys: {[key]}")
    return np.asarray(data[key])


class KdNpzLayout:

    name = "kd-npz"

    def matches(self, inventory: Inventory) -> bool:
        return inventory.container == "npz" and AXIS_ORDER_KEY in inventory.entries

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
        if select is not None:
            raise ValueError("kd-npz has no sample axis")
        arrays = read_kd_npz(inventory.path)
        coords: dict[str, torch.Tensor | np.ndarray | Sequence[float]] = dict(
            arrays.axes
        )
        fields: dict[str, torch.Tensor | np.ndarray] = dict(arrays.fields)
        return PDEDataset.from_arrays(
            coords=coords,
            fields=fields,
            lhs=lhs,
            periodic=periodic,
            name=name,
            ground_truth=None,
            dtype=dtype,
        )


KD_NPZ = KdNpzLayout()


__all__ = [
    "AXIS_ENDPOINT_ATOL",
    "AXIS_ORDER_KEY",
    "KD_NPZ",
    "UNIFORM_RTOL",
    "KdNpzArrays",
    "KdNpzLayout",
    "decode_axis_order",
    "is_uniform_spacing",
    "load_field_array",
    "read_kd_npz",
    "reconstruct_uniform_axis",
    "regularize_axis",
]
