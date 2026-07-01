
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from kd.data.schema import AxisInfo, FieldData


def to_float_tensor(
    values: Any,
    dtype: torch.dtype,
    *,
    kind: str,
    label: str,
) -> torch.Tensor:
    if not is_floating_dtype(dtype):
        raise TypeError(f"dtype must be a floating-point torch.dtype, got {dtype!r}.")
    try:
        tensor = torch.as_tensor(values, dtype=dtype)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(
            f"Could not convert {kind} '{label}' to torch.Tensor (dtype={dtype}): {exc}"
        ) from exc
    return tensor


def is_floating_dtype(dtype: torch.dtype) -> bool:
    return torch.empty(0, dtype=dtype).is_floating_point()


def parse_lhs_spec(
    lhs: str,
    *,
    fields: dict[str, Any],
    coords: dict[str, Any],
) -> tuple[str, str, int]:
    from kd.core.expr.naming import parse_derivative_name

    parsed = parse_derivative_name(
        lhs,
        known_fields=set(fields.keys()),
        known_axes=set(coords.keys()),
    )
    if parsed is None:
        raise ValueError(
            f"lhs spec '{lhs}' is malformed or references an unknown "
            f"field/axis: expected '{{field}}_{{axis...}}' format "
            f"(e.g. 'u_t' for du/dt, 'u_tt' for d²u/dt²). Available fields: "
            f"{list(fields.keys())}, axes: {list(coords.keys())}."
        )
    return parsed


def _check_strictly_increasing(tensor: torch.Tensor, axis_name: str) -> None:
    if tensor.numel() < 2:
        return
    diffs = torch.diff(tensor)
    if bool((diffs > 0).all()):
        return
    bad_idx = int((diffs <= 0).nonzero(as_tuple=False)[0].item())
    raise ValueError(
        f"coords['{axis_name}'] must be strictly increasing; "
        f"violation at index {bad_idx}: "
        f"values[{bad_idx}]={float(tensor[bad_idx]):.6g}, "
        f"values[{bad_idx + 1}]={float(tensor[bad_idx + 1]):.6g}. "
        f"If your data is reversed, apply np.flip / torch.flip "
        f"on both the coord and the corresponding field axis "
        f"before calling from_arrays."
    )


def build_axes_dict(
    coords: dict[str, torch.Tensor | np.ndarray | Sequence[float]],
    *,
    dtype: torch.dtype,
    periodic: Iterable[str] | None,
) -> dict[str, AxisInfo]:
    from kd.data.schema import AxisInfo

    periodic_set: set[str] = set(periodic) if periodic is not None else set()
    unknown_periodic = periodic_set - set(coords.keys())
    if unknown_periodic:
        raise ValueError(
            f"periodic references unknown axes: {sorted(unknown_periodic)}. "
            f"Available axes: {list(coords.keys())}."
        )

    axes_dict: dict[str, AxisInfo] = {}
    for axis_name, axis_values in coords.items():
        tensor = to_float_tensor(axis_values, dtype, kind="coord", label=axis_name)
        if tensor.dim() != 1:
            raise ValueError(
                f"coords['{axis_name}'] must be 1D, got {tensor.dim()}D shape "
                f"{tuple(tensor.shape)}."
            )
        _check_strictly_increasing(tensor, axis_name)
        axes_dict[axis_name] = AxisInfo(
            name=axis_name,
            values=tensor,
            is_periodic=axis_name in periodic_set,
        )
    return axes_dict


def build_fields_dict(
    fields: dict[str, torch.Tensor | np.ndarray],
    *,
    dtype: torch.dtype,
) -> dict[str, FieldData]:
    from kd.data.schema import FieldData

    fields_dict: dict[str, FieldData] = {}
    for field_name, field_values in fields.items():
        tensor = to_float_tensor(field_values, dtype, kind="field", label=field_name)
        fields_dict[field_name] = FieldData(name=field_name, values=tensor)
    return fields_dict


def annotate_shape_error(
    exc: ValueError,
    *,
    axes_dict: dict[str, AxisInfo],
    fields_dict: dict[str, FieldData],
    axis_order: list[str],
) -> ValueError:
    shape_summary = ", ".join(
        f"{n}=len({axes_dict[n].values.numel()})" for n in axis_order
    )
    field_summary = ", ".join(
        f"{n}.shape={tuple(f.values.shape)}" for n, f in fields_dict.items()
    )
    return ValueError(
        f"PDEDataset.from_arrays failed: {exc} "
        f"[coords: {shape_summary}; fields: {field_summary}; "
        f"axis_order={axis_order}]"
    )
