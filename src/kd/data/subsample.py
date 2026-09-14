
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np
import torch

from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset

__all__ = ["ratio_subsample", "stride_subsample"]


def _require_grid(dataset: PDEDataset, what: str) -> None:
    if dataset.topology is not DataTopology.GRID:
        raise ValueError(
            f"{what} needs a GRID dataset, got {dataset.topology.name} "
            f"({dataset.name!r})"
        )
    if dataset.axes is None or dataset.axis_order is None or dataset.fields is None:
        raise ValueError(f"{what}: dataset {dataset.name!r} carries no grid payload")


def _strides(dataset: PDEDataset, stride: int | Mapping[str, int]) -> dict[str, int]:
    assert dataset.axis_order is not None
    if isinstance(stride, Mapping):
        unknown = set(stride) - set(dataset.axis_order)
        if unknown:
            raise ValueError(
                f"stride names axes {sorted(unknown)} that {dataset.name!r} "
                f"does not have (axes: {dataset.axis_order})"
            )
        raw = {axis: stride.get(axis, 1) for axis in dataset.axis_order}
    else:
        raw = dict.fromkeys(dataset.axis_order, stride)
    per_axis: dict[str, int] = {}
    for axis, step in raw.items():
        if isinstance(step, bool | float) or not isinstance(step, Integral):
            raise ValueError(
                f"stride for axis {axis!r} must be an integer >= 1, got {step!r}"
            )
        if int(step) < 1:
            raise ValueError(
                f"stride for axis {axis!r} must be an integer >= 1, got {step!r}"
            )
        per_axis[axis] = int(step)
    return per_axis


def stride_subsample(
    dataset: PDEDataset, stride: int | Mapping[str, int]
) -> PDEDataset:
    _require_grid(dataset, "stride_subsample")
    assert dataset.axes is not None and dataset.axis_order is not None
    assert dataset.fields is not None
    per_axis = _strides(dataset, stride)
    axes: dict[str, AxisInfo] = {}
    for axis in dataset.axis_order:
        info = dataset.axes[axis]
        step = per_axis[axis]
        n = info.values.numel()
        if info.is_periodic and n % step != 0:
            raise ValueError(
                f"axis {axis!r} is periodic with {n} points; a stride of {step} "
                "does not divide it, so the wrap-around gap would differ from "
                "the new spacing. Pick a stride that divides the length"
            )
        kept = (n + step - 1) // step
        if kept < 2 and n > 1:
            raise ValueError(
                f"axis {axis!r} has {n} points; a stride of {step} keeps {kept}, "
                "which is not a grid"
            )
        axes[axis] = AxisInfo(
            name=info.name,
            values=info.values[::step].clone(),
            is_periodic=info.is_periodic,
            allow_nan=info.allow_nan,
        )
    slices = tuple(slice(None, None, per_axis[axis]) for axis in dataset.axis_order)
    fields = {
        name: FieldData(
            name=field.name,
            values=field.values[slices].clone(),
            allow_nan=field.allow_nan,
        )
        for name, field in dataset.fields.items()
    }
    return PDEDataset(
        name=dataset.name,
        task_type=dataset.task_type,
        topology=DataTopology.GRID,
        axes=axes,
        axis_order=list(dataset.axis_order),
        fields=fields,
        lhs_field=dataset.lhs_field,
        lhs_axis=dataset.lhs_axis,
        lhs_order=dataset.lhs_order,
        noise_level=dataset.noise_level,
        ground_truth=dataset.ground_truth,
        source=dataset.source,
    )


def _lhs_spec(dataset: PDEDataset) -> str:
    if dataset.lhs_order == 0:
        return ""
    return f"{dataset.lhs_field}_{dataset.lhs_axis * dataset.lhs_order}"


def ratio_subsample(
    dataset: PDEDataset,
    ratio: float,
    *,
    seed: int,
    boundary_trim: float = 0.1,
) -> PDEDataset:
    _require_grid(dataset, "ratio_subsample")
    assert dataset.axes is not None and dataset.axis_order is not None
    assert dataset.fields is not None
    if not isinstance(ratio, int | float) or isinstance(ratio, bool):
        raise ValueError(f"ratio must be a number in (0, 1], got {ratio!r}")
    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio!r}")
    if not 0.0 <= float(boundary_trim) < 0.5:
        raise ValueError(f"boundary_trim must be in [0, 0.5), got {boundary_trim!r}")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    seed = int(seed)

    interior: list[torch.Tensor] = []
    for axis in dataset.axis_order:
        n = dataset.axes[axis].values.numel()
        cut = int(n * float(boundary_trim))
        lo, hi = cut, n - cut
        if hi - lo < 1:
            raise ValueError(
                f"boundary_trim {boundary_trim} leaves no interior on axis "
                f"{axis!r} ({n} points)"
            )
        interior.append(torch.arange(lo, hi))
    mesh = torch.meshgrid(*interior, indexing="ij")
    index_columns = [m.reshape(-1) for m in mesh]
    n_interior = index_columns[0].numel()
    n_keep = max(1, int(round(float(ratio) * n_interior)))
    generator = torch.Generator().manual_seed(seed)
    chosen = torch.randperm(n_interior, generator=generator)[:n_keep]
    picked = [column[chosen] for column in index_columns]

    coords: dict[str, torch.Tensor | np.ndarray[Any, Any] | Sequence[float]] = {
        axis: dataset.axes[axis].values[picked[i]].clone()
        for i, axis in enumerate(dataset.axis_order)
    }
    fields: dict[str, torch.Tensor | np.ndarray[Any, Any]] = {
        name: field.values[tuple(picked)].clone()
        for name, field in dataset.fields.items()
    }
    lhs = _lhs_spec(dataset)
    if lhs and (dataset.lhs_field not in fields or dataset.lhs_axis not in coords):
        raise ValueError(
            f"dataset {dataset.name!r} declares LHS {lhs!r} over a field or axis "
            "it does not carry"
        )


    dtype = torch.float32
    for tensor in list(dataset.fields.values()) + list(dataset.axes.values()):
        dtype = torch.promote_types(dtype, tensor.values.dtype)
    allow_nan = any(f.allow_nan for f in dataset.fields.values()) or any(
        a.allow_nan for a in dataset.axes.values()
    )
    sample = PDEDataset.from_scatter(
        coords,
        fields,
        lhs=lhs,
        name=dataset.name,
        ground_truth=dataset.ground_truth,
        dtype=dtype,
        allow_nan=allow_nan,
    )

    sample.task_type = dataset.task_type
    sample.noise_level = dataset.noise_level
    sample.source = dataset.source
    return sample
