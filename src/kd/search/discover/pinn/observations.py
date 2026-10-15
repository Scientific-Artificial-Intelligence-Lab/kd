
from __future__ import annotations

import torch
from torch import Tensor

from kd.data.schema import AxisInfo, FieldData, PDEDataset

MIN_OBSERVATIONS = 10


def _grid_layout(
    dataset: PDEDataset,
) -> tuple[list[str], dict[str, AxisInfo], dict[str, FieldData]]:
    if dataset.axis_order is None or dataset.axes is None or dataset.fields is None:
        raise ValueError(
            f"dataset {dataset.name!r} carries no grid values: axis_order, axes "
            "and fields must all be set to sample observations or read bounds."
        )
    return dataset.axis_order, dataset.axes, dataset.fields


def sample_observations(
    dataset: PDEDataset,
    *,
    obs_ratio: float,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    axis_order, axes, fields = _grid_layout(dataset)
    grids = torch.meshgrid(*(axes[name].values for name in axis_order), indexing="ij")
    n_total = grids[0].numel()
    n_obs = max(MIN_OBSERVATIONS, int(n_total * obs_ratio))
    generator = torch.Generator().manual_seed(seed)
    index = torch.randperm(n_total, generator=generator)[:n_obs]
    coords = {
        name: grid.flatten()[index].float().to(device=device)
        for name, grid in zip(axis_order, grids, strict=True)
    }
    targets = {
        name: field.values.flatten()[index].float().to(device=device)
        for name, field in fields.items()
    }
    return coords, targets


def domain_bounds(dataset: PDEDataset) -> dict[str, tuple[float, float]]:
    axis_order, axes, _ = _grid_layout(dataset)
    return {
        name: (float(axes[name].values.min()), float(axes[name].values.max()))
        for name in axis_order
    }


__all__ = ["domain_bounds", "sample_observations"]
