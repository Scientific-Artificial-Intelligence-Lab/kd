
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from kd.data._factory import validate_scatter_point_shapes
from kd.data.layouts.kd_npz import regularize_axis
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    compute_dataset_fingerprint,
)
from kd.data.source import DatasetSource
from kd.models import FieldModel, FieldModelTrainer

__all__ = ["resample_to_grid"]


def resample_to_grid(
    dataset: PDEDataset,
    *,
    axes: dict[str, torch.Tensor | np.ndarray | Sequence[float]],
    hidden_sizes: Sequence[int] = (64, 64),
    activation: str = "tanh",
    max_epochs: int = 10000,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int | None = 100,
    val_ratio: float = 0.2,
    seed: int = 0,
    restore_best: bool = False,
    device: str | torch.device | None = None,
) -> PDEDataset:
    coords, targets, dtype = _scatter_payload(dataset)
    output_axes = _target_axes(dataset, axes, dtype)
    if max_epochs < 1:
        raise ValueError("max_epochs must be >= 1 to train the resampling model")
    model_options: dict[str, Any] = {
        "hidden_sizes": list(hidden_sizes),
        "activation": activation,
    }
    training: dict[str, Any] = {
        "max_epochs": max_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "val_ratio": val_ratio,
        "seed": seed,
        "restore_best": restore_best,
        "device": str(device) if device is not None else "cpu",
    }
    model, epochs_run = _fit_surrogate(coords, targets, model_options, training)
    return replace(
        dataset,
        name=f"{dataset.name}_resampled",
        topology=DataTopology.GRID,
        axes=output_axes,
        axis_order=list(output_axes),
        fields=_sample_fields(model, output_axes),
        source=_resampled_source(
            dataset, output_axes, model_options, training, epochs_run
        ),
    )


def _scatter_payload(
    dataset: PDEDataset,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.dtype]:
    if dataset.topology is not DataTopology.SCATTERED:
        raise ValueError("resample_to_grid requires a SCATTERED dataset")
    if not dataset.axes or not dataset.axis_order or not dataset.fields:
        raise ValueError("resample_to_grid requires scattered coordinate/field data")
    validate_scatter_point_shapes(dataset.axes, dataset.axis_order, dataset.fields)
    if dataset.get_shape()[0] < 2:
        raise ValueError("resample_to_grid needs at least two observed points")
    dtype = torch.float32
    items: list[tuple[str, AxisInfo | FieldData]] = [
        *dataset.axes.items(),
        *dataset.fields.items(),
    ]
    for name, item in items:
        if not torch.isfinite(item.values).all():
            raise ValueError(
                f"resample_to_grid: {name!r} must contain only finite values"
            )
        dtype = torch.promote_types(dtype, item.values.dtype)
    coords = {
        axis: dataset.get_coords(axis).detach().to(device="cpu", dtype=dtype)
        for axis in dataset.axis_order
    }
    targets = {
        name: field.values.detach().to(device="cpu", dtype=dtype)
        for name, field in dataset.fields.items()
    }
    return coords, targets, dtype


def _target_axes(
    dataset: PDEDataset,
    axes: dict[str, torch.Tensor | np.ndarray | Sequence[float]],
    dtype: torch.dtype,
) -> dict[str, AxisInfo]:
    assert dataset.axes is not None
    if set(axes) != set(dataset.axes):
        raise ValueError(
            f"axes must name exactly the input coordinates: {list(dataset.axes)}"
        )
    result: dict[str, AxisInfo] = {}
    for name, raw in axes.items():
        axis = AxisInfo(
            name=name,
            values=torch.as_tensor(raw, dtype=dtype),
            is_periodic=dataset.axes[name].is_periodic,
        )
        values = regularize_axis(name, axis.values.detach().cpu().numpy())
        axis.values = torch.as_tensor(values, dtype=dtype, device="cpu").clone()
        result[name] = axis
    return result


def _fit_surrogate(
    coords: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    model_options: dict[str, Any],
    training: dict[str, Any],
) -> tuple[FieldModel, int]:


    with torch.random.fork_rng(devices=[]):
        model = FieldModel(list(coords), list(targets), **model_options)
        trainer = FieldModelTrainer(
            model,
            lr=training["lr"],
            weight_decay=training["weight_decay"],
        )
        outcome = trainer.fit(
            coords,
            targets,
            **{
                key: value
                for key, value in training.items()
                if key not in {"lr", "weight_decay"}
            },
        )
    return model, outcome.epochs_run


@torch.no_grad()
def _sample_fields(
    model: FieldModel, axes: dict[str, AxisInfo]
) -> dict[str, FieldData]:
    mesh = torch.meshgrid(*(axis.values for axis in axes.values()), indexing="ij")
    coords = {name: values.reshape(-1) for name, values in zip(axes, mesh, strict=True)}
    shape = tuple(axis.values.numel() for axis in axes.values())
    return {
        name: FieldData(name=name, values=values.reshape(shape).detach())
        for name, values in model(**coords).items()
    }


def _resampled_source(
    dataset: PDEDataset,
    axes: dict[str, AxisInfo],
    model_options: dict[str, Any],
    training: dict[str, Any],
    epochs_run: int,
) -> DatasetSource:
    return DatasetSource(
        path="",
        sha256="",
        container="derived",
        layout="resample_to_grid",
        select=None,
        hints={
            "origin": {
                "name": dataset.name,
                "fingerprint": compute_dataset_fingerprint(dataset),
                "source": dataset.source.to_dict()
                if dataset.source is not None
                else None,
            },
            "axes": {name: axis.values.tolist() for name, axis in axes.items()},
            "model": model_options,
            "training": training,
            "epochs_run": epochs_run,
        },
    )
