
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
import torch
from torch import Tensor

from kd.data.diagnostics import MIN_STENCIL_POINTS, noise_estimate
from kd.data.schema import (
    AxisInfo,
    FieldData,
    PDEDataset,
    compute_dataset_fingerprint,
)
from kd.data.subsample import _require_grid
from kd.models.field_model import FieldModel
from kd.models.field_recipe import FieldRecipe
from kd.models.trainer import FieldModelTrainer, TrainingResult

__all__ = [
    "IMPLEMENTATION",
    "NetworkSmoothing",
    "NetworkSmoothingQuality",
    "network_smooth",
    "surrogate_field",
]

logger = logging.getLogger(__name__)

IMPLEMENTATION: Final[str] = "kd-network-smoothing-v1"


@dataclass(frozen=True)
class NetworkSmoothingQuality:

    requested_epochs: int
    executed_epochs: int
    stopping_reason: Literal["max_epochs", "early_stopped"]
    final_train_loss: float
    best_val_loss: float | None
    loss_scale: float
    sigma_hat: dict[str, float | None]
    residual_ratio: dict[str, float | None]
    device: str
    cuda_max_memory_allocated_bytes: int | None
    unavailable: dict[str, str]


@dataclass(frozen=True)
class NetworkSmoothing:

    dataset: PDEDataset
    model: FieldModel
    training: TrainingResult
    recipe: FieldRecipe
    identity: dict[str, Any]
    quality: NetworkSmoothingQuality


def surrogate_field(
    dataset: PDEDataset, model: FieldModel, *, batch_size: int = 8192
) -> PDEDataset:
    if batch_size < 1:
        raise ValueError(
            f"surrogate_field: batch_size must be a positive integer, got "
            f"{batch_size!r}"
        )
    field = _single_field(dataset, what="surrogate_field")
    _require_model_names(dataset, model, field)
    predicted = _predict(dataset, model, field, batch_size)
    return _sampled_dataset(dataset, field, predicted)


def network_smooth(
    dataset: PDEDataset,
    *,
    recipe: FieldRecipe,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> NetworkSmoothing:
    field = _single_field(dataset, what="network_smooth")
    coords, target = _training_points(dataset, field)
    train_on = target.device if device is None else device
    model, training = _fit(coords, {field: target}, recipe, seed, train_on)
    used = _device_used(torch.device(train_on))
    peak = int(torch.cuda.max_memory_allocated(used)) if used.type == "cuda" else None
    predicted = surrogate_field(dataset, model)
    logger.info(
        "network_smooth: %s trained %d of %d epochs on %s in %.1f s",
        recipe.name,
        training.epochs_run,
        recipe.max_epochs,
        used,
        training.elapsed_seconds,
    )
    return NetworkSmoothing(
        dataset=predicted,
        model=model,
        training=training,
        recipe=recipe,
        identity=_identity(dataset, field, recipe, seed, model),
        quality=_quality(dataset, predicted, field, recipe, training, used, peak),
    )


def _fit(
    coords: dict[str, Tensor],
    targets: dict[str, Tensor],
    recipe: FieldRecipe,
    seed: int,
    device: str | torch.device,
) -> tuple[FieldModel, TrainingResult]:
    with torch.random.fork_rng(devices=[]):
        model = FieldModel(
            coord_names=list(coords),
            field_names=list(targets),
            hidden_sizes=list(recipe.hidden_sizes),
            activation=recipe.activation,
        )
    training = FieldModelTrainer(model, lr=recipe.lr).fit(
        coords,
        targets,
        max_epochs=recipe.max_epochs,
        patience=recipe.patience,
        val_ratio=recipe.val_ratio,
        seed=seed,
        restore_best=recipe.restore_best,
        device=device,
        training_dtype=recipe.training_dtype,
    )
    return model, training


def _single_field(dataset: PDEDataset, *, what: str) -> str:
    _require_grid(dataset, what=what)
    assert dataset.fields is not None
    names = list(dataset.fields)
    if len(names) != 1:
        raise ValueError(
            f"{what} takes a single-field dataset; {dataset.name!r} has fields {names}"
        )
    return names[0]


def _require_model_names(dataset: PDEDataset, model: FieldModel, field: str) -> None:
    assert dataset.axis_order is not None
    if sorted(model.coord_names) != sorted(dataset.axis_order):
        raise ValueError(
            f"surrogate_field: the model's coordinates {model.coord_names} are "
            f"not the dataset's axes {dataset.axis_order}"
        )
    if model.field_names != [field]:
        raise ValueError(
            f"surrogate_field: the model's fields {model.field_names} are not "
            f"the dataset's field [{field!r}]"
        )


def _predict(
    dataset: PDEDataset, model: FieldModel, field: str, batch_size: int
) -> Tensor:
    assert dataset.axes is not None and dataset.axis_order is not None
    assert dataset.fields is not None
    weight = next(model.parameters())
    names = dataset.axis_order
    axes = [
        dataset.axes[name].values.to(device=weight.device, dtype=weight.dtype)
        for name in names
    ]
    shape = tuple(int(axis.numel()) for axis in axes)
    source = dataset.fields[field].values
    out = torch.empty(math.prod(shape), dtype=source.dtype, device=source.device)
    modes = [module.training for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, out.numel(), batch_size):
                stop = min(start + batch_size, out.numel())
                index = torch.arange(start, stop, device=weight.device)
                position = torch.unravel_index(index, shape)
                batch = {
                    name: axis[at]
                    for name, axis, at in zip(names, axes, position, strict=True)
                }
                out[start:stop] = model(**batch)[field].to(out)
    finally:
        for module, mode in zip(model.modules(), modes, strict=True):
            module.training = mode
    bad = int((~torch.isfinite(out)).sum())
    if bad:
        raise ValueError(
            f"surrogate_field: the model predicted {bad} non-finite values of "
            f"{field!r} on the grid"
        )
    return out.reshape(shape)


def _sampled_dataset(dataset: PDEDataset, field: str, values: Tensor) -> PDEDataset:
    assert dataset.axes is not None and dataset.axis_order is not None
    assert dataset.fields is not None
    source = dataset.fields[field]
    axes = {
        name: AxisInfo(info.name, info.values.clone(), False, info.allow_nan)
        for name, info in dataset.axes.items()
    }
    return PDEDataset(
        name=dataset.name,
        task_type=dataset.task_type,
        topology=dataset.topology,
        axes=axes,
        axis_order=list(dataset.axis_order),
        fields={field: FieldData(source.name, values, allow_nan=source.allow_nan)},
        lhs_field=dataset.lhs_field,
        lhs_axis=dataset.lhs_axis,
        lhs_order=dataset.lhs_order,
        noise_level=dataset.noise_level,
        ground_truth=None,
        source=dataset.source,
    )


def _training_points(
    dataset: PDEDataset, field: str
) -> tuple[dict[str, Tensor], Tensor]:
    assert dataset.axes is not None and dataset.axis_order is not None
    assert dataset.fields is not None
    values = dataset.fields[field].values.detach()
    if torch.isnan(values).any():
        raise ValueError(
            f"network_smooth: field {field!r} holds NaN; every epoch's loss "
            "would be NaN"
        )
    grids = torch.meshgrid(
        *(
            dataset.axes[name].values.to(dtype=values.dtype, device=values.device)
            for name in dataset.axis_order
        ),
        indexing="ij",
    )
    coords = {
        name: grid.reshape(-1)
        for name, grid in zip(dataset.axis_order, grids, strict=True)
    }
    return coords, values.reshape(-1)


def _device_used(device: torch.device) -> torch.device:
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _identity(
    dataset: PDEDataset,
    field: str,
    recipe: FieldRecipe,
    seed: int,
    model: FieldModel,
) -> dict[str, Any]:
    assert dataset.axes is not None and dataset.axis_order is not None
    values = dataset.get_field(field).detach().cpu().contiguous()
    return {
        "implementation": IMPLEMENTATION,
        "dataset": compute_dataset_fingerprint(dataset),
        "field_sha256": {field: hashlib.sha256(values.numpy().tobytes()).hexdigest()},
        "axis_order": list(dataset.axis_order),
        "field_order": [field],
        "dtype": str(values.dtype).removeprefix("torch."),
        "source_periodic": {
            name: info.is_periodic for name, info in dataset.axes.items()
        },
        "recipe": recipe.to_dict(),
        "seed": int(seed),
        "model_sha256": _model_sha256(model),
    }


def _model_sha256(model: FieldModel) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        data = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{data.dtype}:{tuple(data.shape)};".encode())
        digest.update(data.numpy().tobytes())
    return digest.hexdigest()


def _quality(
    dataset: PDEDataset,
    predicted: PDEDataset,
    field: str,
    recipe: FieldRecipe,
    training: TrainingResult,
    used: torch.device,
    peak: int | None,
) -> NetworkSmoothingQuality:
    observed = _as_float64(dataset.get_field(field))
    fitted = _as_float64(predicted.get_field(field))
    unavailable: dict[str, str] = {}
    sigma = _sigma_hat(observed)
    ratio: float | None = None
    if sigma is None:
        unavailable[f"sigma_hat.{field}"] = (
            f"no axis has at least {MIN_STENCIL_POINTS} points"
        )
        unavailable[f"residual_ratio.{field}"] = "sigma_hat is unavailable"
    elif sigma == 0.0:
        unavailable[f"residual_ratio.{field}"] = (
            "sigma_hat is zero: the second differences have zero MAD"
        )
    else:
        ratio = float(np.sqrt(np.mean((observed - fitted) ** 2))) / sigma
    if training.best_val_loss is None:
        unavailable["best_val_loss"] = "the fit had no validation split"
    if peak is None:
        unavailable["cuda_max_memory_allocated_bytes"] = (
            f"the fit ran on {used}, not on a CUDA device"
        )
    return NetworkSmoothingQuality(
        requested_epochs=recipe.max_epochs,
        executed_epochs=training.epochs_run,
        stopping_reason="early_stopped" if training.early_stopped else "max_epochs",
        final_train_loss=training.final_loss,
        best_val_loss=training.best_val_loss,
        loss_scale=float(np.var(observed)),
        sigma_hat={field: sigma},
        residual_ratio={field: ratio},
        device=str(used),
        cuda_max_memory_allocated_bytes=peak,
        unavailable=unavailable,
    )


def _as_float64(values: Tensor) -> np.ndarray:
    return values.detach().to(device="cpu", dtype=torch.float64).numpy()


def _sigma_hat(observed: np.ndarray) -> float | None:
    axes = [
        axis
        for axis in range(observed.ndim)
        if observed.shape[axis] >= MIN_STENCIL_POINTS
    ]
    if not axes:
        return None
    std = float(np.std(observed))
    return min(noise_estimate(observed, axis=axis) * std for axis in axes)
