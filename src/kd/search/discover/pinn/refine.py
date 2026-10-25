
from __future__ import annotations

import time
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
from torch import Tensor

from kd.core.expr.registry import FunctionRegistry
from kd.data.schema import PDEDataset
from kd.search.discover.config import PINNConfig
from kd.search.discover.pinn.collocation import generate_collocation_points
from kd.search.discover.pinn.cycle import pretrain_split
from kd.search.discover.pinn.executor import PINNExecutor, make_pinn_dataset_from
from kd.search.discover.pinn.model import PINNModel, _chunked_eval_residual_loss
from kd.search.discover.pinn.observations import domain_bounds, sample_observations

_PROBE_POINTS = 16


@dataclass(frozen=True)
class JointFit:

    coefficients: tuple[float, ...]
    initial_data_loss: float
    initial_residual_loss: float
    data_loss: float
    residual_loss: float
    epochs_run: int
    stopped_at_deadline: bool
    pretrain_epochs_run: int
    pretrain_stopped_early: bool
    seconds: float
    fields: dict[str, Tensor]


class _RefinementNetwork(PINNModel):

    def __init__(
        self,
        coord_names: list[str],
        field_names: list[str],
        config: PINNConfig,
        factors: Tensor,
        *,
        trainable: bool,
        device: torch.device,
    ) -> None:
        super().__init__(coord_names, field_names, config, device=device)
        self.coefficient_factors = nn.Parameter(
            factors.to(dtype=torch.float32, device=device), requires_grad=trainable
        )


class _ScaledExecutor(PINNExecutor):

    def __init__(self, registry: FunctionRegistry, scale: Tensor) -> None:
        super().__init__(registry)
        self.scale = scale

    def compute_residual(
        self,
        model: nn.Module,
        terms: list[str],
        coefficients: list[float] | Tensor,
        coords: dict[str, Tensor],
        dataset_metadata: PDEDataset,
        *,
        lhs_field: str,
        lhs_axis: str,
    ) -> Tensor:
        factors = torch.as_tensor(
            coefficients, dtype=self.scale.dtype, device=self.scale.device
        )
        return super().compute_residual(
            model,
            terms,
            self.scale * factors,
            coords,
            dataset_metadata,
            lhs_field=lhs_field,
            lhs_axis=lhs_axis,
        )


def coefficient_scale(initial: tuple[float, ...]) -> list[float]:
    largest = max(abs(c) for c in initial)
    fallback = largest if largest > 0 else 1.0
    return [abs(c) if c != 0 else fallback for c in initial]


def fit_jointly(
    dataset: PDEDataset,
    terms: list[str],
    initial: tuple[float, ...],
    config: PINNConfig,
    *,
    trainable: bool,
    device: torch.device,
    seed: int,
    max_seconds: float | None,
) -> JointFit:
    obs_coords, obs_targets = sample_observations(
        dataset, obs_ratio=1.0, seed=seed, device=device
    )
    scale = torch.tensor(coefficient_scale(initial), device=device)
    model = _RefinementNetwork(
        list(obs_coords),
        list(obs_targets),
        config,
        torch.tensor(initial) / scale.cpu(),
        trainable=trainable,
        device=device,
    )
    colloc = generate_collocation_points(
        domain_bounds(dataset),
        config.n_collocation,
        config.collocation_cut_ratio,
        seed=seed,
        device=device,
    )
    metadata = make_pinn_dataset_from(dataset)
    registry = FunctionRegistry.create_default()
    executor = _ScaledExecutor(registry, scale)
    started = time.perf_counter()
    pretrained = pretrain_split(model, obs_coords, obs_targets, config, split_seed=seed)
    _probe_terms(model, PINNExecutor(registry), terms, colloc, metadata)

    def train(epochs: int, deadline: float | None) -> tuple[float, float, int, bool]:
        result = model.train_pinn(
            terms=terms,
            coefficients=model.coefficient_factors,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc,
            dataset_metadata=metadata,


            config=replace(config, pinn_epoch=epochs, early_stop_patience=epochs + 1),
            deadline=deadline,
        )
        return (
            result.data_loss,
            result.physics_loss,
            result.epochs_run,
            result.stopped_early,
        )

    def residual_mse() -> float:
        return _residual_mse(
            model, executor, terms, colloc, metadata, config.colloc_chunk_size
        )

    initial_data, _, _, _ = train(0, None)
    initial_residual = residual_mse()
    deadline = None if max_seconds is None else started + max_seconds
    data_loss, _, epochs_run, stopped = train(config.pinn_epoch, deadline)
    seconds = time.perf_counter() - started

    residual_loss = residual_mse()
    refined = (
        initial
        if not trainable
        else tuple((scale * model.coefficient_factors).detach().cpu().tolist())
    )
    return JointFit(
        coefficients=refined,
        initial_data_loss=initial_data,
        initial_residual_loss=initial_residual,
        data_loss=data_loss,
        residual_loss=residual_loss,
        epochs_run=epochs_run,
        stopped_at_deadline=stopped,
        pretrain_epochs_run=pretrained.epochs_run,
        pretrain_stopped_early=pretrained.stopped_early,
        seconds=seconds,
        fields=_grid_prediction(model, dataset),
    )


def _residual_mse(
    model: _RefinementNetwork,
    executor: _ScaledExecutor,
    terms: list[str],
    colloc: dict[str, Tensor],
    metadata: PDEDataset,
    chunk_size: int | None,
) -> float:
    coords = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in colloc.items()
    }
    return _chunked_eval_residual_loss(
        model,
        executor,
        terms,
        model.coefficient_factors,
        coords,
        metadata,
        weight=1.0,
        chunk_size=chunk_size,
    )


def _probe_terms(
    model: PINNModel,
    executor: PINNExecutor,
    terms: list[str],
    colloc: dict[str, Tensor],
    metadata: PDEDataset,
) -> None:
    for term in terms:
        coords = {
            name: value[:_PROBE_POINTS].detach().clone().requires_grad_(True)
            for name, value in colloc.items()
        }
        try:
            executor.compute_residual(
                model=model,
                terms=[term],
                coefficients=[1.0],
                coords=coords,
                dataset_metadata=metadata,
                lhs_field=metadata.lhs_field,
                lhs_axis=metadata.lhs_axis,
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"refine_pinn: term {term!r} cannot be evaluated on the network: {exc}"
            ) from exc


@torch.no_grad()
def _grid_prediction(model: PINNModel, dataset: PDEDataset) -> dict[str, Tensor]:
    assert dataset.axes is not None and dataset.axis_order is not None
    assert dataset.fields is not None
    model.eval()
    device = next(model.parameters()).device
    grids = torch.meshgrid(
        *(dataset.axes[name].values for name in dataset.axis_order), indexing="ij"
    )
    coords = {
        name: grid.flatten().to(device=device, dtype=torch.float32)
        for name, grid in zip(dataset.axis_order, grids, strict=True)
    }
    predicted = model(**coords)
    return {
        name: predicted[name].reshape(field.values.shape)
        for name, field in dataset.fields.items()
    }
