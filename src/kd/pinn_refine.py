
from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import torch

from kd.core.equation._sketch_terms import lower_term
from kd.core.expr.registry import FunctionRegistry
from kd.core.expr.terms import split_terms
from kd.core.platform.builder import resolve_lhs_defaults
from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset
from kd.search.discover.config import PINNConfig

_DISCOVER = PINNConfig()


@dataclass(frozen=True)
class PinnRefinement:

    terms: tuple[str, ...]
    initial_coefficients: tuple[float, ...]
    coefficients: tuple[float, ...]
    initial_data_loss: float
    initial_residual_loss: float
    data_loss: float
    residual_loss: float
    epochs_run: int
    pretrain_epochs: int
    pretrain_epochs_run: int
    pretrain_stopped_early: bool
    seconds: float
    stopped_at_deadline: bool
    device: str
    dataset: PDEDataset


def refine_pinn(
    dataset: PDEDataset,
    terms: Sequence[str],
    coefficients: Sequence[float],
    *,
    epochs: int,
    pretrain_epochs: int,
    train_coefficients: bool = True,
    device: str | torch.device | None = None,
    seed: int = 0,
    max_seconds: float | None = None,
    hidden_layers: int = _DISCOVER.number_layer,
    hidden_width: int = _DISCOVER.n_hidden,
    activation: Literal["tanh", "sin", "relu"] = _DISCOVER.activation,
    lr: float = _DISCOVER.lr,
    residual_weight: float = _DISCOVER.coef_pde,
    n_collocation: int = _DISCOVER.n_collocation,
    collocation_cut_ratio: float = _DISCOVER.collocation_cut_ratio,
) -> PinnRefinement:
    resolved = _checked_dataset(dataset)
    _check_budget(epochs, pretrain_epochs, max_seconds)
    irs, initial = _checked_law(terms, coefficients)
    run_device = _resolve_device(resolved, device)
    config = PINNConfig(
        number_layer=hidden_layers,
        n_hidden=hidden_width,
        activation=activation,
        pretrain_epoch=pretrain_epochs,
        pinn_epoch=epochs,
        lr=lr,
        coef_pde=residual_weight,
        n_collocation=n_collocation,
        collocation_cut_ratio=collocation_cut_ratio,
    )

    from kd.search.discover.pinn.refine import fit_jointly

    with _seeded(run_device, seed):
        fit = fit_jointly(
            resolved,
            irs,
            initial,
            config,
            trainable=train_coefficients,
            device=run_device,
            seed=seed,
            max_seconds=max_seconds,
        )
    return PinnRefinement(
        terms=tuple(terms),
        initial_coefficients=initial,
        coefficients=fit.coefficients,
        initial_data_loss=fit.initial_data_loss,
        initial_residual_loss=fit.initial_residual_loss,
        data_loss=fit.data_loss,
        residual_loss=fit.residual_loss,
        epochs_run=fit.epochs_run,
        pretrain_epochs=pretrain_epochs,
        pretrain_epochs_run=fit.pretrain_epochs_run,
        pretrain_stopped_early=fit.pretrain_stopped_early,
        seconds=fit.seconds,
        stopped_at_deadline=fit.stopped_at_deadline,
        device=str(run_device),
        dataset=_predicted_dataset(dataset, fit.fields),
    )


@contextmanager
def _seeded(device: torch.device, seed: int) -> Iterator[None]:
    if device.type != "cuda":
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            yield
        return
    index = torch.cuda.current_device() if device.index is None else device.index
    with torch.random.fork_rng(devices=[index]):
        torch.default_generator.manual_seed(seed)
        torch.cuda.default_generators[index].manual_seed(seed)
        yield


def _checked_dataset(dataset: PDEDataset) -> PDEDataset:
    if dataset.topology is not DataTopology.GRID or dataset.fields is None:
        raise ValueError(
            f"refine_pinn needs a GRID dataset with fields, got "
            f"{dataset.topology.name} ({dataset.name!r})"
        )
    for name, field in dataset.fields.items():
        if torch.isnan(field.values).any():
            raise ValueError(
                f"refine_pinn: field {name!r} holds NaN; every grid point is an "
                "observation of the network"
            )
    if dataset.lhs_order != 1:
        raise ValueError(
            "refine_pinn: the residual's left-hand side is a first derivative; "
            f"the dataset has lhs_order={dataset.lhs_order}"
        )
    return resolve_lhs_defaults(dataset)


def _check_budget(epochs: int, pretrain_epochs: int, max_seconds: float | None) -> None:
    if epochs < 0 or pretrain_epochs < 0:
        raise ValueError(
            "refine_pinn: epochs and pretrain_epochs must be >= 0, got "
            f"{epochs} and {pretrain_epochs}"
        )
    if max_seconds is not None and not max_seconds >= 0:
        raise ValueError(f"refine_pinn: max_seconds must be >= 0, got {max_seconds}")


def _checked_law(
    terms: Sequence[str], coefficients: Sequence[float]
) -> tuple[list[str], tuple[float, ...]]:
    term_list = list(terms)
    values = tuple(float(c) for c in coefficients)
    if not term_list:
        raise ValueError("refine_pinn: terms must not be empty")
    if len(values) != len(term_list):
        raise ValueError(
            f"refine_pinn: {len(term_list)} terms but {len(values)} coefficients"
        )
    if not all(torch.isfinite(torch.tensor(values)).tolist()):
        raise ValueError(f"refine_pinn: coefficients must be finite, got {values}")
    registry = FunctionRegistry.create_default()
    irs = []
    for term in term_list:
        try:
            ir = lower_term(term)
            parts = split_terms(ir, registry)
        except ValueError as exc:
            raise ValueError(
                f"refine_pinn: term {term!r} is not canonical funcall IR or "
                f"product notation: {exc}"
            ) from exc
        if len(parts) != 1:
            raise ValueError(
                f"refine_pinn: term {term!r} is a sum of {len(parts)} terms "
                f"{parts}; pass them as separate terms"
            )
        irs.append(ir)
    return irs, values


def _resolve_device(
    dataset: PDEDataset, device: str | torch.device | None
) -> torch.device:
    if device is not None:
        return torch.device(device)
    assert dataset.fields is not None
    return next(iter(dataset.fields.values())).values.device


def _predicted_dataset(
    dataset: PDEDataset, predicted: dict[str, torch.Tensor]
) -> PDEDataset:
    assert dataset.axes is not None and dataset.axis_order is not None
    assert dataset.fields is not None
    fields = {
        name: FieldData(
            field.name,
            predicted[name].to(dtype=field.values.dtype, device=field.values.device),
            allow_nan=field.allow_nan,
        )
        for name, field in dataset.fields.items()
    }
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
        fields=fields,
        lhs_field=dataset.lhs_field,
        lhs_axis=dataset.lhs_axis,
        lhs_order=dataset.lhs_order,
        noise_level=dataset.noise_level,
        ground_truth=dataset.ground_truth,
        source=dataset.source,
    )


__all__ = ["PinnRefinement", "refine_pinn"]
