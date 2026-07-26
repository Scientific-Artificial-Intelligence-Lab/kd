
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

import torch

from kd.core.evaluator import Evaluator
from kd.core.executor.context import ExecutionContext
from kd.core.executor.surrogate_context import SurrogateContext
from kd.core.expr import FunctionRegistry, PythonExecutor
from kd.core.linear_solve.least_squares import LeastSquaresSolver
from kd.core.platform.requirements import DerivativeReqs
from kd.data.derivatives.autograd import AutogradProvider
from kd.data.derivatives.finite_diff import FiniteDiffProvider
from kd.models.field_model import FieldModel
from kd.models.trainer import FieldModelTrainer
from kd.search.protocol import PlatformComponents

if TYPE_CHECKING:
    from kd.data.derivatives.base import DerivativeProvider
    from kd.data.schema import PDEDataset
    from kd.models.trainer import TrainingResult

logger = logging.getLogger(__name__)


_DEFAULT_LHS_FIELD = "u"
_DEFAULT_LHS_AXIS = "t"



_DEFAULT_SURROGATE_HIDDEN_SIZES: tuple[int, ...] = (64, 64, 64, 64, 64)
_DEFAULT_SURROGATE_ACTIVATION = "tanh"


__all__ = [
    "PlatformBuilder",
    "_resolve_derivative_requirements",
    "resolve_lhs_defaults",
]


def _resolve_device(device: str | None) -> torch.device | None:
    if device is None:
        return None
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _resolve_derivative_requirements(plugin: Any) -> DerivativeReqs:
    raw = getattr(plugin, "derivative_requirements", None)
    if raw is None:
        return DerivativeReqs()
    if callable(raw):
        raise TypeError(
            f"{type(plugin).__name__}.derivative_requirements must be a "
            f"@property returning DerivativeReqs, got a callable/method. "
            f"Decorate the definition with @property."
        )
    if not isinstance(raw, DerivativeReqs):
        raise TypeError(
            f"{type(plugin).__name__}.derivative_requirements must return "
            f"a DerivativeReqs instance, got {type(raw).__name__}."
        )
    return raw


def resolve_lhs_defaults(dataset: PDEDataset) -> PDEDataset:
    if dataset.lhs_order == 0:
        return dataset

    lhs_field = dataset.lhs_field or _DEFAULT_LHS_FIELD
    lhs_axis = dataset.lhs_axis or _DEFAULT_LHS_AXIS

    field_names = set(dataset.fields.keys()) if dataset.fields else set()
    axis_names = set(dataset.axes.keys()) if dataset.axes else set()

    if lhs_field not in field_names:
        raise ValueError(
            f"LHS field '{lhs_field}' not found in dataset (available "
            f"fields: {sorted(field_names)}). Set dataset.lhs_field "
            f"explicitly."
        )
    if lhs_axis not in axis_names:
        raise ValueError(
            f"LHS axis '{lhs_axis}' not found in dataset (available "
            f"axes: {sorted(axis_names)}). Set dataset.lhs_axis "
            f"explicitly."
        )




    if dataset.lhs_field != lhs_field or dataset.lhs_axis != lhs_axis:
        return dataclasses.replace(dataset, lhs_field=lhs_field, lhs_axis=lhs_axis)
    return dataset


class PlatformBuilder:

    def __init__(
        self,
        dataset: PDEDataset,
        reqs: DerivativeReqs,
        device: str | None = None,
    ) -> None:
        self._dataset = dataset
        self._reqs = reqs
        self._device: torch.device | None = _resolve_device(device)







        self._surrogate_training: TrainingResult | None = None

    def build(self) -> PlatformComponents:


        self._surrogate_training = None
        if self._reqs.provider_kind == "none":






            dataset = self._resolve_lhs(self._dataset)
            registry = FunctionRegistry.create_default()
            executor = PythonExecutor(registry)
            return PlatformComponents(
                dataset=dataset,
                executor=executor,
                evaluator=None,
                context=None,
                registry=registry,
            )
        dataset = self._resolve_lhs(self._dataset)
        provider = self._build_provider(dataset)
        context = self._build_context(dataset, provider)
        registry = FunctionRegistry.create_default()
        executor = PythonExecutor(registry)
        evaluator = self._build_evaluator(dataset, provider, executor, context)
        return PlatformComponents(
            dataset=dataset,
            executor=executor,
            evaluator=evaluator,
            context=context,
            registry=registry,
        )





    @staticmethod
    def _resolve_lhs(dataset: PDEDataset) -> PDEDataset:
        return resolve_lhs_defaults(dataset)

    def _build_provider(self, dataset: PDEDataset) -> DerivativeProvider:
        kind = self._reqs.provider_kind
        if kind == "finite_diff":
            return FiniteDiffProvider(dataset, max_order=self._reqs.max_atomic_order)
        if kind == "autograd":
            coords = self._build_autograd_coords(dataset, device=self._device)
            model = self._resolve_surrogate_model(dataset, coords)






            model.eval()
            return AutogradProvider(
                model,
                coords,
                dataset,
                max_order=self._reqs.max_atomic_order,
            )


        raise ValueError(f"Unsupported provider_kind: {kind!r}")

    def _build_context(
        self,
        dataset: PDEDataset,
        provider: DerivativeProvider,
    ) -> ExecutionContext:
        if self._reqs.needs_surrogate:








            return SurrogateContext(
                dataset,
                provider,
                surrogate_field=dataset.lhs_field,
                training_result=self._surrogate_training,
                device=self._device,
            )


        if self._device is not None:
            return ExecutionContext(
                dataset=dataset,
                derivative_provider=provider,
                device=self._device,
            )
        return ExecutionContext(dataset=dataset, derivative_provider=provider)

    def _build_evaluator(
        self,
        dataset: PDEDataset,
        provider: DerivativeProvider,
        executor: PythonExecutor,
        context: ExecutionContext,
    ) -> Evaluator:
        if dataset.lhs_order == 0:
            raise NotImplementedError(
                "PlatformBuilder cannot build a platform evaluator for a "
                "homogeneous (lhs_order=0) dataset; use a provider_kind='none' "
                "plugin-private homogeneous path."
            )
        solver = LeastSquaresSolver()
        lhs = (
            provider.get_derivative(
                dataset.lhs_field,
                dataset.lhs_axis,
                self._reqs.lhs_order,
            )
            .detach()
            .flatten()
        )



        lhs = lhs.to(context.device)
        return Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=lhs,
        )





    def _resolve_surrogate_model(
        self,
        dataset: PDEDataset,
        coords: dict[str, torch.Tensor],
    ) -> torch.nn.Module:
        if self._reqs.surrogate_model is not None:
            return self._align_model_to_coords(self._reqs.surrogate_model, coords)
        if not self._reqs.needs_surrogate:




            model = self._build_default_field_model(dataset)
            return self._align_model_to_coords(model, coords)
        return self._train_default_surrogate(dataset, coords)

    def _train_default_surrogate(
        self,
        dataset: PDEDataset,
        coords: dict[str, torch.Tensor],
    ) -> torch.nn.Module:
        if dataset.fields is None:
            raise ValueError(
                "Default surrogate training requires dataset.fields to be set."
            )
        model = self._build_default_field_model(dataset)
        target = dataset.get_field(dataset.lhs_field).flatten().detach()




        aligned = self._align_model_to_coords(model, coords)


        assert isinstance(aligned, FieldModel)
        model = aligned


        coord_dtype = next(model.parameters()).dtype
        if target.dtype != coord_dtype:
            target = target.to(dtype=coord_dtype)
        all_kwargs: dict[str, Any] = dict(self._reqs.surrogate_train_kwargs or {})




        ctor_keys = ("lr", "weight_decay")
        ctor_kwargs = {k: all_kwargs.pop(k) for k in ctor_keys if k in all_kwargs}
        trainer = FieldModelTrainer(model, **ctor_kwargs)

        any_coord = next(iter(coords.values()))
        if target.device != any_coord.device:
            target = target.to(device=any_coord.device)




        self._surrogate_training = trainer.fit(
            coords,
            {dataset.lhs_field: target},
            **all_kwargs,
        )
        return model

    def _build_default_field_model(self, dataset: PDEDataset) -> FieldModel:
        if dataset.axis_order is None or dataset.fields is None:
            raise ValueError(
                "Default FieldModel construction requires dataset.axis_order "
                "and dataset.fields to be set."
            )
        arch_kwargs: dict[str, Any] = dict(self._reqs.surrogate_arch_kwargs or {})
        hidden_sizes = arch_kwargs.pop(
            "hidden_sizes", list(_DEFAULT_SURROGATE_HIDDEN_SIZES)
        )
        activation = arch_kwargs.pop("activation", _DEFAULT_SURROGATE_ACTIVATION)
        return FieldModel(
            coord_names=list(dataset.axis_order),
            field_names=[dataset.lhs_field],
            hidden_sizes=list(hidden_sizes),
            activation=activation,
            **arch_kwargs,
        )

    @staticmethod
    def _align_model_to_coords(
        model: torch.nn.Module,
        coords: dict[str, torch.Tensor],
    ) -> torch.nn.Module:
        any_coord = next(iter(coords.values()))
        target_device = any_coord.device
        target_dtype = any_coord.dtype


        try:
            current = next(model.parameters())
        except StopIteration:

            return model.to(device=target_device, dtype=target_dtype)
        if current.device != target_device or current.dtype != target_dtype:
            model = model.to(device=target_device, dtype=target_dtype)
        return model

    @staticmethod
    def _build_training_coords(dataset: PDEDataset) -> dict[str, torch.Tensor]:
        if dataset.axes is None or dataset.axis_order is None:
            raise ValueError("Dataset must have axes and axis_order")
        axis_values = [dataset.axes[name].values for name in dataset.axis_order]
        grids = torch.meshgrid(*axis_values, indexing="ij")
        return {
            name: grid.flatten().detach().clone()
            for name, grid in zip(dataset.axis_order, grids, strict=True)
        }

    @staticmethod
    def _build_autograd_coords(
        dataset: PDEDataset, device: torch.device | None = None
    ) -> dict[str, torch.Tensor]:
        if dataset.axes is None or dataset.axis_order is None:
            raise ValueError("Dataset must have axes and axis_order")
        axis_values = [dataset.axes[name].values for name in dataset.axis_order]
        grids = torch.meshgrid(*axis_values, indexing="ij")

        def _coord(grid: torch.Tensor) -> torch.Tensor:
            flat = grid.flatten().detach().clone()
            if device is not None:
                flat = flat.to(device)
            return flat.requires_grad_(True)

        return {
            name: _coord(grid)
            for name, grid in zip(dataset.axis_order, grids, strict=True)
        }
