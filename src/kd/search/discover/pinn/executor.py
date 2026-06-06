
from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from kd.core.executor import ExecutionContext
from kd.core.expr import (
    FunctionRegistry,
    PythonExecutor,
)
from kd.data.derivatives.autograd import (
    AutogradProvider,
)
from kd.data.schema import (
    DataTopology,
    PDEDataset,
    TaskType,
)
from kd.search.discover.pinn._memory_log import _log_memory

logger = logging.getLogger(__name__)

_DATASET_NAME = "pinn_collocation"
_MAX_EXECUTOR_DEPTH = 1000


def make_pinn_dataset(
    axis_names: list[str],
    field_names: list[str],
    *,
    lhs_field: str,
    lhs_axis: str,
) -> PDEDataset:
    _validate_dataset_names(axis_names, field_names, lhs_field, lhs_axis)
    return PDEDataset(
        name=_DATASET_NAME,
        task_type=TaskType.PDE,
        topology=DataTopology.SCATTERED,
        axis_order=list(axis_names),
        fields=None,
        lhs_field=lhs_field,
        lhs_axis=lhs_axis,
    )


def make_pinn_dataset_from(source: PDEDataset) -> PDEDataset:
    if not source.fields:
        raise ValueError(
            "make_pinn_dataset_from: source.fields must be populated "
            "(non-empty dict); loaders should return PDEDataset with "
            "fields set."
        )
    axis_names = list(source.axis_order) if source.axis_order is not None else []
    field_names = list(source.fields.keys())
    return make_pinn_dataset(
        axis_names=axis_names,
        field_names=field_names,
        lhs_field=source.lhs_field,
        lhs_axis=source.lhs_axis,
    )


class PINNExecutor:

    def __init__(self, registry: FunctionRegistry) -> None:
        self._registry = registry

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
        _log_memory("compute_residual_start", logger)
        _validate_term_inputs(terms, coefficients)
        provider = AutogradProvider(model, coords, dataset_metadata)
        device = _infer_device(coords)
        context = ExecutionContext(
            dataset=dataset_metadata,
            derivative_provider=provider,
            device=device,
        )
        executor = PythonExecutor(self._registry, max_depth=_MAX_EXECUTOR_DEPTH)
        field_value = provider.get_field(lhs_field)
        _log_memory("after_get_field", logger)
        lhs = provider.diff(field_value, lhs_axis, order=1)
        term_values: list[Tensor] = []
        for i, term in enumerate(terms):
            term_values.append(_evaluate_term(executor, term, context))
            _log_memory(f"after_diff_term_{i}", logger)
        rhs = _combine_terms(lhs, term_values, coefficients)
        result = lhs - rhs
        _log_memory("compute_residual_end", logger)
        return result


def _validate_dataset_names(
    axis_names: Sequence[str],
    field_names: Sequence[str],
    lhs_field: str,
    lhs_axis: str,
) -> None:
    _validate_name_list(axis_names, "axis_names")
    _validate_name_list(field_names, "field_names")
    if lhs_field not in field_names:
        raise ValueError(f"lhs_field '{lhs_field}' must be present in field_names")
    if lhs_axis not in axis_names:
        raise ValueError(f"lhs_axis '{lhs_axis}' must be present in axis_names")


def _validate_name_list(names: Sequence[str], label: str) -> None:
    if not names:
        raise ValueError(f"{label} must not be empty")
    if any(not name.strip() for name in names):
        raise ValueError(f"{label} must not contain empty or whitespace names")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must not contain duplicates")


def _validate_term_inputs(
    terms: Sequence[str],
    coefficients: Sequence[float] | Tensor,
) -> None:
    if not terms:
        raise ValueError("terms must not be empty")
    if any(not term.strip() for term in terms):
        raise ValueError("terms must not contain empty expressions")
    if len(coefficients) != len(terms):
        raise ValueError("coefficients length must match terms length")


def _infer_device(coords: dict[str, Tensor]) -> torch.device:
    try:
        first_tensor = next(iter(coords.values()))
    except StopIteration as exc:
        raise ValueError("coords must not be empty") from exc
    return first_tensor.device


def _evaluate_term(
    executor: PythonExecutor,
    code: str,
    context: ExecutionContext,
) -> Tensor:
    return executor.execute(code, context).value


def _combine_terms(
    lhs: Tensor,
    term_values: Sequence[Tensor],
    coefficients: Sequence[float] | Tensor,
) -> Tensor:
    coeff_tensor = _normalize_coefficients(coefficients, lhs)
    aligned_terms = [_align_term_shape(value, lhs) for value in term_values]
    stacked_terms = torch.stack(aligned_terms, dim=0)
    view_shape = (coeff_tensor.shape[0],) + (1,) * lhs.dim()
    return (coeff_tensor.view(view_shape) * stacked_terms).sum(dim=0)


def _normalize_coefficients(
    coefficients: Sequence[float] | Tensor,
    reference: Tensor,
) -> Tensor:
    if isinstance(coefficients, Tensor):
        return coefficients.detach().to(
            device=reference.device,
            dtype=reference.dtype,
        )
    return torch.tensor(
        list(coefficients),
        device=reference.device,
        dtype=reference.dtype,
    )


def _align_term_shape(value: Tensor, reference: Tensor) -> Tensor:
    if value.shape == reference.shape:
        return value
    if value.numel() == 1:
        return value.reshape(()).expand_as(reference)
    return value


__all__ = ["PINNExecutor", "make_pinn_dataset", "make_pinn_dataset_from"]
