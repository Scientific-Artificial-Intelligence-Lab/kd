
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from torch import Tensor

from kd.core.executor.context import ExecutionContext
from kd.core.expr.executor import (
    PythonExecutor,
    _is_diff_operator,
    _parse_diff_name,
    _parse_expression,
)
from kd.core.expr.naming import parse_compound_derivative
from kd.core.expr.registry import FunctionRegistry
from kd.data.derivatives.base import DerivativeProvider
from kd.data.derivatives.finite_diff import (
    DX_ZERO_FLOOR,
    UNIFORM_GRID_RTOL,
    FiniteDiffProvider,
    is_uniform_grid,
)
from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset

logger = logging.getLogger(__name__)


DEFAULT_METHOD = "Radau"


_MAX_STENCIL_ORDER = 3









_ALLOWED_NODE_TYPES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.expr_context,
)


@dataclass
class IntegrationResult:

    success: bool
    predicted_field: Tensor | None = None
    warning: str = ""
    diverged_at_t: float | None = None


@dataclass
class _RHSClassification:

    provider_max_order: int
    references_derivatives: bool


class _UnusedDerivativeProvider(DerivativeProvider):

    def get_derivative(self, field: str, axis: str, order: int) -> Tensor:
        raise RuntimeError(
            "derivative-free RHS classification violated: "
            f"get_derivative({field!r}, {axis!r}, {order}) requested"
        )

    def diff(self, expression: Tensor, axis: str, order: int) -> Tensor:
        raise RuntimeError(
            "derivative-free RHS classification violated: "
            f"diff(..., {axis!r}, {order}) requested"
        )

    def available_derivatives(self) -> list[tuple[str, str, int]]:
        return []


def _check_spatial_uniformity(
    dataset: PDEDataset,
    spatial_axes: list[str],
) -> str | None:
    assert dataset.axes is not None
    for axis_name in spatial_axes:
        vals = dataset.axes[axis_name].values.detach().cpu().numpy().astype(np.float64)




        if vals.size < 2:
            return (
                f"Spatial axis '{axis_name}' must have >=2 points to verify "
                f"uniformity, got {vals.size}"
            )
        diffs = np.diff(vals)
        dx0 = float(diffs[0])
        if abs(dx0) < DX_ZERO_FLOOR:
            return (
                f"Spatial axis '{axis_name}' has degenerate spacing dx={dx0:.6g}; "
                f"finite-difference stencils require nonzero dx"
            )

        if not is_uniform_grid(vals, rtol=UNIFORM_GRID_RTOL):
            max_dev = float(np.max(np.abs(diffs - dx0)))
            return (
                f"Spatial axis '{axis_name}' has non-uniform spacing "
                f"(dx[0]={dx0:.6g}, max deviation={max_dev:.6g}); "
                f"finite-difference stencils require a uniform grid"
            )
    return None


def _classify_rhs(
    tree: ast.Expression,
    dataset: PDEDataset,
    registry: FunctionRegistry,
) -> tuple[_RHSClassification | None, str | None]:
    assert dataset.fields is not None and dataset.axes is not None
    lhs_field = dataset.lhs_field
    field_names = set(dataset.fields.keys())
    axis_names = set(dataset.axes.keys())
    spatial_axes = set(dataset.spatial_axes)

    unsupported_calls: list[str] = []
    bad_axis_derivatives: list[str] = []
    cross_field: list[str] = []
    unknown_symbols: list[str] = []
    over_order: list[str] = []
    orders: list[int] = []
    references_derivatives = False



    call_func_ids = {
        id(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    def _add_unique(bucket: list[str], name: str) -> None:
        if name not in bucket:
            bucket.append(name)

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            return None, (
                "Unsupported syntax in RHS expression for integrate_pde: "
                f"{type(node).__name__} nodes are not part of the platform "
                "expression IR."
            )
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                node.value, (int, float)
            ):
                return None, (
                    "Unsupported constant in RHS expression for "
                    f"integrate_pde: {node.value!r} (only numeric literals "
                    "are supported)."
                )
            continue
        if isinstance(node, ast.Call):
            if node.keywords:
                return None, (
                    "Unsupported syntax in RHS expression for integrate_pde: "
                    "keyword arguments are not part of the platform "
                    "expression IR."
                )
            if not isinstance(node.func, ast.Name):
                return None, (
                    "Unsupported syntax in RHS expression for integrate_pde: "
                    "only simple function calls are supported."
                )
            func_name = node.func.id
            if _is_diff_operator(func_name):




                if len(node.args) != 1:
                    return None, (
                        f"Malformed derivative call in RHS expression for "
                        f"integrate_pde: '{func_name}' expects exactly 1 "
                        f"argument, got {len(node.args)}."
                    )
                try:
                    axis, order = _parse_diff_name(func_name)
                except ValueError:
                    _add_unique(bad_axis_derivatives, func_name)
                    continue
                if axis not in spatial_axes:
                    _add_unique(bad_axis_derivatives, func_name)
                    continue
                if order > _MAX_STENCIL_ORDER:
                    _add_unique(over_order, func_name)
                    continue
                references_derivatives = True
                continue
            if func_name == "lap" or not registry.has(func_name):
                _add_unique(unsupported_calls, func_name)
            continue
        if isinstance(node, ast.Name) and id(node) not in call_func_ids:
            name = node.id
            if name == lhs_field:
                continue
            if name in field_names:
                _add_unique(cross_field, name)
                continue
            derivative = parse_compound_derivative(
                name,
                known_fields=field_names,
                known_axes=axis_names,
            )
            if derivative is not None:
                deriv_field, segments = derivative
                if deriv_field != lhs_field:
                    _add_unique(cross_field, name)
                    continue
                if any(axis not in spatial_axes for axis, _ in segments):
                    _add_unique(bad_axis_derivatives, name)
                    continue
                if any(order > _MAX_STENCIL_ORDER for _, order in segments):
                    _add_unique(over_order, name)
                    continue



                orders.append(segments[0][1])
                references_derivatives = True
                continue
            if name in spatial_axes:
                continue
            _add_unique(unknown_symbols, name)

    if unsupported_calls:
        return None, (
            "Unsupported function calls in RHS expression for integrate_pde: "
            f"{sorted(unsupported_calls)}. integrate_pde supports field, "
            "coordinate, and explicit derivative symbols only. Expand "
            "context-aware operators such as lap(...) to explicit "
            "derivative symbols."
        )
    if bad_axis_derivatives:
        return None, (
            "integrate_pde skipped: RHS contains derivatives along "
            f"non-spatial or unknown axes {sorted(bad_axis_derivatives)}. "
            "Time integration supports spatial-axis derivatives of the "
            "LHS field only."
        )
    if cross_field:
        return None, (
            f"Cross-field references not supported: {sorted(cross_field)} "
            f"reference fields other than LHS field '{lhs_field}'"
        )
    if unknown_symbols:
        return None, (
            "integrate_pde skipped: RHS contains unrecognised symbols "
            f"{sorted(unknown_symbols)}. Time integration supports field, "
            "spatial-coordinate, and explicit-derivative symbols only."
        )
    if over_order:
        return None, (
            "integrate_pde skipped: derivative order exceeds the supported "
            f"stencil maximum ({_MAX_STENCIL_ORDER}): {sorted(over_order)}."
        )

    return (
        _RHSClassification(
            provider_max_order=max(orders, default=0),
            references_derivatives=references_derivatives,
        ),
        None,
    )


def _build_spatial_slice(
    dataset: PDEDataset,
    spatial_axes: list[str],
    initial_state: NDArray[np.float64],
) -> tuple[PDEDataset, Tensor]:
    assert dataset.axes is not None
    slice_axes = {
        name: AxisInfo(
            name=name,
            values=dataset.axes[name].values.detach().to("cpu", torch.float64),
            is_periodic=dataset.axes[name].is_periodic,
        )
        for name in spatial_axes
    }
    state_tensor = torch.from_numpy(initial_state.copy())
    slice_dataset = PDEDataset(
        name=f"{dataset.name}::integration-slice",
        task_type=dataset.task_type,
        topology=DataTopology.GRID,
        axes=slice_axes,
        axis_order=list(spatial_axes),
        fields={
            dataset.lhs_field: FieldData(name=dataset.lhs_field, values=state_tensor)
        },
        lhs_field="",
        lhs_axis="",
    )
    return slice_dataset, state_tensor


def _check_divergence(
    y: NDArray[np.floating[Any]],
    t: NDArray[np.floating[Any]],
) -> float | None:
    for i in range(y.shape[1]):
        if not np.all(np.isfinite(y[:, i])):
            return float(t[i])
    return None


def _reconstruct_field(
    y: NDArray[np.floating[Any]],
    spatial_shape: tuple[int, ...],
    n_times: int,
    time_dim: int,
    device: torch.device,
) -> Tensor:
    output_shape = list(spatial_shape)
    output_shape.insert(time_dim, n_times)

    field_np = np.zeros(output_shape, dtype=np.float64)
    for i in range(n_times):
        u_spatial = y[:, i].reshape(spatial_shape)
        idx: list[slice | int] = [slice(None)] * len(output_shape)
        idx[time_dim] = i
        field_np[tuple(idx)] = u_spatial

    return torch.tensor(field_np, dtype=torch.float64, device=device)


def integrate_pde(
    rhs: str,
    dataset: PDEDataset,
    *,
    method: str = DEFAULT_METHOD,
    max_step: float | None = None,
) -> IntegrationResult:

    if dataset.topology != DataTopology.GRID:
        return IntegrationResult(
            success=False,
            warning=(
                f"Integration requires GRID topology, got {dataset.topology.value}"
            ),
        )

    if dataset.axes is None or dataset.axis_order is None or dataset.fields is None:
        return IntegrationResult(
            success=False,
            warning="Dataset missing axes, axis_order, or fields",
        )

    if not dataset.lhs_field or not dataset.lhs_axis:
        return IntegrationResult(
            success=False,
            warning="Dataset missing lhs_field or lhs_axis",
        )

    time_axis = dataset.lhs_axis
    spatial_axes = dataset.spatial_axes

    if time_axis not in dataset.axes:
        return IntegrationResult(
            success=False,
            warning=f"Time axis '{time_axis}' not found in dataset axes",
        )






    if not spatial_axes:
        return IntegrationResult(
            success=False,
            warning=(
                "integrate_pde skipped: dataset has no spatial axes "
                f"(axis_order contains only the evolution axis '{time_axis}'). "
                "Method-of-Lines integration requires at least one spatial "
                "axis; pure-ODE datasets are not supported."
            ),
        )

    t_vals = dataset.axes[time_axis].values.detach().cpu().numpy().astype(np.float64)
    t_span = (float(t_vals[0]), float(t_vals[-1]))


    field_name = dataset.lhs_field
    field_data = (
        dataset.fields[field_name].values.detach().cpu().numpy().astype(np.float64)
    )

    non_uniform_warning = _check_spatial_uniformity(dataset, spatial_axes)
    if non_uniform_warning is not None:
        return IntegrationResult(success=False, warning=non_uniform_warning)




    try:
        parsed = _parse_expression(rhs)
    except (TypeError, ValueError) as exc:
        return IntegrationResult(
            success=False,
            warning=f"Invalid RHS expression for integrate_pde: {exc}",
        )


    registry = FunctionRegistry.create_default()
    classification, rejection = _classify_rhs(parsed.tree, dataset, registry)
    if classification is None:
        return IntegrationResult(success=False, warning=rejection or "")

    executor = PythonExecutor(registry)


    time_dim = dataset.axis_order.index(time_axis)
    spatial_shape = tuple(dataset.axes[a].values.numel() for a in spatial_axes)
    u0_grid = np.ascontiguousarray(
        np.take(field_data, 0, axis=time_dim).reshape(spatial_shape)
    )
    u0 = u0_grid.ravel()

    slice_dataset, state_tensor = _build_spatial_slice(dataset, spatial_axes, u0_grid)


    assert slice_dataset.axes is not None
    dirichlet_axes = [
        idx
        for idx, name in enumerate(spatial_axes)
        if not slice_dataset.axes[name].is_periodic
    ]

    references_derivatives = classification.references_derivatives
    provider_max_order = max(classification.provider_max_order, 1)
    static_context: ExecutionContext | None = None
    if not references_derivatives:



        static_context = ExecutionContext(
            dataset=slice_dataset,
            derivative_provider=_UnusedDerivativeProvider(),
        )

    def ode_rhs(
        _t: float,
        u_flat: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        state = np.ascontiguousarray(u_flat.reshape(spatial_shape))
        state_tensor.copy_(torch.from_numpy(state))

        if references_derivatives:


            provider: DerivativeProvider = FiniteDiffProvider(
                slice_dataset,
                max_order=provider_max_order,
            )
            context = ExecutionContext(
                dataset=slice_dataset,
                derivative_provider=provider,
            )
        else:
            assert static_context is not None
            context = static_context

        value = executor.execute(rhs, context).value

        if value.dim() == 0:


            dudt = np.full(spatial_shape, float(value.item()), dtype=np.float64)
        else:



            dudt = np.array(value.detach().cpu().numpy(), dtype=np.float64)
            if dudt.shape != spatial_shape:
                raise ValueError(
                    f"RHS evaluation produced shape {dudt.shape}, expected "
                    f"{spatial_shape}"
                )

        for axis_idx in dirichlet_axes:
            idx_first: list[slice | int] = [slice(None)] * len(spatial_shape)
            idx_first[axis_idx] = 0
            dudt[tuple(idx_first)] = 0.0

            idx_last: list[slice | int] = [slice(None)] * len(spatial_shape)
            idx_last[axis_idx] = -1
            dudt[tuple(idx_last)] = 0.0

        return dudt.ravel()

    solve_kwargs: dict[str, object] = {
        "method": method,
        "t_eval": t_vals,
        "dense_output": False,
    }
    if max_step is not None:
        solve_kwargs["max_step"] = max_step

    try:
        sol = solve_ivp(ode_rhs, t_span, u0, **solve_kwargs)
    except Exception as exc:
        return IntegrationResult(
            success=False,
            warning=f"solve_ivp failed: {exc}",
        )

    if sol.status == -1:
        return IntegrationResult(
            success=False,
            warning=f"solve_ivp integration failed: {sol.message}",
        )


    diverged_at_t = _check_divergence(sol.y, sol.t)

    device = torch.device("cpu")
    predicted = _reconstruct_field(
        sol.y,
        spatial_shape,
        len(sol.t),
        time_dim,
        device,
    )

    if diverged_at_t is not None:
        return IntegrationResult(
            success=False,
            predicted_field=predicted,
            diverged_at_t=diverged_at_t,
            warning=f"Integration diverged at t={diverged_at_t:.4g}",
        )

    return IntegrationResult(success=True, predicted_field=predicted)


__all__ = [
    "IntegrationResult",
    "integrate_pde",
]
