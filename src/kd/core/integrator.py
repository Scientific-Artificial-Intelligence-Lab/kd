
from __future__ import annotations

import ast
import logging
import math
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from torch import Tensor

from kd.core.executor.context import ExecutionContext
from kd.core.expr.columns import ColumnContext
from kd.core.expr.executor import (
    UNITY_TOKEN,
    PythonExecutor,
    _is_diff_operator,
    _parse_diff_name,
    _parse_expression,
    _value_leaf_names,
)
from kd.core.expr.naming import parse_compound_derivative
from kd.core.expr.registry import FunctionRegistry
from kd.core.interrupt import SearchInterrupted
from kd.core.oom import is_cpu_alloc_failure
from kd.data.derivatives.base import DerivativeProvider
from kd.data.derivatives.finite_diff import (
    DX_ZERO_FLOOR,
    MAX_SUPPORTED_ORDER,
    UNIFORM_GRID_RTOL,
    FiniteDiffProvider,
    is_uniform_grid,
)
from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset
from kd.data.trajectory import Trajectory, derivative_target

logger = logging.getLogger(__name__)


DEFAULT_METHOD = "Radau"



_MAX_STENCIL_ORDER = MAX_SUPPORTED_ORDER


def assemble_rhs(
    terms: Sequence[str],
    coefficients: Sequence[float],
    keep: Sequence[int],
) -> str:
    non_finite = [i for i in keep if not math.isfinite(coefficients[i])]
    if non_finite:
        detail = ", ".join(
            f"term '{terms[i]}' has coefficient {coefficients[i]!r}" for i in non_finite
        )
        raise ValueError(
            f"Cannot assemble integration RHS: non-finite coefficient(s) — "
            f"{detail}. Coefficients must be finite; a NaN/Inf here points "
            "at a degenerate upstream fit/refit, not at the RHS terms."
        )
    survivors = [i for i in keep if coefficients[i] != 0.0]
    if not survivors:
        return "0"
    return " + ".join(f"({coefficients[i]!r})*({terms[i]})" for i in survivors)










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
    timed_out: bool = False


class _DeadlineExpired(Exception):
    pass


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

    def diff(
        self,
        expression: Tensor,
        axis: str,
        order: int,
        *,
        is_periodic: bool | None = None,
    ) -> Tensor:
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


def _add_unique(bucket: list[str], name: str) -> None:
    if name not in bucket:
        bucket.append(name)


@dataclass(frozen=True)
class _RHSSymbols:

    diff_calls: tuple[str, ...]
    unsupported_calls: tuple[str, ...]
    names: tuple[str, ...]


def _syntax_rejection(node: ast.AST, entry: str) -> str | None:
    if not isinstance(node, _ALLOWED_NODE_TYPES):
        return (
            f"Unsupported syntax in RHS expression for {entry}: "
            f"{type(node).__name__} nodes are not part of the platform "
            "expression IR."
        )
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            return (
                f"Unsupported constant in RHS expression for {entry}: "
                f"{node.value!r} (only numeric literals are supported)."
            )
        return None
    if not isinstance(node, ast.Call):
        return None
    if node.keywords:
        return (
            f"Unsupported syntax in RHS expression for {entry}: keyword "
            "arguments are not part of the platform expression IR."
        )
    if not isinstance(node.func, ast.Name):
        return (
            f"Unsupported syntax in RHS expression for {entry}: only simple "
            "function calls are supported."
        )



    if _is_diff_operator(node.func.id) and len(node.args) != 1:
        return (
            f"Malformed derivative call in RHS expression for {entry}: "
            f"'{node.func.id}' expects exactly 1 argument, got {len(node.args)}."
        )
    return None


def _rhs_symbols(
    tree: ast.Expression,
    registry: FunctionRegistry,
    entry: str,
) -> _RHSSymbols | str:
    diff_calls: list[str] = []
    unsupported_calls: list[str] = []
    for node in ast.walk(tree):
        rejection = _syntax_rejection(node, entry)
        if rejection is not None:
            return rejection
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id
            if _is_diff_operator(func_name):
                _add_unique(diff_calls, func_name)
            elif func_name == "lap" or not registry.has(func_name):
                _add_unique(unsupported_calls, func_name)
    names = sorted(_value_leaf_names(tree.body) - {UNITY_TOKEN})
    return _RHSSymbols(tuple(diff_calls), tuple(unsupported_calls), tuple(names))


def _classify_rhs(
    tree: ast.Expression,
    dataset: PDEDataset,
    registry: FunctionRegistry,
) -> tuple[_RHSClassification | None, str | None]:
    assert dataset.fields is not None and dataset.axes is not None
    symbols = _rhs_symbols(tree, registry, "integrate_pde")
    if isinstance(symbols, str):
        return None, symbols
    spatial_axes = set(dataset.spatial_axes)
    buckets: dict[str, list[str]] = {
        "bad_axis": [],
        "cross_field": [],
        "unknown": [],
        "over_order": [],
    }
    references_derivatives = False
    for func_name in symbols.diff_calls:
        try:
            axis, order = _parse_diff_name(func_name)
        except ValueError:
            _add_unique(buckets["bad_axis"], func_name)
            continue
        if axis not in spatial_axes:
            _add_unique(buckets["bad_axis"], func_name)
        elif order > _MAX_STENCIL_ORDER:
            _add_unique(buckets["over_order"], func_name)
        else:
            references_derivatives = True
    orders = _classify_grid_names(symbols.names, dataset, buckets)
    rejection = _grid_rejection(symbols.unsupported_calls, buckets, dataset.lhs_field)
    if rejection is not None:
        return None, rejection
    return (
        _RHSClassification(
            provider_max_order=max(orders, default=0),
            references_derivatives=references_derivatives or bool(orders),
        ),
        None,
    )


def _classify_grid_names(
    names: Sequence[str],
    dataset: PDEDataset,
    buckets: dict[str, list[str]],
) -> list[int]:
    assert dataset.fields is not None and dataset.axes is not None
    lhs_field = dataset.lhs_field
    field_names = set(dataset.fields.keys())
    axis_names = set(dataset.axes.keys())
    spatial_axes = set(dataset.spatial_axes)
    orders: list[int] = []
    for name in names:
        if name == lhs_field:
            continue
        if name in field_names:
            _add_unique(buckets["cross_field"], name)
            continue
        derivative = parse_compound_derivative(
            name, known_fields=field_names, known_axes=axis_names
        )
        if derivative is not None:
            deriv_field, segments = derivative
            if deriv_field != lhs_field:
                _add_unique(buckets["cross_field"], name)
            elif any(axis not in spatial_axes for axis, _ in segments):
                _add_unique(buckets["bad_axis"], name)
            elif any(order > _MAX_STENCIL_ORDER for _, order in segments):
                _add_unique(buckets["over_order"], name)
            else:



                orders.append(segments[0][1])
            continue
        if name not in spatial_axes:
            _add_unique(buckets["unknown"], name)
    return orders


def _grid_rejection(
    unsupported_calls: Sequence[str],
    buckets: dict[str, list[str]],
    lhs_field: str,
) -> str | None:
    if unsupported_calls:
        return (
            "Unsupported function calls in RHS expression for integrate_pde: "
            f"{sorted(unsupported_calls)}. integrate_pde supports field, "
            "coordinate, and explicit derivative symbols only. Expand "
            "context-aware operators such as lap(...) to explicit "
            "derivative symbols."
        )
    if buckets["bad_axis"]:
        return (
            "integrate_pde skipped: RHS contains derivatives along "
            f"non-spatial or unknown axes {sorted(buckets['bad_axis'])}. "
            "Time integration supports spatial-axis derivatives of the "
            "LHS field only."
        )
    if buckets["cross_field"]:
        return (
            f"Cross-field references not supported: {sorted(buckets['cross_field'])} "
            f"reference fields other than LHS field '{lhs_field}'"
        )
    if buckets["unknown"]:
        return (
            "integrate_pde skipped: RHS contains unrecognised symbols "
            f"{sorted(buckets['unknown'])}. Time integration supports field, "
            "spatial-coordinate, and explicit-derivative symbols only."
        )
    if buckets["over_order"]:
        return (
            "integrate_pde skipped: derivative order exceeds the supported "
            f"stencil maximum ({_MAX_STENCIL_ORDER}): {sorted(buckets['over_order'])}."
        )
    return None


def _trajectory_rejection(
    rhs: Mapping[str, str],
    params: Collection[str],
    time_var: str,
    registry: FunctionRegistry,
) -> str | None:
    admitted = {*rhs, *params, time_var}
    refusals: list[str] = []
    for state, code in rhs.items():
        refusal = _trajectory_law_refusal(code, rhs, admitted, registry)
        if refusal:
            refusals.append(f"{derivative_target(state)}: {refusal}")
    if not refusals:
        return None
    return (
        f"integrate_trajectory refused {'; '.join(refusals)}. A right-hand side "
        f"may use the states {list(rhs)}, the parameters {list(params)}, the "
        f"time '{time_var}', 'one', numeric literals and registry functions."
    )


def _trajectory_law_refusal(
    code: str,
    states: Collection[str],
    admitted: Collection[str],
    registry: FunctionRegistry,
) -> str:
    try:
        tree = _parse_expression(code).tree
    except ValueError as exc:
        return f"invalid expression ({exc})"
    symbols = _rhs_symbols(tree, registry, "integrate_trajectory")
    if isinstance(symbols, str):
        return symbols
    targets = {derivative_target(state) for state in states}
    derivatives = list(symbols.diff_calls)
    target_names: list[str] = []
    unknown: list[str] = []
    for name in symbols.names:
        if name in admitted:
            continue
        if name in targets:
            target_names.append(name)
        elif parse_compound_derivative(name, known_fields=set(states)) is not None:
            derivatives.append(name)
        else:
            unknown.append(name)
    found = (
        ("unsupported function calls", symbols.unsupported_calls),
        ("derivatives", derivatives),
        ("target names", target_names),
        ("unrecognised symbols", unknown),
    )
    return ", ".join(f"{label} {sorted(names)}" for label, names in found if names)


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
    deadline: float | None = None,
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

    return _solve_with_deadline(
        ode_rhs,
        t_vals,
        u0,
        lambda y: _reconstruct_field(
            y, spatial_shape, y.shape[1], time_dim, torch.device("cpu")
        ),
        method=method,
        max_step=max_step,
        deadline=deadline,
    )


def _solve_with_deadline(
    rhs: Callable[[float, NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
    t_eval: NDArray[np.float64],
    y0: NDArray[np.float64],
    to_field: Callable[[NDArray[np.floating[Any]]], Tensor],
    *,
    method: str,
    max_step: float | None,
    deadline: float | None,
    rtol: float | None = None,
    atol: float | None = None,
) -> IntegrationResult:

    def timed_rhs(t: float, y: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        if deadline is not None and time.monotonic() >= deadline:
            raise _DeadlineExpired
        return rhs(t, y)

    t_span = (float(t_eval[0]), float(t_eval[-1]))
    solve_kwargs: dict[str, object] = {
        "method": method,
        "t_eval": t_eval,
        "dense_output": False,
    }
    optional = {"max_step": max_step, "rtol": rtol, "atol": atol}
    solve_kwargs.update(
        {key: value for key, value in optional.items() if value is not None}
    )

    try:
        if not np.all(np.isfinite(timed_rhs(t_span[0], y0))):
            raise ValueError(
                "RHS evaluation produced non-finite values at the initial state"
            )
        sol = solve_ivp(timed_rhs, t_span, y0, **solve_kwargs)
    except _DeadlineExpired:
        return IntegrationResult(
            success=False,
            warning="integration stopped at its deadline during an RHS evaluation",
            timed_out=True,
        )
    except (SearchInterrupted, torch.cuda.OutOfMemoryError):

        raise
    except Exception as exc:
        if is_cpu_alloc_failure(exc):
            raise
        return IntegrationResult(
            success=False,
            warning=f"solve_ivp failed: {exc}",
        )
    return _solution_result(sol, to_field)


def _solution_result(
    sol: Any,
    to_field: Callable[[NDArray[np.floating[Any]]], Tensor],
) -> IntegrationResult:
    if sol.status == -1:
        return IntegrationResult(
            success=False,
            warning=f"solve_ivp integration failed: {sol.message}",
        )

    diverged_at_t = _check_divergence(sol.y, sol.t)
    predicted = to_field(sol.y)
    if diverged_at_t is not None:
        return IntegrationResult(
            success=False,
            predicted_field=predicted,
            diverged_at_t=diverged_at_t,
            warning=f"Integration diverged at t={diverged_at_t:.4g}",
        )
    return IntegrationResult(success=True, predicted_field=predicted)


def integrate_trajectory(
    rhs: Mapping[str, str],
    trajectory: Trajectory,
    *,
    time_var: str,
    method: str = DEFAULT_METHOD,
    max_step: float | None = None,
    deadline: float | None = None,
    rtol: float,
    atol: float,
) -> IntegrationResult:
    registry = FunctionRegistry.create_default()
    rejection = _trajectory_rejection(rhs, tuple(trajectory.params), time_var, registry)
    if rejection is not None:
        return IntegrationResult(success=False, warning=rejection)
    executor = PythonExecutor(registry)
    state = torch.zeros(len(rhs), dtype=torch.float64, device="cpu")
    clock = torch.zeros(1, dtype=torch.float64, device="cpu")
    columns = {name: state[row: row + 1] for row, name in enumerate(rhs)}
    for name, value in trajectory.params.items():
        columns[name] = torch.tensor([value], dtype=torch.float64, device="cpu")
    columns[time_var] = clock
    unity = torch.ones(1, dtype=torch.float64, device="cpu")
    context = cast(ExecutionContext, ColumnContext(columns, unity))
    programs = tuple(rhs.values())

    def ode_rhs(t: float, y: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        state.copy_(torch.from_numpy(np.ascontiguousarray(y, dtype=np.float64)))
        clock.fill_(t)
        values = [executor.execute(code, context).value.item() for code in programs]
        return np.array(values, dtype=np.float64)

    return _solve_with_deadline(
        ode_rhs,
        np.asarray(trajectory.t, dtype=np.float64),
        np.array(trajectory.state[:, 0], dtype=np.float64),
        lambda y: torch.from_numpy(np.array(y, dtype=np.float64)),
        method=method,
        max_step=max_step,
        deadline=deadline,
        rtol=rtol,
        atol=atol,
    )


__all__ = [
    "IntegrationResult",
    "assemble_rhs",
    "integrate_pde",
    "integrate_trajectory",
]
