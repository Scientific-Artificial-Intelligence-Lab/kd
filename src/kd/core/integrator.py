
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy
import torch
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from sympy.core.function import AppliedUndef
from torch import Tensor

from kd.core.expr.naming import parse_compound_derivative
from kd.data.derivatives.finite_diff import (
    DX_ZERO_FLOOR,
    UNIFORM_GRID_RTOL,
    central_diff,
    is_uniform_grid,
)
from kd.data.schema import DataTopology, PDEDataset

logger = logging.getLogger(__name__)


DEFAULT_METHOD = "Radau"


@dataclass
class IntegrationResult:

    success: bool
    predicted_field: Tensor | None = None
    warning: str = ""
    diverged_at_t: float | None = None


@dataclass
class _ParsedSymbols:

    state_vars: set[str] = field(default_factory=set)
    derivatives: dict[str, tuple[str, list[tuple[str, int]]]] = field(
        default_factory=dict
    )
    coordinates: set[str] = field(default_factory=set)
    unsupported_functions: set[str] = field(default_factory=set)
    unknown_symbols: set[str] = field(default_factory=set)






_DERIV_PLACEHOLDER_RE = re.compile(r"^d\d+_[a-zA-Z]\w*$")


def _is_derivative_placeholder(name: str) -> bool:
    return bool(_DERIV_PLACEHOLDER_RE.match(name))


def _classify_symbols(
    rhs_expr: sympy.Expr,
    field_names: set[str],
    coord_names: set[str],
) -> _ParsedSymbols:
    parsed = _ParsedSymbols()
    for call in rhs_expr.atoms(AppliedUndef):
        parsed.unsupported_functions.add(str(call.func))

    for sym in rhs_expr.free_symbols:
        name = str(sym)
        derivative = parse_compound_derivative(
            name,
            known_fields=field_names,
            known_axes=coord_names,
        )
        if derivative is not None:
            parsed.derivatives[name] = derivative
            continue
        if name in field_names:
            parsed.state_vars.add(name)
            continue
        if name in coord_names:
            parsed.coordinates.add(name)
            continue




        parsed.unknown_symbols.add(name)
    return parsed


def _finite_diff(
    u: NDArray[np.floating[Any]],
    dx: float,
    order: int,
    periodic: bool,
) -> NDArray[np.floating[Any]]:
    return _finite_diff_along_axis(u, 0, dx, order, periodic)


@dataclass
class _SpatialAxisInfo:

    name: str
    values: NDArray[np.floating[Any]]
    dx: float
    periodic: bool
    axis_index: int


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


def _build_spatial_info(
    dataset: PDEDataset,
    spatial_axes: list[str],
) -> list[_SpatialAxisInfo]:
    assert dataset.axes is not None
    info_list: list[_SpatialAxisInfo] = []
    for idx, axis_name in enumerate(spatial_axes):
        axis = dataset.axes[axis_name]
        vals = axis.values.detach().cpu().numpy().astype(np.float64)
        dx = float(vals[1] - vals[0]) if len(vals) > 1 else 1.0
        info_list.append(
            _SpatialAxisInfo(
                name=axis_name,
                values=vals,
                dx=dx,
                periodic=axis.is_periodic,
                axis_index=idx,
            )
        )
    return info_list


def _build_lambdify_args(
    parsed: _ParsedSymbols,
    coord_names: set[str],
) -> list[sympy.Symbol]:
    args: list[sympy.Symbol] = []
    for name in sorted(parsed.state_vars):
        args.append(sympy.Symbol(name))
    for name in sorted(parsed.derivatives.keys()):
        args.append(sympy.Symbol(name))
    for name in sorted(coord_names & parsed.coordinates):
        args.append(sympy.Symbol(name))
    return args


def _finite_diff_along_axis(
    u: NDArray[np.floating[Any]],
    axis_index: int,
    dx: float,
    order: int,
    periodic: bool,
) -> NDArray[np.floating[Any]]:
    tensor = torch.from_numpy(np.ascontiguousarray(u))
    deriv = central_diff(tensor, dx, axis=axis_index, order=order, is_periodic=periodic)
    return np.asarray(deriv.numpy())


def _mol_rhs(
    u_flat: NDArray[np.floating[Any]],
    rhs_func: Any,
    parsed: _ParsedSymbols,
    spatial_info: list[_SpatialAxisInfo],
    spatial_shape: tuple[int, ...],
    sym_args: list[sympy.Symbol],
    lhs_field: str,
) -> NDArray[np.floating[Any]]:
    for _name, (fld, _orders) in parsed.derivatives.items():
        assert fld == lhs_field, (
            f"_mol_rhs single-field invariant violated: derivative "
            f"references '{fld}' but lhs_field is '{lhs_field}'"
        )
    for svar in parsed.state_vars:
        assert svar == lhs_field, (
            f"_mol_rhs single-field invariant violated: state var "
            f"'{svar}' does not match lhs_field '{lhs_field}'"
        )
    u = u_flat.reshape(spatial_shape)
    axis_lookup = {info.name: info for info in spatial_info}


    deriv_values: dict[str, NDArray[np.floating[Any]]] = {}
    for name, (_fld, axis_orders) in parsed.derivatives.items():
        deriv = u
        for axis, order in axis_orders:
            axis_info = axis_lookup.get(axis)
            if axis_info is None:
                raise ValueError(
                    f"Spatial axis '{axis}' not found for derivative '{name}'",
                )
            if len(spatial_shape) == 1:
                deriv = _finite_diff(
                    deriv,
                    axis_info.dx,
                    order,
                    axis_info.periodic,
                )
            else:
                deriv = _finite_diff_along_axis(
                    deriv,
                    axis_info.axis_index,
                    axis_info.dx,
                    order,
                    axis_info.periodic,
                )
        deriv_values[name] = deriv


    coord_grids: dict[str, NDArray[np.floating[Any]]] = {}
    for info in spatial_info:
        if info.name in parsed.coordinates:
            if len(spatial_info) == 1:
                coord_grids[info.name] = info.values
            else:
                shape = [1] * len(spatial_shape)
                shape[info.axis_index] = len(info.values)
                coord_grids[info.name] = np.broadcast_to(
                    info.values.reshape(shape),
                    spatial_shape,
                )








    call_args: list[Any] = []
    for sym in sym_args:
        name = str(sym)
        if name in parsed.state_vars:
            call_args.append(u)
        elif name in deriv_values:
            call_args.append(deriv_values[name])
        elif name in coord_grids:
            call_args.append(coord_grids[name])
        else:
            raise AssertionError(
                f"_mol_rhs received unrecognised symbol '{name}' in "
                "sym_args; _classify_symbols should have routed it via "
                "parsed.unknown_symbols before "
                "lambdify."
            )

    dudt_raw: Any = rhs_func(*call_args)

    if np.isscalar(dudt_raw):
        dudt = np.full(spatial_shape, dudt_raw, dtype=np.float64)
    else:


        dudt = np.array(dudt_raw, dtype=np.float64)


    for info in spatial_info:
        if not info.periodic:
            idx_first: list[Any] = [slice(None)] * len(spatial_shape)
            idx_first[info.axis_index] = 0
            dudt[tuple(idx_first)] = 0.0

            idx_last: list[Any] = [slice(None)] * len(spatial_shape)
            idx_last[info.axis_index] = -1
            dudt[tuple(idx_last)] = 0.0

    return dudt.ravel()


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
        idx: list[Any] = [slice(None)] * len(output_shape)
        idx[time_dim] = i
        field_np[tuple(idx)] = u_spatial

    return torch.tensor(field_np, dtype=torch.float64, device=device)


def integrate_pde(
    rhs_expr: sympy.Expr,
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

    t_vals = dataset.axes[time_axis].values.detach().cpu().numpy().astype(np.float64)
    t_span = (float(t_vals[0]), float(t_vals[-1]))


    field_name = dataset.lhs_field
    field_data = (
        dataset.fields[field_name].values.detach().cpu().numpy().astype(np.float64)
    )

    non_uniform_warning = _check_spatial_uniformity(dataset, spatial_axes)
    if non_uniform_warning is not None:
        return IntegrationResult(success=False, warning=non_uniform_warning)

    spatial_info = _build_spatial_info(dataset, spatial_axes)


    field_names = set(dataset.fields.keys())
    coord_names = set(spatial_axes)
    parsed = _classify_symbols(rhs_expr, field_names, coord_names)

    if parsed.unsupported_functions:
        unsupported = sorted(parsed.unsupported_functions)
        return IntegrationResult(
            success=False,
            warning=(
                "Unsupported function calls in RHS expression for integrate_pde: "
                f"{unsupported}. integrate_pde supports field, coordinate, and "
                "explicit derivative symbols only. Expand context-aware operators "
                "such as lap(...) to explicit derivative symbols, or add explicit "
                "lambdify support for the function."
            ),
        )










    if parsed.unknown_symbols:
        unknowns = sorted(parsed.unknown_symbols)
        placeholders = [n for n in unknowns if _is_derivative_placeholder(n)]
        detail = (
            f"nested-derivative placeholders {placeholders}"
            if placeholders
            else f"unrecognised symbols {unknowns}"
        )
        return IntegrationResult(
            success=False,
            warning=(
                f"integrate_pde skipped: RHS contains {detail} "
                "Time integration supports field, spatial-coordinate, and "
                "explicit-derivative symbols only."
            ),
        )





    cross_fields: list[str] = []
    for dname, (dfld, _axis_orders) in parsed.derivatives.items():
        if dfld != field_name:
            cross_fields.append(dname)
    for svar in sorted(parsed.state_vars):
        if svar != field_name:
            cross_fields.append(svar)

    if cross_fields:
        return IntegrationResult(
            success=False,
            warning=(
                f"Cross-field references not supported: "
                f"{cross_fields} reference fields other than "
                f"LHS field '{field_name}'"
            ),
        )


    sym_args = _build_lambdify_args(parsed, coord_names)
    try:
        rhs_func = sympy.lambdify(sym_args, rhs_expr, modules=["numpy"])
    except Exception as exc:
        return IntegrationResult(
            success=False,
            warning=f"Failed to lambdify RHS expression: {exc}",
        )


    time_dim = dataset.axis_order.index(time_axis)
    u0 = np.take(field_data, 0, axis=time_dim).ravel().astype(np.float64)

    spatial_shape = tuple(dataset.axes[a].values.numel() for a in spatial_axes)


    def ode_rhs(
        _t: float,
        u_flat: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        return _mol_rhs(
            u_flat,
            rhs_func,
            parsed,
            spatial_info,
            spatial_shape,
            sym_args,
            field_name,
        )

    solve_kwargs: dict[str, Any] = {
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
