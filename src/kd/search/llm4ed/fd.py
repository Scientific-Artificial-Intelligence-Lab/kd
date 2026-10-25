
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


OPERAND_ORDER = ("x", "u_x", "u_xx", "u_xxx", "u")


def operand_order(axes: Sequence[str]) -> tuple[str, ...]:
    derivatives = [f"u_{axis * order}" for axis in axes for order in (1, 2, 3)]
    return (*axes, *derivatives, "u")


def _require_float64(name: str, array: FloatArray) -> None:
    if array.dtype != np.float64:
        raise ValueError(f"{name} must be float64, got {array.dtype}")


def _axis_values(name: str, axis: FloatArray) -> FloatArray:
    _require_float64(name, axis)
    if axis.ndim == 2 and axis.shape[1] == 1:
        return axis[:, 0]
    if axis.ndim == 1:
        return axis
    raise ValueError(f"{name} must be 1-D or a (n, 1) column, got {axis.shape}")


def _spatial_spacing(x: FloatArray, n_rows: int) -> float:
    values = _axis_values("x", x)
    if values.shape[0] != n_rows:
        raise ValueError(f"x has {values.shape[0]} entries but u has {n_rows} rows")
    if values.shape[0] < 3:
        raise ValueError("x needs at least 3 entries to read x[2] - x[1]")
    return float(values[2] - values[1])


def finite_diff(u: FloatArray, dx: float) -> FloatArray:
    _require_float64("u", u)
    n = u.size
    if n < 3:
        raise ValueError(f"u needs at least 3 points, got {n}")
    ux = np.zeros(n)
    ux[1: n - 1] = (u[2:n] - u[0: n - 2]) / (2 * dx)
    ux[0] = (-3.0 / 2 * u[0] + 2 * u[1] - u[2] / 2) / dx
    ux[n - 1] = (3.0 / 2 * u[n - 1] - 2 * u[n - 2] + u[n - 3] / 2) / dx
    return ux


def diff(u: FloatArray, x: FloatArray) -> FloatArray:
    _require_float64("u", u)
    if u.ndim != 2:
        raise ValueError(f"u must be 2-D (space, time), got {u.ndim}-D")
    n, _m = u.shape
    dx = _spatial_spacing(x, n)
    if n < 3:
        raise ValueError(f"u needs at least 3 rows, got {n}")
    ux = np.zeros_like(u)
    ux[1: n - 1,:] = (u[2:n,:] - u[0: n - 2,:]) / (2 * dx)
    ux[0,:] = (-3.0 / 2 * u[0,:] + 2 * u[1,:] - u[2,:] / 2) / dx
    ux[n - 1,:] = (3.0 / 2 * u[n - 1,:] - 2 * u[n - 2,:] + u[n - 3,:] / 2) / dx
    return ux


def diff2(u: FloatArray, x: FloatArray) -> FloatArray:
    _require_float64("u", u)
    if u.ndim != 2:
        raise ValueError(f"u must be 2-D (space, time), got {u.ndim}-D")
    n, _m = u.shape
    dx = _spatial_spacing(x, n)
    if n < 4:
        raise ValueError(f"u needs at least 4 rows, got {n}")
    uxx = np.zeros_like(u)
    uxx[1: n - 1,:] = (u[2:n,:] - 2 * u[1: n - 1,:] + u[0: n - 2,:]) / dx**2
    uxx[0,:] = (2 * u[0,:] - 5 * u[1,:] + 4 * u[2,:] - u[3,:]) / dx**2
    uxx[n - 1,:] = (
        2 * u[n - 1,:] - 5 * u[n - 2,:] + 4 * u[n - 3,:] - u[n - 4,:]
    ) / dx**2
    return uxx


def diff3(u: FloatArray, x: FloatArray) -> FloatArray:
    return diff(diff2(u, x), x)


def build_operand_columns(
    u: FloatArray,
    x: FloatArray,
    t: FloatArray,
) -> tuple[FloatArray, dict[str, FloatArray]]:
    _require_float64("u", u)
    if u.ndim != 2:
        raise ValueError(f"u must be 2-D (space, time), got {u.ndim}-D")
    n, m = u.shape
    x_values = _axis_values("x", x)
    t_values = _axis_values("t", t)
    if x_values.shape[0] != n:
        raise ValueError(f"x has {x_values.shape[0]} entries for {n} rows")
    if t_values.shape[0] != m:
        raise ValueError(f"t has {t_values.shape[0]} entries for {m} columns")
    if n < 4:
        raise ValueError(f"u needs at least 4 spatial rows, got {n}")
    if m < 3:
        raise ValueError(f"u needs at least 3 time columns, got {m}")


    dt = float(t_values[1] - t_values[0])
    ut = np.zeros((n, m))
    for idx in range(n):
        ut[idx,:] = finite_diff(u[idx,:], dt)

    u_x = diff(u, x_values)
    u_xx = diff2(u, x_values)
    u_xxx = diff3(u, x_values)

    features: dict[str, FloatArray] = {}
    features["x"] = np.repeat(x_values.reshape(-1, 1), m, axis=1).reshape(-1)
    features["u_x"] = u_x.reshape(-1)
    features["u_xx"] = u_xx.reshape(-1)
    features["u_xxx"] = u_xxx.reshape(-1)
    features["u"] = u.reshape(-1)
    return ut.reshape(-1, 1), features


def _along_axis(
    template: Callable[[FloatArray, FloatArray], FloatArray],
    u: FloatArray,
    values: FloatArray,
    axis: int,
) -> FloatArray:
    moved = np.moveaxis(u, axis, 0)
    flat = np.ascontiguousarray(moved).reshape(moved.shape[0], -1)
    return np.moveaxis(template(flat, values).reshape(moved.shape), 0, axis)


def _time_derivative(u: FloatArray, t: FloatArray) -> FloatArray:
    t_values = _axis_values("t", t)
    m = u.shape[-1]
    if t_values.shape[0] != m:
        raise ValueError(f"t has {t_values.shape[0]} entries for {m} columns")
    if m < 3:
        raise ValueError(f"u needs at least 3 time columns, got {m}")

    dt = float(t_values[1] - t_values[0])
    ut = np.zeros_like(u)
    flat_u = u.reshape(-1, m)
    flat_ut = ut.reshape(-1, m)
    for idx in range(flat_u.shape[0]):
        flat_ut[idx,:] = finite_diff(flat_u[idx,:], dt)
    return ut


def _axis_columns(
    u: FloatArray, index: int, name: str, values: FloatArray
) -> dict[str, FloatArray]:
    if values.shape[0] != u.shape[index]:
        raise ValueError(
            f"{name} has {values.shape[0]} entries for {u.shape[index]} rows"
        )
    shape = [1] * u.ndim
    shape[index] = values.shape[0]
    columns = {name: np.broadcast_to(values.reshape(shape), u.shape).reshape(-1)}
    for order, template in ((1, diff), (2, diff2), (3, diff3)):
        column = _along_axis(template, u, values, index)
        columns[f"u_{name * order}"] = column.reshape(-1)
    return columns


def build_operand_columns_axes(
    u: FloatArray,
    axes: Mapping[str, FloatArray],
    t: FloatArray,
) -> tuple[FloatArray, dict[str, FloatArray]]:
    names = tuple(axes)
    if names == ("x",):
        return build_operand_columns(u, axes["x"], t)
    if len(names) == 1:
        lhs, columns = build_operand_columns(u, axes[names[0]], t)
        return lhs, dict(zip(operand_order(names), columns.values(), strict=True))
    _require_float64("u", u)
    if u.ndim != len(names) + 1:
        raise ValueError(
            f"u must be {len(names) + 1}-D for axes {names}, got {u.ndim}-D"
        )
    u = np.ascontiguousarray(u)
    ut = _time_derivative(u, t)
    coordinates: dict[str, FloatArray] = {}
    derivatives: dict[str, FloatArray] = {}
    for index, name in enumerate(names):
        columns = _axis_columns(u, index, name, _axis_values(name, axes[name]))
        coordinates[name] = columns.pop(name)
        derivatives.update(columns)
    features = {**coordinates, **derivatives, "u": u.reshape(-1)}
    return ut.reshape(-1, 1), features
