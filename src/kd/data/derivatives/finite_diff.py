
from __future__ import annotations

import math

import numpy as np
import torch
from numpy.typing import NDArray

from kd.data.derivatives.base import DerivativeProvider
from kd.data.schema import DataTopology, PDEDataset


_MIN_DX = 1e-15





MAX_SUPPORTED_ORDER = 3


_MIN_POINTS_FOR_ORDER = {
    1: 5,
    2: 5,
    3: 5,
}


assert set(_MIN_POINTS_FOR_ORDER) == set(range(1, MAX_SUPPORTED_ORDER + 1))






UNIFORM_GRID_RTOL = 1e-4




DX_ZERO_FLOOR = 1e-30


def coordinate_diff(
    f: torch.Tensor,
    coords: torch.Tensor,
    axis: int,
    order: int,
    *,
    is_periodic: bool = False,
) -> torch.Tensor:
    _validate_coordinate_diff(f, coords, axis, order)
    if is_uniform_grid(coords, rtol=UNIFORM_GRID_RTOL):
        dx = (coords[1:] - coords[:-1])[0].item()
        return central_diff(f, dx, axis, order, is_periodic=is_periodic)
    if is_periodic:
        raise ValueError(
            "Periodic derivatives require a uniform grid, got non-uniform coords"
        )
    indices, counts = _coordinate_stencils(f.shape[axis], order)
    nodes = coords.detach().cpu().to(torch.float64).numpy()
    weights = _coordinate_weights(nodes, indices, counts, order)
    return _apply_coordinate_weights(f, axis, indices, weights)


def _validate_coordinate_diff(
    f: torch.Tensor, coords: torch.Tensor, axis: int, order: int
) -> None:
    if order < 1 or order > MAX_SUPPORTED_ORDER:
        raise ValueError(f"order must be in [1, {MAX_SUPPORTED_ORDER}], got {order}")
    if axis < 0 or axis >= f.dim():
        raise ValueError(f"axis {axis} out of range for {f.dim()}D tensor")
    if torch.isnan(f).any():
        raise ValueError("Input tensor contains NaN values")
    if torch.isinf(f).any():
        raise ValueError("Input tensor contains Inf values")
    n_points = f.shape[axis]
    min_points = _MIN_POINTS_FOR_ORDER[order]
    if n_points < min_points:
        raise ValueError(
            f"Order {order} derivative requires at least {min_points} points, "
            f"got {n_points}"
        )
    if coords.ndim != 1:
        raise ValueError("coords must be 1-D")
    if not torch.isfinite(coords).all():
        raise ValueError("coords must be finite")
    if not torch.all(coords[1:] > coords[:-1]):
        raise ValueError("coords must be strictly increasing")
    if coords.numel() != n_points:
        raise ValueError(
            f"coords length {coords.numel()} must equal f.shape[{axis}]={n_points}"
        )


def _coordinate_stencils(
    n: int, order: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    indices = np.repeat(np.arange(n, dtype=np.int64)[:, None], 5, axis=1)
    indices[2:-2] = np.arange(n - 4)[:, None] + np.arange(5)
    counts = np.full(n, 5, dtype=np.int64)
    if order < 3:
        indices[1, :3] = np.arange(3)
        indices[-2, :3] = np.arange(n - 3, n)
        counts[[1, -2]] = 3
        width = 2 if order == 1 else 4
    else:
        indices[1, :4] = np.arange(1, 5)
        indices[-2, :4] = np.arange(n - 5, n - 1)
        counts[[1, -2]] = 4
        width = 4
    indices[0, :width] = np.arange(width)
    indices[-1, :width] = np.arange(n - width, n)
    counts[[0, -1]] = width
    return indices, counts


def _fornberg_weights(nodes: NDArray[np.float64], order: int) -> NDArray[np.float64]:
    n_rows, width = nodes.shape
    coefficients = np.zeros((n_rows, width, order + 1), dtype=np.float64)
    coefficients[:, 0, 0] = 1.0
    previous_product = np.ones(n_rows, dtype=np.float64)
    offset = nodes[:, 0]
    for i in range(1, width):
        max_order = min(i, order)
        product = np.ones(n_rows, dtype=np.float64)
        previous_offset, offset = offset, nodes[:, i]
        for j in range(i):
            separation = nodes[:, i] - nodes[:, j]
            product *= separation
            if j == i - 1:
                for k in range(max_order, 0, -1):
                    coefficients[:, i, k] = (
                        previous_product
                        * (
                            k * coefficients[:, i - 1, k - 1]
                            - previous_offset * coefficients[:, i - 1, k]
                        )
                        / product
                    )
                coefficients[:, i, 0] = (
                    -previous_product
                    * previous_offset
                    * coefficients[:, i - 1, 0]
                    / product
                )
            for k in range(max_order, 0, -1):
                coefficients[:, j, k] = (
                    offset * coefficients[:, j, k] - k * coefficients[:, j, k - 1]
                ) / separation
            coefficients[:, j, 0] *= offset / separation
        previous_product = product
    return coefficients[:, :, order]


def _coordinate_weights(
    coords: NDArray[np.float64],
    indices: NDArray[np.int64],
    counts: NDArray[np.int64],
    order: int,
) -> NDArray[np.float64]:
    weights = np.zeros(indices.shape, dtype=np.float64)
    for width in np.unique(counts):
        rows = counts == width
        offsets = coords[indices[rows, :width]] - coords[rows, None]

        spacing = np.max(np.diff(offsets, axis=1), axis=1)
        local_weights = _fornberg_weights(offsets / spacing[:, None], order)
        for _ in range(order):
            local_weights /= spacing[:, None]
        weights[rows, :width] = local_weights
    return weights


def _apply_coordinate_weights(
    f: torch.Tensor,
    axis: int,
    indices: NDArray[np.int64],
    weights: NDArray[np.float64],
) -> torch.Tensor:
    f_moved = f.movedim(axis, 0)
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=f.device)
    weight_tensor = torch.as_tensor(weights, dtype=f.dtype, device=f.device)

    samples = f_moved[index_tensor] - f_moved[:, None]
    result = torch.einsum("nk,nk...->n...", weight_tensor, samples)
    return result.movedim(0, axis)


def central_diff(
    f: torch.Tensor,
    dx: float,
    axis: int,
    order: int,
    is_periodic: bool = False,
) -> torch.Tensor:

    if math.isinf(dx):
        raise ValueError(f"dx must be finite, got {dx}")
    if dx <= 0:
        raise ValueError(f"dx must be positive, got {dx}")
    if dx < _MIN_DX:
        raise ValueError(f"dx is too small ({dx}), must be >= {_MIN_DX}")


    if order < 1 or order > MAX_SUPPORTED_ORDER:
        raise ValueError(f"order must be in [1, {MAX_SUPPORTED_ORDER}], got {order}")


    if axis < 0 or axis >= f.dim():
        raise ValueError(f"axis {axis} out of range for {f.dim()}D tensor")


    if torch.isnan(f).any():
        raise ValueError("Input tensor contains NaN values")
    if torch.isinf(f).any():
        raise ValueError("Input tensor contains Inf values")


    n_points = f.shape[axis]
    min_points = _MIN_POINTS_FOR_ORDER[order]
    if n_points < min_points:
        raise ValueError(
            f"Order {order} derivative requires at least {min_points} points, "
            f"got {n_points}"
        )


    f_moved = f.movedim(axis, 0)

    if is_periodic:
        return _central_diff_periodic(f_moved, dx, order).movedim(0, axis)


    result_moved = torch.zeros_like(f_moved)

    if order == 1:


        result_moved[2:-2] = (
            -f_moved[4:] + 8 * f_moved[3:-1] - 8 * f_moved[1:-3] + f_moved[:-4]
        ) / (12 * dx)



        result_moved[1] = (f_moved[2] - f_moved[0]) / (2 * dx)
        result_moved[-2] = (f_moved[-1] - f_moved[-3]) / (2 * dx)


        result_moved[0] = (f_moved[1] - f_moved[0]) / dx
        result_moved[-1] = (f_moved[-1] - f_moved[-2]) / dx

    elif order == 2:


        result_moved[2:-2] = (
            -f_moved[4:]
            + 16 * f_moved[3:-1]
            - 30 * f_moved[2:-2]
            + 16 * f_moved[1:-3]
            - f_moved[:-4]
        ) / (12 * dx**2)



        result_moved[1] = (f_moved[2] - 2 * f_moved[1] + f_moved[0]) / (dx**2)
        result_moved[-2] = (f_moved[-1] - 2 * f_moved[-2] + f_moved[-3]) / (dx**2)



        result_moved[0] = (
            2 * f_moved[0] - 5 * f_moved[1] + 4 * f_moved[2] - f_moved[3]
        ) / (dx**2)

        result_moved[-1] = (
            2 * f_moved[-1] - 5 * f_moved[-2] + 4 * f_moved[-3] - f_moved[-4]
        ) / (dx**2)

    elif order == 3:



        result_moved[2:-2] = (
            f_moved[4:] - 2 * f_moved[3:-1] + 2 * f_moved[1:-3] - f_moved[:-4]
        ) / (2 * dx**3)




        result_moved[0] = (
            -f_moved[0] + 3 * f_moved[1] - 3 * f_moved[2] + f_moved[3]
        ) / (dx**3)
        result_moved[1] = (
            -f_moved[1] + 3 * f_moved[2] - 3 * f_moved[3] + f_moved[4]
        ) / (dx**3)



        result_moved[-1] = (
            f_moved[-1] - 3 * f_moved[-2] + 3 * f_moved[-3] - f_moved[-4]
        ) / (dx**3)
        result_moved[-2] = (
            f_moved[-2] - 3 * f_moved[-3] + 3 * f_moved[-4] - f_moved[-5]
        ) / (dx**3)


    return result_moved.movedim(0, axis)


def _central_diff_periodic(
    f_moved: torch.Tensor,
    dx: float,
    order: int,
) -> torch.Tensor:
    pad = 2

    f_padded = torch.cat([f_moved[-pad:], f_moved, f_moved[:pad]], dim=0)

    if order == 1:
        result = (
            -f_padded[4:] + 8 * f_padded[3:-1] - 8 * f_padded[1:-3] + f_padded[:-4]
        ) / (12 * dx)
    elif order == 2:
        result = (
            -f_padded[4:]
            + 16 * f_padded[3:-1]
            - 30 * f_padded[2:-2]
            + 16 * f_padded[1:-3]
            - f_padded[:-4]
        ) / (12 * dx**2)
    else:
        result = (
            f_padded[4:] - 2 * f_padded[3:-1] + 2 * f_padded[1:-3] - f_padded[:-4]
        ) / (2 * dx**3)

    return result


def is_uniform_grid(
    coords: torch.Tensor | np.ndarray,
    rtol: float = UNIFORM_GRID_RTOL,
) -> bool:
    arr = (
        coords.detach().cpu().numpy()
        if isinstance(coords, torch.Tensor)
        else np.asarray(coords)
    )
    if arr.ndim != 1 or arr.size < 2:
        return False
    diffs = np.diff(arr.astype(np.float64))
    dx0 = float(diffs[0])



    if not np.isfinite(dx0):
        return False




    if dx0 <= 0:
        return False
    if dx0 < DX_ZERO_FLOOR:
        return False
    return bool(np.allclose(diffs, dx0, rtol=rtol, atol=0.0))


def _check_uniform_grid(values: torch.Tensor, axis_name: str) -> float:
    if values.numel() < 2:
        raise ValueError(f"Axis '{axis_name}' must have at least 2 points")

    diffs = values[1:] - values[:-1]
    dx = diffs[0].item()




    if not math.isfinite(dx):
        raise ValueError(
            f"Axis '{axis_name}' has non-finite spacing dx={dx}; "
            f"finite-difference stencils require finite dx."
        )




    if dx < 0:
        raise ValueError(
            f"Axis '{axis_name}' has decreasing spacing dx={dx:.6g}; "
            f"finite-difference stencils require monotonic increasing "
            f"coordinates (flip the array before fitting)."
        )




    if abs(dx) < DX_ZERO_FLOOR:
        raise ValueError(
            f"Axis '{axis_name}' has degenerate spacing dx={dx:.6g}; "
            f"finite-difference stencils require nonzero dx."
        )


    if not is_uniform_grid(values, rtol=UNIFORM_GRID_RTOL):



        max_dev = float((diffs - dx).abs().max().item())
        raise ValueError(
            f"Axis '{axis_name}' has non-uniform spacing "
            f"(dx[0]={dx:.6g}, max deviation={max_dev:.6g})"
        )

    return dx


class FiniteDiffProvider(DerivativeProvider):

    def __init__(
        self,
        dataset: PDEDataset,
        max_order: int = 3,
        accuracy: int = 4,
    ) -> None:

        if accuracy != 4:
            raise ValueError(
                f"Only accuracy=4 is supported currently, got {accuracy}. "
                f"Other accuracy levels will be added currently."
            )


        if dataset.topology != DataTopology.GRID:
            raise ValueError(
                f"FiniteDiffProvider requires Grid topology, "
                f"got {dataset.topology.value}"
            )


        if max_order < 1:
            raise ValueError(f"max_order must be >= 1, got {max_order}")
        if max_order > MAX_SUPPORTED_ORDER:
            raise ValueError(
                f"max_order must be <= {MAX_SUPPORTED_ORDER}, got {max_order}"
            )

        self._dataset = dataset
        self._max_order = max_order
        self._accuracy = accuracy


        self._dx: dict[str, float] = {}
        self._axis_indices: dict[str, int] = {}
        self._is_periodic: dict[str, bool] = {}

        if dataset.axes is None or dataset.axis_order is None:
            raise ValueError("Dataset must have axes and axis_order defined")

        for i, axis_name in enumerate(dataset.axis_order):
            axis_info = dataset.axes[axis_name]
            dx = _check_uniform_grid(axis_info.values, axis_name)
            self._dx[axis_name] = dx
            self._axis_indices[axis_name] = i
            self._is_periodic[axis_name] = axis_info.is_periodic


        self._cache: dict[tuple[str, str, int], torch.Tensor] = {}
        self._precompute_derivatives()

    @property
    def coords(self) -> dict[str, torch.Tensor]:
        if self._dataset.axes is None:
            return {}
        return {name: axis.values for name, axis in self._dataset.axes.items()}

    def _precompute_derivatives(self) -> None:
        if self._dataset.fields is None:
            return

        for field_name, field_data in self._dataset.fields.items():
            for axis_name in self._axis_indices:
                axis_idx = self._axis_indices[axis_name]
                dx = self._dx[axis_name]

                for order in range(1, self._max_order + 1):
                    periodic = self._is_periodic[axis_name]
                    deriv = central_diff(
                        field_data.values,
                        dx,
                        axis_idx,
                        order,
                        is_periodic=periodic,
                    )
                    self._cache[(field_name, axis_name, order)] = deriv

    def get_derivative(
        self,
        field: str,
        axis: str,
        order: int,
    ) -> torch.Tensor:

        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        if order > self._max_order:
            raise ValueError(f"order {order} exceeds max_order {self._max_order}")


        if self._dataset.fields is None or field not in self._dataset.fields:
            raise KeyError(f"Field '{field}' not found in dataset")


        if axis not in self._axis_indices:
            raise KeyError(f"Axis '{axis}' not found in dataset")


        key = (field, axis, order)
        return self._cache[key].clone()

    def diff(
        self,
        expression: torch.Tensor,
        axis: str,
        order: int,
        *,
        is_periodic: bool | None = None,
    ) -> torch.Tensor:
        if axis not in self._axis_indices:
            raise KeyError(f"Axis '{axis}' not found in dataset")

        axis_idx = self._axis_indices[axis]


        if self._dataset.axes is not None and axis in self._dataset.axes:
            expected_size = len(self._dataset.axes[axis].values)
            actual_size = (
                expression.shape[axis_idx] if axis_idx < expression.dim() else -1
            )
            if actual_size != expected_size:
                raise ValueError(
                    f"Expression shape[{axis_idx}]={actual_size} doesn't match "
                    f"grid axis '{axis}' size={expected_size}"
                )

        dx = self._dx[axis]
        stencil = self._is_periodic[axis] if is_periodic is None else is_periodic

        return central_diff(expression, dx, axis_idx, order, is_periodic=stencil)

    def available_derivatives(self) -> list[tuple[str, str, int]]:
        return list(self._cache.keys())
