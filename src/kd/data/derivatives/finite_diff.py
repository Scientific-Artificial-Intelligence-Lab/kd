
from __future__ import annotations

import math

import numpy as np
import torch

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
        is_periodic = self._is_periodic[axis]

        return central_diff(expression, dx, axis_idx, order, is_periodic=is_periodic)

    def available_derivatives(self) -> list[tuple[str, str, int]]:
        return list(self._cache.keys())
