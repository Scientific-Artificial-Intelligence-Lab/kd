
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _slice_nd_to_2d(
    field: NDArray[np.floating],
    axes: tuple[int, int],
) -> NDArray[np.floating]:
    ndim = field.ndim
    ax0, ax1 = axes


    if ax0 == ax1:
        raise ValueError(f"axes must be distinct, got ({ax0}, {ax1})")
    for ax in (ax0, ax1):
        if ax < 0 or ax >= ndim:
            raise ValueError(f"Axis {ax} out of range for {ndim}-D array")


    if ndim == 2:
        return field


    keep = {ax0, ax1}
    idx: list[int | slice] = []
    for i in range(ndim):
        if i in keep:
            idx.append(slice(None))
        else:
            idx.append(field.shape[i] // 2)

    return field[tuple(idx)]


def _pick_time_steps(n_t: int, n: int = 3) -> list[int]:
    if n_t < 1:
        raise ValueError(f"n_t must be >= 1, got {n_t}")

    actual_n = min(n, n_t)

    if actual_n == 1:
        return [0]


    indices = np.linspace(0, n_t - 1, actual_n)
    result = sorted(set(int(round(x)) for x in indices))
    return result
