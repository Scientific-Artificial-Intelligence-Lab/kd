
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from kd.data.schema import PDEDataset


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


def _pick_animation_frames(n_t: int, max_frames: int = 24) -> list[int]:
    if max_frames < 1:
        raise ValueError(f"max_frames must be >= 1, got {max_frames}")
    return _pick_time_steps(n_t, max_frames)


def _imshow_extent_for_spatial_axes(
    dataset: PDEDataset,
    spatial_axes: list[str],
) -> tuple[tuple[float, float, float, float], str, str]:
    vertical_axis = spatial_axes[0]
    horizontal_axis = spatial_axes[1]
    h_coords = np.asarray(
        dataset.get_coords(horizontal_axis).detach().cpu().numpy(),
        dtype=np.float64,
    )
    v_coords = np.asarray(
        dataset.get_coords(vertical_axis).detach().cpu().numpy(),
        dtype=np.float64,
    )
    extent = (
        float(h_coords[0]),
        float(h_coords[-1]),
        float(v_coords[0]),
        float(v_coords[-1]),
    )
    return extent, horizontal_axis, vertical_axis
