
from __future__ import annotations

import math
from typing import Final

import numpy as np
import torch

from kd.data.schema import PDEDataset
from kd.search.eqgpt.config import EqGPTConfig

DISK_R_RANGE: Final[tuple[float, float]] = (0.5, 1.45)
DISK_N_R: Final[int] = 100
DISK_N_THETA: Final[int] = 100




_MAX_LATTICE_CELLS: Final[int] = 16_000_000

_REFERENCE_GRIDS: Final[dict[str, tuple[np.ndarray, np.ndarray]]] = {
    "eqgpt-laplacian-eitech": (
        np.arange(4.3, 141.0 + 136.7 / 800, 136.7 / 800),
        np.arange(19.6, 54.6 + 35.0 / 200, 35.0 / 200),
    ),
    "eqgpt-laplacian-smile": (
        np.arange(-4.0, 4.0 + 8.0 / 250, 8.0 / 250),
        np.arange(-4.0, 4.0 + 8.0 / 250, 8.0 / 250),
    ),
}


def build_steady_eval_dataset(
    dataset: PDEDataset, config: EqGPTConfig
) -> PDEDataset:
    if config.steady_polar_eval:
        return _polar_eval_dataset(dataset)
    delete_num = config.steady_boundary_delete_num
    if delete_num is not None:
        return _corroded_eval_dataset(dataset, int(delete_num))
    return PDEDataset.from_scatter(
        coords={axis: dataset.get_coords(axis) for axis in dataset.axis_order or []},
        fields={"u": dataset.get_field("u")},
        lhs="",
        name=f"{dataset.name}-steady-eval",
    )


def _polar_eval_dataset(dataset: PDEDataset) -> PDEDataset:
    radii = torch.linspace(*DISK_R_RANGE, DISK_N_R, dtype=torch.float32)
    theta = torch.linspace(0.0, 2.0 * math.pi, DISK_N_THETA, dtype=torch.float32)
    r_flat = radii.repeat_interleave(DISK_N_THETA)
    theta_flat = theta.repeat(DISK_N_R)
    x = r_flat * torch.cos(theta_flat)
    y = r_flat * torch.sin(theta_flat)
    return PDEDataset.from_scatter(
        coords={"x": x, "y": y},
        fields={"u": torch.zeros_like(x)},
        lhs="",
        name=f"{dataset.name}-steady-polar-eval",
    )


def _corroded_eval_dataset(dataset: PDEDataset, delete_num: int) -> PDEDataset:
    x = dataset.get_coords("x").detach().cpu().numpy()
    y = dataset.get_coords("y").detach().cpu().numpy()
    u = dataset.get_field("u").detach().cpu().numpy()
    x_grid, y_grid = _corrosion_grid(dataset.name, x, y)
    x_idx = _lattice_indices(x, x_grid)
    y_idx = _lattice_indices(y, y_grid)
    n_x, n_y = x_grid.size, y_grid.size
    if n_x * n_y > _MAX_LATTICE_CELLS:
        raise ValueError(
            f"steady boundary corrosion: reconstructed lattice {n_x}x{n_y} "
            f"({n_x * n_y} cells) exceeds {_MAX_LATTICE_CELLS} -- the axis "
            "coordinates are not a clean ordered grid (the data is genuinely "
            "scattered or exposes too many coordinate levels). "
            "Corrosion requires gridded steady data."
        )
    source_index = np.full((n_x, n_y), -1, dtype=np.int64)
    source_index[x_idx, y_idx] = np.arange(x.size, dtype=np.int64)
    occupancy = (source_index >= 0).astype(np.int64)
    prefix = np.pad(occupancy, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    kept: list[tuple[int, int, int]] = []
    window_area = (2 * delete_num) ** 2
    for index in np.lexsort((y_idx, x_idx)):
        i, j = int(x_idx[index]), int(y_idx[index])
        if not (delete_num <= i < n_x - delete_num):
            continue
        if not (delete_num <= j < n_y - delete_num):
            continue
        lo_i, hi_i = i - delete_num, i + delete_num
        lo_j, hi_j = j - delete_num, j + delete_num
        occupied = (
            prefix[hi_i, hi_j]
            - prefix[lo_i, hi_j]
            - prefix[hi_i, lo_j]
            + prefix[lo_i, lo_j]
        )
        if int(occupied) == window_area:
            kept.append((i, j, int(index)))
    if not kept:
        raise ValueError("steady boundary corrosion removed every evaluation point")
    return PDEDataset.from_scatter(
        coords={
            "x": np.asarray([x_grid[i] for i, _j, _index in kept]),
            "y": np.asarray([y_grid[j] for _i, j, _index in kept]),
        },
        fields={"u": np.asarray([u[index] for _i, _j, index in kept])},
        lhs="",
        name=f"{dataset.name}-steady-corroded-eval",
    )


def _corrosion_grid(
    dataset_name: str, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    reference = _REFERENCE_GRIDS.get(dataset_name)
    if reference is not None:
        return reference
    return np.unique(x), np.unique(y)


def _lattice_indices(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if grid.size < 2:
        raise ValueError("steady boundary corrosion needs at least two axis values")
    spacing = float(grid[1] - grid[0])
    indices: np.ndarray = np.asarray(
        np.rint((values - grid[0]) / spacing), dtype=np.int64
    )
    if (indices < 0).any() or (indices >= grid.size).any():
        raise ValueError("steady boundary corrosion coordinates fall outside its grid")
    tolerance = abs(spacing) * 1.0e-5
    if not np.allclose(values, grid[indices], rtol=1.0e-7, atol=tolerance):
        raise ValueError("steady boundary corrosion coordinates are off its grid")
    return indices


__all__ = [
    "DISK_N_R",
    "DISK_N_THETA",
    "DISK_R_RANGE",
    "build_steady_eval_dataset",
]
