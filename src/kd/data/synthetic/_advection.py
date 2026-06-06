"""Synthetic advection equation data generation.

Equation: u_t + c1*u_x [+ c2*u_y [+ c3*u_z]] = 0
Analytic solution: u = prod_i(sin(k_i * (x_i - c_i * t)))

Supports N-dimensional advection (1D, 2D, 3D, ...).
"""

from __future__ import annotations

import math

import torch

from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd.data.synthetic._common import (
    AXIS_T as _AXIS_T,
)
from kd.data.synthetic._common import (
    FIELD_U as _FIELD_U,
)
from kd.data.synthetic._common import (
    SPATIAL_AXIS_NAMES as _SPATIAL_AXIS_NAMES,
)
from kd.data.synthetic._common import (
    broadcast_grids as _broadcast_grids,
)


_SPATIAL_MIN = 0.0
_SPATIAL_MAX = 2.0 * math.pi
_T_MIN = 0.0
_T_MAX = 1.0


def generate_advection_data(
    speeds: tuple[float, ...],
    waves: tuple[float, ...],
    grid_sizes: tuple[int, ...],
    nt: int,
    noise_level: float = 0.0,
    device: torch.device | None = None,
    seed: int | None = None,
) -> PDEDataset:
    """Generate synthetic advection equation data.

    Equation: u_t + c1*u_x [+ c2*u_y] = 0
    Analytic solution: u = prod_i(sin(k_i * (x_i - c_i * t)))

    The number of spatial dimensions is determined by len(speeds),
    which must equal len(waves) and len(grid_sizes).

    Args:
        speeds: Advection speeds for each spatial dimension.
        waves: Wave numbers for each spatial dimension.
        grid_sizes: Number of grid points for each spatial dimension.
        nt: Number of temporal grid points.
        noise_level: Standard deviation of Gaussian noise (default: 0.0).
        device: Target device for tensors (default: CPU).
        seed: Random seed for reproducibility (default: None).

    Returns:
        PDEDataset with analytic advection solution.

    Raises:
        ValueError: If parameters are invalid or inconsistent.
    """
    _validate_advection_params(speeds, waves, grid_sizes, nt, noise_level)

    if device is None:
        device = torch.device("cpu")
    dtype = torch.float32 if device.type == "mps" else torch.float64

    if seed is not None:
        torch.manual_seed(seed)

    ndim = len(speeds)
    spatial_names = list(_SPATIAL_AXIS_NAMES[:ndim])


    spatial_coords = [
        torch.linspace(_SPATIAL_MIN, _SPATIAL_MAX, gs + 1, dtype=dtype, device=device)[
            :-1
        ]
        for gs in grid_sizes
    ]
    t_coord = torch.linspace(_T_MIN, _T_MAX, nt, dtype=dtype, device=device)



    all_coords = spatial_coords + [t_coord]
    grids = _broadcast_grids(all_coords)


    t_grid = grids[-1]
    u = torch.ones_like(t_grid)
    for i in range(ndim):
        u = u * torch.sin(waves[i] * (grids[i] - speeds[i] * t_grid))


    if noise_level > 0.0:
        u = u + torch.randn_like(u) * noise_level


    axis_order = spatial_names + [_AXIS_T]
    axes: dict[str, AxisInfo] = {}
    for i, name in enumerate(spatial_names):
        axes[name] = AxisInfo(name=name, values=spatial_coords[i], is_periodic=True)
    axes[_AXIS_T] = AxisInfo(name=_AXIS_T, values=t_coord, is_periodic=False)


    ground_truth = _build_advection_ground_truth(speeds, spatial_names)


    name = f"advection-{ndim}d"

    return PDEDataset(
        name=name,
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes=axes,
        axis_order=axis_order,
        fields={_FIELD_U: FieldData(name=_FIELD_U, values=u)},
        lhs_field=_FIELD_U,
        lhs_axis=_AXIS_T,
        noise_level=noise_level,
        ground_truth=ground_truth,
    )


def _validate_advection_params(
    speeds: tuple[float, ...],
    waves: tuple[float, ...],
    grid_sizes: tuple[int, ...],
    nt: int,
    noise_level: float,
) -> None:
    """Validate advection generator parameters."""
    if len(speeds) != len(waves):
        raise ValueError(
            f"speeds and waves must have same length, "
            f"got {len(speeds)} and {len(waves)}"
        )
    if len(speeds) != len(grid_sizes):
        raise ValueError(
            f"speeds and grid_sizes must have same length, "
            f"got {len(speeds)} and {len(grid_sizes)}"
        )
    if len(speeds) == 0:
        raise ValueError("At least one spatial dimension is required")

    for i, s in enumerate(speeds):
        if not math.isfinite(s):
            raise ValueError(f"speeds[{i}] must be finite, got {s}")
    for i, w in enumerate(waves):
        if not math.isfinite(w):
            raise ValueError(f"waves[{i}] must be finite, got {w}")
    for i, gs in enumerate(grid_sizes):
        if gs <= 0:
            raise ValueError(f"grid_sizes[{i}] must be positive, got {gs}")
    if nt <= 0:
        raise ValueError(f"nt must be positive, got {nt}")
    if not math.isfinite(noise_level):
        raise ValueError(f"noise_level must be finite, got {noise_level}")
    if noise_level < 0:
        raise ValueError(f"noise_level must be non-negative, got {noise_level}")


def _build_advection_ground_truth(
    speeds: tuple[float, ...],
    spatial_names: list[str],
) -> str:
    """Build human-readable ground truth equation string.

    Examples:
        1D: "u_t + 1.7 * u_x = 0"
        2D: "u_t + 1.1 * u_x + -0.7 * u_y = 0"
    """
    terms = ["u_t"]
    for speed, name in zip(speeds, spatial_names, strict=True):
        terms.append(f"{speed} * u_{name}")
    return " + ".join(terms) + " = 0"
