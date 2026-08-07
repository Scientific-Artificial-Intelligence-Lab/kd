
from __future__ import annotations

import math

import torch

from kd.data.noise import xu2020_relative
from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType

_FIELD_U = "u"
_AXIS_X = "x"
_AXIS_T = "t"
_X_MIN = 0.0
_X_MAX = 2.0 * math.pi
_T_MIN = 0.0
_T_MAX = 1.0


def make_axes(
    nx: int,
    nt: int,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.dtype, torch.device]:
    if nx < 5:
        raise ValueError(f"nx must be >= 5, got {nx}")
    if nt < 2:
        raise ValueError(f"nt must be >= 2, got {nt}")
    if device is None:
        device = torch.device("cpu")
    dtype = torch.float32 if device.type == "mps" else torch.float64
    x = torch.linspace(_X_MIN, _X_MAX, nx + 1, dtype=dtype, device=device)[:-1]
    t = torch.linspace(_T_MIN, _T_MAX, nt, dtype=dtype, device=device)
    return x, t, dtype, device


def add_relative_noise(
    u: torch.Tensor,
    noise_level: float,
    *,
    seed: int | None,
) -> torch.Tensor:
    return xu2020_relative(u, noise_level, seed=seed)


def build_dataset(
    *,
    name: str,
    x: torch.Tensor,
    t: torch.Tensor,
    u: torch.Tensor,
    noise_level: float,
    ground_truth: str,
    lhs_order: int = 1,
) -> PDEDataset:
    return PDEDataset(
        name=name,
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            _AXIS_X: AxisInfo(_AXIS_X, x, is_periodic=True),
            _AXIS_T: AxisInfo(_AXIS_T, t, is_periodic=False),
        },
        axis_order=[_AXIS_X, _AXIS_T],
        fields={_FIELD_U: FieldData(_FIELD_U, u)},
        lhs_field=_FIELD_U,
        lhs_axis=_AXIS_T,
        lhs_order=lhs_order,
        noise_level=noise_level,
        ground_truth=ground_truth,
    )
