
from __future__ import annotations

import math

import torch

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
    if noise_level < 0.0 or not math.isfinite(noise_level):
        raise ValueError(
            f"noise_level must be finite and non-negative, got {noise_level}"
        )
    if noise_level == 0.0:
        return u
    generator = None
    if seed is not None:
        generator = torch.Generator(device=u.device).manual_seed(seed)
    noise = torch.randn(
        u.shape,
        dtype=u.dtype,
        device=u.device,
        generator=generator,
    )
    signal_std = u.std(correction=0)
    if float(signal_std.item()) == 0.0:
        raise ValueError(
            "signal has zero std; relative noise is undefined "
            "(use absolute noise scale instead)"
        )
    noise_std = noise.std(correction=0)
    if float(noise_std.item()) == 0.0:
        return u
    noise = noise / noise_std * (noise_level * signal_std)
    return u + noise


def build_dataset(
    *,
    name: str,
    x: torch.Tensor,
    t: torch.Tensor,
    u: torch.Tensor,
    noise_level: float,
    ground_truth: str,
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
        noise_level=noise_level,
        ground_truth=ground_truth,
    )
