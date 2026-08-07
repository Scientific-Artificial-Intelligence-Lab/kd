
from __future__ import annotations

import torch

from kd.data.schema import PDEDataset
from kd.data.synthetic._xu2020_common import (
    add_relative_noise,
    build_dataset,
    make_axes,
)


def generate_wave_xu2020_data(
    nx: int = 256,
    nt: int = 101,
    noise_level: float = 0.0,
    device: torch.device | None = None,
    seed: int | None = None,
) -> PDEDataset:
    x, t, _dtype, _device = make_axes(nx, nt, device=device)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(xg) * torch.cos(tg) + 0.5 * torch.sin(2.0 * xg) * torch.cos(2.0 * tg)
    u = add_relative_noise(u, noise_level, seed=seed)
    return build_dataset(
        name="xu2020-wave",
        x=x,
        t=t,
        u=u,
        noise_level=noise_level,
        ground_truth="u_tt = u_xx",
        lhs_order=2,
    )
