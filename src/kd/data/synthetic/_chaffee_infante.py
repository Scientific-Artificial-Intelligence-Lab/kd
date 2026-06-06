
from __future__ import annotations

import torch

from kd.data.schema import PDEDataset
from kd.data.synthetic._xu2020_common import (
    add_relative_noise,
    build_dataset,
    make_axes,
)

_CUBE_CLAMP = 1e3


def generate_chaffee_infante_xu2020_data(
    nx: int = 256,
    nt: int = 101,
    noise_level: float = 0.0,
    device: torch.device | None = None,
    seed: int | None = None,
) -> PDEDataset:
    x, t, dtype, device = make_axes(nx, nt, device=device)
    u = _solve_chaffee_infante(x, t, dtype=dtype, device=device)
    u = add_relative_noise(u, noise_level, seed=seed)
    return build_dataset(
        name="xu2020-chaffee-infante",
        x=x,
        t=t,
        u=u,
        noise_level=noise_level,
        ground_truth="u_t = u_xx + u^3 - u",
    )


def _solve_chaffee_infante(
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    nx = x.numel()
    nt = t.numel()
    dx = float((x[1] - x[0]).item())
    k = torch.fft.fftfreq(nx, d=dx, device=device) * 2.0 * torch.pi
    u = 0.5 * torch.sin(x)
    out = torch.zeros(nx, nt, dtype=dtype, device=device)
    out[:, 0] = u
    if nt == 1:
        return out






    cfl_bound = 0.14 * dx * dx
    current = float(t[0].item())
    for j in range(1, nt):
        target = float(t[j].item())
        dt_base = min((target - current) / 10.0, 0.001, cfl_bound)
        while current < target - 1e-14:
            dt = min(dt_base, target - current)
            u = _rk4_step(u, k, dt)
            current += dt
        out[:, j] = u
    return out


def _rk4_step(u: torch.Tensor, k: torch.Tensor, dt: float) -> torch.Tensor:
    k1 = _rhs(u, k)
    k2 = _rhs(u + 0.5 * dt * k1, k)
    k3 = _rhs(u + 0.5 * dt * k2, k)
    k4 = _rhs(u + dt * k3, k)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _rhs(u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    u_hat = torch.fft.fft(u)
    uxx = torch.fft.ifft((1j * k) ** 2 * u_hat).real
    u_safe = torch.clamp(u, min=-_CUBE_CLAMP, max=_CUBE_CLAMP)
    result: torch.Tensor = uxx + u_safe * u_safe * u_safe - u
    return result
