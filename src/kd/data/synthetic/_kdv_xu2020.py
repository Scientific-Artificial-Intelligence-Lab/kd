
from __future__ import annotations

import torch

from kd.data.schema import PDEDataset
from kd.data.synthetic._xu2020_common import (
    add_relative_noise,
    build_dataset,
    make_axes,
)


def generate_kdv_xu2020_data(
    nx: int = 256,
    nt: int = 101,
    b: float = 0.0025,
    noise_level: float = 0.0,
    device: torch.device | None = None,
    seed: int | None = None,
) -> PDEDataset:
    if b <= 0.0:
        raise ValueError(f"b must be positive, got {b}")
    x, t, dtype, device = make_axes(nx, nt, device=device)
    u = _solve_kdv(x, t, b, dtype=dtype, device=device)
    u = add_relative_noise(u, noise_level, seed=seed)
    return build_dataset(
        name="xu2020-kdv",
        x=x,
        t=t,
        u=u,
        noise_level=noise_level,
        ground_truth=f"u_t = -u*u_x - {b}*u_xxx",
    )


def _solve_kdv(
    x: torch.Tensor,
    t: torch.Tensor,
    b: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    nx = x.numel()
    nt = t.numel()
    dx = float((x[1] - x[0]).item())
    k = torch.fft.fftfreq(nx, d=dx, device=device) * 2.0 * torch.pi
    u = 0.25 * torch.cos(x) + 0.10 * torch.cos(2.0 * x)
    out = torch.zeros(nx, nt, dtype=dtype, device=device)
    out[:, 0] = u
    if nt == 1:
        return out







    cfl_bound = 0.045 * dx**3 / b
    current = float(t[0].item())
    for j in range(1, nt):
        target = float(t[j].item())
        dt_base = min((target - current) / 5.0, 0.002, cfl_bound)
        while current < target - 1e-14:
            dt = min(dt_base, target - current)
            u = _rk4_step(u, k, b, dt)
            current += dt
        out[:, j] = u
    return out


def _rk4_step(u: torch.Tensor, k: torch.Tensor, b: float, dt: float) -> torch.Tensor:
    k1 = _rhs(u, k, b)
    k2 = _rhs(u + 0.5 * dt * k1, k, b)
    k3 = _rhs(u + 0.5 * dt * k2, k, b)
    k4 = _rhs(u + dt * k3, k, b)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _rhs(u: torch.Tensor, k: torch.Tensor, b: float) -> torch.Tensor:
    u_hat = torch.fft.fft(u)
    ux = torch.fft.ifft(1j * k * u_hat).real
    uxxx = torch.fft.ifft((1j * k) ** 3 * u_hat).real
    result: torch.Tensor = -u * ux - b * uxxx
    return result
