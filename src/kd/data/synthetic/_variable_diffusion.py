"""Known-conductivity diffusion on [0, 1] with zero Dirichlet ends.

The conservative spatial operator uses face fluxes
F[i+1/2] = H(K[i], K[i+1]) * (u[i+1] - u[i]) / dx, where H is the
harmonic mean. Forward Euler gives weights r*K_left, 1-r*(K_left+K_right),
and r*K_right, with r=dt/dx**2. Positive K and
dt <= dx**2 / max(K_left+K_right) make these weights nonnegative and
sum to one, giving a discrete maximum principle and stability. Since each
face K <= max(K), dx**2/(2*max(K)) is a sufficient bound. We use
dt <= 0.2*dx**2/max(K), 40% of that bound, subdividing each saved interval
into equal steps so output times remain uniform.

A piecewise-constant K, such as the zoned values 0.5 / 1.5 / 0.25 of
Chang & Zhang (2019), Computational Geosciences 23:895-910,
doi:10.1007/s10596-019-09847-2, is passed to the generator as node values.
Such data is kd-generated and dimensionless, not the paper's MODFLOW data or
its 2-D setup. Generic central differences do not implement the harmonic
interface flux: fit piecewise-constant data away from the jumps and trim the
initial/final time rows (``rows=kd.data.interface_mask(...)`` in
``kd.evaluate_terms``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from kd.data.schema import PDEDataset

_EQUATION = "u_t = diff_x(mul(K, u_x))"


@torch.no_grad()
def generate_variable_diffusion_data(
    k_profile: Callable[[Tensor], Tensor] | Tensor | Sequence[float],
    *,
    nx: int = 101,
    nt: int = 161,
    t_max: float = 0.04,
) -> PDEDataset:
    """Integrate u_t = (K u_x)_x, returning CPU float64 fields u and K.

    ``k_profile`` is a callable of the 1-D x tensor or an array of ``nx``
    finite, strictly positive node values. ``nx`` and the number of saved
    frames ``nt`` must be at least five (the platform FD minimum).
    The initial condition is sin(pi*x) + 0.5*sin(3*pi*x); endpoints stay zero.
    ``t_max`` must be finite and positive. K is broadcast over saved times.
    The default 0.00025 output interval resolves saved-time increments;
    the internal Euler step is at most 0.2*dx**2/max(K).

    K is a known input field next to ``u``. How each algorithm uses it:

    - SGA puts ``K`` and ``K_x`` into its variable pool automatically.
    - PySINDy and PySR use K only through terms you write, for example
      ``PySINDyConfig(terms=("diff_x(mul(K, u_x))", ...))``.
    - DISCOVER sees K only when ``library.state_vars`` lists it.
    - DLGA fits a surrogate of ``u`` only and refuses library terms that
      reference K.
    - EqGPT and LLM4ED do not use K.
    """
    if nx < 5:
        raise ValueError("nx must be at least 5")
    if nt < 5:
        raise ValueError("nt must be at least 5")
    if not math.isfinite(t_max) or t_max <= 0:
        raise ValueError("t_max must be finite and positive")
    x = torch.linspace(0, 1, nx, dtype=torch.float64, device="cpu")
    t = torch.linspace(0, t_max, nt, dtype=torch.float64, device="cpu")
    k = torch.as_tensor(
        k_profile(x) if callable(k_profile) else k_profile,
        dtype=torch.float64,
        device="cpu",
    ).detach()
    if k.shape != x.shape:
        raise ValueError("k_profile must have shape (nx,)")
    if not torch.isfinite(k).all() or not (k > 0).all():
        raise ValueError("k_profile must contain finite, strictly positive values")
    dx = 1.0 / (nx - 1)


    face_k = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:])
    saved_dt = t_max / (nt - 1)
    substeps = math.ceil(saved_dt / (0.2 * dx**2 / k.max().item()))
    dt = saved_dt / substeps
    u = _integrate(x, face_k, dx, dt, substeps, nt)
    return PDEDataset.from_arrays(
        coords={"x": x, "t": t},
        fields={"u": u, "K": k[:, None].expand_as(u)},
        lhs="u_t",
        name="variable-diffusion",
        ground_truth=_EQUATION,
    )


def _flux_divergence(u: Tensor, face_k: Tensor, dx: float) -> Tensor:
    """Return the interior conservative operator (positive dx from the grid)."""
    flux = face_k * (u[1:] - u[:-1]) / dx
    return (flux[1:] - flux[:-1]) / dx


def _integrate(
    x: Tensor,
    face_k: Tensor,
    dx: float,
    dt: float,
    substeps: int,
    nt: int,
) -> Tensor:
    state = torch.sin(math.pi * x) + 0.5 * torch.sin(3 * math.pi * x)
    state[0] = state[-1] = 0
    saved = torch.empty((x.numel(), nt), dtype=x.dtype, device=x.device)
    saved[:, 0] = state
    for frame in range(1, nt):
        for _ in range(substeps):
            state[1:-1] += dt * _flux_divergence(state, face_k, dx)
        saved[:, frame] = state
    return saved


def load_variable_diffusion_smooth() -> PDEDataset:
    """Load the smooth K = 1 + 0.5*sin(2*pi*x) catalog benchmark.

    Fields ``u`` and the known conductivity ``K``; see
    ``generate_variable_diffusion_data`` for which algorithms use K.
    """
    dataset = generate_variable_diffusion_data(
        lambda x: 1 + 0.5 * torch.sin(2 * math.pi * x)
    )
    dataset.name = "variable-diffusion-smooth"
    return dataset
