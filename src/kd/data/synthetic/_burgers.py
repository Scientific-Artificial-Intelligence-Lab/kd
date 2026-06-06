"""Synthetic Burgers equation data generation.

Solves: u_t + u * u_x = nu * u_xx

Uses pseudo-spectral method with RK4 time integration and
integrating factor for the diffusion term.
"""

from __future__ import annotations

import torch

from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)


_FIELD_U = "u"


_AXIS_X = "x"
_AXIS_T = "t"


_X_MIN = -1.0
_X_MAX = 1.0
_T_MIN = 0.0
_T_MAX = 1.0


_CFL_FACTOR = 0.1


_MAX_INTERNAL_STEPS = 10000000


_DEALIAS_FRACTION = 2.0 / 3.0


_MAX_NX = 10000
_MAX_NT = 10000
_MAX_NOISE_LEVEL = 10.0


def generate_burgers_data(
    nx: int = 256,
    nt: int = 101,
    nu: float = 0.1,
    noise_level: float = 0.0,
    device: torch.device | None = None,
    seed: int | None = None,
) -> PDEDataset:
    """Generate synthetic Burgers equation data.

    Solves: u_t + u * u_x = nu * u_xx

    Initial condition: u(x, 0) = -sin(pi * x)
    Boundary condition: Periodic on x in [-1, 1]
    Time domain: t in [0, 1]

    Uses pseudo-spectral method with RK4 time integration and
    integrating factor for the diffusion term. Includes 2/3 dealiasing
    for stability at low viscosity.

    Args:
        nx: Number of spatial grid points (default: 256)
        nt: Number of time steps (default: 101)
        nu: Viscosity coefficient (default: 0.1)
        noise_level: Standard deviation of Gaussian noise to add (default: 0.0)
        device: Target device for tensors (default: CPU)
        seed: Random seed for reproducibility (default: None)

    Returns:
        PDEDataset with:
        - axes: {"x": AxisInfo, "t": AxisInfo}
        - axis_order: ["x", "t"]
        - fields: {"u": FieldData}
        - lhs_field: "u"
        - lhs_axis: "t"
        - ground_truth: equation string

    Raises:
        ValueError: If parameters are invalid.

    Example:
        >>> dataset = generate_burgers_data(nx=128, nt=51, nu=0.1)
        >>> dataset.get_shape()
        (128, 51)
        >>> dataset.get_field("u").shape
        torch.Size([128, 51])
    """

    _validate_parameters(nx, nt, nu, noise_level)


    if device is None:
        device = torch.device("cpu")


    dtype = torch.float32 if device.type == "mps" else torch.float64


    if seed is not None:
        torch.manual_seed(seed)



    x = torch.linspace(_X_MIN, _X_MAX, nx + 1, dtype=dtype, device=device)[:-1]
    t = torch.linspace(_T_MIN, _T_MAX, nt, dtype=dtype, device=device)


    u = _solve_burgers_rk4if(nx, nt, nu, device)


    if noise_level > 0.0:
        noise = torch.randn_like(u) * noise_level
        u = u + noise


    ground_truth = f"u_t + u * u_x - {nu} * u_xx = 0"


    x_axis = AxisInfo(name=_AXIS_X, values=x, is_periodic=True)
    t_axis = AxisInfo(name=_AXIS_T, values=t, is_periodic=False)


    u_field = FieldData(name=_FIELD_U, values=u)


    return PDEDataset(
        name="burgers",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={_AXIS_X: x_axis, _AXIS_T: t_axis},
        axis_order=[_AXIS_X, _AXIS_T],
        fields={_FIELD_U: u_field},
        lhs_field=_FIELD_U,
        lhs_axis=_AXIS_T,
        noise_level=noise_level,
        ground_truth=ground_truth,
    )


def _validate_parameters(nx: int, nt: int, nu: float, noise_level: float) -> None:
    """Validate input parameters.

    Args:
        nx: Number of spatial points
        nt: Number of time points
        nu: Viscosity coefficient
        noise_level: Noise standard deviation

    Raises:
        ValueError: If any parameter is invalid.
    """
    if nx <= 0:
        raise ValueError(f"nx must be positive, got {nx}")
    if nx > _MAX_NX:
        raise ValueError(f"nx must be <= {_MAX_NX} to prevent OOM, got {nx}")
    if nt <= 0:
        raise ValueError(f"nt must be positive, got {nt}")
    if nt > _MAX_NT:
        raise ValueError(f"nt must be <= {_MAX_NT} to prevent OOM, got {nt}")
    if nu <= 0:
        msg = f"nu must be positive (inviscid Burgers is unstable), got {nu}"
        raise ValueError(msg)
    if noise_level < 0:
        raise ValueError(f"noise_level must be non-negative, got {noise_level}")
    if noise_level > _MAX_NOISE_LEVEL:
        raise ValueError(
            f"noise_level must be <= {_MAX_NOISE_LEVEL} to prevent Inf, "
            f"got {noise_level}"
        )


def _solve_burgers_rk4if(
    nx: int,
    nt: int,
    nu: float,
    device: torch.device,
) -> torch.Tensor:
    """Solve Burgers equation using RK4 with Integrating Factor.

    Uses integrating factor to handle the stiff diffusion term exactly,
    which allows larger time steps and better stability for small viscosity.
    Includes 2/3 dealiasing for the nonlinear term.

    For the output grid, we use linspace(-1, 1, nx+1)[:-1] which gives
    nx points on [-1, 1) — proper periodic grid without endpoint duplication.

    Args:
        nx: Number of spatial points
        nt: Number of output time points
        nu: Viscosity coefficient
        device: Computation device

    Returns:
        Solution tensor of shape (nx, nt)
    """

    domain_length = _X_MAX - _X_MIN


    dtype = torch.float32 if device.type == "mps" else torch.float64


    x_output = torch.linspace(_X_MIN, _X_MAX, nx + 1, dtype=dtype, device=device)[:-1]


    dx = domain_length / nx


    t_output = torch.linspace(_T_MIN, _T_MAX, nt, dtype=dtype, device=device)


    k = torch.fft.fftfreq(nx, d=dx, device=device) * 2 * torch.pi



    k_max = torch.pi / dx
    k_cutoff = _DEALIAS_FRACTION * k_max
    dealias_mask = (torch.abs(k) <= k_cutoff).float()



    nu_k2 = nu * (k**2)



    u_max_estimate = 1.5
    dt_advection = _CFL_FACTOR * dx / u_max_estimate


    if nt > 1:
        dt_output = (_T_MAX - _T_MIN) / (nt - 1)
        dt_internal = min(dt_advection, dt_output / 2)
    else:
        dt_internal = dt_advection


    u = -torch.sin(torch.pi * x_output)


    u_all = torch.zeros(nx, nt, dtype=dtype, device=device)
    u_all[:, 0] = u

    if nt == 1:
        return u_all


    current_time = _T_MIN
    output_idx = 1
    step_count = 0

    while output_idx < nt and step_count < _MAX_INTERNAL_STEPS:

        next_output_time = t_output[output_idx].item()
        dt = min(dt_internal, next_output_time - current_time)


        u = _rk4if_step(u, k, nu_k2, dealias_mask, dt)

        current_time += dt
        step_count += 1


        if abs(current_time - next_output_time) < 1e-12:
            u_all[:, output_idx] = u
            output_idx += 1

    return u_all


def _nonlinear_term(
    u: torch.Tensor, k: torch.Tensor, dealias_mask: torch.Tensor
) -> torch.Tensor:
    """Compute nonlinear term: -u * u_x in physical space with dealiasing.

    Uses 2/3 rule for dealiasing to prevent aliasing errors in the
    nonlinear term, which is crucial for stability at low viscosity.

    Args:
        u: Current solution in physical space (1D tensor)
        k: Wave numbers for FFT
        dealias_mask: Mask for dealiasing (1 for kept modes, 0 for removed)

    Returns:
        Nonlinear term -u * u_x in physical space
    """
    u_hat = torch.fft.fft(u)


    u_hat_filtered = u_hat * dealias_mask
    u_filtered = torch.fft.ifft(u_hat_filtered).real


    u_x_hat = 1j * k * u_hat_filtered
    u_x = torch.fft.ifft(u_x_hat).real


    result = -u_filtered * u_x
    result_hat = torch.fft.fft(result)
    result_hat_filtered = result_hat * dealias_mask

    output: torch.Tensor = torch.fft.ifft(result_hat_filtered).real
    return output


def _rk4if_step(
    u: torch.Tensor,
    k: torch.Tensor,
    nu_k2: torch.Tensor,
    dealias_mask: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Perform one RK4-IF (Integrating Factor) time step.

    The Burgers equation in Fourier space is:
        d(u_hat)/dt = -nu * k^2 * u_hat + N_hat

    where N_hat = FFT(-u * u_x) is the nonlinear term.

    Using integrating factor v_hat = exp(nu*k^2*t) * u_hat transforms to:
        d(v_hat)/dt = exp(nu*k^2*t) * N_hat

    We apply RK4 to this transformed equation.

    Args:
        u: Current solution in physical space
        k: Wave numbers
        nu_k2: nu * k^2 for each mode
        dealias_mask: Dealiasing mask for nonlinear terms
        dt: Time step

    Returns:
        Solution at next time step in physical space
    """


    exp_full = torch.exp(-nu_k2 * dt)
    exp_half = torch.exp(-nu_k2 * dt / 2)


    u_hat = torch.fft.fft(u)



    n1 = _nonlinear_term(u, k, dealias_mask)
    k1_hat = torch.fft.fft(n1)



    u2_hat = exp_half * u_hat + (dt / 2) * exp_half * k1_hat
    u2 = torch.fft.ifft(u2_hat).real
    n2 = _nonlinear_term(u2, k, dealias_mask)
    k2_hat = torch.fft.fft(n2)





    u3_hat = exp_half * u_hat + (dt / 2) * k2_hat
    u3 = torch.fft.ifft(u3_hat).real
    n3 = _nonlinear_term(u3, k, dealias_mask)
    k3_hat = torch.fft.fft(n3)


    u4_hat = exp_full * u_hat + dt * exp_half * k3_hat
    u4 = torch.fft.ifft(u4_hat).real
    n4 = _nonlinear_term(u4, k, dealias_mask)
    k4_hat = torch.fft.fft(n4)



    u_new_hat = exp_full * u_hat + (dt / 6) * (
        exp_full * k1_hat + 2 * exp_half * k2_hat + 2 * exp_half * k3_hat + k4_hat
    )

    output: torch.Tensor = torch.fft.ifft(u_new_hat).real
    return output
