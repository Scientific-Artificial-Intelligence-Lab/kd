"""Synthetic PDE data generation.

This package provides functions to generate synthetic PDE data
for testing and validation purposes.

Generators (analytic solutions):
- Burgers equation: u_t + u * u_x = nu * u_xx
- Advection equation: u_t + c1*u_x [+ c2*u_y] = 0
- Diffusion equation: u_t = alpha * (u_xx [+ u_yy])

Data loaders (pre-computed benchmark data):
- Chafee-Infante equation: u_t = u_xx - u + u^3
- KdV equation: u_t = -u * u_x - 0.0025 * u_xxx
- PDE_divide (paper Eq. S4): u_t = -u_x / x + 0.25 * u_xx
- PDE_compound (paper Eq. S5): u_t = u * u_xx + u_x^2
"""

from kd.data.synthetic._advection import generate_advection_data
from kd.data.synthetic._burgers import generate_burgers_data
from kd.data.synthetic._chaffee_infante import (
    generate_chaffee_infante_xu2020_data,
)
from kd.data.synthetic._diffusion import generate_diffusion_data
from kd.data.synthetic._kdv_xu2020 import generate_kdv_xu2020_data
from kd.data.synthetic._loaders import (
    load_burgers,
    load_chafee_infante,
    load_kdv,
    load_pde_compound,
    load_pde_divide,
)
from kd.data.synthetic._wave_xu2020 import generate_wave_xu2020_data

__all__ = [
    "generate_advection_data",
    "generate_burgers_data",
    "generate_chaffee_infante_xu2020_data",
    "generate_diffusion_data",
    "generate_kdv_xu2020_data",
    "generate_wave_xu2020_data",
    "load_burgers",
    "load_chafee_infante",
    "load_kdv",
    "load_pde_compound",
    "load_pde_divide",
]
