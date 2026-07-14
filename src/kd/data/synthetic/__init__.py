"""Synthetic PDE data generation.

This package provides functions to generate synthetic PDE data
for testing and validation purposes.

Generators (analytic solutions):
- Burgers equation: u_t + u * u_x = nu * u_xx
- Advection equation: u_t + c1*u_x [+ c2*u_y] = 0
- Diffusion equation: u_t = alpha * (u_xx [+ u_yy])

Data loaders (pre-computed benchmark data):
- Allen-Cahn equation: u_t = 0.003 * u_xx + u - u^3
- Burgers equation: u_t = -u * u_x + 0.1 * u_xx
- Burgers 2D (EqGPT): u_t = -u*u_x - u*u_y + 0.01*u_xx + 0.01*u_yy
- Chafee-Infante equation: u_t = u_xx - u + u^3
- Convection-Diffusion equation: u_t = -u_x + 0.25 * u_xx
- Eq. 6.2.12 (EqGPT, mixed-derivative): u_t = -0.1*u_x_t - 0.1*u_x
- KdV equation: u_t = -u * u_x - 0.0025 * u_xxx
- Klein-Gordon equation: u_tt = 0.5 * u_xx - 5 * u
- PDE_divide (paper Eq. S4): u_t = -u_x / x + 0.25 * u_xx
- PDE_compound (paper Eq. S5): u_t = u * u_xx + u_x^2
- Wave equation: u_tt = u_xx
"""

from kd.data.synthetic._advection import generate_advection_data
from kd.data.synthetic._burgers import generate_burgers_data
from kd.data.synthetic._chaffee_infante import (
    generate_chaffee_infante_xu2020_data,
)
from kd.data.synthetic._diffusion import generate_diffusion_data
from kd.data.synthetic._eqgpt_grid_loaders import (
    load_burgers_2d,
    load_eq_6_2_12,
)
from kd.data.synthetic._eqgpt_steady_loaders import (
    load_laplacian_eitech,
    load_laplacian_smile,
    load_poisson_disk,
)
from kd.data.synthetic._kdv_xu2020 import generate_kdv_xu2020_data
from kd.data.synthetic._loaders import (
    load_allen_cahn,
    load_burgers,
    load_chafee_infante,
    load_convection_diffusion,
    load_kdv,
    load_klein_gordon,
    load_pde_compound,
    load_pde_divide,
    load_wave,
)
from kd.data.synthetic._wave_xu2020 import generate_wave_xu2020_data

__all__ = [
    "generate_advection_data",
    "generate_burgers_data",
    "generate_chaffee_infante_xu2020_data",
    "generate_diffusion_data",
    "generate_kdv_xu2020_data",
    "generate_wave_xu2020_data",
    "load_allen_cahn",
    "load_burgers",
    "load_burgers_2d",
    "load_chafee_infante",
    "load_convection_diffusion",
    "load_eq_6_2_12",
    "load_kdv",
    "load_klein_gordon",
    "load_laplacian_eitech",
    "load_laplacian_smile",
    "load_pde_compound",
    "load_pde_divide",
    "load_poisson_disk",
    "load_wave",
]
