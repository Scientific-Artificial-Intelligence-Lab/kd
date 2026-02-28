"""N-D synthetic data factory for examples / N-D 合成数据工厂。

Provides reusable functions to generate multi-spatial-dimension test data
so that individual example scripts stay DRY.
"""

from __future__ import annotations

import numpy as np


def make_diffusion_2d(
    nx: int = 20,
    ny: int = 20,
    nt: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate multi-mode 2D diffusion data on a regular grid.

    u = sin(x)sin(y)exp(-2t) + 0.5*sin(2x)sin(y)exp(-5t)

    Satisfies u_t = u_xx + u_yy (2D diffusion equation).
    Multi-mode ensures u_xx + u_yy cannot be simplified to c*u,
    forcing algorithms to discover spatial derivatives.

    Returns:
        (x, y, t, u) where x/y/t are 1-D coordinate arrays
        and u has shape (nx, ny, nt).
    """
    x = np.linspace(0, 2 * np.pi, nx)
    y = np.linspace(0, 2 * np.pi, ny)
    t = np.linspace(0, 1, nt)

    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    u = (np.sin(X) * np.sin(Y) * np.exp(-2 * T)
         + 0.5 * np.sin(2 * X) * np.sin(Y) * np.exp(-5 * T))

    return x, y, t, u
