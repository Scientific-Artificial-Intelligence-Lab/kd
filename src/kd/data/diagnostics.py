
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MIN_STENCIL_POINTS",
    "axis_noise_limited",
    "noise_estimate",
    "second_derivative_gap",
    "first_derivative_gap",
]

MIN_STENCIL_POINTS = 5


def axis_noise_limited(values: NDArray[np.float64], axis: int) -> bool:
    u = _prepare_axis(values, axis)
    n = u.shape[0]
    m = u.size // n
    centered = u - u.mean(axis=0, keepdims=True)
    energy = float(np.sum(centered**2))
    if energy == 0.0:
        return False
    ratio = float(np.sum(np.diff(u, axis=0) ** 2)) / energy
    variance = 4.0 * (n - 2) / ((n - 1) * (m * (n - 1) + 2))
    return math.isfinite(ratio) and abs(ratio - 2.0) <= 3.0 * math.sqrt(variance)


def _prepare_axis(values: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    u = np.moveaxis(values, axis, 0)
    if u.shape[0] < MIN_STENCIL_POINTS:
        raise ValueError(f"need at least {MIN_STENCIL_POINTS} points along axis {axis}")
    return u


def noise_estimate(values: NDArray[np.float64], axis: int) -> float:
    u = _prepare_axis(values, axis)
    std = float(np.std(values, ddof=0))
    if std == 0.0:
        return 0.0
    d = u[:-2] - 2.0 * u[1:-1] + u[2:]
    mad = float(np.median(np.abs(d - np.median(d))))
    sigma = 1.4826 * mad / math.sqrt(6.0)
    return sigma / std


def second_derivative_gap(values: NDArray[np.float64], axis: int) -> float:
    u = _prepare_axis(values, axis)
    a = u[1:-3] - 2.0 * u[2:-2] + u[3:-1]
    b = u[:-4] - 2.0 * u[2:-2] + u[4:]
    numerator = float(np.linalg.norm(4.0 * a - b))
    denominator = float(np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def first_derivative_gap(values: NDArray[np.float64], axis: int) -> float:
    u = _prepare_axis(values, axis)
    c = u[3:-1] - u[1:-3]
    e = u[4:] - u[:-4]
    numerator = float(np.linalg.norm(2.0 * c - e))
    denominator = float(np.linalg.norm(e))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator
