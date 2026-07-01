
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np


def read_grid_csv(
    path: Path | str,
    *,
    name: str,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{name} grid CSV not found: {csv_path}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            arr = np.loadtxt(csv_path, delimiter=",", ndmin=2, dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{name} grid CSV {csv_path} is malformed: {exc}") from exc

    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"{name} grid CSV {csv_path} is empty or not 2D")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} grid CSV {csv_path} contains NaN or Inf")
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(
            f"{name} grid CSV {csv_path} has shape {arr.shape}; "
            f"expected {expected_shape}"
        )

    return arr


__all__ = ["read_grid_csv"]
