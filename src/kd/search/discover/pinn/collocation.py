
from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from scipy.stats import qmc
from torch import Tensor

_LOCAL_DELTA_DIVISOR = 100.0


_CPU_DEVICE = torch.device("cpu")


def generate_collocation_points(
    bounds: dict[str, tuple[float, float]],
    n_points: int,
    cut_ratio: float = 0.05,
    device: torch.device = _CPU_DEVICE,
    seed: int | None = None,
) -> dict[str, Tensor]:
    names = _validate_bounds(bounds)
    _validate_point_count(n_points)
    _validate_cut_ratio(cut_ratio)
    if n_points == 0:
        return _empty_points(names, device)

    unit_samples = _lhs_samples(len(names), n_points, seed)
    scaled = _scale_samples(unit_samples, bounds, names)
    trimmed = _trim_samples(scaled, cut_ratio)
    return _to_tensor_dict(trimmed, names, device)


def generate_local_samples(
    observation_coords: dict[str, Tensor],
    bounds: dict[str, tuple[float, float]],
    multiplier: int = 20,
    device: torch.device = _CPU_DEVICE,
    seed: int | None = None,
    append_observations: bool = True,
) -> dict[str, Tensor]:
    names = _validate_bounds(bounds)
    if multiplier <= 0:
        return _empty_points(names, device)

    observations = _stack_observations(observation_coords, names, device)
    if observations.shape[0] == 0:
        return _empty_points(names, device)

    repeated = observations.repeat_interleave(multiplier, dim=0)
    lhs = _lhs_tensor(len(names), repeated.shape[0], device, seed)
    deltas = _delta_tensor(bounds, names, device)
    samples = repeated - deltas + (2.0 * deltas * lhs)
    clipped = _clip_to_bounds(samples, bounds, names)

    if append_observations:
        clipped = torch.cat([clipped, observations], dim=0)

    return _split_columns(clipped, names)


def _validate_bounds(bounds: dict[str, tuple[float, float]]) -> list[str]:
    if not bounds:
        raise ValueError("bounds must not be empty")
    for name, (lower, upper) in bounds.items():
        if lower >= upper:
            raise ValueError(f"bounds for '{name}' must satisfy lower < upper")
    return list(bounds)


def _validate_point_count(n_points: int) -> None:
    if n_points < 0:
        raise ValueError(f"n_points must be >= 0, got {n_points}")


def _validate_cut_ratio(cut_ratio: float) -> None:
    if cut_ratio < 0.0 or cut_ratio >= 0.5:
        raise ValueError(f"cut_ratio must be in [0, 0.5), got {cut_ratio}")


def _lhs_samples(n_dims: int, n_points: int, seed: int | None) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    return cast(npt.NDArray[np.float64], sampler.random(n=n_points))


def _scale_samples(
    samples: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    names: Iterable[str],
) -> np.ndarray:
    lowers = np.array([bounds[name][0] for name in names], dtype=np.float64)
    uppers = np.array([bounds[name][1] for name in names], dtype=np.float64)
    return cast(npt.NDArray[np.float64], qmc.scale(samples, lowers, uppers))


def _trim_samples(samples: np.ndarray, cut_ratio: float) -> np.ndarray:
    if cut_ratio == 0.0 or samples.shape[0] == 0:
        return samples

    mask = np.ones(samples.shape[0], dtype=bool)
    for dim in range(samples.shape[1]):
        lower = np.quantile(samples[:, dim], cut_ratio)
        upper = np.quantile(samples[:, dim], 1.0 - cut_ratio)
        mask &= (samples[:, dim] >= lower) & (samples[:, dim] <= upper)
    trimmed = cast(npt.NDArray[np.float64], samples[mask])
    if trimmed.shape[0] == 0:





        raise ValueError(
            f"cut_ratio={cut_ratio} eliminated all {samples.shape[0]} "
            "collocation samples; lower cut_ratio or raise n_points.",
        )
    return trimmed


def _to_tensor_dict(
    samples: np.ndarray,
    names: list[str],
    device: torch.device,
) -> dict[str, Tensor]:
    if samples.shape[0] == 0:
        return _empty_points(names, device)
    tensor = torch.from_numpy(samples.astype(np.float32)).to(device=device)
    return _split_columns(tensor, names)


def _empty_points(names: list[str], device: torch.device) -> dict[str, Tensor]:
    return {
        name: torch.empty(0, dtype=torch.float32, device=device)
        for name in names
    }


def _stack_observations(
    observation_coords: dict[str, Tensor],
    names: list[str],
    device: torch.device,
) -> Tensor:
    columns = [
        observation_coords[name].detach().to(device=device, dtype=torch.float32)
        for name in names
    ]
    return torch.stack(columns, dim=-1)


def _lhs_tensor(
    n_dims: int,
    n_points: int,
    device: torch.device,
    seed: int | None,
) -> Tensor:
    samples = _lhs_samples(n_dims, n_points, seed).astype(np.float32)
    return torch.from_numpy(samples).to(device=device)


def _delta_tensor(
    bounds: dict[str, tuple[float, float]],
    names: list[str],
    device: torch.device,
) -> Tensor:
    deltas = [
        (bounds[name][1] - bounds[name][0]) / _LOCAL_DELTA_DIVISOR
        for name in names
    ]
    return torch.tensor(deltas, dtype=torch.float32, device=device)


def _clip_to_bounds(
    samples: Tensor,
    bounds: dict[str, tuple[float, float]],
    names: list[str],
) -> Tensor:
    lower = torch.tensor(
        [bounds[name][0] for name in names],
        dtype=samples.dtype,
        device=samples.device,
    )
    upper = torch.tensor(
        [bounds[name][1] for name in names],
        dtype=samples.dtype,
        device=samples.device,
    )
    return torch.clamp(samples, min=lower, max=upper)


def _split_columns(samples: Tensor, names: list[str]) -> dict[str, Tensor]:
    return {name: samples[:, idx] for idx, name in enumerate(names)}
