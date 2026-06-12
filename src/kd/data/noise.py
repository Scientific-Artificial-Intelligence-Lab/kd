
from __future__ import annotations

import math
from typing import Literal

import torch



NOISE_SCALE_STD = "std"
NOISE_SCALE_MAX = "max"
NoiseScale = Literal["std", "max"]


def xu2020_relative(
    u: torch.Tensor,
    noise_level: float,
    *,
    seed: int | None,
) -> torch.Tensor:
    if noise_level < 0.0 or not math.isfinite(noise_level):
        raise ValueError(
            f"noise_level must be finite and non-negative, got {noise_level}"
        )
    if noise_level == 0.0:
        return u
    generator = None
    if seed is not None:
        generator = torch.Generator(device=u.device).manual_seed(seed)
    noise = torch.randn(
        u.shape,
        dtype=u.dtype,
        device=u.device,
        generator=generator,
    )
    signal_std = u.std(correction=0)
    if float(signal_std.item()) == 0.0:
        raise ValueError(
            "signal has zero std; relative noise is undefined "
            "(use absolute noise scale instead)"
        )
    noise_std = noise.std(correction=0)
    if float(noise_std.item()) == 0.0:
        return u
    noise = noise / noise_std * (noise_level * signal_std)
    return u + noise


def discover_unnormalized(
    values: torch.Tensor,
    level: float,
    *,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    scale: NoiseScale = "std",
) -> torch.Tensor:
    if level < 0.0:
        raise ValueError(f"noise level must be non-negative, got {level}")
    if scale not in (NOISE_SCALE_STD, NOISE_SCALE_MAX):
        raise ValueError(
            f"scale must be {NOISE_SCALE_STD!r} or {NOISE_SCALE_MAX!r}, got {scale!r}"
        )
    if (seed is None) == (generator is None):
        raise ValueError("exactly one of seed and generator must be provided")
    if generator is None:
        assert seed is not None
        generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(
        values.shape,
        generator=generator,
        dtype=values.dtype,
        device=values.device,
    )
    if scale == NOISE_SCALE_STD:
        sigma = level * torch.std(values, unbiased=True)
    else:
        sigma = level * values.abs().max()
    return values + sigma * noise
