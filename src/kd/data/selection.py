"""Fit-row masks for piecewise-constant coefficient fields on intact grids."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

__all__ = ["interface_mask"]


def interface_mask(
    coefficient: Tensor,
    *,
    axis: int,
    width: int = 3,
    trim: Mapping[int, int] | None = None,
) -> Tensor:
    """Keep rows away from coefficient jumps and chosen grid boundaries.

    A jump is an exact inequality between adjacent values along ``axis``;
    this rule is intended for piecewise-constant fields, not smooth/noisy K.
    At a face j|j+1, remove ``width`` nodes on EACH side, namely
    j-width+1 through j+width inclusive, clipped to the domain. Detection
    is local to each axis fiber; it does not remove unrelated grid lines.
    ``trim={dimension: count}`` additionally removes ``count`` nodes at both
    ends of each specified dimension, e.g. ``{0: 3, 1: 2}`` for (x, t).
    Dimension numbers are nonnegative. Zero width/trim removes nothing.
    The returned bool tensor has the coefficient's shape and device.
    """
    if not 0 <= axis < coefficient.ndim:
        raise ValueError("axis must name a coefficient dimension")
    if width < 0:
        raise ValueError("width must be nonnegative")
    keep = torch.ones_like(coefficient, dtype=torch.bool)
    values = coefficient.movedim(axis, 0)
    mask = keep.movedim(axis, 0)
    jumps = values[1:] != values[:-1]
    n = values.shape[0]
    for offset in range(min(width, n)):
        length = n - 1 - offset
        mask[:length] &= ~jumps[offset:]
        mask[1 + offset:] &= ~jumps[:length]
    for dim, count in (trim or {}).items():
        if not 0 <= dim < coefficient.ndim or count < 0:
            raise ValueError("trim requires valid dimensions and nonnegative counts")
        if count:
            edge = keep.movedim(dim, 0)
            edge[:count] = False
            edge[-count:] = False
    return keep
