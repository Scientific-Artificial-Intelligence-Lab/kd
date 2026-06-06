
from __future__ import annotations

import torch


SPATIAL_AXIS_NAMES = ("x", "y", "z")


AXIS_T = "t"


FIELD_U = "u"


def broadcast_grids(coords: list[torch.Tensor]) -> list[torch.Tensor]:
    ndim = len(coords)
    full_shape = tuple(c.numel() for c in coords)
    grids = []
    for i, c in enumerate(coords):
        shape = [1] * ndim
        shape[i] = c.numel()
        grids.append(c.reshape(shape).expand(full_shape))
    return grids
