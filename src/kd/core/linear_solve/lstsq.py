
from __future__ import annotations

import torch
from torch import Tensor


def deterministic_lstsq(a: Tensor, b: Tensor, *, rcond: float | None = None) -> Tensor:
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        solution: Tensor = torch.linalg.lstsq(a, b, rcond=rcond).solution
        return solution
    finally:



        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)
