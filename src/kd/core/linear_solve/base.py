
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class SolveResult:

    coefficients: torch.Tensor
    residual: float
    r2: float
    condition_number: float | None = None
    selected_indices: list[int] | None = None
    is_valid: bool = True
    error_message: str = ""


class SparseSolver(ABC):

    @abstractmethod
    def solve(
        self,
        theta: torch.Tensor,
        y: torch.Tensor,
    ) -> SolveResult:
        raise NotImplementedError
