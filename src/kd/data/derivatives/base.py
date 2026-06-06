
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class DerivativeProvider(ABC):

    @property
    def coords(self) -> dict[str, torch.Tensor]:
        return {}

    @abstractmethod
    def get_derivative(
        self,
        field: str,
        axis: str,
        order: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def diff(
        self,
        expression: torch.Tensor,
        axis: str,
        order: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def available_derivatives(self) -> list[tuple[str, str, int]]:
        raise NotImplementedError
