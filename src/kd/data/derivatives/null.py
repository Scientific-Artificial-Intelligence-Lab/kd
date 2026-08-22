
from __future__ import annotations

import torch

from kd.data.derivatives.base import DerivativeProvider

_UNAVAILABLE = "derivatives are unavailable for this dataset"


class NullDerivativeProvider(DerivativeProvider):

    def get_derivative(self, field: str, axis: str, order: int) -> torch.Tensor:
        raise NotImplementedError(_UNAVAILABLE)

    def diff(
        self,
        expression: torch.Tensor,
        axis: str,
        order: int,
    ) -> torch.Tensor:
        raise NotImplementedError(_UNAVAILABLE)

    def available_derivatives(self) -> list[tuple[str, str, int]]:
        raise NotImplementedError(_UNAVAILABLE)


__all__ = ["NullDerivativeProvider"]
