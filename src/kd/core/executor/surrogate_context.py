
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from kd.core.executor.context import ExecutionContext
from kd.data.derivatives.base import DerivativeProvider
from kd.data.schema import PDEDataset

if TYPE_CHECKING:
    from kd.models.trainer import TrainingResult


class SurrogateContext(ExecutionContext):

    def __init__(
        self,
        dataset: PDEDataset,
        autograd_provider: DerivativeProvider,
        *,
        surrogate_field: str | None = None,
        constants: dict[str, float] | None = None,
        device: torch.device | None = None,
        training_result: TrainingResult | None = None,
    ) -> None:
        if device is None:
            device = self._resolve_default_device(autograd_provider, dataset)
        super().__init__(
            dataset=dataset,
            derivative_provider=autograd_provider,
            constants={} if constants is None else dict(constants),
            device=device,
        )
        self._surrogate_field = surrogate_field or dataset.lhs_field
        self._cache: dict[str, torch.Tensor] = {}




        self.training_result: TrainingResult | None = training_result

    @staticmethod
    def _resolve_default_device(
        provider: DerivativeProvider,
        dataset: PDEDataset,
    ) -> torch.device:
        provider_device = getattr(provider, "device", None)
        if isinstance(provider_device, torch.device):
            return provider_device
        dataset_device = getattr(dataset, "device", None)
        if isinstance(dataset_device, torch.device):
            return dataset_device
        return torch.device("cpu")

    def get_variable(self, name: str) -> torch.Tensor:
        if name == self._surrogate_field:
            if name not in self._cache:
                get_field = getattr(self.derivative_provider, "get_field", None)
                if get_field is None:
                    raise TypeError("SurrogateContext requires provider.get_field")
                self._cache[name] = get_field(name).detach().flatten().to(self.device)
            return self._cache[name]
        return super().get_variable(name)

    def clear_cache(self) -> None:
        self._cache.clear()
