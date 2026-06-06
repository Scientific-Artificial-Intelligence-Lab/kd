
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from kd.data import PDEDataset
    from kd.data.derivatives import DerivativeProvider


@dataclass
class ExecutionContext:

    dataset: PDEDataset
    derivative_provider: DerivativeProvider
    constants: dict[str, float] = field(default_factory=dict)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def get_variable(self, name: str) -> torch.Tensor:

        if self.dataset.fields is not None and name in self.dataset.fields:
            return self.dataset.fields[name].values.to(self.device)


        if self.dataset.axes is not None and name in self.dataset.axes:
            coord = self.dataset.axes[name].values.to(self.device)

            return self._broadcast_coord(name, coord)

        raise KeyError(f"Variable '{name}' not found in dataset")

    def _broadcast_coord(self, axis_name: str, coord: torch.Tensor) -> torch.Tensor:
        if self.dataset.axis_order is None or self.dataset.axes is None:
            raise ValueError("Dataset must have axis_order and axes defined")


        field_shape = self.dataset.get_shape()


        axis_idx = self.dataset.axis_order.index(axis_name)



        broadcast_shape = [1] * len(field_shape)
        broadcast_shape[axis_idx] = coord.shape[0]


        coord_reshaped = coord.view(*broadcast_shape)
        return coord_reshaped.expand(*field_shape)

    def get_derivative(self, field_name: str, axis: str, order: int) -> torch.Tensor:
        return self.derivative_provider.get_derivative(field_name, axis, order).to(
            self.device
        )

    def get_constant(self, name: str) -> float:
        if name not in self.constants:
            raise KeyError(f"Constant '{name}' not found in context")
        return self.constants[name]

    def diff(self, expression: torch.Tensor, axis: str, order: int = 1) -> torch.Tensor:
        return self.derivative_provider.diff(expression, axis, order)

    @property
    def spatial_axes(self) -> list[str]:
        return self.dataset.spatial_axes
