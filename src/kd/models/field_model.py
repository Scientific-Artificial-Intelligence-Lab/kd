
from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


_ACTIVATIONS: dict[str, Callable[[], Callable[[Tensor], Tensor]]] = {
    "tanh": lambda: nn.Tanh(),
    "relu": lambda: nn.ReLU(),
    "sin": lambda: torch.sin,
}


_STD_FLOOR = 1.0


class FieldModel(nn.Module):

    def __init__(
        self,
        coord_names: list[str],
        field_names: list[str],
        hidden_sizes: list[int] | None = None,
        activation: str = "tanh",
    ) -> None:
        super().__init__()


        if not coord_names:
            raise ValueError("coord_names must not be empty")
        if not field_names:
            raise ValueError("field_names must not be empty")
        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        if not hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Supported: {sorted(_ACTIVATIONS)}"
            )


        self.coord_names = list(coord_names)
        self.field_names = list(field_names)
        self.n_coords = len(coord_names)
        self.n_fields = len(field_names)


        act_factory = _ACTIVATIONS[activation]
        layers: list[nn.Module | Callable[[Tensor], Tensor]] = []
        in_size = self.n_coords
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(act_factory())
            in_size = h


        wrapped: list[nn.Module] = []
        for layer in layers:
            if isinstance(layer, nn.Module):
                wrapped.append(layer)
            else:
                wrapped.append(_FuncModule(layer))
        self.trunk = nn.Sequential(*wrapped)


        self.head = nn.Linear(in_size, self.n_fields)


        self._register_norm_buffers()





    def _register_norm_buffers(self) -> None:
        for name in self.coord_names:
            self.register_buffer(f"coord_{name}_mean", torch.tensor(0.0))
            self.register_buffer(f"coord_{name}_std", torch.tensor(1.0))
        for name in self.field_names:
            self.register_buffer(f"field_{name}_mean", torch.tensor(0.0))
            self.register_buffer(f"field_{name}_std", torch.tensor(1.0))

    def set_normalization(
        self,
        coord_stats: dict[str, tuple[Tensor, Tensor]],
        field_stats: dict[str, tuple[Tensor, Tensor]],
    ) -> None:
        for name in self.coord_names:
            mean, std = coord_stats[name]
            std = _floor_std(std)

            getattr(self, f"coord_{name}_mean").copy_(mean)
            getattr(self, f"coord_{name}_std").copy_(std)

        for name in self.field_names:
            mean, std = field_stats[name]
            std = _floor_std(std)
            getattr(self, f"field_{name}_mean").copy_(mean)
            getattr(self, f"field_{name}_std").copy_(std)







    def forward(self, **coords: Tensor) -> dict[str, Tensor]:

        normed: list[Tensor] = []
        for name in self.coord_names:
            c = coords[name]
            mean: Tensor = getattr(self, f"coord_{name}_mean")
            std: Tensor = getattr(self, f"coord_{name}_std")
            normed.append((c - mean) / std)

        inp = torch.stack(normed, dim=-1)


        hidden = self.trunk(inp)
        raw_out = self.head(hidden)


        result: dict[str, Tensor] = {}
        for i, name in enumerate(self.field_names):
            f_mean: Tensor = getattr(self, f"field_{name}_mean")
            f_std: Tensor = getattr(self, f"field_{name}_std")
            result[name] = raw_out[..., i] * f_std + f_mean

        return result







class _FuncModule(nn.Module):

    def __init__(self, func: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self._func = func

    def forward(self, x: Tensor) -> Tensor:
        return self._func(x)


def _floor_std(std: Tensor) -> Tensor:
    if std.item() == 0.0:
        return torch.tensor(_STD_FLOOR, dtype=std.dtype, device=std.device)
    return std
