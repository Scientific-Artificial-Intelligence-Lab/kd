
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor


class ColumnDataset:

    def __init__(self, names: Sequence[str]) -> None:
        self.fields: dict[str, None] = dict.fromkeys(names)
        self.axes: None = None


class ColumnContext:

    def __init__(self, columns: Mapping[str, Tensor], unity: Tensor) -> None:
        self._columns = columns
        self._unity = unity
        self.dataset = ColumnDataset(tuple(columns))
        self.device = unity.device

    def unity_column(self) -> Tensor:
        return self._unity

    def get_variable(self, name: str) -> Tensor:
        return self._columns[name]

    def get_derivative(self, *_args: Any, **_kwargs: Any) -> Tensor:
        raise KeyError("named columns have no derivatives")

    def get_constant(self, name: str) -> float:
        raise KeyError(f"Unknown constant: {name}")


__all__ = ["ColumnContext", "ColumnDataset"]
