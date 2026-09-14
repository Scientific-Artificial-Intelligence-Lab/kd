
from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import torch

from kd.data.containers import Inventory
from kd.data.layouts.kd_npz import KD_NPZ
from kd.data.layouts.pdebench import PDEBENCH_1D, PDEBENCH_CFD
from kd.data.schema import PDEDataset


class Layout(Protocol):

    name: str

    def matches(self, inventory: Inventory) -> bool:
        ...

    def build(
        self,
        inventory: Inventory,
        *,
        select: dict[str, int] | None,
        lhs: str,
        periodic: Iterable[str] | None,
        name: str,
        dtype: torch.dtype,
    ) -> PDEDataset:
        ...


LAYOUTS: tuple[Layout, ...] = (PDEBENCH_1D, PDEBENCH_CFD, KD_NPZ)


def recognize(inventory: Inventory) -> list[Layout]:
    return [layout for layout in LAYOUTS if layout.matches(inventory)]


def layout_by_name(name: str) -> Layout:
    for layout in LAYOUTS:
        if layout.name == name:
            return layout
    names = [layout.name for layout in LAYOUTS]
    raise KeyError(f"unknown layout '{name}'; available layouts: {names}")


__all__ = ["LAYOUTS", "Layout", "layout_by_name", "recognize"]
