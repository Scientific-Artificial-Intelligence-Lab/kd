
from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import torch

from kd.data.containers import Inventory
from kd.data.layouts.kd_npz import KD_NPZ
from kd.data.layouts.pdebench import PDEBENCH_1D, PDEBENCH_CFD
from kd.data.layouts.plain_grid import PLAIN_GRID
from kd.data.layouts.scattered_npz import SCATTERED_NPZ
from kd.data.schema import PDEDataset
from kd.data.tabular_file import TABULAR_NPZ


class Recognizer(Protocol):

    name: str

    def matches(self, inventory: Inventory) -> bool:
        ...


class Layout(Recognizer, Protocol):

    def build(
        self,
        inventory: Inventory,
        *,
        select: dict[str, int] | None,
        lhs: str | None,
        periodic: Iterable[str] | None,
        name: str,
        dtype: torch.dtype,
    ) -> PDEDataset:
        ...


LAYOUTS: tuple[Layout, ...] = (
    PDEBENCH_1D,
    PDEBENCH_CFD,
    KD_NPZ,
    SCATTERED_NPZ,
    PLAIN_GRID,
)
RECOGNIZERS: tuple[Recognizer, ...] = (*LAYOUTS, TABULAR_NPZ)


def recognize(inventory: Inventory) -> list[Recognizer]:
    return [layout for layout in RECOGNIZERS if layout.matches(inventory)]


def layout_by_name(name: str) -> Layout:
    for layout in LAYOUTS:
        if layout.name == name:
            return layout
    names = [layout.name for layout in RECOGNIZERS]
    raise KeyError(f"unknown layout '{name}'; available layouts: {names}")


__all__ = [
    "LAYOUTS",
    "RECOGNIZERS",
    "Layout",
    "Recognizer",
    "layout_by_name",
    "recognize",
]
