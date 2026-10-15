
from collections.abc import Iterable
from typing import Final

import numpy as np
import torch

from kd.data.containers import Inventory
from kd.data.layouts.kd_npz import decode_axis_order
from kd.data.schema import PDEDataset

SCATTERED_NPZ_FORMAT: Final[str] = "kd-scattered-npz-v1"


class ScatteredNpzLayout:

    name = "kd-scattered-npz"

    def matches(self, inventory: Inventory) -> bool:
        entry = inventory.entries.get("format")
        return (
            inventory.container == "npz"
            and entry is not None
            and entry.shape == ()
            and entry.read().item() == SCATTERED_NPZ_FORMAT
        )

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
        if select is not None:
            raise ValueError("kd-scattered-npz has no sample axis; omit select")
        if periodic is not None and tuple(periodic):
            raise ValueError(
                "kd-scattered-npz does not support periodic axes; pass periodic=[]"
            )
        with np.load(inventory.path, allow_pickle=False) as data:
            if data["format"].item() != SCATTERED_NPZ_FORMAT:
                raise ValueError(f"format must be {SCATTERED_NPZ_FORMAT}")
            axes = decode_axis_order(data["axis_order"])
            fields = decode_axis_order(data["field_names"])
            return PDEDataset.from_scatter(
                coords={axis: data[f"coord__{axis}"] for axis in axes},
                fields={field: data[f"field__{field}"] for field in fields},
                lhs=lhs,
                name=name,
                dtype=dtype,
            )


SCATTERED_NPZ = ScatteredNpzLayout()
