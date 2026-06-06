
from __future__ import annotations

from pathlib import Path

import torch

from kd.data.schema import PDEDataset
from kd.search.discover.data.loader import _build_pde_dataset_from_npz

_DEFAULT_DATA_PATH = Path("data/allen_cahn_2d_smoke.npz")
_DATASET_NAME = "allen_cahn_2d_smoke"
_AXIS_ORDER = ["x", "y", "t"]
_PERIODIC_AXES = frozenset({"x", "y"})


def load_allen_cahn_2d(
    data_path: Path | str = _DEFAULT_DATA_PATH,
    *,
    dtype: torch.dtype = torch.float32,
) -> PDEDataset:
    dataset = _build_pde_dataset_from_npz(
        data_path,
        name=_DATASET_NAME,
        lhs_field="u",
        lhs_axis="t",
        periodic_axes=_PERIODIC_AXES,
        dtype=dtype,
    )
    if dataset.axis_order != _AXIS_ORDER:
        raise ValueError(f"Allen-Cahn 2D axis_order must be {_AXIS_ORDER}")
    return dataset
