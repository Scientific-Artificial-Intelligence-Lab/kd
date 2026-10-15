
from pathlib import Path
from typing import Final

import numpy as np

from kd.data.regression import TabularDataset
from kd.data.tabular_bridge import dataset_from_tabular

TABULAR_NPZ_FORMAT: Final[str] = "kd-tabular-npz-v1"


def load_tabular(path: Path, *, name: str | None = None) -> TabularDataset:
    with np.load(path, allow_pickle=False) as data:
        if data["format"].item() != TABULAR_NPZ_FORMAT:
            raise ValueError(f"format must be {TABULAR_NPZ_FORMAT}")
        if data["var_names"].ndim != 1:
            raise ValueError("var_names must be a string vector")
        for key in ("X", "y"):
            if data[key].dtype.kind not in "fiu":
                raise ValueError(f"{key} must be a real numeric array")
        table = TabularDataset(
            X=np.asarray(data["X"], dtype=np.float64),
            y=np.asarray(data["y"], dtype=np.float64),
            var_names=tuple(data["var_names"].tolist()),
            target_name=data["target_name"].item(),
            name=path.stem if name is None else name,
            source=str(path),
            description="",
        )
    dataset_from_tabular(table)
    return table
