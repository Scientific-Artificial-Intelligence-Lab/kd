
from pathlib import Path
from typing import Final

import numpy as np

from kd.data.containers import Inventory, read_real
from kd.data.regression import TabularDataset
from kd.data.source import DatasetSource, file_sha256
from kd.data.tabular_bridge import dataset_from_tabular

TABULAR_NPZ_FORMAT: Final[str] = "kd-tabular-npz-v1"
TABULAR_NPZ_LAYOUT: Final[str] = "kd-tabular-npz"


class TabularNpzRecognizer:

    name = TABULAR_NPZ_LAYOUT

    def matches(self, inventory: Inventory) -> bool:
        entry = inventory.entries.get("format")
        return (
            inventory.container == "npz"
            and entry is not None
            and entry.shape == ()
            and entry.read().item() == TABULAR_NPZ_FORMAT
        )


TABULAR_NPZ = TabularNpzRecognizer()


def load_tabular(path: Path, *, name: str | None = None) -> TabularDataset:
    path = path.resolve()
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
            source="",
            description="",
            file_source=DatasetSource(
                path=str(path),
                sha256=file_sha256(path),
                container="npz",
                layout=TABULAR_NPZ_LAYOUT,
                hints={},
                select=None,
            ),
        )
    dataset_from_tabular(table)
    return table


def _table_mapping(
    inventory: Inventory, lhs: str, fields: dict[str, str] | None
) -> dict[str, str]:
    if fields is None:
        if lhs not in inventory.entries:
            raise ValueError(
                f"unknown target {lhs!r}; available columns: {list(inventory.entries)}"
            )
        return {key: key for key in inventory.entries}
    if lhs not in fields:
        raise ValueError(
            f"fields must include the target {lhs!r} as a logical name; "
            f"got fields={fields}"
        )
    return dict(fields)


def load_table_columns(
    inventory: Inventory, *, lhs: str, fields: dict[str, str] | None, name: str
) -> tuple[TabularDataset, dict[str, str]]:
    mapping = _table_mapping(inventory, lhs, fields)
    if len(mapping) < 2:
        raise ValueError(
            f"tabular needs at least one input column besides the target {lhs!r}"
        )
    advice = "; select columns with fields=" if fields is None else ""
    columns = {}
    for label, key in mapping.items():
        values = read_real(inventory.entry(key), inventory.container, advice=advice)
        if values.ndim != 1:
            raise ValueError(
                f"column/array {key!r} must be a real numeric vector; got shape "
                f"{values.shape}{advice}"
            )
        columns[label] = values
    lengths = {label: values.shape[0] for label, values in columns.items()}
    if len(set(lengths.values())) != 1 or not lengths[lhs]:
        raise ValueError(f"tabular columns need one shape with rows; got {lengths}")
    inputs = [label for label in mapping if label != lhs]
    table = TabularDataset(
        X=np.column_stack([columns[label] for label in inputs]),
        y=columns[lhs],
        var_names=tuple(inputs),
        target_name=lhs,
        name=name,
        source="",
        description="",
    )
    dataset_from_tabular(table)
    return table, mapping


__all__ = [
    "TABULAR_NPZ",
    "TABULAR_NPZ_FORMAT",
    "TABULAR_NPZ_LAYOUT",
    "TabularNpzRecognizer",
    "load_table_columns",
    "load_tabular",
]
