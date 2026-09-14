
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import h5py
import numpy as np

from kd.data._columns import column_indices

ContainerKind = Literal["npy", "npz", "mat", "mat73", "hdf5", "csv"]
KEYLESS_ARRAY_KEY: Final[str] = "array"

_MATLAB_METADATA_KEYS = frozenset({"__header__", "__version__", "__globals__"})
_SUPPORTED_SUFFIXES = (".npy", ".npz", ".mat", ".h5", ".hdf5", ".csv")


@dataclass(frozen=True, kw_only=True)
class ArrayEntry:

    key: str
    shape: tuple[int, ...]
    dtype: str
    numeric: bool
    read: Callable[[], np.ndarray]
    read_index0: Callable[[int], np.ndarray]


@dataclass(frozen=True, kw_only=True)
class Inventory:

    path: Path
    container: ContainerKind
    entries: dict[str, ArrayEntry]

    def describe(self) -> str:
        lines = [f"{self.container} {self.path}"]
        lines.extend(
            f"{entry.key} shape={entry.shape} dtype={entry.dtype}"
            for entry in self.entries.values()
        )
        return "\n".join(lines)


def read_inventory(path: str | Path) -> Inventory:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"array container not found: {resolved}")

    suffix = resolved.suffix.lower()
    if suffix == ".xlsx":
        raise ValueError(f"xlsx is not an array container: {resolved}")
    if suffix == ".npy":
        return _read_npy_inventory(resolved)
    if suffix == ".npz":
        return _read_npz_inventory(resolved)
    if suffix == ".mat":
        if h5py.is_hdf5(resolved):
            return _read_hdf5_inventory(resolved, container="mat73")
        return _read_mat_inventory(resolved)
    if suffix in {".h5", ".hdf5"}:
        return _read_hdf5_inventory(resolved, container="hdf5")
    if suffix == ".csv":
        return _read_csv_inventory(resolved)
    supported = ", ".join(_SUPPORTED_SUFFIXES)
    raise ValueError(
        f"unsupported container suffix '{resolved.suffix}' for {resolved}; "
        f"supported suffixes: {supported}"
    )


def as_axis_vector(key: str, values: np.ndarray) -> np.ndarray:
    original_shape = values.shape
    squeezed = np.squeeze(values)
    if squeezed.ndim != 1:
        raise ValueError(f"'{key}' with shape {original_shape} is not a 1-D coordinate")
    return np.asarray(squeezed, dtype=np.float64)


def _entry(
    key: str,
    array: np.ndarray,
    *,
    read: Callable[[], np.ndarray],
    read_index0: Callable[[int], np.ndarray],
) -> ArrayEntry:
    return ArrayEntry(
        key=key,
        shape=tuple(array.shape),
        dtype=str(array.dtype),
        numeric=bool(np.issubdtype(array.dtype, np.number)),
        read=read,
        read_index0=read_index0,
    )


def _read_npy_inventory(path: Path) -> Inventory:
    array = np.load(path, mmap_mode="r")
    entry = _entry(
        KEYLESS_ARRAY_KEY,
        array,
        read=lambda: np.load(path, mmap_mode="r"),
        read_index0=lambda index: np.asarray(np.load(path, mmap_mode="r")[index]),
    )
    return Inventory(path=path, container="npy", entries={KEYLESS_ARRAY_KEY: entry})


def _read_npz_array(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data[key])


def _read_npz_index0(path: Path, key: str, index: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data[key][index])


def _deferred_readers(
    path: Path,
    key: str,
    *,
    read_array: Callable[[Path, str], np.ndarray],
    read_index0: Callable[[Path, str, int], np.ndarray],
) -> tuple[Callable[[], np.ndarray], Callable[[int], np.ndarray]]:
    def read() -> np.ndarray:
        return read_array(path, key)

    def read_leading_index(index: int) -> np.ndarray:
        return read_index0(path, key, index)

    return read, read_leading_index


def _read_npz_inventory(path: Path) -> Inventory:
    entries: dict[str, ArrayEntry] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            array = np.asarray(data[key])
            read, read_index0 = _deferred_readers(
                path,
                key,
                read_array=_read_npz_array,
                read_index0=_read_npz_index0,
            )
            entries[key] = _entry(
                key,
                array,
                read=read,
                read_index0=read_index0,
            )
    return Inventory(path=path, container="npz", entries=entries)


def _load_mat(path: Path) -> dict[str, object]:
    import scipy.io as sio

    return dict(sio.loadmat(path))


def _read_mat_array(path: Path, key: str) -> np.ndarray:
    return np.asarray(_load_mat(path)[key])


def _read_mat_index0(path: Path, key: str, index: int) -> np.ndarray:
    return np.asarray(_read_mat_array(path, key)[index])


def _read_mat_inventory(path: Path) -> Inventory:
    entries: dict[str, ArrayEntry] = {}
    for key, value in _load_mat(path).items():
        if key in _MATLAB_METADATA_KEYS:
            continue
        array = np.asarray(value)
        read, read_index0 = _deferred_readers(
            path,
            key,
            read_array=_read_mat_array,
            read_index0=_read_mat_index0,
        )
        entries[key] = _entry(
            key,
            array,
            read=read,
            read_index0=read_index0,
        )
    return Inventory(path=path, container="mat", entries=entries)


def _read_hdf5_array(path: Path, key: str) -> np.ndarray:
    with h5py.File(path, "r") as data:
        return np.asarray(data[key][...])


def _read_hdf5_index0(path: Path, key: str, index: int) -> np.ndarray:
    with h5py.File(path, "r") as data:
        return np.asarray(data[key][index])


def _read_hdf5_inventory(path: Path, *, container: ContainerKind) -> Inventory:
    entries: dict[str, ArrayEntry] = {}
    with h5py.File(path, "r") as data:

        def add_dataset(key: str, item: h5py.Group | h5py.Dataset) -> None:
            if not isinstance(item, h5py.Dataset):
                return
            read, read_index0 = _deferred_readers(
                path,
                key,
                read_array=_read_hdf5_array,
                read_index0=_read_hdf5_index0,
            )
            entries[key] = ArrayEntry(
                key=key,
                shape=tuple(item.shape),
                dtype=str(item.dtype),
                numeric=bool(np.issubdtype(item.dtype, np.number)),
                read=read,
                read_index0=read_index0,
            )

        data.visititems(add_dataset)
    return Inventory(path=path, container=container, entries=entries)


def _read_csv_array(path: Path) -> np.ndarray:


    return np.loadtxt(path, delimiter=",", ndmin=2, dtype=np.float64)


def _read_csv_inventory(path: Path) -> Inventory:
    columns = _read_csv_columns(path)
    if columns is not None:
        entries = {key: _csv_entry(key, values) for key, values in columns.items()}
        return Inventory(path=path, container="csv", entries=entries)
    array = _read_csv_array(path)
    entry = _entry(
        KEYLESS_ARRAY_KEY,
        array,
        read=lambda: _read_csv_array(path),
        read_index0=lambda index: _read_csv_array(path)[index],
    )
    return Inventory(path=path, container="csv", entries={KEYLESS_ARRAY_KEY: entry})


def _csv_entry(key: str, values: np.ndarray) -> ArrayEntry:
    return _entry(
        key,
        values,
        read=lambda: values,
        read_index0=lambda index: np.asarray(values[index]),
    )


def _read_csv_columns(path: Path) -> dict[str, np.ndarray] | None:
    import csv

    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = (
            (number, row)
            for number, row in enumerate(csv.reader(handle), 1)
            if row and not row[0].lstrip().startswith("#")
        )
        _, header = next(rows, (0, []))
        if not header:
            return None
        try:

            np.asarray(
                ",".join(header).partition("#")[0].rstrip(", ").split(","),
                dtype=np.float64,
            )
        except ValueError:
            pass
        else:
            return None
        if any(not name for name in header):
            raise ValueError("CSV headers must be non-empty column names")
        indices = column_indices(header, header, kind="CSV")
        columns: dict[str, list[str]] = {name: [] for name in indices}
        for number, row in rows:
            if len(row) != len(header):
                raise ValueError(
                    f"CSV row {number} has {len(row)} columns; header has {len(header)}"
                )
            for name, index in indices.items():
                columns[name].append(row[index])
    return {name: _csv_column(values) for name, values in columns.items()}


def _csv_column(values: list[str]) -> np.ndarray:
    numeric = ["nan" if value in {"", "Indeterminate"} else value for value in values]
    try:
        return np.asarray(numeric, dtype=np.float64)
    except ValueError:
        return np.asarray(values, dtype=str)


__all__ = [
    "ArrayEntry",
    "ContainerKind",
    "Inventory",
    "KEYLESS_ARRAY_KEY",
    "as_axis_vector",
    "read_inventory",
]
