
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.io as sio
import torch

from kd.data.layouts.kd_npz import read_kd_npz




from kd.data.noise import (
    NOISE_SCALE_MAX as _NOISE_SCALE_MAX,
)
from kd.data.noise import (
    NOISE_SCALE_STD as _NOISE_SCALE_STD,
)
from kd.data.noise import (
    NoiseScale as NoiseScale,
)
from kd.data.noise import (
    discover_unnormalized,
)
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)


_BURGERS_FIELD_U = "usol"
_BURGERS_FIELD_X = "x"
_BURGERS_FIELD_T = "t"


_FISHER_FIELD_U = "U"



_FISHER_EDGE_TRIM: slice = slice(1, -1)




_KDV_FIELD_U = "uu"
_KDV_FIELD_T = "tt"





_PDE_COMPOUND_NX = 100
_PDE_COMPOUND_NT = 251
_PDE_COMPOUND_X_RANGE = (1.0, 2.0)
_PDE_COMPOUND_T_RANGE = (0.0, 0.5)


_PDE_COMPOUND_X_TRIM: slice = slice(10, 90)






_PDE_DIVIDE_NX = 100
_PDE_DIVIDE_NT = 251
_PDE_DIVIDE_X_RANGE = (1.0, 2.0)
_PDE_DIVIDE_T_RANGE = (0.0, 1.0)


def load_chafee_infante_npy(directory: str | Path) -> PDEDataset:
    d = Path(directory)
    u_np = _load_npy(d / "chafee_infante_CI.npy")
    x_np = _load_npy(d / "chafee_infante_x.npy").flatten()
    t_np = _load_npy(d / "chafee_infante_t.npy").flatten()

    if u_np.ndim != 2:
        raise ValueError(f"Expected CI to be 2D (N_x, N_t), got shape {u_np.shape}")

    x_tensor = torch.from_numpy(x_np).to(torch.float64)
    t_tensor = torch.from_numpy(t_np).to(torch.float64)
    u_tensor = torch.from_numpy(u_np).to(torch.float64)

    return PDEDataset(
        name="chafee_infante",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x_tensor),
            "t": AxisInfo(name="t", values=t_tensor),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u_tensor)},
        lhs_field="u",
        lhs_axis="t",
    )


def _load_npy(path: Path) -> npt.NDArray[Any]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return np.load(str(path))


def load_burgers_mat(path: str | Path) -> PDEDataset:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Data file not found: {resolved}")

    data = sio.loadmat(str(resolved))




    x_np = data[_BURGERS_FIELD_X].flatten()
    t_np = data[_BURGERS_FIELD_T].flatten()
    u_np = data[_BURGERS_FIELD_U]
    if u_np.ndim != 2:
        raise ValueError(f"Expected usol to be 2D (N_x, N_t), got shape {u_np.shape}")
    if np.iscomplexobj(u_np):
        u_np = np.real(u_np)


    x_tensor = torch.from_numpy(x_np).to(torch.float64)
    t_tensor = torch.from_numpy(t_np).to(torch.float64)
    u_tensor = torch.from_numpy(u_np).to(torch.float64)

    return PDEDataset(
        name="burgers",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x_tensor),
            "t": AxisInfo(name="t", values=t_tensor),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u_tensor)},
        lhs_field="u",
        lhs_axis="t",
    )


def load_fisher_linear_mat(path: str | Path) -> PDEDataset:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Data file not found: {resolved}")

    data = sio.loadmat(str(resolved))
    u_raw = data[_FISHER_FIELD_U]
    if u_raw.ndim != 2:
        raise ValueError(f"Expected U to be 2D (N_t, N_x), got shape {u_raw.shape}")

    u_np = u_raw[_FISHER_EDGE_TRIM, _FISHER_EDGE_TRIM].T
    x_np = data[_BURGERS_FIELD_X].flatten()[_FISHER_EDGE_TRIM]
    t_np = data[_BURGERS_FIELD_T].flatten()[_FISHER_EDGE_TRIM]

    x_tensor = torch.from_numpy(x_np).to(torch.float64)
    t_tensor = torch.from_numpy(t_np).to(torch.float64)
    u_tensor = torch.from_numpy(np.ascontiguousarray(u_np)).to(torch.float64)

    return PDEDataset(
        name="fisher_linear",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x_tensor),
            "t": AxisInfo(name="t", values=t_tensor),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u_tensor)},
        lhs_field="u",
        lhs_axis="t",
    )


def load_fisher_nonlinear_mat(path: str | Path) -> PDEDataset:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Data file not found: {resolved}")

    data = sio.loadmat(str(resolved))
    u_raw = data[_FISHER_FIELD_U]
    if u_raw.ndim != 2:
        raise ValueError(f"Expected U to be 2D (N_t, N_x), got shape {u_raw.shape}")


    u_np = u_raw[_FISHER_EDGE_TRIM, _FISHER_EDGE_TRIM].T
    x_np = data[_BURGERS_FIELD_X].flatten()[_FISHER_EDGE_TRIM]
    t_np = data[_BURGERS_FIELD_T].flatten()[_FISHER_EDGE_TRIM]

    x_tensor = torch.from_numpy(x_np).to(torch.float64)
    t_tensor = torch.from_numpy(t_np).to(torch.float64)
    u_tensor = torch.from_numpy(np.ascontiguousarray(u_np)).to(torch.float64)

    return PDEDataset(
        name="fisher_nonlinear",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x_tensor),
            "t": AxisInfo(name="t", values=t_tensor),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u_tensor)},
        lhs_field="u",
        lhs_axis="t",
    )


def load_pde_compound_npy(path: str | Path) -> PDEDataset:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Data file not found: {resolved}")

    raw = _load_npy(resolved)
    if raw.ndim != 2:
        raise ValueError(f"Expected PDE_compound npy to be 2D, got shape {raw.shape}")


    u_full = raw.T
    if u_full.shape != (_PDE_COMPOUND_NX, _PDE_COMPOUND_NT):
        raise ValueError(
            f"Expected PDE_compound transposed shape "
            f"({_PDE_COMPOUND_NX}, {_PDE_COMPOUND_NT}), got "
            f"{u_full.shape}"
        )
    u_np = np.ascontiguousarray(u_full[_PDE_COMPOUND_X_TRIM,:])
    x_full = np.linspace(
        _PDE_COMPOUND_X_RANGE[0],
        _PDE_COMPOUND_X_RANGE[1],
        _PDE_COMPOUND_NX,
    )
    x_np = x_full[_PDE_COMPOUND_X_TRIM]
    t_np = np.linspace(
        _PDE_COMPOUND_T_RANGE[0],
        _PDE_COMPOUND_T_RANGE[1],
        _PDE_COMPOUND_NT,
    )

    x_tensor = torch.from_numpy(x_np).to(torch.float64)
    t_tensor = torch.from_numpy(t_np).to(torch.float64)
    u_tensor = torch.from_numpy(u_np).to(torch.float64)

    return PDEDataset(
        name="pde_compound",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x_tensor),
            "t": AxisInfo(name="t", values=t_tensor),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u_tensor)},
        lhs_field="u",
        lhs_axis="t",
    )


def load_pde_divide_npy(path: str | Path) -> PDEDataset:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Data file not found: {resolved}")

    raw = _load_npy(resolved)
    if raw.ndim != 2:
        raise ValueError(f"Expected PDE_divide npy to be 2D, got shape {raw.shape}")

    u_full = raw.T
    if u_full.shape != (_PDE_DIVIDE_NX, _PDE_DIVIDE_NT):
        raise ValueError(
            f"Expected PDE_divide transposed shape "
            f"({_PDE_DIVIDE_NX}, {_PDE_DIVIDE_NT}), got {u_full.shape}"
        )
    u_np = np.ascontiguousarray(u_full)
    x_np = np.linspace(
        _PDE_DIVIDE_X_RANGE[0],
        _PDE_DIVIDE_X_RANGE[1],
        _PDE_DIVIDE_NX,
    )
    t_np = np.linspace(
        _PDE_DIVIDE_T_RANGE[0],
        _PDE_DIVIDE_T_RANGE[1],
        _PDE_DIVIDE_NT,
    )

    x_tensor = torch.from_numpy(x_np).to(torch.float64)
    t_tensor = torch.from_numpy(t_np).to(torch.float64)
    u_tensor = torch.from_numpy(u_np).to(torch.float64)

    return PDEDataset(
        name="pde_divide",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x_tensor),
            "t": AxisInfo(name="t", values=t_tensor),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u_tensor)},
        lhs_field="u",
        lhs_axis="t",
    )


def load_kdv_mat(path: str | Path) -> PDEDataset:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Data file not found: {resolved}")

    data = sio.loadmat(str(resolved))
    u_raw = data[_KDV_FIELD_U]
    if u_raw.ndim != 2:
        raise ValueError(f"Expected uu to be 2D (N_x, N_t), got shape {u_raw.shape}")

    u_np = np.ascontiguousarray(u_raw)
    x_np = data[_BURGERS_FIELD_X].flatten()
    t_np = data[_KDV_FIELD_T].flatten()

    x_tensor = torch.from_numpy(x_np).to(torch.float64)
    t_tensor = torch.from_numpy(t_np).to(torch.float64)
    u_tensor = torch.from_numpy(u_np).to(torch.float64)

    return PDEDataset(
        name="kdv",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x_tensor),
            "t": AxisInfo(name="t", values=t_tensor),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u_tensor)},
        lhs_field="u",
        lhs_axis="t",
    )


def add_gaussian_noise(
    dataset: PDEDataset,
    level: float,
    seed: int,
    *,
    scale: NoiseScale = "std",
) -> PDEDataset:
    if level < 0.0:
        raise ValueError(f"noise level must be non-negative, got {level}")
    if scale not in (_NOISE_SCALE_STD, _NOISE_SCALE_MAX):
        raise ValueError(
            f"scale must be {_NOISE_SCALE_STD!r} or {_NOISE_SCALE_MAX!r}, got {scale!r}"
        )
    if dataset.fields is None:
        raise ValueError("dataset.fields must be defined")

    generator = torch.Generator().manual_seed(seed)
    noisy_fields: dict[str, FieldData] = {}
    for name, field in dataset.fields.items():
        noisy_values = discover_unnormalized(
            field.values,
            level,
            generator=generator,
            scale=scale,
        )
        noisy_fields[name] = FieldData(name=name, values=noisy_values)

    axis_order = list(dataset.axis_order) if dataset.axis_order is not None else None
    return PDEDataset(
        name=f"{dataset.name}_noisy",
        task_type=dataset.task_type,
        topology=dataset.topology,
        axes=dataset.axes,
        axis_order=axis_order,
        fields=noisy_fields,
        lhs_field=dataset.lhs_field,
        lhs_axis=dataset.lhs_axis,
        noise_level=level,
        ground_truth=dataset.ground_truth,
    )


def _build_pde_dataset_from_npz(
    path: str | Path,
    *,
    name: str,
    lhs_field: str,
    lhs_axis: str,
    periodic_axes: set[str] | frozenset[str],
    dtype: torch.dtype = torch.float32,
) -> PDEDataset:
    if not dtype.is_floating_point:
        raise ValueError(f"dtype must be a floating torch dtype, got {dtype}")

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Data file not found: {resolved}")

    arrays = read_kd_npz(resolved, field=lhs_field)
    axis_order = arrays.axis_order
    axes_np = arrays.axes
    field_np = arrays.fields[lhs_field]

    expected_shape = tuple(axes_np[axis].shape[0] for axis in axis_order)
    if field_np.shape != expected_shape:
        raise ValueError(
            f"field '{lhs_field}' shape mismatch: expected {expected_shape}, "
            f"got {field_np.shape}"
        )
    if lhs_axis not in axis_order:
        raise ValueError(f"lhs_axis '{lhs_axis}' not found in axis_order")

    axes = {
        axis: AxisInfo(
            name=axis,
            values=_axis_tensor(values, dtype),
            is_periodic=axis in periodic_axes,
        )
        for axis, values in axes_np.items()
    }
    fields = {
        lhs_field: FieldData(
            name=lhs_field,
            values=torch.from_numpy(field_np).to(dtype=dtype),
        ),
    }
    return PDEDataset(
        name=name,
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes=axes,
        axis_order=axis_order,
        fields=fields,
        lhs_field=lhs_field,
        lhs_axis=lhs_axis,
        ground_truth=None,
    )


def _axis_tensor(
    values: npt.NDArray[np.float64],
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.from_numpy(values)
    if dtype == torch.float32:
        return tensor
    return tensor.to(dtype=dtype)
