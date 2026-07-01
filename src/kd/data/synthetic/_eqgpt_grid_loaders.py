
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from kd.data.loaders.csv_grid import read_grid_csv
from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd.data.synthetic._loaders import _load_mat, _resolve_eqgpt_file

_FIELD_U = "u"

_AXIS_X = "x"
_AXIS_Y = "y"
_AXIS_T = "t"

_EQ_6_2_12_FILE = "eqgpt_eq_6_2_12.csv"
_EQ_6_2_12_SUBDIR = ""
_EQ_6_2_12_NAME = "eq-6-2-12"
_EQ_6_2_12_EXPECTED_SHAPE = (501, 501)
_EQ_6_2_12_X_DOMAIN = (0.0, 5.0)
_EQ_6_2_12_T_DOMAIN = (0.0, 10.0)
_EQ_6_2_12_GROUND_TRUTH = "u_t = -0.1*u_x_t - 0.1*u_x"

_BURGERS_2D_FILE = "eqgpt_burgers_2d.mat"
_BURGERS_2D_SUBDIR = ""
_BURGERS_2D_NAME = "burgers-2d"
_BURGERS_2D_U_KEY = "u"
_BURGERS_2D_X_KEY = "x"
_BURGERS_2D_Y_KEY = "y"
_BURGERS_2D_T_KEY = "t"
_BURGERS_2D_GROUND_TRUTH = "u_t = -u*u_x - u*u_y + 0.01*u_xx + 0.01*u_yy"

_UNIFORM_SPACING_RTOL = 1e-3


def _validate_uniform_axis(
    coord: np.ndarray, axis_name: str, dataset_name: str
) -> None:
    diffs = np.diff(coord)
    if diffs.size == 0:
        raise ValueError(
            f"{dataset_name} axis '{axis_name}' must contain at least two points"
        )
    spacing = diffs.mean()
    if not np.allclose(diffs, spacing, rtol=_UNIFORM_SPACING_RTOL, atol=0.0):
        raise ValueError(
            f"{dataset_name} axis '{axis_name}' is not uniformly spaced "
            f"within rtol={_UNIFORM_SPACING_RTOL}"
        )


def load_eq_6_2_12(data_dir: Path | str | None = None) -> PDEDataset:
    csv_path = _resolve_eqgpt_file(_EQ_6_2_12_FILE, _EQ_6_2_12_SUBDIR, data_dir)
    raw = read_grid_csv(
        csv_path,
        name=_EQ_6_2_12_NAME,
        expected_shape=_EQ_6_2_12_EXPECTED_SHAPE,
    )

    u_np = raw.T
    nx, nt = u_np.shape
    x_np = np.linspace(*_EQ_6_2_12_X_DOMAIN, nx, dtype=np.float64)
    t_np = np.linspace(*_EQ_6_2_12_T_DOMAIN, nt, dtype=np.float64)

    u = torch.from_numpy(u_np)
    x = torch.from_numpy(x_np)
    t = torch.from_numpy(t_np)

    return PDEDataset(
        name=_EQ_6_2_12_NAME,
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            _AXIS_X: AxisInfo(name=_AXIS_X, values=x, is_periodic=False),
            _AXIS_T: AxisInfo(name=_AXIS_T, values=t, is_periodic=False),
        },
        axis_order=[_AXIS_X, _AXIS_T],
        fields={_FIELD_U: FieldData(name=_FIELD_U, values=u)},
        lhs_field=_FIELD_U,
        lhs_axis=_AXIS_T,
        lhs_order=1,
        ground_truth=_EQ_6_2_12_GROUND_TRUTH,
    )


def load_burgers_2d(data_dir: Path | str | None = None) -> PDEDataset:
    mat_path = _resolve_eqgpt_file(_BURGERS_2D_FILE, _BURGERS_2D_SUBDIR, data_dir)
    mat_data = _load_mat(mat_path)

    x_np = np.asarray(mat_data[_BURGERS_2D_X_KEY], dtype=np.float64).flatten()
    y_np = np.asarray(mat_data[_BURGERS_2D_Y_KEY], dtype=np.float64).flatten()
    t_np = np.asarray(mat_data[_BURGERS_2D_T_KEY], dtype=np.float64).flatten()

    _validate_uniform_axis(x_np, _AXIS_X, _BURGERS_2D_NAME)
    _validate_uniform_axis(y_np, _AXIS_Y, _BURGERS_2D_NAME)
    _validate_uniform_axis(t_np, _AXIS_T, _BURGERS_2D_NAME)

    u_raw = np.asarray(mat_data[_BURGERS_2D_U_KEY], dtype=np.float64)
    expected_shape = (len(t_np), len(y_np), len(x_np))
    if u_raw.shape != expected_shape:
        raise ValueError(
            f"{_BURGERS_2D_NAME} field '{_FIELD_U}' has shape {u_raw.shape}; "
            f"expected {expected_shape} for raw (t, y, x) layout"
        )

    u_np = np.transpose(u_raw, (2, 1, 0))

    u = torch.from_numpy(u_np)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    t = torch.from_numpy(t_np)

    return PDEDataset(
        name=_BURGERS_2D_NAME,
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            _AXIS_X: AxisInfo(name=_AXIS_X, values=x, is_periodic=False),
            _AXIS_Y: AxisInfo(name=_AXIS_Y, values=y, is_periodic=False),
            _AXIS_T: AxisInfo(name=_AXIS_T, values=t, is_periodic=False),
        },
        axis_order=[_AXIS_X, _AXIS_Y, _AXIS_T],
        fields={_FIELD_U: FieldData(name=_FIELD_U, values=u)},
        lhs_field=_FIELD_U,
        lhs_axis=_AXIS_T,
        lhs_order=1,
        ground_truth=_BURGERS_2D_GROUND_TRUTH,
    )


__all__ = ["load_burgers_2d", "load_eq_6_2_12"]
