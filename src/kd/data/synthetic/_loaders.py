"""Data loaders for benchmark PDE datasets.

Loads pre-computed data from .npy and .mat files for:
- Chafee-Infante equation: u_t = u_xx - u + u^3
- KdV equation: u_t = -u * u_x - 0.0025 * u_xxx
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)

logger = logging.getLogger(__name__)


_FIELD_U = "u"


_AXIS_X = "x"
_AXIS_T = "t"


_DEFAULT_DATA_DIR = "data"


_CI_U_FILE = "chafee_infante_CI.npy"
_CI_X_FILE = "chafee_infante_x.npy"
_CI_T_FILE = "chafee_infante_t.npy"


_KDV_FILE = "KdV_equation.mat"
_KDV_X_KEY = "x"
_KDV_T_KEY = "tt"
_KDV_U_KEY = "uu"
_KDV_EXPECTED_NX = 256


_BURGERS_FILE = "Burgers_equation.mat"
_BURGERS_X_KEY = "x"
_BURGERS_T_KEY = "t"
_BURGERS_U_KEY = "usol"


_PDE_DIVIDE_FILE = "PDE_divide.npy"
_PDE_DIVIDE_X_RANGE = (1.0, 2.0)
_PDE_DIVIDE_T_RANGE = (0.0, 1.0)
_PDE_DIVIDE_NX = 100
_PDE_DIVIDE_NT = 251

_PDE_COMPOUND_FILE = "PDE_compound.npy"
_PDE_COMPOUND_X_RANGE = (1.0, 2.0)
_PDE_COMPOUND_T_RANGE = (0.0, 0.5)
_PDE_COMPOUND_NX = 100
_PDE_COMPOUND_NT = 251


def _load_mat(path: Path) -> dict[str, object]:
    """Load a MATLAB file without making scipy a typed dependency."""
    import scipy.io as sio

    return dict(sio.loadmat(str(path)))


def _find_project_root() -> Path:
    """Find project root by searching for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    return Path(__file__).resolve().parents[4]


def _resolve_data_dir(data_dir: Path | str | None) -> Path:
    """Resolve data directory.

    Search order:
    1. User-provided ``data_dir`` (if not None).
    2. Bundled package data: ``<kd_pkg>/_assets/data/`` — works after
       ``pip install`` because these files are packaged into the wheel.
    3. Source-repo fallback: ``<project_root>/data/`` — used
       during development when running from a checked-out source tree.
    """
    if data_dir is not None:
        return Path(data_dir)



    pkg_data = Path(__file__).resolve().parents[2] / "_assets" / "data"
    if pkg_data.exists():
        return pkg_data

    return _find_project_root() / _DEFAULT_DATA_DIR


def load_chafee_infante(
    data_dir: Path | str | None = None,
) -> PDEDataset:
    """Load Chafee-Infante equation dataset.

    Equation: u_t = u_xx - u + u^3
    Data shape: (301, 200) — 301 spatial points, 200 time points.

    Loads three .npy files:
    - chafee_infante_CI.npy (u field)
    - chafee_infante_x.npy (spatial coordinates)
    - chafee_infante_t.npy (temporal coordinates)

    Args:
        data_dir: Directory containing data files.
            Defaults to data/ relative to project root.

    Returns:
        PDEDataset with Chafee-Infante data.

    Raises:
        FileNotFoundError: If any required data file is missing.
    """
    resolved_dir = _resolve_data_dir(data_dir)


    u_path = resolved_dir / _CI_U_FILE
    x_path = resolved_dir / _CI_X_FILE
    t_path = resolved_dir / _CI_T_FILE

    for path, desc in [
        (u_path, "u field"),
        (x_path, "x coordinates"),
        (t_path, "t coordinates"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Chafee-Infante {desc} file not found: {path}")


    u_np = np.load(u_path)
    x_np = np.load(x_path).flatten().astype(np.float64)
    t_np = np.load(t_path).flatten().astype(np.float64)

    logger.info(
        "Loaded Chafee-Infante data: u=%s, x=%s, t=%s",
        u_np.shape,
        x_np.shape,
        t_np.shape,
    )


    u = torch.from_numpy(np.asarray(u_np, dtype=np.float64))
    x = torch.from_numpy(x_np)
    t = torch.from_numpy(t_np)


    return PDEDataset(
        name="chafee-infante",
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
        ground_truth="u_t = u_xx - u + u^3",
    )


def load_kdv(
    data_dir: Path | str | None = None,
) -> PDEDataset:
    """Load KdV (Korteweg-de Vries) equation dataset.

    Equation: u_t = -u * u_x - 0.0025 * u_xxx
    Data shape: (256, 201) — 256 spatial points, 201 time points.

    Loads .mat file: KdV_equation.mat
    Keys: x (spatial), tt (time), uu (solution)

    Args:
        data_dir: Directory containing data files.
            Defaults to data/ relative to project root.

    Returns:
        PDEDataset with KdV data.

    Raises:
        FileNotFoundError: If the data file is missing.
    """
    resolved_dir = _resolve_data_dir(data_dir)
    mat_path = resolved_dir / _KDV_FILE

    if not mat_path.exists():
        raise FileNotFoundError(f"KdV data file not found: {mat_path}")

    mat_data = _load_mat(mat_path)


    x_np = np.asarray(mat_data[_KDV_X_KEY], dtype=np.float64).flatten()
    t_np = np.asarray(mat_data[_KDV_T_KEY], dtype=np.float64).flatten()
    u_np = np.asarray(mat_data[_KDV_U_KEY], dtype=np.float64)


    if u_np.shape == (len(t_np), len(x_np)):
        u_np = u_np.T



    if x_np.shape[0] > _KDV_EXPECTED_NX:
        raw_nx = x_np.shape[0]
        if raw_nx % _KDV_EXPECTED_NX != 0:
            raise ValueError(
                f"KdV spatial points ({raw_nx}) not evenly divisible "
                f"by expected resolution ({_KDV_EXPECTED_NX})"
            )
        step = raw_nx // _KDV_EXPECTED_NX
        x_np = x_np[::step]
        u_np = u_np[::step,:]

    logger.info(
        "Loaded KdV data: u=%s, x=%s, t=%s",
        u_np.shape,
        x_np.shape,
        t_np.shape,
    )


    u = torch.from_numpy(u_np)
    x = torch.from_numpy(x_np)
    t = torch.from_numpy(t_np)


    return PDEDataset(
        name="kdv",
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
        ground_truth="u_t = -u * u_x - 0.0025 * u_xxx",
    )


def load_burgers(
    data_dir: Path | str | None = None,
) -> PDEDataset:
    """Load the Burgers equation reference dataset from SGA-PDE (Chen et al.).

    Equation: u_t = -u * u_x + nu * u_xx (with nu typically 0.1)
    Data shape: (256, 201) — 256 spatial points, 201 time points.

    Loads .mat file: Burgers_equation.mat
    Keys: x (1, nx), t (1, nt), usol (nx, nt)

    Use this when you want to reproduce the SGA-PDE benchmark.
    For a quick on-the-fly synthetic alternative (no file required),
    see ``generate_burgers_data`` in this same module.

    Args:
        data_dir: Directory containing data files.
            Defaults to bundled package data.

    Returns:
        PDEDataset with Burgers data.

    Raises:
        FileNotFoundError: If the data file is missing.
    """
    resolved_dir = _resolve_data_dir(data_dir)
    mat_path = resolved_dir / _BURGERS_FILE

    if not mat_path.exists():
        raise FileNotFoundError(f"Burgers data file not found: {mat_path}")

    mat_data = _load_mat(mat_path)

    x_np = np.asarray(mat_data[_BURGERS_X_KEY], dtype=np.float64).flatten()
    t_np = np.asarray(mat_data[_BURGERS_T_KEY], dtype=np.float64).flatten()
    u_np = np.asarray(mat_data[_BURGERS_U_KEY], dtype=np.float64)

    if u_np.shape == (len(t_np), len(x_np)):
        u_np = u_np.T

    logger.info(
        "Loaded Burgers data: u=%s, x=%s, t=%s",
        u_np.shape,
        x_np.shape,
        t_np.shape,
    )

    u = torch.from_numpy(u_np)
    x = torch.from_numpy(x_np)
    t = torch.from_numpy(t_np)

    return PDEDataset(
        name="burgers",
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
        ground_truth="u_t = -u * u_x + 0.1 * u_xx",
    )


def _load_paper_pde(
    file_name: str,
    name: str,
    nx: int,
    nt: int,
    x_range: tuple[float, float],
    t_range: tuple[float, float],
    ground_truth: str,
    data_dir: Path | str | None = None,
) -> PDEDataset:
    """Load a paper PDE benchmark (.npy file, u field only).

    The .npy files from SGA-PDE paper (Chen et al.) store u in shape
    (nt, nx). We transpose to (nx, nt) and reconstruct coordinates
    from known domain parameters.

    Args:
        file_name: Name of the .npy file in data_dir.
        name: Dataset name identifier.
        nx, nt: Expected spatial and temporal point counts.
        x_range, t_range: Domain bounds (inclusive endpoints).
        ground_truth: Ground truth equation string.
        data_dir: Directory containing data files.

    Returns:
        PDEDataset with the loaded data.
    """
    resolved_dir = _resolve_data_dir(data_dir)
    npy_path = resolved_dir / file_name

    if not npy_path.exists():
        raise FileNotFoundError(f"{name} data file not found: {npy_path}")

    u_raw = np.load(npy_path)


    if u_raw.shape == (nt, nx):
        u_np = u_raw.T
    elif u_raw.shape == (nx, nt):
        u_np = u_raw
    else:
        raise ValueError(
            f"{name} data shape {u_raw.shape} doesn't match "
            f"expected ({nt}, {nx}) or ({nx}, {nt})"
        )

    x_np = np.linspace(x_range[0], x_range[1], nx, dtype=np.float64)
    t_np = np.linspace(t_range[0], t_range[1], nt, dtype=np.float64)

    logger.info(
        "Loaded %s data: u=%s, x=%s, t=%s", name, u_np.shape, x_np.shape, t_np.shape
    )

    u = torch.from_numpy(np.asarray(u_np, dtype=np.float64))
    x = torch.from_numpy(x_np)
    t = torch.from_numpy(t_np)

    return PDEDataset(
        name=name,
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
        ground_truth=ground_truth,
    )


def load_pde_divide(
    data_dir: Path | str | None = None,
) -> PDEDataset:
    """Load PDE_divide (Eq. S4 from SGA-PDE paper).

    Equation: u_t = -u_x / x + 0.25 * u_xx
    Grid: 100 spatial x 251 temporal, x in [1,2], t in [0,1].
    Features a fractional structure (derivative divided by coordinate).
    """
    return _load_paper_pde(
        file_name=_PDE_DIVIDE_FILE,
        name="pde-divide",
        nx=_PDE_DIVIDE_NX,
        nt=_PDE_DIVIDE_NT,
        x_range=_PDE_DIVIDE_X_RANGE,
        t_range=_PDE_DIVIDE_T_RANGE,
        ground_truth="u_t = -u_x / x + 0.25 * u_xx",
        data_dir=data_dir,
    )


def load_pde_compound(
    data_dir: Path | str | None = None,
) -> PDEDataset:
    """Load PDE_compound (Eq. S5 from SGA-PDE paper).

    Equation: u_t = u * u_xx + u_x^2 (= d(u * u_x)/dx)
    Grid: 100 spatial x 251 temporal, x in [1,2], t in [0,0.5].
    Features compound nonlinearity (product rule derivative).
    """
    return _load_paper_pde(
        file_name=_PDE_COMPOUND_FILE,
        name="pde-compound",
        nx=_PDE_COMPOUND_NX,
        nt=_PDE_COMPOUND_NT,
        x_range=_PDE_COMPOUND_X_RANGE,
        t_range=_PDE_COMPOUND_T_RANGE,
        ground_truth="u_t = u * u_xx + u_x^2",
        data_dir=data_dir,
    )
