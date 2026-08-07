"""Data loaders for benchmark PDE datasets.

Loads pre-computed data from .npy and .mat files. Bundled/reference datasets
(work out of the box from packaged or ``data`` data):
- Chafee-Infante equation: u_t = u_xx - u + u^3
- KdV equation: u_t = -u * u_x - 0.0025 * u_xxx
- Burgers / PDE_divide / PDE_compound (SGA-PDE reference)

EqGPT-derived loaders (``load_allen_cahn`` / ``load_convection_diffusion`` /
``load_wave`` / ``load_klein_gordon``) resolve via :func:`_resolve_eqgpt_file`.
Their upstream ``.mat`` files are bundled under flat ``_assets/data/eqgpt_*``
filenames; pass an explicit ``data_dir`` to load an external copy.
``load_wave`` / ``load_klein_gordon`` are second-order (u_tt) datasets.
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


_ALLEN_CAHN_FILE = "eqgpt_allen_cahn.mat"
_ALLEN_CAHN_SUBDIR = ""
_ALLEN_CAHN_NAME = "allen-cahn"
_ALLEN_CAHN_X_KEY = "x"
_ALLEN_CAHN_T_KEY = "t"
_ALLEN_CAHN_U_KEY = "usol"
_ALLEN_CAHN_GROUND_TRUTH = "u_t = 0.003 * u_xx + u - u^3"


_WAVE_FILE = "eqgpt_wave.mat"
_WAVE_SUBDIR = ""
_WAVE_NAME = "wave"
_WAVE_X_KEY = "x"
_WAVE_T_KEY = "t"
_WAVE_U_KEY = "u"
_WAVE_GROUND_TRUTH = "u_tt = u_xx"


_KG_FILE = "eqgpt_klein_gordon.mat"
_KG_SUBDIR = ""
_KG_NAME = "klein-gordon"
_KG_X_KEY = "x"
_KG_T_KEY = "t"
_KG_U_KEY = "usol"
_KG_GROUND_TRUTH = "u_tt = 0.5 * u_xx - 5 * u"


_LHS_ORDER_SECOND = 2


_CI_U_FILE = "chafee_infante_CI.npy"
_CI_X_FILE = "chafee_infante_x.npy"
_CI_T_FILE = "chafee_infante_t.npy"


_CONVECTION_DIFFUSION_FILE = "eqgpt_convection_diffusion.mat"
_CONVECTION_DIFFUSION_SUBDIR = ""
_CONVECTION_DIFFUSION_NAME = "convection-diffusion"
_CONVECTION_DIFFUSION_X_KEY = "x"
_CONVECTION_DIFFUSION_T_KEY = "t"
_CONVECTION_DIFFUSION_U_KEY = "u"
_CONVECTION_DIFFUSION_GROUND_TRUTH = "u_t = -u_x + 0.25 * u_xx"


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


def _resolve_eqgpt_file(
    filename: str,
    subdir: str,
    data_dir: Path | str | None,
) -> Path:
    """Resolve an EqGPT data file.

    If ``data_dir`` is provided, it is authoritative: only
    ``Path(data_dir) / filename`` is considered (raises if absent there, with
    no fall-through). Otherwise, try the first existing of these
    ``<subdir>``-scoped candidates: the bundled package data, then a source
    checkout's data directory.

    Every no-arg candidate is namespaced by ``subdir`` so a generic filename
    like ``data.mat`` cannot collide with a different equation's file dropped
    into a shared flat directory. A loader whose filename is already
    unambiguous and intentionally lives flat in the data dir may opt out of
    namespacing by passing ``subdir=""``; use that only when the filename
    cannot collide.
    """
    if data_dir is not None:
        explicit_path = Path(data_dir) / filename
        if explicit_path.exists():
            return explicit_path
        raise FileNotFoundError(f"EqGPT data file not found: {explicit_path}")

    project_root = _find_project_root()




    candidates = [
        Path(__file__).resolve().parents[2] / "_assets" / "data" / subdir / filename,
        project_root / _DEFAULT_DATA_DIR / subdir / filename,
    ]
    for path in candidates:
        if path.exists():
            return path

    tried = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"EqGPT data file not found. Tried:\n{tried}")


def _orient_u_field(
    u_np: np.ndarray,
    x_np: np.ndarray,
    t_np: np.ndarray,
    *,
    dataset_name: str,
    u_key: str,
    assume_axis_order: list[str] | None,
) -> np.ndarray:
    """Orient u to (nx, nt), failing loud on a mismatched shape.

    With ``assume_axis_order`` unset (default), use a shape heuristic:
    transpose a (nt, nx) layout, accept (nx, nt) as-is, else raise. This
    heuristic is ambiguous on a square grid (nx == nt), so callers with a
    known layout pin it explicitly via ``assume_axis_order``:
    ``["x", "t"]`` means u is already (nx, nt) (no transpose);
    ``["t", "x"]`` means u is (nt, nx) and is transposed.
    """
    nx, nt = len(x_np), len(t_np)
    if assume_axis_order is not None:
        if assume_axis_order == [_AXIS_X, _AXIS_T]:
            expected, transpose = (nx, nt), False
        elif assume_axis_order == [_AXIS_T, _AXIS_X]:
            expected, transpose = (nt, nx), True
        else:
            raise ValueError(
                f"{dataset_name} assume_axis_order must be a permutation of "
                f"['{_AXIS_X}', '{_AXIS_T}'], got {assume_axis_order}"
            )
        if u_np.shape != expected:
            raise ValueError(
                f"{dataset_name} field '{u_key}' has shape {u_np.shape}; "
                f"expected {expected} for axis_order {assume_axis_order}"
            )
        return u_np.T if transpose else u_np


    if u_np.shape == (nt, nx):
        return u_np.T
    if u_np.shape == (nx, nt):
        return u_np
    raise ValueError(
        f"{dataset_name} field '{u_key}' has shape {u_np.shape}; "
        f"expected (nx, nt)=({nx}, {nt}) or its transpose"
    )


def _load_eqgpt_mat_dataset(
    *,
    filename: str,
    subdir: str,
    dataset_name: str,
    x_key: str,
    t_key: str,
    u_key: str,
    ground_truth: str,
    x_is_periodic: bool,
    data_dir: Path | str | None,
    lhs_order: int = 1,
    assume_axis_order: list[str] | None = None,
) -> PDEDataset:
    """Load an EqGPT .mat benchmark with axes ordered as (x, t).

    ``lhs_order`` selects the LHS time-derivative order (1 -> u_t, 2 -> u_tt).
    ``assume_axis_order`` pins the raw field's axis layout when the (nx, nt)
    vs (nt, nx) shape heuristic is ambiguous (square grids); see
    :func:`_orient_u_field`.
    """
    mat_path = _resolve_eqgpt_file(filename, subdir, data_dir)
    mat_data = _load_mat(mat_path)

    x_np = np.asarray(mat_data[x_key], dtype=np.float64).flatten()
    t_np = np.asarray(mat_data[t_key], dtype=np.float64).flatten()
    u_np = np.asarray(mat_data[u_key], dtype=np.float64)




    u_np = _orient_u_field(
        u_np,
        x_np,
        t_np,
        dataset_name=dataset_name,
        u_key=u_key,
        assume_axis_order=assume_axis_order,
    )

    logger.info(
        "Loaded %s data: u=%s, x=%s, t=%s",
        dataset_name,
        u_np.shape,
        x_np.shape,
        t_np.shape,
    )

    u = torch.from_numpy(u_np)
    x = torch.from_numpy(x_np)
    t = torch.from_numpy(t_np)

    return PDEDataset(
        name=dataset_name,
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            _AXIS_X: AxisInfo(name=_AXIS_X, values=x, is_periodic=x_is_periodic),
            _AXIS_T: AxisInfo(name=_AXIS_T, values=t, is_periodic=False),
        },
        axis_order=[_AXIS_X, _AXIS_T],
        fields={_FIELD_U: FieldData(name=_FIELD_U, values=u)},
        lhs_field=_FIELD_U,
        lhs_axis=_AXIS_T,
        lhs_order=lhs_order,
        ground_truth=ground_truth,
    )


def load_allen_cahn(
    data_dir: Path | str | None = None,
) -> PDEDataset:
    """Load the Allen-Cahn reaction-diffusion benchmark (EqGPT).

    Equation: u_t = 0.003 * u_xx + u - u^3
    Data shape: (256, 201) — 256 spatial points, 201 time points.
    x in [-1, 1) is a periodic grid (excludes +1), t in [0, 10].

    Loads ``eqgpt_allen_cahn.mat`` (key ``usol`` already (nx, nt)). Pass an
    explicit ``data_dir`` to load a different local copy.

    Args:
        data_dir: Directory holding ``eqgpt_allen_cahn.mat``. If None, falls
            back to the bundled package data, then a source checkout's data
            directory.

    Returns:
        PDEDataset with Allen-Cahn data.

    Raises:
        FileNotFoundError: If the data file cannot be resolved.
    """
    return _load_eqgpt_mat_dataset(
        filename=_ALLEN_CAHN_FILE,
        subdir=_ALLEN_CAHN_SUBDIR,
        dataset_name=_ALLEN_CAHN_NAME,
        x_key=_ALLEN_CAHN_X_KEY,
        t_key=_ALLEN_CAHN_T_KEY,
        u_key=_ALLEN_CAHN_U_KEY,
        ground_truth=_ALLEN_CAHN_GROUND_TRUTH,
        x_is_periodic=True,
        data_dir=data_dir,
    )


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


def load_convection_diffusion(
    data_dir: Path | str | None = None,
) -> PDEDataset:
    """Load the convection-diffusion benchmark (EqGPT).

    Equation: u_t = -u_x + 0.25 * u_xx
    Data shape: (256, 100) — 256 spatial points, 100 time points.
    x in [0, 2] (closed, non-periodic), t in [0, 1].

    Loads ``eqgpt_convection_diffusion.mat`` (key ``u``, raw (nt, nx) —
    transposed to (nx, nt)).
    Like :func:`load_allen_cahn`, the EqGPT copy is bundled under
    flat ``_assets/data/eqgpt_convection_diffusion.mat``; pass an explicit
    ``data_dir`` to load a different local copy.

    Args:
        data_dir: Directory holding ``eqgpt_convection_diffusion.mat``. If
            None, falls back to the bundled package data, then a source
            checkout's data directory.

    Returns:
        PDEDataset with convection-diffusion data.

    Raises:
        FileNotFoundError: If the data file cannot be resolved.
    """
    return _load_eqgpt_mat_dataset(
        filename=_CONVECTION_DIFFUSION_FILE,
        subdir=_CONVECTION_DIFFUSION_SUBDIR,
        dataset_name=_CONVECTION_DIFFUSION_NAME,
        x_key=_CONVECTION_DIFFUSION_X_KEY,
        t_key=_CONVECTION_DIFFUSION_T_KEY,
        u_key=_CONVECTION_DIFFUSION_U_KEY,
        ground_truth=_CONVECTION_DIFFUSION_GROUND_TRUTH,
        x_is_periodic=False,
        data_dir=data_dir,
    )


def load_wave(
    data_dir: Path | str | None = None,
) -> PDEDataset:
    """Load the wave equation benchmark (EqGPT), a second-order LHS dataset.

    Equation: u_tt = u_xx
    Data shape: (161, 321) — 161 spatial points, 321 time points.
    x in [0, pi], t in [0, 2*pi]; both axes non-periodic.

    Loads ``eqgpt_wave.mat`` (key ``u``, already (nx, nt)). The returned
    dataset has ``lhs_order=2`` (u_tt). Pass an explicit ``data_dir`` to load a
    different local copy.

    Args:
        data_dir: Directory holding ``eqgpt_wave.mat``. If None, falls back to
            the bundled package data, then a source checkout's data directory.

    Returns:
        PDEDataset with wave data (lhs_order=2).

    Raises:
        FileNotFoundError: If the data file cannot be resolved.
    """
    return _load_eqgpt_mat_dataset(
        filename=_WAVE_FILE,
        subdir=_WAVE_SUBDIR,
        dataset_name=_WAVE_NAME,
        x_key=_WAVE_X_KEY,
        t_key=_WAVE_T_KEY,
        u_key=_WAVE_U_KEY,
        ground_truth=_WAVE_GROUND_TRUTH,
        x_is_periodic=False,
        data_dir=data_dir,
        lhs_order=_LHS_ORDER_SECOND,
        assume_axis_order=[_AXIS_X, _AXIS_T],
    )


def load_klein_gordon(
    data_dir: Path | str | None = None,
) -> PDEDataset:
    """Load the Klein-Gordon benchmark (EqGPT), a second-order LHS dataset.

    Equation: u_tt = 0.5 * u_xx - 5 * u
    Data shape: (201, 201) — 201 spatial points, 201 time points (square grid).
    x in [-1, 1], t in [0, 3]; both axes non-periodic.

    Loads ``eqgpt_klein_gordon.mat`` (key ``usol``, already (nx, nt)). Because
    the grid is square the (nx, nt) vs (nt, nx) shape heuristic is ambiguous,
    so the layout is pinned explicitly to ``["x", "t"]`` (no transpose);
    transposing would flip x and t and corrupt the recovered coefficients. The
    returned dataset has ``lhs_order=2`` (u_tt). Pass an explicit ``data_dir``
    to load a different local copy.

    Args:
        data_dir: Directory holding ``eqgpt_klein_gordon.mat``. If None, falls
            back to the bundled package data, then a source checkout's data
            directory.

    Returns:
        PDEDataset with Klein-Gordon data (lhs_order=2).

    Raises:
        FileNotFoundError: If the data file cannot be resolved.
    """
    return _load_eqgpt_mat_dataset(
        filename=_KG_FILE,
        subdir=_KG_SUBDIR,
        dataset_name=_KG_NAME,
        x_key=_KG_X_KEY,
        t_key=_KG_T_KEY,
        u_key=_KG_U_KEY,
        ground_truth=_KG_GROUND_TRUTH,
        x_is_periodic=False,
        data_dir=data_dir,
        lhs_order=_LHS_ORDER_SECOND,
        assume_axis_order=[_AXIS_X, _AXIS_T],
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
