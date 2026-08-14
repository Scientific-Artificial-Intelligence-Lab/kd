
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from kd.search.discover.data.loader import load_pde_divide_npy
from kd.search.discover.paths import REFERENCE_DATA_DIR


_DATA_PATH = REFERENCE_DATA_DIR / "PDE_divide.npy"



_NX = 100
_NT = 251

_X_REF = np.linspace(1.0, 2.0, 100)
_T_REF = np.linspace(0.0, 1.0, 251)


@pytest.fixture(scope="module")
def pde_divide_dataset() -> object:
    if not _DATA_PATH.exists():
        pytest.skip(f"PDE_divide data file not present: {_DATA_PATH}")
    return load_pde_divide_npy(_DATA_PATH)


@pytest.mark.unit
def test_shape_after_transpose_no_trim(pde_divide_dataset: object) -> None:
    fields = pde_divide_dataset.fields
    u = fields["u"].values
    assert tuple(u.shape) == (_NX, _NT), (
        f"u shape {tuple(u.shape)} != ({_NX}, {_NT}); paper PDE_divide "
        f"transposes raw (Nt, Nx)=(251, 100) and does not trim."
    )


@pytest.mark.unit
def test_axis_lengths_match_field_dims(pde_divide_dataset: object) -> None:
    axes = pde_divide_dataset.axes
    fields = pde_divide_dataset.fields
    u = fields["u"].values
    assert axes["x"].values.shape == (u.shape[0],) == (_NX,)
    assert axes["t"].values.shape == (u.shape[1],) == (_NT,)


@pytest.mark.unit
def test_axis_values_match_paper_linspace(
    pde_divide_dataset: object,
) -> None:
    axes = pde_divide_dataset.axes
    x = axes["x"].values.numpy()
    t = axes["t"].values.numpy()
    np.testing.assert_allclose(x, _X_REF, atol=1e-12)
    np.testing.assert_allclose(t, _T_REF, atol=1e-12)


@pytest.mark.unit
def test_dtype_is_float64(pde_divide_dataset: object) -> None:
    axes = pde_divide_dataset.axes
    fields = pde_divide_dataset.fields
    assert fields["u"].values.dtype == torch.float64
    assert axes["x"].values.dtype == torch.float64
    assert axes["t"].values.dtype == torch.float64


@pytest.mark.unit
def test_dataset_metadata_alignment(pde_divide_dataset: object) -> None:
    ds = pde_divide_dataset
    assert ds.axis_order == ["x", "t"]
    assert ds.lhs_field == "u"
    assert ds.lhs_axis == "t"
    assert ds.name == "pde_divide"


@pytest.mark.unit
def test_missing_path_raises(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_file.npy"
    with pytest.raises(FileNotFoundError):
        load_pde_divide_npy(missing)


@pytest.mark.unit
def test_axes_are_strictly_increasing(pde_divide_dataset: object) -> None:
    axes = pde_divide_dataset.axes
    assert torch.all(torch.diff(axes["x"].values) > 0)
    assert torch.all(torch.diff(axes["t"].values) > 0)


@pytest.mark.unit
def test_x_does_not_cross_zero(pde_divide_dataset: object) -> None:
    axes = pde_divide_dataset.axes
    assert torch.all(axes["x"].values > 0)


@pytest.mark.unit
def test_field_has_finite_values(pde_divide_dataset: object) -> None:
    fields = pde_divide_dataset.fields
    assert torch.isfinite(fields["u"].values).all()
