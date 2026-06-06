
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from kd.search.discover.data.loader import load_pde_compound_npy


_DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "refs" / "discover" / "dso" / "dso" / "task" / "pde"
    / "data_new" / "PDE_compound.npy"
)




_NX_AFTER_TRIM = 80
_NT_FULL = 251





_X_FULL_REF = np.linspace(1.0, 2.0, 100)
_T_FULL_REF = np.linspace(0.0, 0.5, 251)
_X_TRIM_SLICE = slice(10, 90)


@pytest.fixture(scope="module")
def pde_compound_dataset() -> object:
    if not _DATA_PATH.exists():
        pytest.skip(f"PDE_compound data file not present: {_DATA_PATH}")
    return load_pde_compound_npy(_DATA_PATH)


@pytest.mark.unit
def test_shape_after_transpose_and_x_trim(
    pde_compound_dataset: object,
) -> None:
    fields = pde_compound_dataset.fields
    u = fields["u"].values
    assert tuple(u.shape) == (_NX_AFTER_TRIM, _NT_FULL), (
        f"u shape {tuple(u.shape)} != ({_NX_AFTER_TRIM}, {_NT_FULL}) "
        f"after paper transpose + x-trim"
    )


@pytest.mark.unit
def test_axis_lengths_match_field_dims(
    pde_compound_dataset: object,
) -> None:
    axes = pde_compound_dataset.axes
    fields = pde_compound_dataset.fields
    u = fields["u"].values
    assert axes["x"].values.shape == (u.shape[0],) == (_NX_AFTER_TRIM,)
    assert axes["t"].values.shape == (u.shape[1],) == (_NT_FULL,)


@pytest.mark.unit
def test_axis_values_match_paper_linspace(
    pde_compound_dataset: object,
) -> None:
    axes = pde_compound_dataset.axes
    x = axes["x"].values.numpy()
    t = axes["t"].values.numpy()
    np.testing.assert_allclose(x, _X_FULL_REF[_X_TRIM_SLICE], atol=1e-12)
    np.testing.assert_allclose(t, _T_FULL_REF, atol=1e-12)


@pytest.mark.unit
def test_dtype_is_float64(pde_compound_dataset: object) -> None:
    axes = pde_compound_dataset.axes
    fields = pde_compound_dataset.fields
    assert fields["u"].values.dtype == torch.float64
    assert axes["x"].values.dtype == torch.float64
    assert axes["t"].values.dtype == torch.float64


@pytest.mark.unit
def test_dataset_metadata_alignment(
    pde_compound_dataset: object,
) -> None:
    ds = pde_compound_dataset
    assert ds.axis_order == ["x", "t"]
    assert ds.lhs_field == "u"
    assert ds.lhs_axis == "t"
    assert ds.name == "pde_compound"


@pytest.mark.unit
def test_missing_path_raises(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_file.npy"
    with pytest.raises(FileNotFoundError):
        load_pde_compound_npy(missing)


@pytest.mark.unit
def test_axes_are_strictly_increasing(
    pde_compound_dataset: object,
) -> None:
    axes = pde_compound_dataset.axes
    assert torch.all(torch.diff(axes["x"].values) > 0)
    assert torch.all(torch.diff(axes["t"].values) > 0)


@pytest.mark.unit
def test_field_has_finite_values(pde_compound_dataset: object) -> None:
    fields = pde_compound_dataset.fields
    assert torch.isfinite(fields["u"].values).all()
