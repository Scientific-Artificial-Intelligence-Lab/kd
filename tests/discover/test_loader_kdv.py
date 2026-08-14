
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kd.search.discover.data.loader import load_kdv_mat
from kd.search.discover.paths import REFERENCE_DATA_DIR



_DATA_PATH = REFERENCE_DATA_DIR / "Kdv.mat"




_NX = 512
_NT = 201


@pytest.fixture(scope="module")
def kdv_dataset() -> object:
    if not _DATA_PATH.exists():
        pytest.skip(f"KdV data file not present: {_DATA_PATH}")
    return load_kdv_mat(_DATA_PATH)


@pytest.mark.unit
def test_shape_no_trim_no_transpose(kdv_dataset: object) -> None:
    fields = kdv_dataset.fields
    u = fields["u"].values
    assert tuple(u.shape) == (_NX, _NT), (
        f"u shape {tuple(u.shape)} != ({_NX}, {_NT}); paper KdV uses "
        f"the raw ``uu`` field with no trim, no transpose."
    )


@pytest.mark.unit
def test_axis_lengths_match_field_dims(kdv_dataset: object) -> None:
    axes = kdv_dataset.axes
    fields = kdv_dataset.fields
    u = fields["u"].values
    assert axes["x"].values.shape == (u.shape[0],)
    assert axes["t"].values.shape == (u.shape[1],)
    assert axes["x"].values.shape == (_NX,)
    assert axes["t"].values.shape == (_NT,)


@pytest.mark.unit
def test_dtype_is_float64(kdv_dataset: object) -> None:
    axes = kdv_dataset.axes
    fields = kdv_dataset.fields
    assert fields["u"].values.dtype == torch.float64
    assert axes["x"].values.dtype == torch.float64
    assert axes["t"].values.dtype == torch.float64


@pytest.mark.unit
def test_dataset_metadata_alignment(kdv_dataset: object) -> None:
    ds = kdv_dataset
    assert ds.axis_order == ["x", "t"]
    assert ds.lhs_field == "u"
    assert ds.lhs_axis == "t"
    assert ds.name == "kdv"


@pytest.mark.unit
def test_missing_path_raises(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_file.mat"
    with pytest.raises(FileNotFoundError):
        load_kdv_mat(missing)


@pytest.mark.unit
def test_axes_are_strictly_increasing(kdv_dataset: object) -> None:
    axes = kdv_dataset.axes
    assert torch.all(torch.diff(axes["x"].values) > 0)
    assert torch.all(torch.diff(axes["t"].values) > 0)


@pytest.mark.unit
def test_field_has_finite_values(kdv_dataset: object) -> None:
    fields = kdv_dataset.fields
    assert torch.isfinite(fields["u"].values).all()
