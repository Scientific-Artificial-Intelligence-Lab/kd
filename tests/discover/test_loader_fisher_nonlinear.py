
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kd.search.discover.data.loader import load_fisher_nonlinear_mat




_DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "refs" / "discover" / "dso" / "dso" / "task" / "pde"
    / "data_new" / "fisher_nonlin_groundtruth.mat"
)





_NX_AFTER_TRIM = 199
_NT_AFTER_TRIM = 99


@pytest.fixture(scope="module")
def fisher_nonlinear_dataset() -> object:
    if not _DATA_PATH.exists():
        pytest.skip(f"Fisher_nonlinear data file not present: {_DATA_PATH}")
    return load_fisher_nonlinear_mat(_DATA_PATH)


@pytest.mark.unit
def test_shape_after_trim_and_transpose(
    fisher_nonlinear_dataset: object,
) -> None:
    fields = fisher_nonlinear_dataset.fields
    u = fields["u"].values
    assert tuple(u.shape) == (_NX_AFTER_TRIM, _NT_AFTER_TRIM), (
        f"u shape {tuple(u.shape)} != ({_NX_AFTER_TRIM}, {_NT_AFTER_TRIM}) "
        f"after paper trim + transpose"
    )


@pytest.mark.unit
def test_axis_lengths_match_field_dims(
    fisher_nonlinear_dataset: object,
) -> None:
    axes = fisher_nonlinear_dataset.axes
    fields = fisher_nonlinear_dataset.fields
    u = fields["u"].values
    assert axes["x"].values.shape == (u.shape[0],)
    assert axes["t"].values.shape == (u.shape[1],)
    assert axes["x"].values.shape == (_NX_AFTER_TRIM,)
    assert axes["t"].values.shape == (_NT_AFTER_TRIM,)


@pytest.mark.unit
def test_dtype_is_float64(fisher_nonlinear_dataset: object) -> None:
    axes = fisher_nonlinear_dataset.axes
    fields = fisher_nonlinear_dataset.fields
    assert fields["u"].values.dtype == torch.float64
    assert axes["x"].values.dtype == torch.float64
    assert axes["t"].values.dtype == torch.float64


@pytest.mark.unit
def test_dataset_metadata_alignment(
    fisher_nonlinear_dataset: object,
) -> None:
    ds = fisher_nonlinear_dataset
    assert ds.axis_order == ["x", "t"]
    assert ds.lhs_field == "u"
    assert ds.lhs_axis == "t"



    assert ds.name == "fisher_nonlinear"


@pytest.mark.unit
def test_missing_path_raises(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_file.mat"
    with pytest.raises(FileNotFoundError):
        load_fisher_nonlinear_mat(missing)


@pytest.mark.unit
def test_axes_are_strictly_increasing(
    fisher_nonlinear_dataset: object,
) -> None:
    axes = fisher_nonlinear_dataset.axes
    assert torch.all(torch.diff(axes["x"].values) > 0)
    assert torch.all(torch.diff(axes["t"].values) > 0)


@pytest.mark.unit
def test_field_has_finite_values(
    fisher_nonlinear_dataset: object,
) -> None:
    fields = fisher_nonlinear_dataset.fields
    assert torch.isfinite(fields["u"].values).all()
