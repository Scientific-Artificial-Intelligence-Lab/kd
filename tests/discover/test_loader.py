
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch

from kd.data.schema import PDEDataset, TaskType
from kd.search.discover.data.loader import load_burgers_mat



_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BURGERS_MAT = _PROJECT_ROOT / "refs/discover/dso/dso/task/pde/data_new/burgers.mat"
BURGERS2_MAT = _PROJECT_ROOT / "refs/discover/dso/dso/task/pde/data_new/burgers2.mat"


N_X = 256
N_T = 201
N_T_BURGERS2 = 101



_REFS_AVAILABLE = BURGERS_MAT.exists() and BURGERS2_MAT.exists()
_SKIP_WHEN_NO_REFS = pytest.mark.skipif(
    not _REFS_AVAILABLE,
    reason=(
        f"refs/ data not available at {BURGERS_MAT.parent}; "
        "this fixture/test requires the local baseline mirror."
    ),
)


@pytest.fixture(scope="module")
def dataset() -> PDEDataset:
    if not _REFS_AVAILABLE:
        pytest.skip("refs/ data not available (required by fixture)")
    return load_burgers_mat(BURGERS_MAT)





@_SKIP_WHEN_NO_REFS
class TestLoadBurgersUnit:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_loads_without_error(self) -> None:
        ds = load_burgers_mat(BURGERS_MAT)
        assert isinstance(ds, PDEDataset)

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_returns_pde_task_type(self, dataset: PDEDataset) -> None:
        assert dataset.task_type == TaskType.PDE

    @pytest.mark.unit
    def test_dataset_name(self, dataset: PDEDataset) -> None:
        assert dataset.name == "burgers"

    @pytest.mark.unit
    def test_correct_field_names(self, dataset: PDEDataset) -> None:
        assert dataset.fields is not None
        assert "u" in dataset.fields
        assert dataset.axes is not None
        assert "x" in dataset.axes
        assert "t" in dataset.axes

    @pytest.mark.unit
    def test_grid_dimensions(self, dataset: PDEDataset) -> None:
        assert dataset.axes is not None
        assert dataset.axes["x"].values.shape == (N_X,)
        assert dataset.axes["t"].values.shape == (N_T,)

    @pytest.mark.unit
    def test_field_shape_matches_grid(self, dataset: PDEDataset) -> None:
        assert dataset.fields is not None
        assert dataset.fields["u"].values.shape == (N_X, N_T)

    @pytest.mark.unit
    def test_axis_order(self, dataset: PDEDataset) -> None:
        assert dataset.axis_order == ["x", "t"]

    @pytest.mark.unit
    def test_u_is_float64_tensor(self, dataset: PDEDataset) -> None:
        assert dataset.fields is not None
        u = dataset.fields["u"].values
        assert isinstance(u, torch.Tensor)
        assert u.dtype == torch.float64

    @pytest.mark.unit
    def test_axes_are_float64(self, dataset: PDEDataset) -> None:
        assert dataset.axes is not None
        assert dataset.axes["x"].values.dtype == torch.float64
        assert dataset.axes["t"].values.dtype == torch.float64

    @pytest.mark.unit
    def test_x_is_1d_tensor(self, dataset: PDEDataset) -> None:
        assert dataset.axes is not None
        x = dataset.axes["x"].values
        assert isinstance(x, torch.Tensor)
        assert x.dim() == 1

    @pytest.mark.unit
    def test_t_is_1d_tensor(self, dataset: PDEDataset) -> None:
        assert dataset.axes is not None
        t = dataset.axes["t"].values
        assert isinstance(t, torch.Tensor)
        assert t.dim() == 1

    @pytest.mark.unit
    def test_axis_names_match_keys(self, dataset: PDEDataset) -> None:
        assert dataset.axes is not None
        assert dataset.axes["x"].name == "x"
        assert dataset.axes["t"].name == "t"

    @pytest.mark.unit
    def test_field_name_matches_key(self, dataset: PDEDataset) -> None:
        assert dataset.fields is not None
        assert dataset.fields["u"].name == "u"

    @pytest.mark.unit
    def test_no_nan_in_u(self, dataset: PDEDataset) -> None:
        assert dataset.fields is not None
        assert not torch.isnan(dataset.fields["u"].values).any()

    @pytest.mark.unit
    def test_no_inf_in_u(self, dataset: PDEDataset) -> None:
        assert dataset.fields is not None
        assert not torch.isinf(dataset.fields["u"].values).any()

    @pytest.mark.unit
    def test_no_nan_inf_in_axes(self, dataset: PDEDataset) -> None:
        assert dataset.axes is not None
        for name in ("x", "t"):
            v = dataset.axes[name].values
            assert not torch.isnan(v).any(), f"NaN in {name}"
            assert not torch.isinf(v).any(), f"Inf in {name}"

    @pytest.mark.unit
    def test_lhs_field(self, dataset: PDEDataset) -> None:
        assert dataset.lhs_field == "u"

    @pytest.mark.unit
    def test_lhs_axis(self, dataset: PDEDataset) -> None:
        assert dataset.lhs_axis == "t"

    @pytest.mark.unit
    def test_x_monotonically_increasing(self, dataset: PDEDataset) -> None:
        assert dataset.axes is not None
        x = dataset.axes["x"].values
        assert torch.all(torch.diff(x) > 0)

    @pytest.mark.unit
    def test_t_starts_at_zero_and_increasing(self, dataset: PDEDataset) -> None:
        assert dataset.axes is not None
        t = dataset.axes["t"].values
        assert t[0].item() == pytest.approx(0.0, abs=1e-10)
        assert torch.all(torch.diff(t) > 0)

    @pytest.mark.unit
    def test_accepts_string_path(self) -> None:
        ds = load_burgers_mat(str(BURGERS_MAT))
        assert isinstance(ds, PDEDataset)


class TestLoadBurgersErrorHandling:

    @pytest.mark.unit
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_burgers_mat(tmp_path / "nonexistent.mat")

    @pytest.mark.unit
    def test_missing_key_raises(self, tmp_path: Path) -> None:
        bad_mat = tmp_path / "bad.mat"
        sio.savemat(str(bad_mat), {"x": [1.0], "t": [0.0]})
        with pytest.raises(KeyError):
            load_burgers_mat(bad_mat)


@_SKIP_WHEN_NO_REFS
class TestLoadBurgers2Unit:

    @pytest.mark.unit
    def test_burgers2_loads_without_complex_cast_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dataset = load_burgers_mat(BURGERS2_MAT)

        messages = [str(w.message) for w in caught]
        assert not any(
            "Casting complex values to real discards the imaginary part" in msg
            for msg in messages
        )
        assert dataset.fields is not None
        assert dataset.fields["u"].values.shape == (N_X, N_T_BURGERS2)
        assert dataset.fields["u"].values.dtype == torch.float64

    @pytest.mark.unit
    def test_burgers2_uses_real_component_of_usol(self) -> None:
        raw = sio.loadmat(str(BURGERS2_MAT))["usol"]
        dataset = load_burgers_mat(BURGERS2_MAT)
        assert dataset.fields is not None
        expected = torch.from_numpy(np.real(raw)).to(torch.float64)
        assert torch.allclose(dataset.fields["u"].values, expected)





@_SKIP_WHEN_NO_REFS
class TestLoaderInvariants:

    @pytest.mark.integration
    def test_kd_post_init_validation_passes(
        self, dataset: PDEDataset
    ) -> None:
        assert dataset.get_shape() == (N_X, N_T)

    @pytest.mark.integration
    def test_get_coords_returns_correct_tensors(
        self, dataset: PDEDataset
    ) -> None:
        assert dataset.axes is not None
        x_coords = dataset.get_coords("x")
        assert torch.equal(x_coords, dataset.axes["x"].values)
        t_coords = dataset.get_coords("t")
        assert torch.equal(t_coords, dataset.axes["t"].values)

    @pytest.mark.integration
    def test_get_field_returns_u_tensor(self, dataset: PDEDataset) -> None:
        assert dataset.fields is not None
        u = dataset.get_field("u")
        assert torch.equal(u, dataset.fields["u"].values)
