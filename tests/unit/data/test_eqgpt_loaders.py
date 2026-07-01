
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kd.data import synthetic
from kd.data.schema import DataTopology, PDEDataset, TaskType






def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent


    return Path(__file__).resolve().parents[3]


_PROJECT_ROOT = _find_project_root()
_ASSETS_DATA_ROOT = _PROJECT_ROOT / "src" / "kd" / "_assets" / "data"






_AC_FILE = "eqgpt_allen_cahn.mat"
_AC_DIR = _ASSETS_DATA_ROOT
_AC_PATH = _AC_DIR / _AC_FILE

_AC_NX = 256
_AC_NT = 201
_AC_EXPECTED_SHAPE = (_AC_NX, _AC_NT)
_AC_GROUND_TRUTH = "u_t = 0.003 * u_xx + u - u^3"
_AC_NAME = "allen-cahn"

_AC_X_LO = -1.0
_AC_X_HI_EXCLUDED = 1.0
_AC_X_LAST = 0.9921875
_AC_T_LO = 0.0
_AC_T_HI = 10.0







_CD_FILE = "eqgpt_convection_diffusion.mat"
_CD_DIR = _ASSETS_DATA_ROOT
_CD_PATH = _CD_DIR / _CD_FILE

_CD_NX = 256
_CD_NT = 100
_CD_EXPECTED_SHAPE = (_CD_NX, _CD_NT)
_CD_GROUND_TRUTH = "u_t = -u_x + 0.25 * u_xx"
_CD_NAME = "convection-diffusion"
_CD_X_LO = 0.0
_CD_X_HI = 2.0
_CD_T_LO = 0.0
_CD_T_HI = 1.0


_ENDPOINT_ATOL = 1e-6






skip_no_allen_cahn = pytest.mark.skipif(
    not _AC_PATH.exists(),
    reason=f"EqGPT Allen-Cahn data not found: {_AC_PATH}",
)
skip_no_convection_diffusion = pytest.mark.skipif(
    not _CD_PATH.exists(),
    reason=f"EqGPT convection-diffusion data not found: {_CD_PATH}",
)







class TestAllenCahnSmoke:

    @pytest.mark.smoke
    def test_function_exists_and_callable(self) -> None:
        assert callable(synthetic.load_allen_cahn)

    @skip_no_allen_cahn
    @pytest.mark.smoke
    def test_explicit_dir_returns_pde_dataset(self) -> None:
        ds = synthetic.load_allen_cahn(data_dir=_AC_DIR)
        assert isinstance(ds, PDEDataset)







class TestAllenCahnFallback:

    @skip_no_allen_cahn
    @pytest.mark.unit
    def test_no_arg_returns_pde_dataset(self) -> None:
        ds = synthetic.load_allen_cahn()
        assert isinstance(ds, PDEDataset)

    @skip_no_allen_cahn
    @pytest.mark.unit
    def test_no_arg_shape(self) -> None:
        ds = synthetic.load_allen_cahn()
        assert ds.get_shape() == _AC_EXPECTED_SHAPE







@skip_no_allen_cahn
class TestAllenCahnLoader:

    @pytest.mark.unit
    def test_field_shape(self) -> None:
        ds = synthetic.load_allen_cahn(data_dir=_AC_DIR)
        u = ds.get_field("u")
        assert tuple(u.shape) == _AC_EXPECTED_SHAPE

    @pytest.mark.unit
    def test_dataset_shape(self) -> None:
        ds = synthetic.load_allen_cahn(data_dir=_AC_DIR)
        assert ds.get_shape() == _AC_EXPECTED_SHAPE

    @pytest.mark.unit
    def test_axis_lengths(self) -> None:
        ds = synthetic.load_allen_cahn(data_dir=_AC_DIR)
        assert len(ds.get_coords("x")) == _AC_NX
        assert len(ds.get_coords("t")) == _AC_NT

    @pytest.mark.unit
    def test_x_axis_monotonic_increasing(self) -> None:
        ds = synthetic.load_allen_cahn(data_dir=_AC_DIR)
        x = ds.get_coords("x")
        assert (x[1:] > x[:-1]).all(), "x not strictly increasing"

    @pytest.mark.unit
    def test_t_axis_monotonic_increasing(self) -> None:
        ds = synthetic.load_allen_cahn(data_dir=_AC_DIR)
        t = ds.get_coords("t")
        assert (t[1:] > t[:-1]).all(), "t not strictly increasing"

    @pytest.mark.unit
    def test_x_span(self) -> None:
        ds = synthetic.load_allen_cahn(data_dir=_AC_DIR)
        x = ds.get_coords("x")
        torch.testing.assert_close(
            x[0],
            torch.tensor(_AC_X_LO, dtype=x.dtype),
            rtol=0.0,
            atol=_ENDPOINT_ATOL,
        )

        assert x[-1].item() < _AC_X_HI_EXCLUDED
        torch.testing.assert_close(
            x[-1],
            torch.tensor(_AC_X_LAST, dtype=x.dtype),
            rtol=0.0,
            atol=_ENDPOINT_ATOL,
        )

    @pytest.mark.unit
    def test_t_span(self) -> None:
        ds = synthetic.load_allen_cahn(data_dir=_AC_DIR)
        t = ds.get_coords("t")
        torch.testing.assert_close(
            t[0],
            torch.tensor(_AC_T_LO, dtype=t.dtype),
            rtol=0.0,
            atol=_ENDPOINT_ATOL,
        )
        torch.testing.assert_close(
            t[-1],
            torch.tensor(_AC_T_HI, dtype=t.dtype),
            rtol=0.0,
            atol=_ENDPOINT_ATOL,
        )

    @pytest.mark.unit
    def test_ground_truth_exact(self) -> None:
        ds = synthetic.load_allen_cahn(data_dir=_AC_DIR)
        assert ds.ground_truth == _AC_GROUND_TRUTH

    @pytest.mark.unit
    def test_lhs_field_axis_order(self) -> None:
        ds = synthetic.load_allen_cahn(data_dir=_AC_DIR)
        assert ds.lhs_field == "u"
        assert ds.lhs_axis == "t"
        assert ds.axis_order == ["x", "t"]
        assert ds.lhs_order == 1

    @pytest.mark.unit
    def test_task_type_and_topology(self) -> None:
        ds = synthetic.load_allen_cahn(data_dir=_AC_DIR)
        assert ds.task_type == TaskType.PDE
        assert ds.topology == DataTopology.GRID

    @pytest.mark.unit
    def test_periodic_flags(self) -> None:
        ds = synthetic.load_allen_cahn(data_dir=_AC_DIR)
        assert ds.axes is not None
        assert ds.axes["x"].is_periodic is True
        assert ds.axes["t"].is_periodic is False

    @pytest.mark.unit
    def test_name(self) -> None:
        ds = synthetic.load_allen_cahn(data_dir=_AC_DIR)
        assert ds.name == _AC_NAME

    @pytest.mark.numerical
    def test_dtype_float64(self) -> None:
        ds = synthetic.load_allen_cahn(data_dir=_AC_DIR)
        assert ds.get_field("u").dtype == torch.float64
        assert ds.get_coords("x").dtype == torch.float64
        assert ds.get_coords("t").dtype == torch.float64

    @pytest.mark.numerical
    def test_no_nan_inf(self) -> None:
        ds = synthetic.load_allen_cahn(data_dir=_AC_DIR)
        assert torch.isfinite(ds.get_field("u")).all()
        assert torch.isfinite(ds.get_coords("x")).all()
        assert torch.isfinite(ds.get_coords("t")).all()







class TestAllenCahnNegative:

    @pytest.mark.unit
    def test_missing_dir_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            synthetic.load_allen_cahn(data_dir=Path("/nonexistent/eqgpt/path"))

    @pytest.mark.unit
    def test_empty_dir_raises_file_not_found(self) -> None:
        import tempfile

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(FileNotFoundError),
        ):
            synthetic.load_allen_cahn(data_dir=Path(tmpdir))







class TestConvectionDiffusionSmoke:

    @pytest.mark.smoke
    def test_function_exists_and_callable(self) -> None:
        assert callable(synthetic.load_convection_diffusion)

    @skip_no_convection_diffusion
    @pytest.mark.smoke
    def test_explicit_dir_returns_pde_dataset(self) -> None:
        ds = synthetic.load_convection_diffusion(data_dir=_CD_DIR)
        assert isinstance(ds, PDEDataset)







class TestConvectionDiffusionFallback:

    @skip_no_convection_diffusion
    @pytest.mark.unit
    def test_no_arg_returns_pde_dataset(self) -> None:
        ds = synthetic.load_convection_diffusion()
        assert isinstance(ds, PDEDataset)

    @skip_no_convection_diffusion
    @pytest.mark.unit
    def test_no_arg_shape(self) -> None:
        ds = synthetic.load_convection_diffusion()
        assert ds.get_shape() == _CD_EXPECTED_SHAPE







@skip_no_convection_diffusion
class TestConvectionDiffusionLoader:

    @pytest.mark.unit
    def test_field_shape_is_transposed(self) -> None:
        ds = synthetic.load_convection_diffusion(data_dir=_CD_DIR)
        u = ds.get_field("u")
        assert tuple(u.shape) == _CD_EXPECTED_SHAPE

    @pytest.mark.unit
    def test_dataset_shape(self) -> None:
        ds = synthetic.load_convection_diffusion(data_dir=_CD_DIR)
        assert ds.get_shape() == _CD_EXPECTED_SHAPE

    @pytest.mark.unit
    def test_axis_lengths(self) -> None:
        ds = synthetic.load_convection_diffusion(data_dir=_CD_DIR)
        assert len(ds.get_coords("x")) == _CD_NX
        assert len(ds.get_coords("t")) == _CD_NT

    @pytest.mark.unit
    def test_x_axis_monotonic_increasing(self) -> None:
        ds = synthetic.load_convection_diffusion(data_dir=_CD_DIR)
        x = ds.get_coords("x")
        assert (x[1:] > x[:-1]).all(), "x not strictly increasing"

    @pytest.mark.unit
    def test_t_axis_monotonic_increasing(self) -> None:
        ds = synthetic.load_convection_diffusion(data_dir=_CD_DIR)
        t = ds.get_coords("t")
        assert (t[1:] > t[:-1]).all(), "t not strictly increasing"

    @pytest.mark.unit
    def test_x_span(self) -> None:
        ds = synthetic.load_convection_diffusion(data_dir=_CD_DIR)
        x = ds.get_coords("x")
        torch.testing.assert_close(
            x[0],
            torch.tensor(_CD_X_LO, dtype=x.dtype),
            rtol=0.0,
            atol=_ENDPOINT_ATOL,
        )
        torch.testing.assert_close(
            x[-1],
            torch.tensor(_CD_X_HI, dtype=x.dtype),
            rtol=0.0,
            atol=_ENDPOINT_ATOL,
        )

    @pytest.mark.unit
    def test_t_span(self) -> None:
        ds = synthetic.load_convection_diffusion(data_dir=_CD_DIR)
        t = ds.get_coords("t")
        torch.testing.assert_close(
            t[0],
            torch.tensor(_CD_T_LO, dtype=t.dtype),
            rtol=0.0,
            atol=_ENDPOINT_ATOL,
        )
        torch.testing.assert_close(
            t[-1],
            torch.tensor(_CD_T_HI, dtype=t.dtype),
            rtol=0.0,
            atol=_ENDPOINT_ATOL,
        )

    @pytest.mark.unit
    def test_ground_truth_exact(self) -> None:
        ds = synthetic.load_convection_diffusion(data_dir=_CD_DIR)
        assert ds.ground_truth == _CD_GROUND_TRUTH

    @pytest.mark.unit
    def test_lhs_field_axis_order(self) -> None:
        ds = synthetic.load_convection_diffusion(data_dir=_CD_DIR)
        assert ds.lhs_field == "u"
        assert ds.lhs_axis == "t"
        assert ds.axis_order == ["x", "t"]
        assert ds.lhs_order == 1

    @pytest.mark.unit
    def test_task_type_and_topology(self) -> None:
        ds = synthetic.load_convection_diffusion(data_dir=_CD_DIR)
        assert ds.task_type == TaskType.PDE
        assert ds.topology == DataTopology.GRID

    @pytest.mark.unit
    def test_periodic_flags(self) -> None:
        ds = synthetic.load_convection_diffusion(data_dir=_CD_DIR)
        assert ds.axes is not None
        assert ds.axes["x"].is_periodic is False
        assert ds.axes["t"].is_periodic is False

    @pytest.mark.unit
    def test_name(self) -> None:
        ds = synthetic.load_convection_diffusion(data_dir=_CD_DIR)
        assert ds.name == _CD_NAME

    @pytest.mark.numerical
    def test_dtype_float64(self) -> None:
        ds = synthetic.load_convection_diffusion(data_dir=_CD_DIR)
        assert ds.get_field("u").dtype == torch.float64
        assert ds.get_coords("x").dtype == torch.float64
        assert ds.get_coords("t").dtype == torch.float64

    @pytest.mark.numerical
    def test_no_nan_inf(self) -> None:
        ds = synthetic.load_convection_diffusion(data_dir=_CD_DIR)
        assert torch.isfinite(ds.get_field("u")).all()
        assert torch.isfinite(ds.get_coords("x")).all()
        assert torch.isfinite(ds.get_coords("t")).all()







class TestConvectionDiffusionNegative:

    @pytest.mark.unit
    def test_missing_dir_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            synthetic.load_convection_diffusion(
                data_dir=Path("/nonexistent/eqgpt/path")
            )

    @pytest.mark.unit
    def test_empty_dir_raises_file_not_found(self) -> None:
        import tempfile

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(FileNotFoundError),
        ):
            synthetic.load_convection_diffusion(data_dir=Path(tmpdir))
