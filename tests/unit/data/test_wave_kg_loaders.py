
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
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
_WAVE_DIR = _ASSETS_DATA_ROOT
_KG_DIR = _ASSETS_DATA_ROOT








_WAVE_FILE = "eqgpt_wave.mat"
_WAVE_PATH = _WAVE_DIR / _WAVE_FILE

_WAVE_NX = 161
_WAVE_NT = 321
_WAVE_SHAPE = (_WAVE_NX, _WAVE_NT)
_WAVE_X_LO = 0.0
_WAVE_X_HI = math.pi
_WAVE_T_LO = 0.0
_WAVE_T_HI = 2.0 * math.pi

_WAVE_DX = (_WAVE_X_HI - _WAVE_X_LO) / (_WAVE_NX - 1)
_WAVE_DT = (_WAVE_T_HI - _WAVE_T_LO) / (_WAVE_NT - 1)
_WAVE_GROUND_TRUTH = "u_tt = u_xx"









_KG_FILE = "eqgpt_klein_gordon.mat"
_KG_PATH = _KG_DIR / _KG_FILE

_KG_NX = 201
_KG_NT = 201
_KG_SHAPE = (_KG_NX, _KG_NT)
_KG_X_LO = -1.0
_KG_X_HI = 1.0
_KG_T_LO = 0.0
_KG_T_HI = 3.0
_KG_DX = (_KG_X_HI - _KG_X_LO) / (_KG_NX - 1)
_KG_DT = (_KG_T_HI - _KG_T_LO) / (_KG_NT - 1)
_KG_GROUND_TRUTH = "u_tt = 0.5 * u_xx - 5 * u"


_KG_A = 0.5
_KG_B = -5.0


_ENDPOINT_ATOL = 1e-6






skip_no_wave = pytest.mark.skipif(
    not _WAVE_PATH.exists(),
    reason=f"wave data not found: {_WAVE_PATH}",
)
skip_no_kg = pytest.mark.skipif(
    not _KG_PATH.exists(),
    reason=f"Klein-Gordon data not found: {_KG_PATH}",
)














def _second_difference(field: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    fwd = np.roll(field, -1, axis=axis)
    bwd = np.roll(field, 1, axis=axis)
    return (fwd - 2.0 * field + bwd) / spacing**2


def _fit_loaded_pde(ds: PDEDataset, *, with_u_term: bool) -> tuple[np.ndarray, float]:
    assert ds.axis_order is not None
    x_axis = ds.axis_order.index("x")
    t_axis = ds.axis_order.index("t")

    u = ds.get_field("u").detach().cpu().numpy().astype(np.float64)
    x = ds.get_coords("x").detach().cpu().numpy().astype(np.float64)
    t = ds.get_coords("t").detach().cpu().numpy().astype(np.float64)
    dx = float(np.diff(x).mean())
    dt = float(np.diff(t).mean())

    u_xx = _second_difference(u, dx, axis=x_axis)
    u_tt = _second_difference(u, dt, axis=t_axis)

    interior = tuple(slice(1, -1) for _ in range(u.ndim))
    target = u_tt[interior].ravel()
    columns = [u_xx[interior].ravel()]
    if with_u_term:
        columns.append(u[interior].ravel())
    design = np.stack(columns, axis=1)

    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    prediction = design @ coef
    nmse = float(np.mean((target - prediction) ** 2) / np.mean(target**2))
    return coef, nmse







class TestWaveSmoke:

    @pytest.mark.smoke
    def test_function_exists_and_callable(self) -> None:
        assert callable(synthetic.load_wave)

    @pytest.mark.smoke
    def test_exported_top_level(self) -> None:
        import kd

        loader = getattr(kd, "load_wave", None)
        assert callable(loader)

    @skip_no_wave
    @pytest.mark.smoke
    def test_explicit_dir_returns_pde_dataset(self) -> None:
        ds = synthetic.load_wave(data_dir=_WAVE_DIR)
        assert isinstance(ds, PDEDataset)







class TestWaveFallback:

    @skip_no_wave
    @pytest.mark.unit
    def test_no_arg_returns_pde_dataset(self) -> None:
        ds = synthetic.load_wave()
        assert isinstance(ds, PDEDataset)







@skip_no_wave
class TestWaveLoader:

    @pytest.mark.unit
    def test_field_shape(self) -> None:
        ds = synthetic.load_wave(data_dir=_WAVE_DIR)
        u = ds.get_field("u")
        assert tuple(u.shape) == _WAVE_SHAPE

    @pytest.mark.unit
    def test_dataset_shape(self) -> None:
        ds = synthetic.load_wave(data_dir=_WAVE_DIR)
        assert ds.get_shape() == _WAVE_SHAPE

    @pytest.mark.unit
    def test_axis_order_is_x_then_t(self) -> None:
        ds = synthetic.load_wave(data_dir=_WAVE_DIR)
        assert ds.axis_order == ["x", "t"]

    @pytest.mark.unit
    def test_axis_lengths(self) -> None:
        ds = synthetic.load_wave(data_dir=_WAVE_DIR)
        assert len(ds.get_coords("x")) == _WAVE_NX
        assert len(ds.get_coords("t")) == _WAVE_NT

    @pytest.mark.unit
    def test_x_axis_monotonic_increasing(self) -> None:
        ds = synthetic.load_wave(data_dir=_WAVE_DIR)
        x = ds.get_coords("x")
        assert (x[1:] > x[:-1]).all(), "x not strictly increasing"

    @pytest.mark.unit
    def test_t_axis_monotonic_increasing(self) -> None:
        ds = synthetic.load_wave(data_dir=_WAVE_DIR)
        t = ds.get_coords("t")
        assert (t[1:] > t[:-1]).all(), "t not strictly increasing"

    @pytest.mark.unit
    def test_x_span_and_spacing(self) -> None:
        ds = synthetic.load_wave(data_dir=_WAVE_DIR)
        x = ds.get_coords("x")
        torch.testing.assert_close(
            x[0], torch.tensor(_WAVE_X_LO, dtype=x.dtype), rtol=0.0, atol=_ENDPOINT_ATOL
        )
        torch.testing.assert_close(
            x[-1],
            torch.tensor(_WAVE_X_HI, dtype=x.dtype),
            rtol=0.0,
            atol=_ENDPOINT_ATOL,
        )
        dx = (x[1:] - x[:-1]).mean().item()
        assert dx == pytest.approx(_WAVE_DX, rel=1e-6)

    @pytest.mark.unit
    def test_t_span_and_spacing(self) -> None:
        ds = synthetic.load_wave(data_dir=_WAVE_DIR)
        t = ds.get_coords("t")
        torch.testing.assert_close(
            t[0], torch.tensor(_WAVE_T_LO, dtype=t.dtype), rtol=0.0, atol=_ENDPOINT_ATOL
        )
        torch.testing.assert_close(
            t[-1],
            torch.tensor(_WAVE_T_HI, dtype=t.dtype),
            rtol=0.0,
            atol=_ENDPOINT_ATOL,
        )
        dt = (t[1:] - t[:-1]).mean().item()
        assert dt == pytest.approx(_WAVE_DT, rel=1e-6)

    @pytest.mark.unit
    def test_ground_truth_exact(self) -> None:
        ds = synthetic.load_wave(data_dir=_WAVE_DIR)
        assert ds.ground_truth == _WAVE_GROUND_TRUTH

    @pytest.mark.unit
    def test_lhs_spec_second_order(self) -> None:
        ds = synthetic.load_wave(data_dir=_WAVE_DIR)
        assert ds.lhs_field == "u"
        assert ds.lhs_axis == "t"
        assert ds.lhs_order == 2

    @pytest.mark.unit
    def test_task_type_and_topology(self) -> None:
        ds = synthetic.load_wave(data_dir=_WAVE_DIR)
        assert ds.task_type == TaskType.PDE
        assert ds.topology == DataTopology.GRID

    @pytest.mark.unit
    def test_no_periodic_axes(self) -> None:
        ds = synthetic.load_wave(data_dir=_WAVE_DIR)
        assert ds.axes is not None
        assert ds.axes["x"].is_periodic is False
        assert ds.axes["t"].is_periodic is False

    @pytest.mark.numerical
    def test_dtype_float64(self) -> None:
        ds = synthetic.load_wave(data_dir=_WAVE_DIR)
        assert ds.get_field("u").dtype == torch.float64
        assert ds.get_coords("x").dtype == torch.float64
        assert ds.get_coords("t").dtype == torch.float64

    @pytest.mark.numerical
    def test_no_nan_inf(self) -> None:
        ds = synthetic.load_wave(data_dir=_WAVE_DIR)
        assert torch.isfinite(ds.get_field("u")).all()
        assert torch.isfinite(ds.get_coords("x")).all()
        assert torch.isfinite(ds.get_coords("t")).all()

    @pytest.mark.numerical
    def test_physics_u_tt_equals_u_xx(self) -> None:
        ds = synthetic.load_wave(data_dir=_WAVE_DIR)
        (coef_uxx,), nmse = _fit_loaded_pde(ds, with_u_term=False)
        assert coef_uxx == pytest.approx(1.0, rel=1e-3)
        assert nmse < 1e-6







class TestWaveNegative:

    @pytest.mark.unit
    def test_missing_dir_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            synthetic.load_wave(data_dir=Path("/nonexistent/wave/path"))

    @pytest.mark.unit
    def test_empty_dir_raises_file_not_found(self) -> None:
        import tempfile

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(FileNotFoundError),
        ):
            synthetic.load_wave(data_dir=Path(tmpdir))







class TestKleinGordonSmoke:

    @pytest.mark.smoke
    def test_function_exists_and_callable(self) -> None:
        assert callable(synthetic.load_klein_gordon)

    @pytest.mark.smoke
    def test_exported_top_level(self) -> None:
        import kd

        loader = getattr(kd, "load_klein_gordon", None)
        assert callable(loader)

    @skip_no_kg
    @pytest.mark.smoke
    def test_explicit_dir_returns_pde_dataset(self) -> None:
        ds = synthetic.load_klein_gordon(data_dir=_KG_DIR)
        assert isinstance(ds, PDEDataset)







class TestKleinGordonFallback:

    @skip_no_kg
    @pytest.mark.unit
    def test_no_arg_returns_pde_dataset(self) -> None:
        ds = synthetic.load_klein_gordon()
        assert isinstance(ds, PDEDataset)







@skip_no_kg
class TestKleinGordonLoader:

    @pytest.mark.unit
    def test_field_shape(self) -> None:
        ds = synthetic.load_klein_gordon(data_dir=_KG_DIR)
        u = ds.get_field("u")
        assert tuple(u.shape) == _KG_SHAPE

    @pytest.mark.unit
    def test_dataset_shape(self) -> None:
        ds = synthetic.load_klein_gordon(data_dir=_KG_DIR)
        assert ds.get_shape() == _KG_SHAPE

    @pytest.mark.unit
    def test_axis_order_is_x_then_t(self) -> None:
        ds = synthetic.load_klein_gordon(data_dir=_KG_DIR)
        assert ds.axis_order == ["x", "t"]

    @pytest.mark.unit
    def test_axis_lengths(self) -> None:
        ds = synthetic.load_klein_gordon(data_dir=_KG_DIR)
        assert len(ds.get_coords("x")) == _KG_NX
        assert len(ds.get_coords("t")) == _KG_NT

    @pytest.mark.unit
    def test_x_axis_monotonic_increasing(self) -> None:
        ds = synthetic.load_klein_gordon(data_dir=_KG_DIR)
        x = ds.get_coords("x")
        assert (x[1:] > x[:-1]).all(), "x not strictly increasing"

    @pytest.mark.unit
    def test_t_axis_monotonic_increasing(self) -> None:
        ds = synthetic.load_klein_gordon(data_dir=_KG_DIR)
        t = ds.get_coords("t")
        assert (t[1:] > t[:-1]).all(), "t not strictly increasing"

    @pytest.mark.unit
    def test_x_span_and_spacing(self) -> None:
        ds = synthetic.load_klein_gordon(data_dir=_KG_DIR)
        x = ds.get_coords("x")
        torch.testing.assert_close(
            x[0], torch.tensor(_KG_X_LO, dtype=x.dtype), rtol=0.0, atol=_ENDPOINT_ATOL
        )
        torch.testing.assert_close(
            x[-1], torch.tensor(_KG_X_HI, dtype=x.dtype), rtol=0.0, atol=_ENDPOINT_ATOL
        )
        dx = (x[1:] - x[:-1]).mean().item()
        assert dx == pytest.approx(_KG_DX, rel=1e-6)

    @pytest.mark.unit
    def test_t_span_and_spacing(self) -> None:
        ds = synthetic.load_klein_gordon(data_dir=_KG_DIR)
        t = ds.get_coords("t")
        torch.testing.assert_close(
            t[0], torch.tensor(_KG_T_LO, dtype=t.dtype), rtol=0.0, atol=_ENDPOINT_ATOL
        )
        torch.testing.assert_close(
            t[-1], torch.tensor(_KG_T_HI, dtype=t.dtype), rtol=0.0, atol=_ENDPOINT_ATOL
        )
        dt = (t[1:] - t[:-1]).mean().item()
        assert dt == pytest.approx(_KG_DT, rel=1e-6)

    @pytest.mark.unit
    def test_ground_truth_exact(self) -> None:
        ds = synthetic.load_klein_gordon(data_dir=_KG_DIR)
        assert ds.ground_truth == _KG_GROUND_TRUTH

    @pytest.mark.unit
    def test_lhs_spec_second_order(self) -> None:
        ds = synthetic.load_klein_gordon(data_dir=_KG_DIR)
        assert ds.lhs_field == "u"
        assert ds.lhs_axis == "t"
        assert ds.lhs_order == 2

    @pytest.mark.unit
    def test_task_type_and_topology(self) -> None:
        ds = synthetic.load_klein_gordon(data_dir=_KG_DIR)
        assert ds.task_type == TaskType.PDE
        assert ds.topology == DataTopology.GRID

    @pytest.mark.unit
    def test_no_periodic_axes(self) -> None:
        ds = synthetic.load_klein_gordon(data_dir=_KG_DIR)
        assert ds.axes is not None
        assert ds.axes["x"].is_periodic is False
        assert ds.axes["t"].is_periodic is False

    @pytest.mark.numerical
    def test_dtype_float64(self) -> None:
        ds = synthetic.load_klein_gordon(data_dir=_KG_DIR)
        assert ds.get_field("u").dtype == torch.float64
        assert ds.get_coords("x").dtype == torch.float64
        assert ds.get_coords("t").dtype == torch.float64

    @pytest.mark.numerical
    def test_no_nan_inf(self) -> None:
        ds = synthetic.load_klein_gordon(data_dir=_KG_DIR)
        assert torch.isfinite(ds.get_field("u")).all()
        assert torch.isfinite(ds.get_coords("x")).all()
        assert torch.isfinite(ds.get_coords("t")).all()








@skip_no_kg
class TestKleinGordonOrientationProof:

    @pytest.mark.numerical
    def test_recovers_klein_gordon_coefficients(self) -> None:
        ds = synthetic.load_klein_gordon(data_dir=_KG_DIR)
        (coef_uxx, coef_u), nmse = _fit_loaded_pde(ds, with_u_term=True)



        assert coef_uxx == pytest.approx(_KG_A, rel=5e-2)
        assert coef_u == pytest.approx(_KG_B, rel=5e-2)

        assert coef_u < 0.0, "u coefficient sign flipped -> KG loaded transposed"

        assert nmse < 1e-4







class TestKleinGordonNegative:

    @pytest.mark.unit
    def test_missing_dir_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            synthetic.load_klein_gordon(data_dir=Path("/nonexistent/kg/path"))

    @pytest.mark.unit
    def test_empty_dir_raises_file_not_found(self) -> None:
        import tempfile

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(FileNotFoundError),
        ):
            synthetic.load_klein_gordon(data_dir=Path(tmpdir))
