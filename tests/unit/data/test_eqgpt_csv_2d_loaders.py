
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
_EQ_DIR = _ASSETS_DATA_ROOT
_B2D_DIR = _ASSETS_DATA_ROOT















_EQ_FILE = "eqgpt_eq_6_2_12.csv"
_EQ_PATH = _EQ_DIR / _EQ_FILE

_EQ_NX = 501
_EQ_NT = 501
_EQ_SHAPE = (_EQ_NX, _EQ_NT)
_EQ_X_LO = 0.0
_EQ_X_HI = 5.0
_EQ_T_LO = 0.0
_EQ_T_HI = 10.0
_EQ_DX = (_EQ_X_HI - _EQ_X_LO) / (_EQ_NX - 1)
_EQ_DT = (_EQ_T_HI - _EQ_T_LO) / (_EQ_NT - 1)
_EQ_GROUND_TRUTH = "u_t = -0.1*u_x_t - 0.1*u_x"

_EQ_COEF = -0.1
_EQ_COEF_REL = 0.4



_EQ_NMSE_SANITY = 0.1













_B2D_FILE = "eqgpt_burgers_2d.mat"
_B2D_PATH = _B2D_DIR / _B2D_FILE

_B2D_NX = 101
_B2D_NY = 51
_B2D_NT = 100
_B2D_SHAPE = (_B2D_NX, _B2D_NY, _B2D_NT)
_B2D_X_LO, _B2D_X_HI = -1.0, 1.0
_B2D_Y_LO, _B2D_Y_HI = -1.0, 1.0
_B2D_T_LO, _B2D_T_HI = 0.0, 1.98
_B2D_DX = (_B2D_X_HI - _B2D_X_LO) / (_B2D_NX - 1)
_B2D_DY = (_B2D_Y_HI - _B2D_Y_LO) / (_B2D_NY - 1)
_B2D_DT = (_B2D_T_HI - _B2D_T_LO) / (_B2D_NT - 1)
_B2D_GROUND_TRUTH = "u_t = -u*u_x - u*u_y + 0.01*u_xx + 0.01*u_yy"
_B2D_ADVECTION_COEF = -1.0
_B2D_DIFFUSION_COEF = 0.01
_B2D_COEF_REL = 0.1
_B2D_NMSE_CEILING = 1e-3

_ENDPOINT_ATOL = 1e-6





skip_no_eq = pytest.mark.skipif(
    not _EQ_PATH.exists(),
    reason=f"Eq. 6.2.12 data not found: {_EQ_PATH}",
)
skip_no_b2d = pytest.mark.skipif(
    not _B2D_PATH.exists(),
    reason=f"2D Burgers data not found: {_B2D_PATH}",
)









def _first_difference(field: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    fwd = np.roll(field, -1, axis=axis)
    bwd = np.roll(field, 1, axis=axis)
    return (fwd - bwd) / (2.0 * spacing)


def _second_difference(field: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    fwd = np.roll(field, -1, axis=axis)
    bwd = np.roll(field, 1, axis=axis)
    return (fwd - 2.0 * field + bwd) / spacing**2


def _interior(ndim: int) -> tuple[slice, ...]:
    return tuple(slice(1, -1) for _ in range(ndim))


def _fit_eq_6_2_12(ds: PDEDataset) -> tuple[np.ndarray, float]:
    assert ds.axis_order is not None
    x_axis = ds.axis_order.index("x")
    t_axis = ds.axis_order.index("t")

    u = ds.get_field("u").detach().cpu().numpy().astype(np.float64)
    x = ds.get_coords("x").detach().cpu().numpy().astype(np.float64)
    t = ds.get_coords("t").detach().cpu().numpy().astype(np.float64)
    dx = float(np.diff(x).mean())
    dt = float(np.diff(t).mean())

    u_t = _first_difference(u, dt, t_axis)
    u_x = _first_difference(u, dx, x_axis)
    u_x_t = _first_difference(_first_difference(u, dx, x_axis), dt, t_axis)

    sl = _interior(u.ndim)
    target = u_t[sl].ravel()
    design = np.stack([u_x_t[sl].ravel(), u_x[sl].ravel()], axis=1)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    prediction = design @ coef
    nmse = float(np.mean((target - prediction) ** 2) / np.mean(target**2))
    return coef, nmse


def _fit_burgers_2d(ds: PDEDataset) -> tuple[np.ndarray, float]:
    assert ds.axis_order is not None
    x_axis = ds.axis_order.index("x")
    y_axis = ds.axis_order.index("y")
    t_axis = ds.axis_order.index("t")

    u = ds.get_field("u").detach().cpu().numpy().astype(np.float64)
    dx = float(np.diff(ds.get_coords("x").detach().cpu().numpy()).mean())
    dy = float(np.diff(ds.get_coords("y").detach().cpu().numpy()).mean())
    dt = float(np.diff(ds.get_coords("t").detach().cpu().numpy()).mean())

    u_t = _first_difference(u, dt, t_axis)
    u_x = _first_difference(u, dx, x_axis)
    u_y = _first_difference(u, dy, y_axis)
    u_xx = _second_difference(u, dx, x_axis)
    u_yy = _second_difference(u, dy, y_axis)

    sl = _interior(u.ndim)
    target = u_t[sl].ravel()
    design = np.stack(
        [
            (u * u_x)[sl].ravel(),
            (u * u_y)[sl].ravel(),
            u_xx[sl].ravel(),
            u_yy[sl].ravel(),
        ],
        axis=1,
    )
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    prediction = design @ coef
    nmse = float(np.mean((target - prediction) ** 2) / np.mean(target**2))
    return coef, nmse


def _write_burgers_2d_mat(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    *,
    seed: int = 0,
) -> None:
    import scipy.io as sio

    nx, ny, nt = len(x), len(y), len(t)
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((nt, ny, nx)).astype(np.float64)
    sio.savemat(
        str(path),
        {
            "u": u,
            "x": np.asarray(x, dtype=np.float64).reshape(1, nx),
            "y": np.asarray(y, dtype=np.float64).reshape(1, ny),
            "t": np.asarray(t, dtype=np.float64).reshape(1, nt),
        },
    )







class TestEq6212Smoke:

    @pytest.mark.smoke
    def test_function_exists_and_callable(self) -> None:
        assert callable(synthetic.load_eq_6_2_12)

    @pytest.mark.smoke
    def test_exported_through_three_layers(self) -> None:
        import kd
        import kd.data

        assert callable(getattr(kd, "load_eq_6_2_12", None))
        assert callable(getattr(kd.data, "load_eq_6_2_12", None))
        assert callable(getattr(kd.data.synthetic, "load_eq_6_2_12", None))

    @skip_no_eq
    @pytest.mark.smoke
    def test_explicit_dir_returns_pde_dataset(self) -> None:
        ds = synthetic.load_eq_6_2_12(data_dir=_EQ_DIR)
        assert isinstance(ds, PDEDataset)


class TestEq6212Fallback:

    @skip_no_eq
    @pytest.mark.unit
    def test_no_arg_returns_pde_dataset(self) -> None:
        ds = synthetic.load_eq_6_2_12()
        assert isinstance(ds, PDEDataset)







@skip_no_eq
class TestEq6212Loader:

    @pytest.mark.unit
    def test_field_shape(self) -> None:
        ds = synthetic.load_eq_6_2_12(data_dir=_EQ_DIR)
        assert tuple(ds.get_field("u").shape) == _EQ_SHAPE

    @pytest.mark.unit
    def test_dataset_shape_and_axis_order(self) -> None:
        ds = synthetic.load_eq_6_2_12(data_dir=_EQ_DIR)
        assert ds.axis_order == ["x", "t"]
        assert ds.get_shape() == _EQ_SHAPE

    @pytest.mark.unit
    def test_axis_lengths(self) -> None:
        ds = synthetic.load_eq_6_2_12(data_dir=_EQ_DIR)
        assert len(ds.get_coords("x")) == _EQ_NX
        assert len(ds.get_coords("t")) == _EQ_NT

    @pytest.mark.unit
    def test_axes_monotonic_increasing(self) -> None:
        ds = synthetic.load_eq_6_2_12(data_dir=_EQ_DIR)
        x = ds.get_coords("x")
        t = ds.get_coords("t")
        assert (x[1:] > x[:-1]).all(), "x not strictly increasing"
        assert (t[1:] > t[:-1]).all(), "t not strictly increasing"

    @pytest.mark.unit
    def test_x_span_and_spacing(self) -> None:
        ds = synthetic.load_eq_6_2_12(data_dir=_EQ_DIR)
        x = ds.get_coords("x")
        torch.testing.assert_close(
            x[0], torch.tensor(_EQ_X_LO, dtype=x.dtype), rtol=0.0, atol=_ENDPOINT_ATOL
        )
        torch.testing.assert_close(
            x[-1], torch.tensor(_EQ_X_HI, dtype=x.dtype), rtol=0.0, atol=_ENDPOINT_ATOL
        )
        assert (x[1:] - x[:-1]).mean().item() == pytest.approx(_EQ_DX, rel=1e-6)

    @pytest.mark.unit
    def test_t_span_and_spacing(self) -> None:
        ds = synthetic.load_eq_6_2_12(data_dir=_EQ_DIR)
        t = ds.get_coords("t")
        torch.testing.assert_close(
            t[0], torch.tensor(_EQ_T_LO, dtype=t.dtype), rtol=0.0, atol=_ENDPOINT_ATOL
        )
        torch.testing.assert_close(
            t[-1], torch.tensor(_EQ_T_HI, dtype=t.dtype), rtol=0.0, atol=_ENDPOINT_ATOL
        )
        assert (t[1:] - t[:-1]).mean().item() == pytest.approx(_EQ_DT, rel=1e-6)

    @pytest.mark.unit
    def test_ground_truth_exact_and_mixed_term(self) -> None:
        ds = synthetic.load_eq_6_2_12(data_dir=_EQ_DIR)
        assert ds.ground_truth == _EQ_GROUND_TRUTH
        assert "u_x_t" in _EQ_GROUND_TRUTH
        assert "u_x" in _EQ_GROUND_TRUTH

    @pytest.mark.unit
    def test_lhs_spec_first_order(self) -> None:
        ds = synthetic.load_eq_6_2_12(data_dir=_EQ_DIR)
        assert ds.lhs_field == "u"
        assert ds.lhs_axis == "t"
        assert ds.lhs_order == 1

    @pytest.mark.unit
    def test_task_type_and_topology(self) -> None:
        ds = synthetic.load_eq_6_2_12(data_dir=_EQ_DIR)
        assert ds.task_type == TaskType.PDE
        assert ds.topology == DataTopology.GRID

    @pytest.mark.unit
    def test_axes_non_periodic(self) -> None:
        ds = synthetic.load_eq_6_2_12(data_dir=_EQ_DIR)
        assert ds.axes is not None
        assert ds.axes["x"].is_periodic is False
        assert ds.axes["t"].is_periodic is False

    @pytest.mark.numerical
    def test_dtype_float64(self) -> None:
        ds = synthetic.load_eq_6_2_12(data_dir=_EQ_DIR)
        assert ds.get_field("u").dtype == torch.float64
        assert ds.get_coords("x").dtype == torch.float64
        assert ds.get_coords("t").dtype == torch.float64

    @pytest.mark.numerical
    def test_no_nan_inf(self) -> None:
        ds = synthetic.load_eq_6_2_12(data_dir=_EQ_DIR)
        assert torch.isfinite(ds.get_field("u")).all()
        assert torch.isfinite(ds.get_coords("x")).all()
        assert torch.isfinite(ds.get_coords("t")).all()







@skip_no_eq
class TestEq6212OrientationProof:

    @pytest.mark.numerical
    def test_recovers_mixed_derivative_coefficients(self) -> None:
        ds = synthetic.load_eq_6_2_12(data_dir=_EQ_DIR)
        (coef_uxt, coef_ux), nmse = _fit_eq_6_2_12(ds)



        assert coef_uxt < 0.0, "u_x_t coefficient sign flipped"
        assert coef_ux < 0.0, "u_x coefficient sign flipped"


        assert coef_uxt == pytest.approx(_EQ_COEF, rel=_EQ_COEF_REL)
        assert coef_ux == pytest.approx(_EQ_COEF, rel=_EQ_COEF_REL)

        assert nmse < _EQ_NMSE_SANITY







@skip_no_eq
class TestEq6212MixedDerivativeEvaluate:

    @pytest.mark.numerical
    def test_u_x_t_term_evaluates_finite(self) -> None:
        from kd.evaluate import evaluate_terms, validate_terms

        ds = synthetic.load_eq_6_2_12(data_dir=_EQ_DIR)

        report = validate_terms(ds, ["u_x_t", "u_x"], max_order=2)
        assert "u_x_t" in report.valid
        assert "u_x" in report.valid
        assert report.rejected == []

        result = evaluate_terms(ds, ["u_x_t", "u_x"], max_order=2)
        assert result.is_valid
        assert math.isfinite(result.mse)
        assert result.nmse < 0.1
        assert result.coefficients is not None
        assert torch.isfinite(result.coefficients).all()

        coef_uxt = float(result.coefficients[0])
        coef_ux = float(result.coefficients[1])
        assert coef_uxt == pytest.approx(_EQ_COEF, rel=0.5)
        assert coef_ux == pytest.approx(_EQ_COEF, rel=0.5)







class TestEq6212Negative:

    @pytest.mark.unit
    def test_missing_dir_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            synthetic.load_eq_6_2_12(data_dir=Path("/nonexistent/eq6212/path"))

    @pytest.mark.unit
    def test_empty_dir_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            synthetic.load_eq_6_2_12(data_dir=tmp_path)

    @pytest.mark.unit
    def test_wrong_shape_csv_raises_value_error(self, tmp_path: Path) -> None:
        wrong = np.arange(100, dtype=np.float64).reshape(10, 10)
        np.savetxt(tmp_path / _EQ_FILE, wrong, delimiter=",")
        with pytest.raises(ValueError):
            synthetic.load_eq_6_2_12(data_dir=tmp_path)







class TestBurgers2DSmoke:

    @pytest.mark.smoke
    def test_function_exists_and_callable(self) -> None:
        assert callable(synthetic.load_burgers_2d)

    @pytest.mark.smoke
    def test_exported_through_three_layers(self) -> None:
        import kd
        import kd.data

        assert callable(getattr(kd, "load_burgers_2d", None))
        assert callable(getattr(kd.data, "load_burgers_2d", None))
        assert callable(getattr(kd.data.synthetic, "load_burgers_2d", None))

    @skip_no_b2d
    @pytest.mark.smoke
    def test_explicit_dir_returns_pde_dataset(self) -> None:
        ds = synthetic.load_burgers_2d(data_dir=_B2D_DIR)
        assert isinstance(ds, PDEDataset)


class TestBurgers2DFallback:

    @skip_no_b2d
    @pytest.mark.unit
    def test_no_arg_returns_pde_dataset(self) -> None:
        ds = synthetic.load_burgers_2d()
        assert isinstance(ds, PDEDataset)







@skip_no_b2d
class TestBurgers2DLoader:

    @pytest.mark.unit
    def test_field_shape(self) -> None:
        ds = synthetic.load_burgers_2d(data_dir=_B2D_DIR)
        assert tuple(ds.get_field("u").shape) == _B2D_SHAPE

    @pytest.mark.unit
    def test_dataset_shape_and_axis_order(self) -> None:
        ds = synthetic.load_burgers_2d(data_dir=_B2D_DIR)
        assert ds.axis_order == ["x", "y", "t"]
        assert ds.get_shape() == _B2D_SHAPE

    @pytest.mark.unit
    def test_axis_lengths(self) -> None:
        ds = synthetic.load_burgers_2d(data_dir=_B2D_DIR)
        assert len(ds.get_coords("x")) == _B2D_NX
        assert len(ds.get_coords("y")) == _B2D_NY
        assert len(ds.get_coords("t")) == _B2D_NT

    @pytest.mark.unit
    def test_axes_monotonic_increasing(self) -> None:
        ds = synthetic.load_burgers_2d(data_dir=_B2D_DIR)
        for axis in ("x", "y", "t"):
            coords = ds.get_coords(axis)
            assert (coords[1:] > coords[:-1]).all(), f"{axis} not strictly increasing"

    @pytest.mark.unit
    def test_x_span_and_spacing(self) -> None:
        ds = synthetic.load_burgers_2d(data_dir=_B2D_DIR)
        x = ds.get_coords("x")
        torch.testing.assert_close(
            x[0], torch.tensor(_B2D_X_LO, dtype=x.dtype), rtol=0.0, atol=_ENDPOINT_ATOL
        )
        torch.testing.assert_close(
            x[-1], torch.tensor(_B2D_X_HI, dtype=x.dtype), rtol=0.0, atol=_ENDPOINT_ATOL
        )
        assert (x[1:] - x[:-1]).mean().item() == pytest.approx(_B2D_DX, rel=1e-6)

    @pytest.mark.unit
    def test_y_span_and_spacing(self) -> None:
        ds = synthetic.load_burgers_2d(data_dir=_B2D_DIR)
        y = ds.get_coords("y")
        torch.testing.assert_close(
            y[0], torch.tensor(_B2D_Y_LO, dtype=y.dtype), rtol=0.0, atol=_ENDPOINT_ATOL
        )
        torch.testing.assert_close(
            y[-1], torch.tensor(_B2D_Y_HI, dtype=y.dtype), rtol=0.0, atol=_ENDPOINT_ATOL
        )
        assert (y[1:] - y[:-1]).mean().item() == pytest.approx(_B2D_DY, rel=1e-6)

    @pytest.mark.unit
    def test_t_span_and_spacing(self) -> None:
        ds = synthetic.load_burgers_2d(data_dir=_B2D_DIR)
        t = ds.get_coords("t")
        torch.testing.assert_close(
            t[0], torch.tensor(_B2D_T_LO, dtype=t.dtype), rtol=0.0, atol=_ENDPOINT_ATOL
        )
        torch.testing.assert_close(
            t[-1], torch.tensor(_B2D_T_HI, dtype=t.dtype), rtol=0.0, atol=_ENDPOINT_ATOL
        )
        assert (t[1:] - t[:-1]).mean().item() == pytest.approx(_B2D_DT, rel=1e-6)

    @pytest.mark.unit
    def test_ground_truth_exact(self) -> None:
        ds = synthetic.load_burgers_2d(data_dir=_B2D_DIR)
        assert ds.ground_truth == _B2D_GROUND_TRUTH

    @pytest.mark.unit
    def test_lhs_spec_first_order(self) -> None:
        ds = synthetic.load_burgers_2d(data_dir=_B2D_DIR)
        assert ds.lhs_field == "u"
        assert ds.lhs_axis == "t"
        assert ds.lhs_order == 1

    @pytest.mark.unit
    def test_task_type_and_topology(self) -> None:
        ds = synthetic.load_burgers_2d(data_dir=_B2D_DIR)
        assert ds.task_type == TaskType.PDE
        assert ds.topology == DataTopology.GRID

    @pytest.mark.unit
    def test_axes_non_periodic(self) -> None:
        ds = synthetic.load_burgers_2d(data_dir=_B2D_DIR)
        assert ds.axes is not None
        for axis in ("x", "y", "t"):
            assert ds.axes[axis].is_periodic is False

    @pytest.mark.numerical
    def test_dtype_float64(self) -> None:
        ds = synthetic.load_burgers_2d(data_dir=_B2D_DIR)
        assert ds.get_field("u").dtype == torch.float64
        for axis in ("x", "y", "t"):
            assert ds.get_coords(axis).dtype == torch.float64

    @pytest.mark.numerical
    def test_no_nan_inf(self) -> None:
        ds = synthetic.load_burgers_2d(data_dir=_B2D_DIR)
        assert torch.isfinite(ds.get_field("u")).all()
        for axis in ("x", "y", "t"):
            assert torch.isfinite(ds.get_coords(axis)).all()







@skip_no_b2d
class TestBurgers2DOrientationProof:

    @pytest.mark.numerical
    def test_recovers_burgers_coefficients(self) -> None:
        ds = synthetic.load_burgers_2d(data_dir=_B2D_DIR)
        coef, nmse = _fit_burgers_2d(ds)
        coef_uux, coef_uuy, coef_uxx, coef_uyy = (float(c) for c in coef)

        assert coef_uux == pytest.approx(_B2D_ADVECTION_COEF, rel=_B2D_COEF_REL)
        assert coef_uuy == pytest.approx(_B2D_ADVECTION_COEF, rel=_B2D_COEF_REL)
        assert coef_uxx == pytest.approx(_B2D_DIFFUSION_COEF, rel=_B2D_COEF_REL)
        assert coef_uyy == pytest.approx(_B2D_DIFFUSION_COEF, rel=_B2D_COEF_REL)

        assert coef_uxx > 0.0
        assert coef_uyy > 0.0
        assert nmse < _B2D_NMSE_CEILING







@skip_no_b2d
class TestBurgers2DEvaluate:

    @pytest.mark.numerical
    def test_burgers_terms_evaluate_to_ground_truth(self) -> None:
        from kd.evaluate import evaluate_terms, validate_terms

        ds = synthetic.load_burgers_2d(data_dir=_B2D_DIR)
        terms = ["mul(u,u_x)", "mul(u,u_y)", "u_xx", "u_yy"]

        report = validate_terms(ds, terms, max_order=2)
        assert report.valid == terms
        assert report.rejected == []

        result = evaluate_terms(ds, terms, max_order=2)
        assert result.is_valid
        assert result.nmse < _B2D_NMSE_CEILING
        assert result.coefficients is not None
        assert torch.isfinite(result.coefficients).all()

        coef_uux = float(result.coefficients[0])
        coef_uuy = float(result.coefficients[1])
        coef_uxx = float(result.coefficients[2])
        coef_uyy = float(result.coefficients[3])
        assert coef_uux == pytest.approx(_B2D_ADVECTION_COEF, rel=_B2D_COEF_REL)
        assert coef_uuy == pytest.approx(_B2D_ADVECTION_COEF, rel=_B2D_COEF_REL)
        assert coef_uxx == pytest.approx(_B2D_DIFFUSION_COEF, rel=_B2D_COEF_REL)
        assert coef_uyy == pytest.approx(_B2D_DIFFUSION_COEF, rel=_B2D_COEF_REL)
        assert coef_uxx > 0.0
        assert coef_uyy > 0.0







class TestBurgers2DNonUniformGuard:

    @pytest.mark.unit
    def test_uniform_synthetic_fixture_is_accepted(self, tmp_path: Path) -> None:
        coords = {
            "x": np.linspace(0.0, 1.0, 5),
            "y": np.linspace(0.0, 1.0, 3),
            "t": np.linspace(0.0, 1.0, 4),
        }
        path = tmp_path / _B2D_FILE
        _write_burgers_2d_mat(path, coords["x"], coords["y"], coords["t"])

        ds = synthetic.load_burgers_2d(data_dir=tmp_path)
        assert isinstance(ds, PDEDataset)
        assert ds.axis_order == ["x", "y", "t"]
        assert tuple(ds.get_field("u").shape) == (5, 3, 4)

    @pytest.mark.numerical
    @pytest.mark.parametrize("bad_axis", ["x", "y", "t"])
    def test_non_uniform_axis_raises(self, tmp_path: Path, bad_axis: str) -> None:
        sizes = {"x": 5, "y": 3, "t": 4}
        coords = {ax: np.linspace(0.0, 1.0, n) for ax, n in sizes.items()}
        base = coords[bad_axis].copy()


        spacing = base[1] - base[0]
        base[len(base) // 2] += 0.4 * spacing
        coords[bad_axis] = base

        path = tmp_path / _B2D_FILE
        _write_burgers_2d_mat(path, coords["x"], coords["y"], coords["t"])

        with pytest.raises(ValueError):
            synthetic.load_burgers_2d(data_dir=tmp_path)

    @pytest.mark.unit
    def test_descending_axis_raises(self, tmp_path: Path) -> None:
        coords = {
            "x": np.linspace(1.0, 0.0, 5),
            "y": np.linspace(0.0, 1.0, 3),
            "t": np.linspace(0.0, 1.0, 4),
        }
        path = tmp_path / _B2D_FILE
        _write_burgers_2d_mat(path, coords["x"], coords["y"], coords["t"])

        with pytest.raises(ValueError, match=r"axis 'x'"):
            synthetic.load_burgers_2d(data_dir=tmp_path)

    @pytest.mark.unit
    def test_zero_spacing_axis_raises(self, tmp_path: Path) -> None:
        coords = {
            "x": np.full(5, 0.5),
            "y": np.linspace(0.0, 1.0, 3),
            "t": np.linspace(0.0, 1.0, 4),
        }
        path = tmp_path / _B2D_FILE
        _write_burgers_2d_mat(path, coords["x"], coords["y"], coords["t"])

        with pytest.raises(ValueError, match=r"axis 'x'"):
            synthetic.load_burgers_2d(data_dir=tmp_path)







class TestBurgers2DNegative:

    @pytest.mark.unit
    def test_missing_dir_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            synthetic.load_burgers_2d(data_dir=Path("/nonexistent/burgers2d/path"))

    @pytest.mark.unit
    def test_empty_dir_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            synthetic.load_burgers_2d(data_dir=tmp_path)
