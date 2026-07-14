
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from kd.search.llm4ed import fd

FloatArray = npt.NDArray[np.float64]


def _quadratic_1d() -> FloatArray:
    return np.array([0.0, 1.0, 4.0, 9.0, 16.0])


class TestFiniteDiff1D:

    def test_interior_central(self) -> None:
        ux = fd.finite_diff(_quadratic_1d(), 1.0)

        np.testing.assert_array_equal(ux[1:4], [2.0, 4.0, 6.0])

    def test_boundary_one_sided(self) -> None:
        ux = fd.finite_diff(_quadratic_1d(), 1.0)

        assert ux[0] == 0.0
        assert ux[4] == 8.0

    def test_min_length_enforced(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            fd.finite_diff(np.array([0.0, 1.0]), 1.0)


def _make_2d() -> tuple[FloatArray, FloatArray]:
    col = _quadratic_1d()
    u = np.stack([col, 2.0 * col], axis=1)
    x = np.arange(5, dtype=np.float64).reshape(-1, 1)
    return u, x


class TestDiff2D:

    def test_columns_independent(self) -> None:
        u, x = _make_2d()
        ux = fd.diff(u, x)
        expected = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        np.testing.assert_array_equal(ux[:, 0], expected)
        np.testing.assert_array_equal(ux[:, 1], 2.0 * expected)

    def test_spacing_read_from_x2_minus_x1(self) -> None:
        u, _ = _make_2d()
        x_nonuniform = np.array([-1.0, 0.5, 1.5, 2.5, 3.5]).reshape(-1, 1)
        x_uniform = np.arange(5, dtype=np.float64).reshape(-1, 1)
        np.testing.assert_array_equal(
            fd.diff(u, x_nonuniform), fd.diff(u, x_uniform)
        )

    def test_accepts_1d_x(self) -> None:
        u, x = _make_2d()
        np.testing.assert_array_equal(fd.diff(u, x.ravel()), fd.diff(u, x))

    def test_float64_required(self) -> None:
        u, x = _make_2d()
        with pytest.raises(ValueError, match="float64"):
            fd.diff(u.astype(np.float32), x)

    def test_min_rows_enforced(self) -> None:
        u, x = _make_2d()
        with pytest.raises(ValueError, match="at least 3"):
            fd.diff(u[:2,:], x[:2])


class TestDiff2:

    def test_exact_for_quadratic_everywhere(self) -> None:
        u, x = _make_2d()
        uxx = fd.diff2(u, x)


        np.testing.assert_array_equal(uxx[:, 0], np.full(5, 2.0))
        np.testing.assert_array_equal(uxx[:, 1], np.full(5, 4.0))

    def test_min_rows_enforced(self) -> None:
        u, x = _make_2d()
        with pytest.raises(ValueError, match="at least 4"):
            fd.diff2(u[:3,:], x[:3])


class TestDiff3:

    def test_exact_for_cubic_everywhere(self) -> None:


        x = np.arange(6, dtype=np.float64).reshape(-1, 1)
        u = (x**3) @ np.ones((1, 2))
        uxxx = fd.diff3(u, x)
        np.testing.assert_allclose(uxxx, np.full((6, 2), 6.0), rtol=1e-12)

    def test_composition_order_is_diff2_first(self) -> None:
        x = np.arange(6, dtype=np.float64).reshape(-1, 1)
        u = (x**3) @ np.ones((1, 2))
        diff2_first = fd.diff(fd.diff2(u, x), x)
        diff_first = fd.diff2(fd.diff(u, x), x)

        assert not np.allclose(diff2_first, diff_first)
        np.testing.assert_array_equal(fd.diff3(u, x), diff2_first)


class TestBuildOperandColumns:

    @staticmethod
    def _field() -> tuple[FloatArray, FloatArray, FloatArray]:
        rng = np.random.RandomState(7)
        u = rng.rand(6, 5)
        x = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
        t = np.linspace(0.0, 0.4, 5).reshape(-1, 1)
        return u, x, t

    def test_feature_dict_keys_and_order(self) -> None:
        u, x, t = self._field()
        _, features = fd.build_operand_columns(u, x, t)



        assert tuple(features.keys()) == fd.OPERAND_ORDER
        assert fd.OPERAND_ORDER == ("x", "u_x", "u_xx", "u_xxx", "u")

    def test_shapes_and_dtype(self) -> None:
        u, x, t = self._field()
        lhs, features = fd.build_operand_columns(u, x, t)
        assert lhs.shape == (30, 1)
        assert lhs.dtype == np.float64
        for name, column in features.items():
            assert column.shape == (30,), name
            assert column.dtype == np.float64, name

    def test_u_column_is_row_major_flatten(self) -> None:
        u, x, t = self._field()
        _, features = fd.build_operand_columns(u, x, t)
        np.testing.assert_array_equal(features["u"], u.reshape(-1))

    def test_x_column_repeats_across_time(self) -> None:
        u, x, t = self._field()
        _, features = fd.build_operand_columns(u, x, t)
        m = u.shape[1]
        np.testing.assert_array_equal(features["x"][:m], np.full(m, x[0, 0]))
        np.testing.assert_array_equal(features["x"][m: 2 * m], np.full(m, x[1, 0]))

    def test_ut_rows_use_finite_diff_over_time(self) -> None:
        u, x, t = self._field()
        lhs, _ = fd.build_operand_columns(u, x, t)
        dt = float(t[1, 0] - t[0, 0])
        m = u.shape[1]
        np.testing.assert_array_equal(lhs[:m, 0], fd.finite_diff(u[0,:], dt))
        np.testing.assert_array_equal(lhs[m: 2 * m, 0], fd.finite_diff(u[1,:], dt))

    def test_dt_read_from_t1_minus_t0(self) -> None:
        u, x, t = self._field()
        t_nonuniform = t.copy()
        t_nonuniform[2:] += 0.35
        lhs_uniform, _ = fd.build_operand_columns(u, x, t)
        lhs_nonuniform, _ = fd.build_operand_columns(u, x, t_nonuniform)
        np.testing.assert_array_equal(lhs_uniform, lhs_nonuniform)

    def test_derivative_columns_match_templates(self) -> None:
        u, x, t = self._field()
        _, features = fd.build_operand_columns(u, x, t)
        np.testing.assert_array_equal(features["u_x"], fd.diff(u, x).reshape(-1))
        np.testing.assert_array_equal(features["u_xx"], fd.diff2(u, x).reshape(-1))
        np.testing.assert_array_equal(features["u_xxx"], fd.diff3(u, x).reshape(-1))

    def test_accepts_1d_axes(self) -> None:
        u, x, t = self._field()
        lhs_2d, feats_2d = fd.build_operand_columns(u, x, t)
        lhs_1d, feats_1d = fd.build_operand_columns(u, x.ravel(), t.ravel())
        np.testing.assert_array_equal(lhs_2d, lhs_1d)
        for name in feats_2d:
            np.testing.assert_array_equal(feats_2d[name], feats_1d[name])

    def test_float64_required(self) -> None:
        u, x, t = self._field()
        with pytest.raises(ValueError, match="float64"):
            fd.build_operand_columns(u.astype(np.float32), x, t)

    def test_min_grid_enforced(self) -> None:
        u, x, t = self._field()
        with pytest.raises(ValueError, match="at least 4"):
            fd.build_operand_columns(u[:3,:], x[:3], t)
        with pytest.raises(ValueError, match="at least 3"):
            fd.build_operand_columns(u[:, :2], x, t[:2])
