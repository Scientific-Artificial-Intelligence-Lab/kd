
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from kd.search.llm4ed.stridge import (
    ERROR_ABNORMAL_COEF,
    ERROR_LSTSQ,
    SparseSolveResult,
    TrainStridgeResult,
    sparse_solve,
    train_stridge,
    valid_coef,
)

FloatArray = npt.NDArray[np.float64]


def _sparse_system(
    n: int = 200,
    d: int = 6,
    coeffs: tuple[float, ...] = (1.5, 0.0, -2.0, 0.0, 3.0, 0.0),
    seed: int = 42,
) -> tuple[FloatArray, FloatArray]:
    rng = np.random.RandomState(seed)
    theta = rng.randn(n, d)
    y = theta @ np.asarray(coeffs)
    return theta, y


class TestTrainStridge:

    def test_recovers_sparse_support(self) -> None:
        theta, y = _sparse_system()
        result = train_stridge(theta, y)
        assert isinstance(result, TrainStridgeResult)
        assert set(np.nonzero(result.coefficients)[0]) == {0, 2, 4}
        np.testing.assert_allclose(
            result.coefficients[[0, 2, 4]], [1.5, -2.0, 3.0], rtol=1e-8
        )

    def test_off_support_coefficients_are_exact_zeros(self) -> None:
        theta, y = _sparse_system()
        result = train_stridge(theta, y)
        assert (result.coefficients[[1, 3, 5]] == 0.0).all()

    def test_err_is_in_sample_mse_of_best(self) -> None:
        theta, y = _sparse_system()
        result = train_stridge(theta, y)
        mse = float(np.mean((y - theta @ result.coefficients) ** 2))
        assert result.err == pytest.approx(mse, rel=1e-12)

    def test_accepts_column_y(self) -> None:
        theta, y = _sparse_system()
        flat = train_stridge(theta, y)
        column = train_stridge(theta, y.reshape(-1, 1))



        np.testing.assert_array_equal(flat.coefficients, column.coefficients)

    def test_output_dtype_and_shape(self) -> None:
        theta, y = _sparse_system()
        result = train_stridge(theta, y)
        assert result.coefficients.dtype == np.float64
        assert result.coefficients.shape == (theta.shape[1],)

    def test_float64_required(self) -> None:
        theta, y = _sparse_system()
        with pytest.raises(ValueError, match="float64"):
            train_stridge(theta.astype(np.float32), y.astype(np.float32))

    def test_all_below_tol_branch_bypasses_inner_solve(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import kd.search.llm4ed.stridge as stridge_mod

        def _boom(*args: object, **kwargs: object) -> object:
            raise AssertionError("_stridge must not be called on this input")

        monkeypatch.setattr(stridge_mod, "_stridge", _boom)
        theta, _ = _sparse_system()
        result = train_stridge(theta, np.zeros(theta.shape[0]))
        np.testing.assert_array_equal(
            result.coefficients, np.zeros(theta.shape[1])
        )


class TestValidCoef:

    @pytest.mark.parametrize(
        ("coefficients", "expected"),
        [
            ([], True),
            ([0.0], True),
            ([1.0, -5.0], True),
            ([1e-4], True),
            ([1e4], True),
            ([9.9e-5], False),
            ([-9.9e-5], False),
            ([1.1e4], False),
            ([-1.1e4], False),
            ([0.0, 2e4], False),
            ([1.0, 1e-5], False),
        ],
    )
    def test_gate(self, coefficients: list[float], expected: bool) -> None:
        assert valid_coef(np.asarray(coefficients, dtype=np.float64)) is expected

    def test_nan_passes_the_gate(self) -> None:
        assert valid_coef(np.asarray([np.nan])) is True


class TestSparseSolve:

    def test_valid_path(self) -> None:
        theta, y = _sparse_system()
        outcome = sparse_solve(theta, y)
        assert isinstance(outcome, SparseSolveResult)
        assert outcome.valid is True
        assert outcome.error_type is None
        assert outcome.coefficients is not None
        assert outcome.y_hat is not None

        assert outcome.y_hat.shape == (theta.shape[0], 1)
        np.testing.assert_allclose(
            outcome.y_hat,
            (theta @ outcome.coefficients).reshape(-1, 1),
            rtol=1e-12,
        )

    def test_abnormal_coef_rejected(self) -> None:
        theta, y = _sparse_system(coeffs=(2e4, 0.0, 0.0, 0.0, 0.0, 0.0))
        outcome = sparse_solve(theta, y)
        assert outcome.valid is False
        assert outcome.error_type == ERROR_ABNORMAL_COEF

        assert outcome.coefficients is not None
        assert outcome.y_hat is not None

    def test_lstsq_error_on_nonfinite_theta(self) -> None:
        theta, y = _sparse_system()
        theta = theta.copy()
        theta[0, 0] = np.nan
        outcome = sparse_solve(theta, y)
        assert outcome.valid is False
        assert outcome.error_type == ERROR_LSTSQ
        assert outcome.coefficients is None
        assert outcome.y_hat is None

    def test_error_strings_match_edl_verbatim(self) -> None:
        assert ERROR_LSTSQ == "lstsq_error"
        assert ERROR_ABNORMAL_COEF == "abnormal coef"

    def test_probe_coef_error_is_ignored(self) -> None:
        rng = np.random.RandomState(3)
        theta = rng.randn(300, 3)


        y = theta @ np.asarray([1.5, 5e-5, -2.0])
        probe = np.linalg.lstsq(theta, y.reshape(-1, 1), rcond=None)[0]
        assert not valid_coef(probe.ravel())
        outcome = sparse_solve(theta, y)
        assert outcome.valid is True
        assert outcome.error_type is None
