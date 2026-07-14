
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]


ERROR_LSTSQ: Final[str] = "lstsq_error"
ERROR_ABNORMAL_COEF: Final[str] = "abnormal coef"


COEF_MIN: Final[float] = 1e-4
COEF_MAX: Final[float] = 1e4


DEFAULT_LAM: Final[float] = 1e-5
DEFAULT_D_TOL: Final[float] = 1.0
DEFAULT_MAXIT: Final[int] = 100
DEFAULT_STR_ITERS: Final[int] = 10
DEFAULT_L0_PENALTY: Final[float] = 1e-5
DEFAULT_NORMALIZE: Final[int] = 2


@dataclass(frozen=True)
class TrainStridgeResult:

    coefficients: FloatArray
    err: float


@dataclass(frozen=True)
class SparseSolveResult:

    valid: bool
    error_type: str | None
    coefficients: FloatArray | None
    y_hat: FloatArray | None


def _validate_theta_y(theta: FloatArray, y: FloatArray) -> FloatArray:
    if theta.ndim != 2:
        raise ValueError(f"theta must be 2-D, got {theta.ndim}-D")
    if theta.dtype != np.float64:
        raise ValueError(f"theta must be float64, got {theta.dtype}")
    if y.dtype != np.float64:
        raise ValueError(f"y must be float64, got {y.dtype}")
    if y.ndim == 1:
        y2d = y.reshape(-1, 1)
    elif y.ndim == 2 and y.shape[1] == 1:
        y2d = y
    else:
        raise ValueError(f"y must be (n,) or (n, 1), got {y.shape}")
    if y2d.shape[0] != theta.shape[0]:
        raise ValueError(
            f"theta has {theta.shape[0]} rows but y has {y2d.shape[0]}"
        )
    return y2d


def _normalized_ridge_estimate(
    theta: FloatArray, y2d: FloatArray, lam: float, normalize: int
) -> FloatArray:
    n, d = theta.shape
    if normalize != 0:
        x = np.zeros((n, d))
        mreg = np.zeros((d, 1))
        for i in range(d):
            mreg[i] = 1.0 / np.linalg.norm(theta[:, i], normalize)
            x[:, i] = mreg[i] * theta[:, i]
    else:
        x = theta
    if lam != 0:
        w = np.linalg.lstsq(
            x.T.dot(x) + lam * np.eye(d), x.T.dot(y2d), rcond=None
        )[0]
    else:
        w = np.linalg.lstsq(x, y2d, rcond=None)[0]
    return np.asarray(w, dtype=np.float64)


def _stridge(
    x0: FloatArray,
    y: FloatArray,
    lam: float,
    maxit: int,
    tol: float,
    normalize: int,
) -> FloatArray:
    n, d = x0.shape
    x = np.zeros((n, d))
    if normalize != 0:
        mreg = np.zeros((d, 1))
        for i in range(d):
            mreg[i] = 1.0 / np.linalg.norm(x0[:, i], normalize)
            x[:, i] = mreg[i] * x0[:, i]
    else:
        x = x0


    if lam != 0:
        w = np.linalg.lstsq(x.T.dot(x) + lam * np.eye(d), x.T.dot(y))[0]
    else:
        w = np.linalg.lstsq(x, y)[0]

    num_relevant = d
    biginds: Any = np.where(abs(w) > tol)[0]
    for j in range(maxit):
        smallinds = np.where(abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]

        if num_relevant == len(new_biginds):
            break
        num_relevant = len(new_biginds)


        if len(new_biginds) == 0:
            if j == 0:
                return np.asarray(w, dtype=np.float64)
            break
        biginds = new_biginds
        w[smallinds] = 0
        if lam != 0:
            w[biginds] = np.linalg.lstsq(
                x[:, biginds].T.dot(x[:, biginds]) + lam * np.eye(len(biginds)),
                x[:, biginds].T.dot(y),
            )[0]
        else:
            w[biginds] = np.linalg.lstsq(x[:, biginds], y)[0]


    if len(biginds) > 0:
        w[biginds] = np.linalg.lstsq(x[:, biginds], y)[0]

    if normalize != 0:
        return np.asarray(np.multiply(mreg, w), dtype=np.float64)
    return np.asarray(w, dtype=np.float64)


def train_stridge(
    theta: FloatArray,
    y: FloatArray,
    *,
    lam: float = DEFAULT_LAM,
    d_tol: float = DEFAULT_D_TOL,
    maxit: int = DEFAULT_MAXIT,
    str_iters: int = DEFAULT_STR_ITERS,
    l0_penalty: float = DEFAULT_L0_PENALTY,
    normalize: int = DEFAULT_NORMALIZE,
) -> TrainStridgeResult:
    y2d = _validate_theta_y(theta, y)
    d_tol = float(d_tol)
    tol = d_tol


    w_best: FloatArray = np.asarray(
        np.linalg.lstsq(theta, y2d, rcond=None)[0], dtype=np.float64
    )
    err_best = float(
        np.mean((y2d - theta.dot(w_best)) ** 2)
        + l0_penalty * np.count_nonzero(w_best)
    )


    w0_norm = _normalized_ridge_estimate(theta, y2d, lam, normalize)

    for iteration in range(maxit):
        if bool(np.all(np.abs(w0_norm) < tol)):


            w = w0_norm
        else:
            w = _stridge(theta, y2d, lam, str_iters, tol, normalize)

        err = float(
            np.mean((y2d - theta.dot(w)) ** 2) + l0_penalty * np.count_nonzero(w)
        )


        if err <= err_best:
            err_best = err
            w_best = w
            tol = tol + d_tol
        else:
            tol = max(0.0, tol - 2 * d_tol)
            d_tol = 2 * d_tol / (maxit - iteration)
            tol = tol + d_tol

    test_err = float(np.mean((y2d - theta.dot(w_best)) ** 2))
    return TrainStridgeResult(
        coefficients=np.asarray(w_best, dtype=np.float64).reshape(-1),
        err=test_err,
    )


def valid_coef(coefficients: FloatArray) -> bool:
    flat = np.asarray(coefficients).reshape(-1)
    for value in flat:
        if value == 0:
            continue
        if np.abs(value) < COEF_MIN or np.abs(value) > COEF_MAX:
            return False
    return True


def sparse_solve(
    theta: FloatArray,
    y: FloatArray,
    *,
    l0_penalty: float = DEFAULT_L0_PENALTY,
) -> SparseSolveResult:
    y2d = _validate_theta_y(theta, y)
    try:


        np.linalg.lstsq(theta, y2d)
    except np.linalg.LinAlgError as exc:
        logger.debug("lstsq probe failed: %s", exc)
        return SparseSolveResult(
            valid=False, error_type=ERROR_LSTSQ, coefficients=None, y_hat=None
        )

    result = train_stridge(theta, y2d, l0_penalty=l0_penalty)
    w = result.coefficients




    y_hat = theta.dot(w).reshape(-1, 1)
    if not valid_coef(w):
        return SparseSolveResult(
            valid=False,
            error_type=ERROR_ABNORMAL_COEF,
            coefficients=w,
            y_hat=y_hat,
        )
    return SparseSolveResult(
        valid=True, error_type=None, coefficients=w, y_hat=y_hat
    )
