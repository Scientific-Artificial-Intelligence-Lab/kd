
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from kd.search.pysindy.config import PySINDyConfig

_INTERCEPT_ATOL = 1e-10


@runtime_checkable
class PySINDyOptimizerBackend(Protocol):

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    def coefficients(self) -> np.ndarray:
        ...


class _PySINDyOptimizerBackend:

    def __init__(self, config: PySINDyConfig) -> None:
        self._config = config
        self._optimizer: Any = None
        self._coefficients: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            from pysindy.optimizers import STLSQ
        except ImportError as exc:
            raise RuntimeError(
                "Model(algorithm='pysindy') requires the optional dependency "
                "`pysindy`; install it with `uv sync --extra pysindy`. "
                f"Original error: {exc}"
            ) from exc

        cfg = self._config
        kwargs: dict[str, Any] = {
            "threshold": cfg.threshold,
            "max_iter": cfg.max_iter,
            "normalize_columns": cfg.normalize_columns,
            "unbias": cfg.unbias,
        }
        kwargs.update(cfg.extra_optimizer_kwargs or {})
        optimizer = STLSQ(**kwargs)
        optimizer.fit(X, y)

        coefficients = self._validated_coefficients(optimizer, X.shape[1])
        self._validate_intercept(optimizer)
        self._optimizer = optimizer
        self._coefficients = coefficients

    def coefficients(self) -> np.ndarray:
        if self._coefficients is None:
            raise RuntimeError(
                "coefficients() called before fit(); call fit(...) first"
            )
        return np.array(self._coefficients, dtype=np.float64, copy=True)

    @staticmethod
    def _validated_coefficients(optimizer: Any, n_features: int) -> np.ndarray:
        if not hasattr(optimizer, "coef_"):
            raise RuntimeError(
                "PySINDy STLSQ missing required coef_; incompatible PySINDy version"
            )
        coefficients = np.asarray(optimizer.coef_, dtype=np.float64)
        if coefficients.shape == (1, n_features):
            coefficients = coefficients.ravel()
        elif coefficients.shape != (n_features,):
            raise RuntimeError(
                "PySINDy STLSQ coef_ has incompatible shape "
                f"{coefficients.shape}; expected ({n_features},) or "
                f"(1, {n_features})"
            )
        if not np.isfinite(coefficients).all():
            raise RuntimeError(
                "PySINDy STLSQ coef_ contains non-finite values; "
                "incompatible optimizer result"
            )
        return np.array(coefficients, dtype=np.float64, copy=True)

    @staticmethod
    def _validate_intercept(optimizer: Any) -> None:
        intercept = np.asarray(getattr(optimizer, "intercept_", 0.0), dtype=np.float64)
        if not np.isfinite(intercept).all() or not np.allclose(
            intercept,
            0.0,
            rtol=0.0,
            atol=_INTERCEPT_ATOL,
        ):
            raise RuntimeError(
                "PySINDy STLSQ produced a nonzero intercept; kd Theta has no "
                "intercept column and this optimizer surface is incompatible"
            )


def default_backend_factory(config: PySINDyConfig) -> PySINDyOptimizerBackend:
    return _PySINDyOptimizerBackend(config)


__all__ = ["PySINDyOptimizerBackend", "default_backend_factory"]
