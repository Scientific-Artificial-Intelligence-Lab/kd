
from __future__ import annotations

import builtins
import importlib
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from kd.search.pysindy.backend import (
    PySINDyOptimizerBackend,
    default_backend_factory,
)
from kd.search.pysindy.config import PySINDyConfig


def _install_optimizer_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    coefficients: np.ndarray,
    intercept: float = 0.0,
) -> type[Any]:
    class StubSTLSQ:
        last_kwargs: dict[str, Any] = {}
        last_X: np.ndarray | None = None
        last_y: np.ndarray | None = None

        def __init__(self, **kwargs: Any) -> None:
            type(self).last_kwargs = dict(kwargs)

        def fit(self, X: np.ndarray, y: np.ndarray) -> StubSTLSQ:
            type(self).last_X = np.array(X, copy=True)
            type(self).last_y = np.array(y, copy=True)
            self.coef_ = np.array(coefficients, copy=True)
            self.intercept_ = intercept
            return self

    optimizers = ModuleType("pysindy.optimizers")
    optimizers.STLSQ = StubSTLSQ
    package = ModuleType("pysindy")
    package.__path__ = []
    package.optimizers = optimizers
    monkeypatch.setitem(sys.modules, "pysindy", package)
    monkeypatch.setitem(sys.modules, "pysindy.optimizers", optimizers)
    return StubSTLSQ


def _sample_problem() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    return X, y


def test_importing_plugin_modules_does_not_import_pysindy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in list(sys.modules):
        if name == "pysindy" or name.startswith("pysindy."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    for name in (
        "kd.search.pysindy.config",
        "kd.search.pysindy.backend",
        "kd.search.pysindy.assembly",
        "kd.search.pysindy.plugin",
        "kd.search.pysindy",
    ):
        importlib.reload(importlib.import_module(name))

    assert "pysindy" not in sys.modules


def test_default_factory_returns_runtime_protocol_without_importing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "pysindy", raising=False)
    monkeypatch.delitem(sys.modules, "pysindy.optimizers", raising=False)

    backend = default_backend_factory(PySINDyConfig())

    assert isinstance(backend, PySINDyOptimizerBackend)
    assert "pysindy" not in sys.modules


def test_missing_dependency_error_names_install_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "pysindy" or name.startswith("pysindy."):
            raise ImportError("simulated missing pysindy")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    backend = default_backend_factory(PySINDyConfig())
    X, y = _sample_problem()

    with pytest.raises(RuntimeError, match=r"kd\[pysindy\]"):
        backend.fit(X, y)


def test_adapter_passes_typed_and_extra_stlsq_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_optimizer_stub(
        monkeypatch,
        coefficients=np.array([[2.0, 0.0, 3.0]]),
    )
    config = PySINDyConfig(
        threshold=0.2,
        max_iter=7,
        normalize_columns=True,
        unbias=False,
        extra_optimizer_kwargs={"alpha": 0.03},
    )
    backend = default_backend_factory(config)
    X, y = _sample_problem()

    backend.fit(X, y)

    assert stub.last_kwargs == {
        "threshold": 0.2,
        "max_iter": 7,
        "normalize_columns": True,
        "unbias": False,
        "alpha": 0.03,
    }
    np.testing.assert_array_equal(stub.last_X, X)
    np.testing.assert_array_equal(stub.last_y, y)


def test_row_coefficients_are_ravelled_to_float64_defensive_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_optimizer_stub(
        monkeypatch,
        coefficients=np.array([[2.0, 0.0, 3.0]], dtype=np.float32),
    )
    backend = default_backend_factory(PySINDyConfig())
    X, y = _sample_problem()
    backend.fit(X, y)

    first = backend.coefficients()
    first[0] = -99.0
    second = backend.coefficients()

    assert second.dtype == np.float64
    np.testing.assert_array_equal(second, np.array([2.0, 0.0, 3.0]))


def test_coefficients_before_fit_fails_loud() -> None:
    backend = default_backend_factory(PySINDyConfig())
    with pytest.raises(RuntimeError, match="fit"):
        backend.coefficients()


@pytest.mark.parametrize(
    "coefficients",
    [
        pytest.param(np.zeros((3, 1)), id="column-vector"),
        pytest.param(np.zeros((2, 3)), id="multi-target-matrix"),
        pytest.param(np.zeros(2), id="wrong-width-vector"),
    ],
)
def test_incompatible_coefficient_shape_fails_loud(
    monkeypatch: pytest.MonkeyPatch,
    coefficients: np.ndarray,
) -> None:
    _install_optimizer_stub(monkeypatch, coefficients=coefficients)
    backend = default_backend_factory(PySINDyConfig())
    X, y = _sample_problem()
    with pytest.raises(RuntimeError, match="shape"):
        backend.fit(X, y)


def test_nonfinite_coefficients_fail_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_optimizer_stub(
        monkeypatch,
        coefficients=np.array([[1.0, np.nan, 0.0]]),
    )
    backend = default_backend_factory(PySINDyConfig())
    X, y = _sample_problem()
    with pytest.raises(RuntimeError, match="finite"):
        backend.fit(X, y)


def test_nonzero_intercept_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_optimizer_stub(
        monkeypatch,
        coefficients=np.array([[1.0, 0.0, 0.0]]),
        intercept=1e-4,
    )
    backend = default_backend_factory(PySINDyConfig())
    X, y = _sample_problem()
    with pytest.raises(RuntimeError, match="intercept"):
        backend.fit(X, y)
