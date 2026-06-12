
from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest
import sympy


from kd.search.pysr.backend import (
    HOFEntry,
    PySRBackend,
    default_backend_factory,
)
from kd.search.pysr.config import PySRConfig

pytestmark = pytest.mark.unit







class _FakeBackend:

    def __init__(self) -> None:
        self.fitted = False
        self._expr: sympy.Expr = sympy.Symbol("c0")
        self._hof: list[HOFEntry] = []

    def fit(self, X: np.ndarray, y: np.ndarray, variable_names: list[str]) -> None:
        self.fitted = True

        first = (
            sympy.Symbol(variable_names[0]) if variable_names else sympy.Symbol("c0")
        )
        self._expr = first
        self._hof = [
            HOFEntry(complexity=1, loss=0.5, sympy_expr=first),
            HOFEntry(complexity=3, loss=0.01, sympy_expr=first + sympy.Float(2.0)),
        ]

    def best_sympy(self) -> Any:
        return self._expr

    def hall_of_fame(self) -> list[HOFEntry]:
        return list(self._hof)


class _IncompleteBackend:

    def fit(self, X: np.ndarray, y: np.ndarray, variable_names: list[str]) -> None:
        return None

    def best_sympy(self) -> Any:
        return sympy.Symbol("c0")







class TestHOFEntry:

    @pytest.mark.smoke
    def test_construct_and_field_access(self) -> None:
        expr = sympy.Symbol("u") + sympy.Symbol("u_x")
        entry = HOFEntry(complexity=5, loss=0.123, sympy_expr=expr)
        assert entry.complexity == 5
        assert entry.loss == 0.123
        assert entry.sympy_expr is expr

    def test_is_namedtuple(self) -> None:
        assert issubclass(HOFEntry, tuple)
        assert HOFEntry._fields == ("complexity", "loss", "sympy_expr")

    def test_unpacking(self) -> None:
        entry = HOFEntry(complexity=2, loss=1.0, sympy_expr=sympy.Symbol("u"))
        complexity, loss, expr = entry
        assert complexity == 2
        assert loss == 1.0
        assert expr == sympy.Symbol("u")

    def test_indexing(self) -> None:
        entry = HOFEntry(complexity=4, loss=0.25, sympy_expr=sympy.Symbol("u_xx"))
        assert entry[0] == 4
        assert entry[1] == 0.25
        assert entry[2] == sympy.Symbol("u_xx")







class TestPySRBackendProtocol:

    def test_fake_satisfies_protocol(self) -> None:
        fake = _FakeBackend()
        assert isinstance(fake, PySRBackend)

    def test_incomplete_backend_does_not_satisfy_protocol(self) -> None:
        incomplete = _IncompleteBackend()
        assert not isinstance(incomplete, PySRBackend)

    def test_plain_object_does_not_satisfy_protocol(self) -> None:
        assert not isinstance(object(), PySRBackend)

    def test_fake_methods_are_callable_through_protocol(self) -> None:
        fake: PySRBackend = _FakeBackend()
        X = np.zeros((4, 2), dtype=float)
        y = np.zeros(4, dtype=float)
        fake.fit(X, y, ["c0", "c1"])
        best = fake.best_sympy()
        assert best == sympy.Symbol("c0")
        hof = fake.hall_of_fame()
        assert len(hof) == 2
        assert all(isinstance(e, HOFEntry) for e in hof)

        assert hof[0].complexity == 1
        assert hof[1].loss == 0.01







class TestDefaultBackendFactory:

    def test_returns_protocol_conforming_instance(self) -> None:
        backend = default_backend_factory(PySRConfig())
        assert backend is not None
        assert isinstance(backend, PySRBackend)

    def test_construction_does_not_import_pysr(self) -> None:
        if "pysr" in sys.modules:
            pytest.skip(
                "pysr already imported by another test; cannot observe lazy seam"
            )
        backend = default_backend_factory(PySRConfig())
        assert backend is not None
        assert "pysr" not in sys.modules, (
            "default_backend_factory imported pysr at construction time; "
            "the import must be deferred to .fit() (lazy / SIGABRT-safe seam)"
        )

    def test_factory_respects_custom_config(self) -> None:
        if "pysr" in sys.modules:
            pytest.skip(
                "pysr already imported by another test; cannot observe lazy seam"
            )
        cfg = PySRConfig(seed=7, niterations=5, terms=("u", "u_x"))
        backend = default_backend_factory(cfg)
        assert isinstance(backend, PySRBackend)
        assert "pysr" not in sys.modules
