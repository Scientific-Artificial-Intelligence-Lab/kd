"""PySR integration for kd: config, backend seam, and IR conversion.

This package exposes:

- :class:`PySRConfig` -- the frozen run configuration.
- :class:`PySRBackend` / :class:`HOFEntry` / :func:`default_backend_factory`
  -- the backend-injection seam (lazy ``pysr`` import; testable without Julia).
- the conversion helpers bridging PySR's SymPy output and kd's funcall IR.

- :class:`PySRPlugin` -- the one-shot-fit search algorithm wrapper exposing
  PySR via kd's iterative ``SearchAlgorithm`` protocol.
- :class:`PySRSymbolicRegressor` -- scikit-learn-style scalar SR bypass
  regressor (no PDE structure, no re-fit), the pure-SR counterpart of
  :class:`kd.search.sindy.SINDyRegressor`.
"""

from __future__ import annotations

from kd.search.pysr.backend import (
    HOFEntry,
    PySRBackend,
    default_backend_factory,
)
from kd.search.pysr.config import PySRConfig
from kd.search.pysr.convert import build_feature_names, pysr_sympy_to_kd_terms
from kd.search.pysr.plugin import PySRPlugin
from kd.search.pysr.sr import PySRSymbolicRegressor

__all__ = [
    "HOFEntry",
    "PySRBackend",
    "PySRConfig",
    "PySRPlugin",
    "PySRSymbolicRegressor",
    "build_feature_names",
    "default_backend_factory",
    "pysr_sympy_to_kd_terms",
]
