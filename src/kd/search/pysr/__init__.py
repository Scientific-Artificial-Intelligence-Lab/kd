"""PySR integration for kd: config, backend seam, and IR conversion.

This package exposes:

- :class:`PySRConfig` -- the frozen run configuration.
- :class:`PySRBackend` / :class:`HOFEntry` / :func:`default_backend_factory`
  -- the backend-injection seam (the default backend runs PySR in a child
  process, so the calling process never loads Julia; testable without Julia).
- the conversion helpers bridging PySR's SymPy output and kd's funcall IR.

- :class:`PySRPlugin` -- the one-shot-fit search algorithm wrapper exposing
  PySR via kd's iterative ``SearchAlgorithm`` protocol.
"""

from __future__ import annotations

from kd.search.pysr.backend import (
    HOFEntry,
    PySRBackend,
    default_backend_factory,
)
from kd.search.pysr.config import PySRConfig
from kd.search.pysr.convert import (
    build_feature_names,
    pysr_sympy_to_kd_terms,
    pysr_sympy_to_tabular_term,
)
from kd.search.pysr.plugin import PySRPlugin

__all__ = [
    "HOFEntry",
    "PySRBackend",
    "PySRConfig",
    "PySRPlugin",
    "build_feature_names",
    "default_backend_factory",
    "pysr_sympy_to_kd_terms",
    "pysr_sympy_to_tabular_term",
]
