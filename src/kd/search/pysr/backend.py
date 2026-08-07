
from __future__ import annotations

import os
from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np

from kd.search.pysr.config import PySRConfig




_JULIA_SIGNALS_ENV_VAR = "PYTHON_JULIACALL_HANDLE_SIGNALS"
_JULIA_SIGNALS_ENV_VALUE = "yes"


_REQUIRED_HOF_COLUMNS = ("complexity", "loss", "sympy_format")


class HOFEntry(NamedTuple):

    complexity: int
    loss: float
    sympy_expr: Any


@runtime_checkable
class PySRBackend(Protocol):

    def fit(self, X: np.ndarray, y: np.ndarray, variable_names: list[str]) -> None:
        ...

    def best_sympy(self) -> Any:
        ...

    def hall_of_fame(self) -> list[HOFEntry]:
        ...


class _PySRRegressorBackend:

    def __init__(self, config: PySRConfig) -> None:
        self._config = config
        self._model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray, variable_names: list[str]) -> None:
        os.environ.setdefault(_JULIA_SIGNALS_ENV_VAR, _JULIA_SIGNALS_ENV_VALUE)
        try:
            from pysr import PySRRegressor
        except ImportError as e:
            raise RuntimeError(
                "Model(algorithm='pysr') requires the optional dependency "
                "`pysr` (which bundles a Julia runtime); install it with "
                f"`uv sync --extra pysr`. Original error: {e}"
            ) from e
        cfg = self._config
        kwargs: dict[str, Any] = {
            "niterations": cfg.niterations,
            "population_size": cfg.population_size,
            "populations": cfg.populations,
            "maxsize": cfg.maxsize,
            "binary_operators": list(cfg.binary_operators),
            "unary_operators": list(cfg.unary_operators),
            "random_state": cfg.seed,





            "temp_equation_file": True,
        }
        kwargs.update(cfg.extra_pysr_kwargs or {})
        self._model = PySRRegressor(**kwargs)
        self._model.fit(X, y, variable_names=variable_names)

    def best_sympy(self) -> Any:
        if self._model is None:
            raise RuntimeError("best_sympy() called before fit(); call fit(...) first")
        return self._model.sympy()

    def hall_of_fame(self) -> list[HOFEntry]:
        if self._model is None:
            raise RuntimeError(
                "hall_of_fame() called before fit(); call fit(...) first"
            )
        equations = self._model.equations_
        for column in _REQUIRED_HOF_COLUMNS:
            if column not in equations.columns:
                raise RuntimeError(
                    f"PySR equations_ table missing required column {column!r}; "
                    "incompatible PySR version "
                    f"(expected columns {_REQUIRED_HOF_COLUMNS})"
                )
        return [
            HOFEntry(
                complexity=int(row["complexity"]),
                loss=float(row["loss"]),
                sympy_expr=row["sympy_format"],
            )
            for _, row in equations.iterrows()
        ]


def default_backend_factory(config: PySRConfig) -> PySRBackend:
    return _PySRRegressorBackend(config)
