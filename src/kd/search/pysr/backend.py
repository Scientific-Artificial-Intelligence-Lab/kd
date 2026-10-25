
from __future__ import annotations

import logging
import os
import pickle
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Mapping
from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np
import sympy

from kd.search.pysr.config import PySRConfig

logger = logging.getLogger(__name__)


_WORKER_MODULE = "kd.search.pysr._worker"




_JULIA_SIGNALS_ENV_VAR = "PYTHON_JULIACALL_HANDLE_SIGNALS"
_JULIA_SIGNALS_ENV_VALUE = "yes"





_JULIA_THREADS_ENV_VARS = ("PYTHON_JULIACALL_THREADS", "JULIA_NUM_THREADS")

_BLAS_THREADS_ENV_VAR = "OMP_NUM_THREADS"






_PYSR_THREADS_WARNING_FILTER = (
    "ignore:PYTHON_JULIACALL_THREADS environment variable is set"
    ":UserWarning:pysr.julia_import"
)


_REQUIRED_HOF_COLUMNS = ("complexity", "loss", "sympy_format")


class HOFEntry(NamedTuple):

    complexity: int
    loss: float
    sympy_expr: Any


@runtime_checkable
class PySRBackend(Protocol):

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: list[str],
        *,
        search_state: bytes | None,
        timeout_in_seconds: float | None,
    ) -> None:
        ...

    def best_sympy(self) -> Any:
        ...

    def hall_of_fame(self) -> list[HOFEntry]:
        ...

    def search_state(self) -> bytes:
        ...


def _regressor_kwargs(
    cfg: PySRConfig, timeout_in_seconds: float | None
) -> dict[str, Any]:
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
    if timeout_in_seconds is None:
        kwargs.pop("timeout_in_seconds", None)
    else:
        kwargs["timeout_in_seconds"] = timeout_in_seconds
    return kwargs


class _PySRRegressorBackend:

    def __init__(self, config: PySRConfig) -> None:
        self._config = config
        self._model: Any = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: list[str],
        *,
        search_state: bytes | None,
        timeout_in_seconds: float | None,
    ) -> None:
        os.environ.setdefault(_JULIA_SIGNALS_ENV_VAR, _JULIA_SIGNALS_ENV_VALUE)
        try:
            from pysr import PySRRegressor
        except ImportError as e:
            raise RuntimeError(
                "Model(algorithm='pysr') requires the optional dependency "
                "`pysr` (which bundles a Julia runtime); install it with "
                f'`pip install "sail-kd[pysr]"`. Original error: {e}'
            ) from e
        if search_state is None:
            self._model = PySRRegressor(
                **_regressor_kwargs(self._config, timeout_in_seconds)
            )
        else:
            self._model = pickle.loads(search_state)




















            self._model.set_params(
                niterations=self._config.niterations,
                random_state=self._config.seed,
                timeout_in_seconds=timeout_in_seconds,
                warm_start=True,
            )





            self._model.output_directory_ = tempfile.mkdtemp()
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

    def search_state(self) -> bytes:
        if self._model is None:
            raise RuntimeError(
                "search_state() called before fit(); call fit(...) first"
            )
        return pickle.dumps(self._model)


def _worker_environment(environ: Mapping[str, str]) -> dict[str, str]:
    env = dict(environ)
    source = next(
        (
            name
            for name in (*_JULIA_THREADS_ENV_VARS, _BLAS_THREADS_ENV_VAR)
            if name in env
        ),
        None,
    )
    if source is None:
        return env
    _warn_on_julia_thread_conflict(env)
    threads = env[source].split(",", 1)[0]
    for name in _JULIA_THREADS_ENV_VARS:
        env.setdefault(name, threads)
    env["PYTHONWARNINGS"] = ",".join(
        entry
        for entry in (env.get("PYTHONWARNINGS"), _PYSR_THREADS_WARNING_FILTER)
        if entry
    )
    logger.debug("PySR worker: %s Julia threads, from %s", threads, source)
    return env


def _warn_on_julia_thread_conflict(environ: Mapping[str, str]) -> None:
    juliacall = environ.get("PYTHON_JULIACALL_THREADS")
    julia_num = environ.get("JULIA_NUM_THREADS")
    if juliacall is None or julia_num is None:
        return
    if juliacall.split(",", 1)[0] != julia_num.split(",", 1)[0]:
        warnings.warn(
            f"PYTHON_JULIACALL_THREADS={juliacall!r} and "
            f"JULIA_NUM_THREADS={julia_num!r} disagree: juliacall starts the "
            f"PySR worker's Julia with --threads={juliacall}, so "
            "JULIA_NUM_THREADS is ignored",
            UserWarning,
            stacklevel=3,
        )


class _SubprocessBackend:

    def __init__(self, config: PySRConfig) -> None:
        self._config = config
        self._best: Any = None
        self._hof: list[HOFEntry] | None = None
        self._search_state: bytes | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: list[str],
        *,
        search_state: bytes | None,
        timeout_in_seconds: float | None,
    ) -> None:
        request = {
            "X": X,
            "y": y,
            "variable_names": list(variable_names),
            "config": self._config,
            "search_state": search_state,
            "timeout_in_seconds": timeout_in_seconds,
        }
        with tempfile.TemporaryDirectory(prefix="kd-pysr-") as tmp:
            request_path = os.path.join(tmp, "request.pkl")
            result_path = os.path.join(tmp, "result.pkl")
            with open(request_path, "wb") as f:
                pickle.dump(request, f)
            cmd = [sys.executable, "-m", _WORKER_MODULE, request_path, result_path]
            logger.debug("launching PySR worker: %s", cmd)
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                env=_worker_environment(os.environ),
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "PySR worker process exited with return code "
                    f"{proc.returncode} without producing a result"
                )
            with open(result_path, "rb") as f:
                result = pickle.load(f)
        if "error" in result:
            raise RuntimeError(
                f"PySR fit failed in the worker process:\n{result['error']}"
            )
        self._best = sympy.sympify(result["best"])
        self._hof = [
            HOFEntry(complexity=complexity, loss=loss, sympy_expr=sympy.sympify(srepr))
            for complexity, loss, srepr in result["hof"]
        ]
        self._search_state = result["search_state"]

    def best_sympy(self) -> Any:
        if self._best is None:
            raise RuntimeError("best_sympy() called before fit(); call fit(...) first")
        return self._best

    def hall_of_fame(self) -> list[HOFEntry]:
        if self._hof is None:
            raise RuntimeError(
                "hall_of_fame() called before fit(); call fit(...) first"
            )
        return list(self._hof)

    def search_state(self) -> bytes:
        if self._search_state is None:
            raise RuntimeError(
                "search_state() called before fit(); call fit(...) first"
            )
        return self._search_state


def default_backend_factory(config: PySRConfig) -> PySRBackend:
    return _SubprocessBackend(config)
