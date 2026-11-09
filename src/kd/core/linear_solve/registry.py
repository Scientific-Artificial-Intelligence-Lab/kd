
from __future__ import annotations

from typing import Any

from kd._registry import Registry, registering_builtins
from kd.core.linear_solve.base import SparseSolver
from kd.core.linear_solve.least_squares import LeastSquaresSolver
from kd.core.linear_solve.stridge import STRidgeSolver
from kd.core.linear_solve.svd_null_space import SVDNullSpaceSolver

_solvers: Registry[type[SparseSolver]] = Registry("sparse_solver")


def register_sparse_solver(
    name: str,
    solver_cls: type[SparseSolver],
) -> None:
    _solvers.register(name, solver_cls)


def sparse_solvers() -> tuple[str, ...]:
    return _solvers.names()


def sparse_solver(name: str, **kwargs: Any) -> SparseSolver:
    return _solvers.get(name)(**kwargs)


with registering_builtins():
    register_sparse_solver("ols", LeastSquaresSolver)
    register_sparse_solver("stridge", STRidgeSolver)
    register_sparse_solver("svd_null_space", SVDNullSpaceSolver)
