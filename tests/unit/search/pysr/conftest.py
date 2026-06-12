
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
import sympy
import torch

from kd.core.evaluator import EvaluationResult
from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd.search.protocol import PlatformComponents
from kd.search.pysr.backend import HOFEntry
from kd.search.pysr.config import PySRConfig




SympyRecipe = Callable[[list[str]], sympy.Expr]
HOFRecipe = Callable[[list[str]], list[HOFEntry]]


def _default_best_recipe(names: list[str]) -> sympy.Expr:
    if len(names) >= 2:
        return sympy.Symbol(names[0]) + sympy.Symbol(names[1])
    return sympy.Symbol(names[0])


def _default_hof_recipe(names: list[str]) -> list[HOFEntry]:
    n = len(names)
    e0 = sympy.Symbol(names[0])
    e1 = e0 + (sympy.Symbol(names[1]) if n >= 2 else sympy.Float(1.0))
    if n >= 3:
        e2 = sympy.Symbol(names[0]) * sympy.Symbol(names[1]) + sympy.Symbol(names[2])
    else:
        e2 = e1
    return [
        HOFEntry(complexity=1, loss=0.5, sympy_expr=e0),
        HOFEntry(complexity=3, loss=0.05, sympy_expr=e1),
        HOFEntry(complexity=5, loss=0.001, sympy_expr=e2),
    ]


def _sqrt_best_recipe(names: list[str]) -> sympy.Expr:
    return sympy.sqrt(sympy.Symbol(names[0]))


def _const_tail_best_recipe(names: list[str]) -> sympy.Expr:
    structural = sympy.Symbol(names[0])
    if len(names) >= 2:
        structural = structural + sympy.Symbol(names[1])
    return structural + sympy.Float(0.5)


def _pure_constant_best_recipe(names: list[str]) -> sympy.Expr:
    return sympy.Float(3.0)


def _const_in_hof_recipe(names: list[str]) -> list[HOFEntry]:
    n = len(names)
    const_entry = HOFEntry(complexity=1, loss=0.9, sympy_expr=sympy.Float(3.0))
    e_single = HOFEntry(complexity=2, loss=0.5, sympy_expr=sympy.Symbol(names[0]))
    second = sympy.Symbol(names[1]) if n >= 2 else sympy.Symbol(names[0])
    e_pair = HOFEntry(
        complexity=3, loss=0.05, sympy_expr=sympy.Symbol(names[0]) + second
    )
    return [const_entry, e_single, e_pair]


def _duplicate_complexity_recipe_factory() -> tuple[SympyRecipe, HOFRecipe]:

    def best_recipe(names: list[str]) -> sympy.Expr:
        if len(names) >= 2:
            return sympy.Symbol(names[0]) + sympy.Symbol(names[1])
        return sympy.Symbol(names[0])

    def hof_recipe(names: list[str]) -> list[HOFEntry]:
        n = len(names)
        first = sympy.Symbol(names[0])
        second = first + (sympy.Symbol(names[1]) if n >= 2 else sympy.Float(1.0))
        return [
            HOFEntry(complexity=4, loss=0.40, sympy_expr=first),
            HOFEntry(complexity=4, loss=0.10, sympy_expr=second),
        ]

    return best_recipe, hof_recipe


def _best_not_in_hof_recipe_factory() -> tuple[SympyRecipe, HOFRecipe]:

    def best_recipe(names: list[str]) -> sympy.Expr:
        if len(names) >= 2:
            return sympy.Symbol(names[0]) * sympy.Symbol(names[1])
        return sympy.Symbol(names[0]) ** 2

    def hof_recipe(names: list[str]) -> list[HOFEntry]:
        n = len(names)
        e0 = HOFEntry(complexity=1, loss=0.5, sympy_expr=sympy.Symbol(names[0]))
        second = sympy.Symbol(names[1]) if n >= 2 else sympy.Symbol(names[0])
        e1 = HOFEntry(
            complexity=3, loss=0.05, sympy_expr=sympy.Symbol(names[0]) + second
        )
        return [e0, e1]

    return best_recipe, hof_recipe


class FakePySRBackend:

    def __init__(
        self,
        *,
        best_recipe: SympyRecipe | None = None,
        hof_recipe: HOFRecipe | None = None,
    ) -> None:
        self._best_recipe = best_recipe or _default_best_recipe
        self._hof_recipe = hof_recipe or _default_hof_recipe
        self.fit_calls = 0
        self.captured_names: list[str] | None = None
        self.captured_X: np.ndarray | None = None
        self.captured_y: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: list[str],
    ) -> None:
        self.fit_calls += 1
        self.captured_X = X
        self.captured_y = y
        self.captured_names = list(variable_names)

    def best_sympy(self) -> sympy.Expr:
        if self.captured_names is None:
            raise RuntimeError("best_sympy() before fit()")
        return self._best_recipe(self.captured_names)

    def hall_of_fame(self) -> list[HOFEntry]:
        if self.captured_names is None:
            raise RuntimeError("hall_of_fame() before fit()")
        return self._hof_recipe(self.captured_names)







_PENALTY_NMSE = 1e10


def make_invalid_result(*, message: str = "kd re-score invalid") -> EvaluationResult:
    return EvaluationResult(
        mse=_PENALTY_NMSE,
        nmse=_PENALTY_NMSE,
        r2=-float("inf"),
        aic=float("inf"),
        complexity=0,
        coefficients=None,
        is_valid=False,
        error_message=message,
    )


def make_backend_factory(
    backend: FakePySRBackend,
) -> Callable[[PySRConfig], FakePySRBackend]:

    def _factory(_config: PySRConfig) -> FakePySRBackend:
        return backend

    return _factory









_N_X = 48
_N_T = 24


@pytest.fixture
def pysr_dataset() -> PDEDataset:
    x = torch.linspace(0.0, 2.0 * math.pi, _N_X, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, _N_T, dtype=torch.float64)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(xg) * torch.exp(-tg)
    return PDEDataset(
        name="pysr-test",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


@pytest.fixture
def real_pysr_components(pysr_dataset: PDEDataset) -> PlatformComponents:
    from kd.search.recorder import VizRecorder

    reqs = DerivativeReqs(
        provider_kind="finite_diff",
        max_atomic_order=2,
        lhs_order=1,
        needs_surrogate=False,
    )
    components = PlatformBuilder(pysr_dataset, reqs).build()
    components.recorder = VizRecorder()
    return components
