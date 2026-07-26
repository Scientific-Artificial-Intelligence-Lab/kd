
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
import torch

from kd.core.evaluator import EvaluationResult
from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs
from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd.search.protocol import PlatformComponents
from kd.search.pysindy.config import PySINDyConfig
from kd.search.recorder import VizRecorder

CoefficientRecipe = Callable[[int], np.ndarray]


def default_coefficient_recipe(n_features: int) -> np.ndarray:
    values = np.zeros(n_features, dtype=np.float64)
    values[min(2, n_features - 1)] = 1.0
    return values


def all_zero_coefficient_recipe(n_features: int) -> np.ndarray:
    return np.zeros(n_features, dtype=np.float64)


def wrong_shape_coefficient_recipe(n_features: int) -> np.ndarray:
    return np.zeros(n_features + 1, dtype=np.float64)


def nonfinite_coefficient_recipe(n_features: int) -> np.ndarray:
    values = np.zeros(n_features, dtype=np.float64)
    values[0] = np.nan
    return values


class FakeSINDyBackend:

    def __init__(self, recipe: CoefficientRecipe | None = None) -> None:
        self._recipe = recipe or default_coefficient_recipe
        self.fit_calls = 0
        self.captured_X: np.ndarray | None = None
        self.captured_y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.fit_calls += 1
        self.captured_X = np.array(X, copy=True)
        self.captured_y = np.array(y, copy=True)

    def coefficients(self) -> np.ndarray:
        if self.captured_X is None:
            raise RuntimeError("coefficients before fit")
        return np.array(self._recipe(self.captured_X.shape[1]), copy=True)


def make_backend_factory(
    backend: FakeSINDyBackend,
) -> Callable[[PySINDyConfig], FakeSINDyBackend]:
    def _factory(_config: PySINDyConfig) -> FakeSINDyBackend:
        return backend

    return _factory


def make_invalid_result() -> EvaluationResult:
    return EvaluationResult(
        mse=1e10,
        nmse=1e10,
        r2=-float("inf"),
        score=float("inf"),
        is_valid=False,
        error_message="synthetic invalid refit",
    )


_N_X = 64
_N_T = 32


@pytest.fixture
def pysindy_dataset() -> PDEDataset:
    x = torch.linspace(0.0, 2.0 * math.pi, _N_X, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, _N_T, dtype=torch.float64)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    u = torch.exp(-tg) * torch.sin(xg) + torch.exp(-4.0 * tg) * torch.sin(2.0 * xg)
    return PDEDataset(
        name="pysindy-two-mode-heat",
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
def real_pysindy_components(pysindy_dataset: PDEDataset) -> PlatformComponents:
    reqs = DerivativeReqs(
        provider_kind="finite_diff",
        max_atomic_order=2,
        lhs_order=1,
        needs_surrogate=False,
    )
    components = PlatformBuilder(pysindy_dataset, reqs).build()
    components.recorder = VizRecorder()
    return components
