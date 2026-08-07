
from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import torch

from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd.search.protocol import PlatformComponents
from tests.unit.search._runner_mocks import RecordingAlgorithm, StatefulAlgorithm






@pytest.fixture
def mock_components() -> PlatformComponents:
    return PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=MagicMock(training_result=None),
        registry=MagicMock(),
    )


@pytest.fixture
def recording_algorithm() -> RecordingAlgorithm:
    return RecordingAlgorithm()


@pytest.fixture
def stateful_algorithm() -> StatefulAlgorithm:
    return StatefulAlgorithm()


@pytest.fixture
def simple_1d_dataset() -> PDEDataset:
    n_points = 100
    x = torch.linspace(0, 2 * math.pi, n_points, dtype=torch.float64)
    u = torch.sin(x)

    return PDEDataset(
        name="test_1d_sin",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo(name="x", values=x)},
        axis_order=["x"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="x",
    )


@pytest.fixture
def simple_2d_dataset() -> PDEDataset:
    n_x = 64
    n_t = 32

    x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
    t = torch.linspace(0, 1, n_t, dtype=torch.float64)


    X, T = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(X) * torch.exp(-T)

    return PDEDataset(
        name="test_2d_sin_exp",
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
def polynomial_1d_dataset() -> PDEDataset:
    n_points = 100
    x = torch.linspace(0, 1, n_points, dtype=torch.float64)
    u = x**3

    return PDEDataset(
        name="test_1d_polynomial",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo(name="x", values=x)},
        axis_order=["x"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="x",
    )


@pytest.fixture
def scattered_dataset() -> PDEDataset:


    x = torch.linspace(0, 1, 10, dtype=torch.float64)
    u = torch.sin(x)

    return PDEDataset(
        name="test_scattered",
        task_type=TaskType.PDE,
        topology=DataTopology.SCATTERED,
        axes={"x": AxisInfo(name="x", values=x)},
        axis_order=["x"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="x",
    )
