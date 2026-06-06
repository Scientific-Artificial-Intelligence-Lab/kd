
from __future__ import annotations

import pytest
import torch

from kd.core.executor import ExecutionContext
from kd.core.expr.executor import PythonExecutor
from kd.core.expr.registry import FunctionRegistry
from kd.data import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd.data.derivatives import FiniteDiffProvider


def _make_burgers_like_context() -> ExecutionContext:
    x = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 7, dtype=torch.float64)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(xx) * torch.exp(-tt)
    dataset = PDEDataset(
        name="lap_1d",
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
    return ExecutionContext(
        dataset=dataset,
        derivative_provider=FiniteDiffProvider(dataset, max_order=2),
    )


def _make_2d_context() -> ExecutionContext:
    x = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    y = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 7, dtype=torch.float64)
    xx, yy, tt = torch.meshgrid(x, y, t, indexing="ij")
    u = torch.sin(xx) + torch.cos(yy) + tt
    dataset = PDEDataset(
        name="lap_2d",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x),
            "y": AxisInfo(name="y", values=y),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "y", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )
    return ExecutionContext(
        dataset=dataset,
        derivative_provider=FiniteDiffProvider(dataset, max_order=2),
    )


@pytest.fixture
def executor() -> PythonExecutor:
    return PythonExecutor(FunctionRegistry.create_default())


@pytest.mark.integration
class TestLapExecutorIntegration:

    def test_lap_1d_matches_diff2_x(self, executor: PythonExecutor) -> None:
        context = _make_burgers_like_context()

        lap_result = executor.execute("lap(u)", context)
        diff_result = executor.execute("diff2_x(u)", context)

        assert lap_result.used_diff is True
        torch.testing.assert_close(
            lap_result.value,
            diff_result.value,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_lap_2d_matches_explicit_sum(self, executor: PythonExecutor) -> None:
        context = _make_2d_context()

        lap_result = executor.execute("lap(u)", context)
        explicit_result = executor.execute("add(diff2_x(u), diff2_y(u))", context)

        assert lap_result.used_diff is True
        torch.testing.assert_close(
            lap_result.value,
            explicit_result.value,
            rtol=1e-10,
            atol=1e-10,
        )
