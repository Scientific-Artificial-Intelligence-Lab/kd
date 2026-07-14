
from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from kd.core.equation import Scalar, make_homogeneous, residual_program
from kd.core.executor import ExecutionContext
from kd.core.executor.surrogate_context import SurrogateContext
from kd.core.expr.executor import PythonExecutor
from kd.core.expr.registry import FunctionRegistry
from kd.data.derivatives import FiniteDiffProvider
from kd.data.derivatives.autograd import AutogradProvider
from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType


@pytest.fixture
def grid_context() -> ExecutionContext:
    n_x, n_t = 6, 5
    x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
    t = torch.linspace(0, 1, n_t, dtype=torch.float64)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(xx) * torch.exp(-tt)
    dataset = PDEDataset(
        name="unity-test",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo("x", x), "t": AxisInfo("t", t)},
        axis_order=["x", "t"],
        fields={"u": FieldData("u", u)},
        lhs_field="u",
        lhs_axis="t",
    )
    return ExecutionContext(
        dataset=dataset,
        derivative_provider=FiniteDiffProvider(dataset, max_order=3),
    )


class _ExactModel(nn.Module):
    def forward(self, *, x: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"u": x + 2.0 * t}


@pytest.fixture
def surrogate_context() -> SurrogateContext:
    x = torch.tensor([0.0, 1.0], dtype=torch.float64)
    t = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    dataset = PDEDataset(
        name="surrogate-unity-test",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo("x", x), "t": AxisInfo("t", t)},
        axis_order=["x", "t"],
        fields={"u": FieldData("u", torch.zeros_like(xg))},
        lhs_field="u",
        lhs_axis="t",
    )
    coords = {
        "x": xg.flatten().detach().requires_grad_(True),
        "t": tg.flatten().detach().requires_grad_(True),
    }
    provider = AutogradProvider(model=_ExactModel(), coords=coords, dataset=dataset)
    return SurrogateContext(dataset, provider, surrogate_field="u")


class TestUnityColumn:
    @pytest.mark.unit
    def test_grid_unity_column_is_field_shaped_ones(
        self, grid_context: ExecutionContext
    ) -> None:
        column = grid_context.unity_column()
        expected = torch.ones(grid_context.dataset.get_shape(), dtype=torch.float32)
        assert column.shape == expected.shape
        torch.testing.assert_close(column, expected)

    @pytest.mark.unit
    def test_surrogate_unity_column_matches_flattened_rows(
        self, surrogate_context: SurrogateContext
    ) -> None:



        column = surrogate_context.unity_column()
        reference = surrogate_context.get_variable("u")
        assert column.shape == reference.shape
        torch.testing.assert_close(column, torch.ones_like(reference).float())

    @pytest.mark.unit
    def test_axes_less_dataset_gives_clear_unity_error(self) -> None:





        dataset = PDEDataset(
            name="tabular",
            task_type=TaskType.REGRESSION,
            fields={"u": FieldData("u", torch.ones(5))},
        )
        context = ExecutionContext(dataset=dataset, derivative_provider=MagicMock())
        with pytest.raises(ValueError, match="unity"):
            context.unity_column()


class TestUnityExecution:
    @pytest.mark.numerical
    def test_execute_one_fast_path(self, grid_context: ExecutionContext) -> None:
        result = PythonExecutor(FunctionRegistry.create_default()).execute(
            "one", grid_context
        )
        expected = torch.ones(grid_context.dataset.get_shape(), dtype=torch.float32)
        torch.testing.assert_close(result.value, expected)

    @pytest.mark.numerical
    def test_execute_one_diff_path(self, grid_context: ExecutionContext) -> None:


        result = PythonExecutor(FunctionRegistry.create_default()).execute(
            "one", grid_context, force_diff_path=True
        )
        expected = torch.ones(grid_context.dataset.get_shape(), dtype=torch.float32)
        torch.testing.assert_close(result.value, expected)

    @pytest.mark.numerical
    def test_residual_program_executes_to_coefficient_fold(
        self, grid_context: ExecutionContext
    ) -> None:



        eq = make_homogeneous(
            [("diff2_x(u)", Scalar(1.0)), ("u", Scalar(0.5)), ("one", Scalar(2.0))]
        )
        executor = PythonExecutor(FunctionRegistry.create_default())
        program = residual_program(eq)

        actual = executor.execute(program, grid_context).value
        d2x = executor.execute("diff2_x(u)", grid_context).value
        u = executor.execute("u", grid_context).value
        ones = executor.execute("one", grid_context).value
        expected = 1.0 * d2x + 0.5 * u + 2.0 * ones

        torch.testing.assert_close(actual, expected)
