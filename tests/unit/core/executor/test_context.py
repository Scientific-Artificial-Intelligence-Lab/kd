
from __future__ import annotations

import math
from unittest.mock import Mock

import pytest
import torch

from kd.core.executor import ExecutionContext
from kd.data import PDEDataset
from kd.data.derivatives import FiniteDiffProvider
from kd.data.schema import AxisInfo, DataTopology, FieldData, TaskType






@pytest.fixture
def simple_2d_dataset() -> PDEDataset:
    n_x = 32
    n_t = 16

    x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
    t = torch.linspace(0, 1, n_t, dtype=torch.float64)


    xx, tt = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(xx) * torch.exp(-tt)
    v = torch.cos(xx) * torch.exp(-tt)

    return PDEDataset(
        name="test_2d",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={
            "u": FieldData(name="u", values=u),
            "v": FieldData(name="v", values=v),
        },
        lhs_field="u",
        lhs_axis="t",
    )


@pytest.fixture
def derivative_provider(simple_2d_dataset: PDEDataset) -> FiniteDiffProvider:
    return FiniteDiffProvider(simple_2d_dataset, max_order=3)


@pytest.fixture
def execution_context(
    simple_2d_dataset: PDEDataset,
    derivative_provider: FiniteDiffProvider,
) -> ExecutionContext:
    return ExecutionContext(
        dataset=simple_2d_dataset,
        derivative_provider=derivative_provider,
        constants={"pi": math.pi, "nu": 0.1},
    )







@pytest.mark.smoke
class TestContextSmoke:

    def test_execution_context_can_be_created(
        self, execution_context: ExecutionContext
    ) -> None:
        assert execution_context is not None
        assert execution_context.dataset is not None
        assert execution_context.derivative_provider is not None







@pytest.mark.unit
class TestExecutionContextVariables:

    def test_get_field_variable(self, execution_context: ExecutionContext) -> None:
        u = execution_context.get_variable("u")
        assert u is not None
        assert u.shape == (32, 16)

        torch.testing.assert_close(
            u[0, 0],
            torch.tensor(0.0, dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_get_second_field(self, execution_context: ExecutionContext) -> None:
        v = execution_context.get_variable("v")
        assert v is not None
        assert v.shape == (32, 16)

        torch.testing.assert_close(
            v[0, 0],
            torch.tensor(1.0, dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_get_coordinate_x(self, execution_context: ExecutionContext) -> None:
        x = execution_context.get_variable("x")
        assert x is not None

        assert x.shape == (32, 16)

    def test_get_coordinate_t(self, execution_context: ExecutionContext) -> None:
        t = execution_context.get_variable("t")
        assert t is not None

        assert t.shape == (32, 16)

    def test_get_variable_not_found(self, execution_context: ExecutionContext) -> None:
        with pytest.raises(KeyError):
            execution_context.get_variable("nonexistent")







@pytest.mark.unit
class TestExecutionContextDerivatives:

    def test_get_first_derivative(self, execution_context: ExecutionContext) -> None:
        u_x = execution_context.get_derivative("u", "x", 1)
        assert u_x is not None
        assert u_x.shape == (32, 16)

    def test_get_second_derivative(self, execution_context: ExecutionContext) -> None:
        u_xx = execution_context.get_derivative("u", "x", 2)
        assert u_xx is not None
        assert u_xx.shape == (32, 16)

    def test_get_time_derivative(self, execution_context: ExecutionContext) -> None:
        u_t = execution_context.get_derivative("u", "t", 1)
        assert u_t is not None
        assert u_t.shape == (32, 16)

    def test_get_derivative_field_not_found(
        self, execution_context: ExecutionContext
    ) -> None:
        with pytest.raises(KeyError):
            execution_context.get_derivative("w", "x", 1)

    def test_get_derivative_axis_not_found(
        self, execution_context: ExecutionContext
    ) -> None:
        with pytest.raises(KeyError):
            execution_context.get_derivative("u", "y", 1)

    def test_get_derivative_order_zero_raises(
        self, execution_context: ExecutionContext
    ) -> None:
        with pytest.raises(ValueError, match="order must be >= 1"):
            execution_context.get_derivative("u", "x", 0)

    def test_get_derivative_negative_order_raises(
        self, execution_context: ExecutionContext
    ) -> None:
        with pytest.raises(ValueError, match="order must be >= 1"):
            execution_context.get_derivative("u", "x", -1)

    def test_get_derivative_order_exceeds_max_raises(
        self, execution_context: ExecutionContext
    ) -> None:
        with pytest.raises(ValueError, match="exceeds max_order"):
            execution_context.get_derivative("u", "x", 4)







@pytest.mark.unit
class TestExecutionContextConstants:

    def test_get_constant(self, execution_context: ExecutionContext) -> None:
        pi = execution_context.get_constant("pi")
        assert abs(pi - math.pi) < 1e-10

    def test_get_constant_nu(self, execution_context: ExecutionContext) -> None:
        nu = execution_context.get_constant("nu")
        assert abs(nu - 0.1) < 1e-10

    def test_get_constant_not_found(
        self, execution_context: ExecutionContext
    ) -> None:
        with pytest.raises(KeyError):
            execution_context.get_constant("nonexistent")







@pytest.mark.unit
class TestExecutionContextSpatialAxes:

    def test_1d_pde_excludes_lhs_axis(
        self, execution_context: ExecutionContext
    ) -> None:
        assert execution_context.spatial_axes == ["x"]

    @pytest.mark.parametrize(
        ("axis_order", "expected"),
        [
            (["x", "y", "t"], ["x", "y"]),
            (["t", "x", "y", "z"], ["x", "y", "z"]),
        ],
    )
    def test_multidimensional_pde_preserves_axis_order(
        self,
        axis_order: list[str],
        expected: list[str],
    ) -> None:
        dataset = PDEDataset(
            name="test_spatial_axes",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes=None,
            axis_order=axis_order,
            fields=None,
            lhs_field="",
            lhs_axis="t",
        )
        context = ExecutionContext(
            dataset=dataset,
            derivative_provider=Mock(),
            constants={},
        )

        assert context.spatial_axes == expected

    def test_empty_lhs_axis_returns_empty_list(self) -> None:
        dataset = PDEDataset(
            name="test_empty_lhs_axis",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes=None,
            axis_order=["x", "t"],
            fields=None,
            lhs_field="",
            lhs_axis="",
        )
        context = ExecutionContext(
            dataset=dataset,
            derivative_provider=Mock(),
            constants={},
        )

        assert context.spatial_axes == []

    def test_missing_axis_order_returns_empty_list(self) -> None:
        dataset = PDEDataset(
            name="test_missing_axis_order",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes=None,
            axis_order=None,
            fields=None,
            lhs_field="",
            lhs_axis="t",
        )
        context = ExecutionContext(
            dataset=dataset,
            derivative_provider=Mock(),
            constants={},
        )

        assert context.spatial_axes == []

    def test_scattered_mode_uses_axis_order_without_axes_or_fields(self) -> None:
        dataset = PDEDataset(
            name="test_scattered_spatial_axes",
            task_type=TaskType.PDE,
            topology=DataTopology.SCATTERED,
            axes=None,
            axis_order=["x", "y", "t"],
            fields=None,
            lhs_field="",
            lhs_axis="t",
        )
        context = ExecutionContext(
            dataset=dataset,
            derivative_provider=Mock(),
            constants={},
        )

        assert context.spatial_axes == ["x", "y"]

    def test_spatial_axes_is_derived_on_each_access(self) -> None:
        dataset = PDEDataset(
            name="test_spatial_axes_not_cached",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes=None,
            axis_order=["x", "y", "t"],
            fields=None,
            lhs_field="",
            lhs_axis="t",
        )
        context = ExecutionContext(
            dataset=dataset,
            derivative_provider=Mock(),
            constants={},
        )

        assert context.spatial_axes == ["x", "y"]
        dataset.lhs_axis = "y"
        assert context.spatial_axes == ["x", "t"]







@pytest.mark.unit
class TestCoordinateBroadcasting:

    def test_x_coordinate_broadcasts_correctly(
        self, execution_context: ExecutionContext
    ) -> None:
        x = execution_context.get_variable("x")

        assert x.shape == (32, 16)

        torch.testing.assert_close(
            x[0,:],
            torch.full((16,), 0.0, dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )

        assert x[1, 0] > x[0, 0]

    def test_t_coordinate_broadcasts_correctly(
        self, execution_context: ExecutionContext
    ) -> None:
        t = execution_context.get_variable("t")

        assert t.shape == (32, 16)

        torch.testing.assert_close(
            t[:, 0],
            torch.full((32,), 0.0, dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )

        assert t[0, 1] > t[0, 0]







@pytest.mark.unit
class TestContextEdgeCases:

    def test_context_with_empty_constants(
        self,
        simple_2d_dataset: PDEDataset,
        derivative_provider: FiniteDiffProvider,
    ) -> None:
        context = ExecutionContext(
            dataset=simple_2d_dataset,
            derivative_provider=derivative_provider,
            constants={},
        )

        with pytest.raises(KeyError):
            context.get_constant("pi")

    def test_context_without_axis_order_raises(self) -> None:
        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        u = torch.randn(10, dtype=torch.float64)

        dataset = PDEDataset(
            name="test_missing_axis_order",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={"x": AxisInfo(name="x", values=x)},
            axis_order=None,
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="",
        )

        context = ExecutionContext(
            dataset=dataset,
            derivative_provider=Mock(),
            constants={},
        )


        with pytest.raises(ValueError, match="axis_order"):
            context.get_variable("x")
