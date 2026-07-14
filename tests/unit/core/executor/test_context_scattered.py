
from __future__ import annotations

import pytest
import torch

from kd.core.executor import ExecutionContext
from kd.data.derivatives.base import DerivativeProvider
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)


class _StubProvider(DerivativeProvider):

    def get_derivative(self, field: str, axis: str, order: int) -> torch.Tensor:
        raise NotImplementedError("stub provider is not consulted by get_variable")

    def diff(self, expression: torch.Tensor, axis: str, order: int) -> torch.Tensor:
        raise NotImplementedError("stub provider is not consulted by get_variable")

    def available_derivatives(self) -> list[tuple[str, str, int]]:
        return []


def _grid_context() -> ExecutionContext:
    t = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    x = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    tt, xx = torch.meshgrid(t, x, indexing="ij")
    u = tt + xx
    dataset = PDEDataset(
        name="grid",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"t": AxisInfo("t", t), "x": AxisInfo("x", x)},
        axis_order=["t", "x"],
        fields={"u": FieldData("u", u)},
        lhs_field="u",
        lhs_axis="t",
        lhs_order=1,
    )
    return ExecutionContext(dataset=dataset, derivative_provider=_StubProvider())







class TestScatteredGetVariable:

    @pytest.mark.unit
    def test_coord_and_field_returned_as_is(self) -> None:
        n = 5
        t = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.tensor([0.5, 1.5, 0.2, 3.1, 2.7], dtype=torch.float64)
        u = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        dataset = PDEDataset.from_scatter(
            coords={"t": t, "x": x},
            fields={"u": u},
        )
        ctx = ExecutionContext(dataset=dataset, derivative_provider=_StubProvider())

        got_u = ctx.get_variable("u")
        got_t = ctx.get_variable("t")
        assert got_u.shape == (n,)
        assert got_t.shape == (n,)
        assert torch.equal(got_u, u)
        assert torch.equal(got_t, t)
        assert ctx.get_variable("x").shape == (n,)
        assert torch.equal(ctx.get_variable("x"), x)







class TestGridGetVariableUnchanged:

    @pytest.mark.unit
    def test_coord_broadcast_byte_identical(self) -> None:
        ctx = _grid_context()
        x = ctx.dataset.get_coords("x")
        expected = x.view(1, 6).expand(5, 6)
        got = ctx.get_variable("x")
        assert got.shape == (5, 6)
        assert torch.equal(got, expected)

    @pytest.mark.unit
    def test_lhs_axis_broadcast_byte_identical(self) -> None:
        ctx = _grid_context()
        t = ctx.dataset.get_coords("t")
        expected = t.view(5, 1).expand(5, 6)
        got = ctx.get_variable("t")
        assert got.shape == (5, 6)
        assert torch.equal(got, expected)

    @pytest.mark.unit
    def test_field_returned_as_is(self) -> None:
        ctx = _grid_context()
        got = ctx.get_variable("u")
        assert got.shape == (5, 6)
        assert torch.equal(got, ctx.dataset.get_field("u"))
