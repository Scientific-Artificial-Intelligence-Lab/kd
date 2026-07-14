
from __future__ import annotations

import math

import pytest
import torch

from kd.data.derivatives.base import DerivativeProvider
from kd.data.derivatives.finite_diff import FiniteDiffProvider
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)


class _MinimalProvider(DerivativeProvider):

    def get_derivative(self, field: str, axis: str, order: int) -> torch.Tensor:
        return torch.zeros(4)

    def diff(
        self, expression: torch.Tensor, axis: str, order: int
    ) -> torch.Tensor:
        raise NotImplementedError

    def available_derivatives(self) -> list[tuple[str, str, int]]:
        return []


def _grid_dataset() -> PDEDataset:
    n_x, n_t = 16, 8
    x = torch.linspace(0.0, 2 * math.pi, n_x, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, n_t, dtype=torch.float64)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(xx) * torch.exp(-tt)
    return PDEDataset(
        name="gen_fd_2d",
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


@pytest.mark.smoke
def test_base_default_generation_is_zero() -> None:
    provider = _MinimalProvider()
    assert provider.generation == 0


@pytest.mark.unit
def test_generation_is_int() -> None:
    provider = _MinimalProvider()
    gen = provider.generation
    assert isinstance(gen, int)


@pytest.mark.unit
def test_real_finite_diff_provider_generation_is_zero() -> None:
    provider = FiniteDiffProvider(_grid_dataset(), max_order=2)
    assert provider.generation == 0


@pytest.mark.unit
def test_generation_is_read_only() -> None:
    provider = _MinimalProvider()
    with pytest.raises(AttributeError):
        provider.generation = 5


@pytest.mark.unit
def test_bump_generation_is_monotonic_and_property_stays_read_only() -> None:
    provider = _MinimalProvider()
    assert provider.generation == 0
    provider._bump_generation()
    provider._bump_generation()
    assert provider.generation == 2
    with pytest.raises(AttributeError):
        provider.generation = 0


@pytest.mark.unit
def test_bump_generation_does_not_leak_across_instances() -> None:
    bumped = _MinimalProvider()
    untouched = _MinimalProvider()

    bumped._bump_generation()

    assert bumped.generation == 1
    assert untouched.generation == 0, "generation leaked to a sibling instance"
    assert DerivativeProvider._generation == 0, "class-level default was mutated"
