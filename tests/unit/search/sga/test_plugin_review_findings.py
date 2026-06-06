
from __future__ import annotations

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
from kd.search.sga.config import SGAConfig







_GRID_SIZE = 10
_TIME_SIZE = 5




_FIXTURE_RNG = torch.Generator()
_FIXTURE_RNG.manual_seed(0)


def _make_dataset() -> PDEDataset:
    x = torch.linspace(0.0, 1.0, _GRID_SIZE)
    t = torch.linspace(0.0, 1.0, _TIME_SIZE)
    u = torch.randn(_GRID_SIZE, _TIME_SIZE, generator=_FIXTURE_RNG)
    return PDEDataset(
        name="byod_test",
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


def _make_context(dataset: PDEDataset) -> MagicMock:
    context = MagicMock()
    context.dataset = dataset

    def get_variable(name: str) -> torch.Tensor:
        if dataset.fields is not None and name in dataset.fields:
            return dataset.fields[name].values
        if dataset.axes is not None and name in dataset.axes:
            return dataset.axes[name].values
        raise KeyError(name)

    def get_derivative(field: str, axis: str, order: int) -> torch.Tensor:



        local_rng = torch.Generator()
        local_rng.manual_seed(hash((field, axis, order)) % (2**31))
        return torch.randn(_GRID_SIZE, _TIME_SIZE, generator=local_rng)

    context.get_variable = get_variable
    context.get_derivative = get_derivative
    return context


@pytest.fixture
def prepared_plugin() -> SGAPlugin:
    from kd.search.sga.plugin import SGAPlugin

    dataset = _make_dataset()
    components = PlatformComponents(
        dataset=dataset,
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=_make_context(dataset),
        registry=MagicMock(),
    )
    plugin = SGAPlugin(SGAConfig(num=4, depth=3, width=3, seed=7, maxit=2))
    plugin.prepare(components)
    return plugin







@pytest.mark.unit
class TestL1BuildResultTargetNonAliased:

    def test_returned_tensor_does_not_share_storage_with_internal_y(
        self,
        prepared_plugin: SGAPlugin,
    ) -> None:
        target = prepared_plugin.build_result_target()
        assert prepared_plugin._y is not None


        assert target.data_ptr() != prepared_plugin._y.data_ptr(), (
            "build_result_target() returned a tensor sharing storage with "
            "the plugin's private _y. In-place mutation by the caller "
            "would silently corrupt SGA state."
        )

    def test_in_place_mutation_does_not_affect_internal_y(
        self,
        prepared_plugin: SGAPlugin,
    ) -> None:
        assert prepared_plugin._y is not None
        original = prepared_plugin._y.clone()

        target = prepared_plugin.build_result_target()
        target.zero_()


        torch.testing.assert_close(prepared_plugin._y, original, rtol=0.0, atol=0.0)

    def test_in_place_mutation_does_not_affect_subsequent_call(
        self,
        prepared_plugin: SGAPlugin,
    ) -> None:
        first = prepared_plugin.build_result_target()

        first.fill_(float("nan"))

        second = prepared_plugin.build_result_target()
        assert torch.isfinite(second).all(), (
            "Second build_result_target() returned a tensor poisoned by "
            "the in-place mutation of the first — they must be independent."
        )

    def test_returned_tensor_is_detached(
        self,
        prepared_plugin: SGAPlugin,
    ) -> None:
        target = prepared_plugin.build_result_target()
        assert target.requires_grad is False
