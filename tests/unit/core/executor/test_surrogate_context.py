
from __future__ import annotations

import torch
import torch.nn as nn

from kd.core.executor.surrogate_context import SurrogateContext
from kd.data.derivatives.autograd import AutogradProvider
from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType


class _ScaledExactModel(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float64))

    def forward(self, *, x: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"u": self.scale * (x + 2.0 * t)}


def _dataset_and_provider(
    model: _ScaledExactModel,
) -> tuple[PDEDataset, AutogradProvider]:
    x = torch.tensor([0.0, 1.0], dtype=torch.float64)
    t = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    raw_u = torch.zeros_like(xg)
    dataset = PDEDataset(
        name="surrogate-test",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo("x", x), "t": AxisInfo("t", t)},
        axis_order=["x", "t"],
        fields={"u": FieldData("u", raw_u)},
        lhs_field="u",
        lhs_axis="t",
    )
    coords = {
        "x": xg.flatten().detach().requires_grad_(True),
        "t": tg.flatten().detach().requires_grad_(True),
    }
    provider = AutogradProvider(model=model, coords=coords, dataset=dataset)
    return dataset, provider


class TestSurrogateContext:
    def test_get_variable_returns_detached_nn_forward_not_raw_dataset(self) -> None:
        model = _ScaledExactModel(scale=3.0)
        dataset, provider = _dataset_and_provider(model)
        context = SurrogateContext(dataset, provider, surrogate_field="u")

        value = context.get_variable("u")
        expected = model(x=provider.coords["x"], t=provider.coords["t"])["u"].detach()

        torch.testing.assert_close(value, expected)
        assert value.requires_grad is False
        assert not torch.equal(value, dataset.fields["u"].values.flatten())

    def test_cache_reuses_value_until_clear_cache(self) -> None:
        model = _ScaledExactModel(scale=1.0)
        dataset, provider = _dataset_and_provider(model)
        context = SurrogateContext(dataset, provider, surrogate_field="u")

        first = context.get_variable("u")
        model.scale.fill_(2.0)
        second = context.get_variable("u")
        context.clear_cache()
        third = context.get_variable("u")

        torch.testing.assert_close(first, second)
        assert not torch.allclose(first, third)

    def test_get_derivative_delegates_to_provider(self) -> None:
        model = _ScaledExactModel(scale=4.0)
        dataset, provider = _dataset_and_provider(model)
        context = SurrogateContext(dataset, provider, surrogate_field="u")

        actual = context.get_derivative("u", "t", 1)
        expected = provider.get_derivative("u", "t", 1)

        torch.testing.assert_close(actual, expected)

    def test_default_device_matches_provider_not_cpu(self) -> None:
        model = _ScaledExactModel(scale=1.0)
        dataset, provider = _dataset_and_provider(model)



        sentinel = torch.device("meta")
        provider.device = sentinel

        context = SurrogateContext(dataset, provider, surrogate_field="u")

        assert context.device == sentinel, (
            f"SurrogateContext defaulted to {context.device!r} but "
            f"provider.device is {sentinel!r}. The context must inherit "
            "the provider's device by default to avoid silent device "
            "mismatches between theta columns and LHS targets."
        )


class TestBackwardCompatAlias:

    def test_dlga_alias_resolves_to_surrogate_context(self) -> None:
        from kd.search.dlga.surrogate import DLGASurrogateContext

        assert DLGASurrogateContext is SurrogateContext, (
            "DLGASurrogateContext must be a literal alias for SurrogateContext, "
            "not a separate class — promoted-and-aliased pattern (POT-5 Step 3)."
        )

    def test_alias_constructs_same_instance_type(self) -> None:
        from kd.search.dlga.surrogate import DLGASurrogateContext

        model = _ScaledExactModel(scale=1.0)
        dataset, provider = _dataset_and_provider(model)
        context = DLGASurrogateContext(dataset, provider, surrogate_field="u")
        assert isinstance(context, SurrogateContext)
