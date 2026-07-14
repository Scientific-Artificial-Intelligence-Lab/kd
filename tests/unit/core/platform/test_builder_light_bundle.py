
from __future__ import annotations

import math

import pytest
import torch

from kd.core.platform.requirements import (
    DerivativeReqs,
    assert_dataset_supported,
)
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.plugin import EqGPTPlugin
from kd.search.protocol import PlatformComponents
from kd.search.sga.plugin import SGAPlugin

pytestmark = pytest.mark.unit







def _grid_dataset() -> PDEDataset:
    x = torch.linspace(0.0, 1.0, 12, dtype=torch.float64)
    t = torch.linspace(0.0, 0.1, 6, dtype=torch.float64)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(2.0 * math.pi * (xg - tg))
    return PDEDataset(
        name="grid-full-bundle",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def _scatter_dataset() -> PDEDataset:
    n = 16
    torch.manual_seed(0)
    t = torch.rand(n, dtype=torch.float64)
    x = torch.rand(n, dtype=torch.float64)
    u = torch.rand(n, dtype=torch.float64)
    return PDEDataset.from_scatter(
        coords={"t": t, "x": x},
        fields={"u": u},
        lhs="u_t",
        name="scatter-light-bundle",
    )


def _wave_reqs() -> DerivativeReqs:
    plugin = EqGPTPlugin(EqGPTConfig.wave_preset())
    reqs = plugin.derivative_requirements

    assert reqs.provider_kind == "none"
    assert DataTopology.SCATTERED in reqs.supported_topologies
    return reqs







class TestLightBundle:

    def test_light_bundle_omits_evaluator_and_context(self) -> None:
        from kd.core.platform.builder import PlatformBuilder

        components = PlatformBuilder(_scatter_dataset(), _wave_reqs()).build()
        assert isinstance(components, PlatformComponents)
        assert components.evaluator is None
        assert components.context is None
        assert components.executor is not None
        assert components.registry is not None
        assert components.dataset is not None

    def test_light_bundle_dataset_is_the_scattered_primary(self) -> None:
        from kd.core.platform.builder import PlatformBuilder

        ds = _scatter_dataset()
        components = PlatformBuilder(ds, _wave_reqs()).build()
        assert components.dataset.topology == DataTopology.SCATTERED
        assert components.dataset.lhs_field == "u"
        assert components.dataset.lhs_axis == "t"







class TestFullBundleRegression:

    def test_default_reqs_build_full_bundle(self) -> None:
        from kd.core.platform.builder import PlatformBuilder

        components = PlatformBuilder(_grid_dataset(), DerivativeReqs()).build()

        assert components.evaluator is not None
        assert components.context is not None
        assert components.executor is not None
        assert components.registry is not None
        assert components.dataset is not None







class TestScatteredGate:

    def test_wave_reqs_accepts_scattered_order1(self) -> None:
        reqs = _wave_reqs()

        assert (
            assert_dataset_supported(
                reqs.lhs_order, DataTopology.SCATTERED, reqs, "eqgpt"
            )
            is None
        )

    def test_default_grid_plugin_rejects_scattered(self) -> None:
        reqs = SGAPlugin().derivative_requirements
        with pytest.raises(NotImplementedError):
            assert_dataset_supported(
                reqs.lhs_order, DataTopology.SCATTERED, reqs, "sga"
            )
