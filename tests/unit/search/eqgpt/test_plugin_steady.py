
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from kd.data.schema import DataTopology, PDEDataset
from kd.search.eqgpt.backend import FakeGPTBackend
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.plugin import EqGPTPlugin
from kd.search.protocol import PlatformComponents

pytestmark = pytest.mark.unit

def _steady_config(**overrides: object) -> EqGPTConfig:




    base: dict[str, object] = {
        "sparsity_alpha": 1.0,
        "steady": True,
        "steady_activation": "sin",
        "start_words": ("S",),
        "steady_train_iters": 20,
    }
    base.update(overrides)
    return EqGPTConfig(**base)


def _steady_scatter_dataset() -> PDEDataset:
    n = 12
    torch.manual_seed(0)
    x = torch.rand(n, dtype=torch.float64)
    y = torch.rand(n, dtype=torch.float64)

    u = x**3 - 3.0 * x * y**2
    return PDEDataset.from_scatter(
        coords={"x": x, "y": y}, fields={"u": u}, lhs="", name="steady_tiny"
    )


def _steady_components(*, evaluator: object | None = None) -> PlatformComponents:
    return PlatformComponents(
        dataset=_steady_scatter_dataset(),
        executor=MagicMock(),
        evaluator=evaluator,
        context=None,
        registry=MagicMock(),
    )







class TestSteadyDerivativeRequirements:
    def test_lhs_order_zero(self) -> None:
        reqs = EqGPTPlugin(_steady_config(), backend=None).derivative_requirements
        assert reqs.lhs_order == 0

    def test_supports_scattered_topology(self) -> None:
        reqs = EqGPTPlugin(_steady_config(), backend=None).derivative_requirements
        assert DataTopology.SCATTERED in reqs.supported_topologies

    def test_declares_no_platform_evaluator_need(self) -> None:
        reqs = EqGPTPlugin(_steady_config(), backend=None).derivative_requirements
        assert reqs.provider_kind == "none"







def test_steady_prepare_does_not_require_platform_evaluator() -> None:
    plugin = EqGPTPlugin(_steady_config(), backend=FakeGPTBackend(57, seed=0))

    plugin.prepare(_steady_components(evaluator=None))
    final = plugin.build_final_result()
    assert final.invalid_reason == "no_candidate"
