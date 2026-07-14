
from __future__ import annotations

import pytest
import torch

from kd.core.platform.requirements import assert_dataset_supported
from kd.data.schema import PDEDataset
from kd.search.dlga import DLGAConfig, DLGAPlugin


def _make_dataset(lhs: str) -> PDEDataset:
    x = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    u = torch.outer(x, t)
    return PDEDataset.from_arrays(coords={"x": x, "t": t}, fields={"u": u}, lhs=lhs)





@pytest.mark.unit
def test_default_plugin_declares_config_target_lhs_order() -> None:
    config = DLGAConfig()
    plugin = DLGAPlugin(config)
    assert plugin.derivative_requirements.lhs_order == config.target_lhs_order


@pytest.mark.unit
def test_wave_preset_plugin_declares_second_order_lhs() -> None:
    plugin = DLGAPlugin(DLGAConfig.wave_preset())
    assert plugin.derivative_requirements.lhs_order == 2


@pytest.mark.unit
@pytest.mark.parametrize("target_order", [1, 2])
def test_plugin_lhs_order_tracks_explicit_config_target(target_order: int) -> None:
    config = DLGAConfig(target_lhs_order=target_order)
    plugin = DLGAPlugin(config)
    assert plugin.derivative_requirements.lhs_order == target_order





@pytest.mark.unit
def test_gate_accepts_order2_dataset_with_wave_preset_plugin() -> None:
    dataset = _make_dataset("u_tt")
    plugin = DLGAPlugin(DLGAConfig.wave_preset())


    assert_dataset_supported(
        dataset.lhs_order, dataset.topology, plugin.derivative_requirements, "dlga"
    )


@pytest.mark.unit
def test_gate_rejects_order2_dataset_with_default_plugin() -> None:
    dataset = _make_dataset("u_tt")
    plugin = DLGAPlugin(DLGAConfig())
    with pytest.raises(NotImplementedError, match="lhs_order"):
        assert_dataset_supported(
            dataset.lhs_order, dataset.topology, plugin.derivative_requirements, "dlga"
        )


@pytest.mark.unit
def test_gate_accepts_order1_dataset_with_default_plugin() -> None:
    dataset = _make_dataset("u_t")
    plugin = DLGAPlugin(DLGAConfig())
    assert_dataset_supported(
        dataset.lhs_order, dataset.topology, plugin.derivative_requirements, "dlga"
    )
