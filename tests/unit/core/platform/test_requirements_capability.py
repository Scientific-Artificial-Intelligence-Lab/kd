
from __future__ import annotations

import re
from dataclasses import FrozenInstanceError

import pytest

from kd.core.platform.requirements import DerivativeReqs
from kd.data.schema import DataTopology
from kd.search.discover.plugin import DISCOVERPlugin
from kd.search.dlga.config import DLGAConfig
from kd.search.dlga.plugin import DLGAPlugin
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.plugin import EqGPTPlugin
from kd.search.llm4ed.config import Llm4edConfig
from kd.search.llm4ed.plugin import Llm4edPlugin
from kd.search.pysr.plugin import PySRPlugin
from kd.search.sga.plugin import SGAPlugin












ALL_PLUGINS = ("sga", "dlga", "discover", "pysr", "eqgpt", "llm4ed")
GRID_ONLY_PLUGINS = ALL_PLUGINS


def _plugin(name: str):
    if name == "sga":
        return SGAPlugin()
    if name == "dlga":
        return DLGAPlugin()
    if name == "discover":
        return DISCOVERPlugin()
    if name == "pysr":
        return PySRPlugin()
    if name == "eqgpt":
        return EqGPTPlugin(EqGPTConfig(sparsity_alpha=0.02, variables=("t", "x")))
    if name == "llm4ed":
        return Llm4edPlugin(Llm4edConfig())
    raise AssertionError(f"unknown plugin name {name!r}")







class TestSupportedTopologiesField:

    @pytest.mark.unit
    def test_default_is_grid_only_frozenset(self) -> None:
        reqs = DerivativeReqs()
        assert reqs.supported_topologies == frozenset({DataTopology.GRID})

    @pytest.mark.unit
    def test_default_value_is_a_frozenset(self) -> None:
        reqs = DerivativeReqs()
        assert isinstance(reqs.supported_topologies, frozenset)

    @pytest.mark.unit
    def test_overridable_with_scattered(self) -> None:
        reqs = DerivativeReqs(
            supported_topologies=frozenset(
                {DataTopology.GRID, DataTopology.SCATTERED}
            )
        )
        assert DataTopology.SCATTERED in reqs.supported_topologies
        assert DataTopology.GRID in reqs.supported_topologies

    @pytest.mark.unit
    def test_field_is_frozen(self) -> None:
        reqs = DerivativeReqs()
        with pytest.raises(FrozenInstanceError):
            reqs.supported_topologies = frozenset({DataTopology.SCATTERED})

    @pytest.mark.unit
    def test_needs_surrogate_post_init_still_fires(self) -> None:
        with pytest.raises(ValueError, match="needs_surrogate"):
            DerivativeReqs(provider_kind="finite_diff", needs_surrogate=True)







class TestAssertDatasetSupportedMatrix:

    @pytest.mark.unit
    @pytest.mark.parametrize("name", ALL_PLUGINS)
    def test_grid_selected_order_passes(self, name: str) -> None:
        from kd.core.platform.requirements import assert_dataset_supported

        reqs = _plugin(name).derivative_requirements

        assert (
            assert_dataset_supported(
                reqs.lhs_order, DataTopology.GRID, reqs, name
            )
            is None
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("name", GRID_ONLY_PLUGINS)
    def test_scattered_raises_topology_for_grid_only_plugins(
        self, name: str
    ) -> None:
        from kd.core.platform.requirements import assert_dataset_supported

        reqs = _plugin(name).derivative_requirements
        with pytest.raises(NotImplementedError) as excinfo:
            assert_dataset_supported(
                reqs.lhs_order, DataTopology.SCATTERED, reqs, name
            )
        message = str(excinfo.value)
        assert name in message, f"topology error must name the algorithm: {message!r}"
        assert re.search(r"(?i)scatter|topolog", message), (
            f"topology error must mention the topology: {message!r}"
        )

    @pytest.mark.unit
    def test_scattered_accepted_for_a_scattered_declaring_reqs(self) -> None:
        from kd.core.platform.requirements import assert_dataset_supported

        reqs = DerivativeReqs(
            supported_topologies=frozenset(
                {DataTopology.GRID, DataTopology.SCATTERED}
            ),
            lhs_order=1,
        )
        assert (
            assert_dataset_supported(1, DataTopology.SCATTERED, reqs, "custom")
            is None
        )

    @pytest.mark.unit
    def test_topology_check_precedes_order_when_both_mismatch(self) -> None:
        from kd.core.platform.requirements import assert_dataset_supported

        reqs = _plugin("sga").derivative_requirements
        with pytest.raises(NotImplementedError) as excinfo:

            assert_dataset_supported(2, DataTopology.SCATTERED, reqs, "sga")
        assert re.search(r"(?i)scatter|topolog", str(excinfo.value))

    @pytest.mark.unit
    @pytest.mark.parametrize("name", GRID_ONLY_PLUGINS)
    def test_grid_order_zero_raises_order_for_all_plugins(self, name: str) -> None:
        from kd.core.platform.requirements import assert_dataset_supported

        reqs = _plugin(name).derivative_requirements
        with pytest.raises(NotImplementedError) as excinfo:
            assert_dataset_supported(0, DataTopology.GRID, reqs, name)
        assert re.search(r"(?i)order", str(excinfo.value))

    @pytest.mark.unit
    def test_grid_order_mismatch_raises_order_for_sga(self) -> None:
        from kd.core.platform.requirements import assert_dataset_supported

        reqs = _plugin("sga").derivative_requirements
        assert reqs.lhs_order == 1
        with pytest.raises(NotImplementedError) as excinfo:
            assert_dataset_supported(2, DataTopology.GRID, reqs, "sga")
        message = str(excinfo.value)
        assert re.search(r"(?i)order", message), (
            f"order-mismatch error must mention the order: {message!r}"
        )
        assert not re.search(r"(?i)scatter", message), (
            f"a GRID order-mismatch must not be reported as a topology error: "
            f"{message!r}"
        )

    @pytest.mark.unit
    def test_order_two_mismatch_names_dlga_as_the_real_remedy(self) -> None:
        from kd.core.platform.requirements import assert_dataset_supported

        reqs = _plugin("sga").derivative_requirements
        with pytest.raises(NotImplementedError) as excinfo:
            assert_dataset_supported(2, DataTopology.GRID, reqs, "sga")
        message = str(excinfo.value)
        assert re.search(r"(?i)dlga", message), (
            f"order-2 mismatch must name the real remedy (dlga): {message!r}"
        )
        assert "target_lhs_order" in message, (
            f"order-2 mismatch must name the actual config knob: {message!r}"
        )







class TestDlgaOrderTwoPreserved:

    @pytest.mark.unit
    def test_order2_config_passes_on_order2_grid(self) -> None:
        from kd.core.platform.requirements import assert_dataset_supported

        config = DLGAConfig(target_lhs_order=2, lhs_auto_select=True)
        reqs = DLGAPlugin(config).derivative_requirements
        assert reqs.lhs_order == 2
        assert (
            assert_dataset_supported(2, DataTopology.GRID, reqs, "dlga") is None
        )

    @pytest.mark.unit
    def test_order1_config_rejected_on_order2_grid(self) -> None:
        from kd.core.platform.requirements import assert_dataset_supported

        reqs = DLGAPlugin(DLGAConfig()).derivative_requirements
        assert reqs.lhs_order == 1
        with pytest.raises(NotImplementedError) as excinfo:
            assert_dataset_supported(2, DataTopology.GRID, reqs, "dlga")
        assert re.search(r"(?i)order", str(excinfo.value))







class TestNoSupportedLhsOrdersField:

    @pytest.mark.unit
    def test_no_supported_lhs_orders_attribute(self) -> None:
        assert not hasattr(DerivativeReqs(), "supported_lhs_orders"), (
            "DerivativeReqs must NOT gain a supported_lhs_orders set — order "
            "capability is the selected-== check in assert_dataset_supported "
            "(see task D-3b-1: a membership set loses DLGA order-2 protection)."
        )
