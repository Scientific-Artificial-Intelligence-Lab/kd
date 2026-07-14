
from __future__ import annotations

import pytest
import torch

from kd.core.executor import ExecutionContext
from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs, assert_dataset_supported
from kd.data.derivatives.finite_diff import FiniteDiffProvider
from kd.data.schema import DataTopology, PDEDataset
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



_N_T = 8
_N_X = 9


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


def _grid_dataset() -> PDEDataset:
    t = torch.linspace(0.0, 1.0, _N_T, dtype=torch.float64)
    x = torch.linspace(0.0, 2.0, _N_X, dtype=torch.float64)
    tt, xx = torch.meshgrid(t, x, indexing="ij")
    u = torch.sin(tt) + torch.cos(xx)
    return PDEDataset.from_arrays(
        coords={"t": t, "x": x},
        fields={"u": u},
        lhs="u_t",
        name="grid-zero-regression",
    )


def _expected_broadcasts(
    dataset: PDEDataset,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u = dataset.get_field("u")
    t = dataset.get_coords("t")
    x = dataset.get_coords("x")
    t_bc = t.view(_N_T, 1).expand(_N_T, _N_X)
    x_bc = x.view(1, _N_X).expand(_N_T, _N_X)
    return u, t_bc, x_bc


def _context_for(name: str, dataset: PDEDataset) -> tuple[ExecutionContext, PDEDataset]:
    reqs = _plugin(name).derivative_requirements
    if reqs.provider_kind == "finite_diff" and not reqs.needs_surrogate:
        components = PlatformBuilder(dataset, reqs).build()
        return components.context, components.dataset

    provider = FiniteDiffProvider(dataset, max_order=min(reqs.max_atomic_order, 3))
    return ExecutionContext(dataset=dataset, derivative_provider=provider), dataset







class TestGridZeroRegression:

    @pytest.mark.unit
    @pytest.mark.parametrize("name", ALL_PLUGINS)
    def test_grid_schema_and_context_byte_identical(self, name: str) -> None:
        dataset = _grid_dataset()
        ctx, resolved = _context_for(name, dataset)


        assert resolved.get_shape() == (_N_T, _N_X)
        assert resolved.spatial_axes == ["x"]


        u_expected, t_bc, x_bc = _expected_broadcasts(resolved)
        assert torch.equal(ctx.get_variable("u"), u_expected)
        assert torch.equal(ctx.get_variable("t"), t_bc)
        assert torch.equal(ctx.get_variable("x"), x_bc)



        reqs = _plugin(name).derivative_requirements
        d1 = ctx.get_derivative("u", "x", 1)
        assert d1.shape == (_N_T, _N_X)
        assert torch.isfinite(d1).all()
        if reqs.max_atomic_order >= 2:
            d2 = ctx.get_derivative("u", "x", 2)
            assert d2.shape == (_N_T, _N_X)
            assert torch.isfinite(d2).all()







class TestDlgaOrderTwoGridPath:

    @pytest.mark.unit
    def test_order2_config_wiring_and_gate(self) -> None:
        reqs = DLGAPlugin(
            DLGAConfig(target_lhs_order=2, lhs_auto_select=True)
        ).derivative_requirements
        assert reqs.lhs_order == 2
        assert assert_dataset_supported(2, DataTopology.GRID, reqs, "dlga") is None

    @pytest.mark.unit
    def test_order2_grid_derivative_computes(self) -> None:
        t = torch.linspace(0.0, 1.0, _N_T, dtype=torch.float64)
        x = torch.linspace(0.0, 2.0, _N_X, dtype=torch.float64)
        tt, xx = torch.meshgrid(t, x, indexing="ij")
        u = torch.sin(tt) + torch.cos(xx)
        dataset = PDEDataset.from_arrays(
            coords={"t": t, "x": x},
            fields={"u": u},
            lhs="u_tt",
            name="grid-order2",
        )
        assert dataset.lhs_order == 2
        assert dataset.spatial_axes == ["x"]
        assert dataset.get_shape() == (_N_T, _N_X)

        provider = FiniteDiffProvider(dataset, max_order=2)
        u_tt = provider.get_derivative("u", "t", 2)
        assert u_tt.shape == (_N_T, _N_X)
        assert torch.isfinite(u_tt).all()







class TestScatterProviderFailLoud:

    @pytest.mark.unit
    def test_finite_diff_rejects_scattered(self) -> None:


        from kd.search.discover.pinn.executor import make_pinn_dataset

        ds = make_pinn_dataset(
            axis_names=["t", "x"],
            field_names=["u"],
            lhs_field="u",
            lhs_axis="t",
        )
        assert ds.topology == DataTopology.SCATTERED
        with pytest.raises(ValueError, match=r"(?i)grid topology"):
            FiniteDiffProvider(ds, max_order=2)

    @pytest.mark.unit
    def test_grid_only_reqs_default_topology(self) -> None:
        assert DerivativeReqs().supported_topologies == frozenset({DataTopology.GRID})
