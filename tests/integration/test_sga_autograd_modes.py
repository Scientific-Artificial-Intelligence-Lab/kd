
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from kd.core.evaluator import Evaluator
from kd.core.executor.context import ExecutionContext
from kd.core.expr import FunctionRegistry, PythonExecutor
from kd.core.linear_solve.least_squares import LeastSquaresSolver
from kd.data.derivatives.autograd import AutogradProvider
from kd.data.derivatives.finite_diff import FiniteDiffProvider
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd.models import FieldModel
from kd.search.protocol import PlatformComponents
from kd.search.runner import ExperimentRunner
from kd.search.sga import SGAConfig, SGAPlugin
from kd.search.sga.evaluate import execute_tree
from kd.search.sga.tree import Node, Tree
from kd.viz.report import ReportResult

_NX = 16
_NT = 8
_X_MAX = 2.0 * math.pi
_T_MAX = 1.0
_SEED = 42


def _build_dataset() -> PDEDataset:
    dtype = torch.float64
    x = torch.linspace(0.0, _X_MAX, _NX + 1, dtype=dtype)[:-1]
    t = torch.linspace(0.0, _T_MAX, _NT, dtype=dtype)
    gx, gt = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(gx) * torch.exp(-gt)
    return PDEDataset(
        name="autograd-test",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=True),
            "t": AxisInfo(name="t", values=t, is_periodic=False),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def _build_components(dataset: PDEDataset) -> PlatformComponents:
    provider = FiniteDiffProvider(dataset, max_order=2)
    context = ExecutionContext(dataset=dataset, derivative_provider=provider)
    registry = FunctionRegistry.create_default()
    executor = PythonExecutor(registry)
    solver = LeastSquaresSolver()
    u_t = provider.get_derivative("u", "t", order=1).flatten()
    evaluator = Evaluator(
        executor=executor,
        solver=solver,
        context=context,
        lhs=u_t,
    )
    return PlatformComponents(
        dataset=dataset,
        executor=executor,
        evaluator=evaluator,
        context=context,
        registry=registry,
    )


def _untrained_field_model(dataset: PDEDataset) -> FieldModel:
    torch.manual_seed(_SEED)
    model = FieldModel(
        coord_names=list(dataset.axes.keys()),
        field_names=list(dataset.fields.keys()),
        hidden_sizes=[8, 8],
    ).to(dtype=torch.float64)
    model.eval()
    return model


def _config(use_autograd: bool, dataset: PDEDataset) -> SGAConfig:
    return SGAConfig(
        num=4,
        depth=3,
        width=3,
        seed=_SEED,
        use_autograd=use_autograd,
        field_model=_untrained_field_model(dataset) if use_autograd else None,
    )


@pytest.fixture(scope="module")
def dataset() -> PDEDataset:
    return _build_dataset()


@pytest.fixture(scope="module")
def fd_components(dataset: PDEDataset) -> PlatformComponents:
    return _build_components(dataset)


@pytest.fixture(scope="module")
def ad_components(dataset: PDEDataset) -> PlatformComponents:
    return _build_components(dataset)


@pytest.fixture(scope="module")
def fd_plugin(dataset: PDEDataset, fd_components: PlatformComponents) -> SGAPlugin:
    plugin = SGAPlugin(_config(use_autograd=False, dataset=dataset))
    plugin.prepare(fd_components)
    return plugin


@pytest.fixture(scope="module")
def ad_plugin(dataset: PDEDataset, ad_components: PlatformComponents) -> SGAPlugin:
    plugin = SGAPlugin(_config(use_autograd=True, dataset=dataset))
    plugin.prepare(ad_components)
    return plugin


@pytest.mark.integration
class TestThreeLayerSemantics:

    def test_layer1_raw_u_invariant_under_use_autograd(
        self,
        dataset: PDEDataset,
        fd_plugin: SGAPlugin,
        ad_plugin: SGAPlugin,
    ) -> None:
        raw_u_flat = dataset.fields["u"].values.flatten()

        torch.testing.assert_close(
            fd_plugin._data_dict["u"], raw_u_flat, rtol=0, atol=0
        )
        torch.testing.assert_close(
            ad_plugin._data_dict["u"], raw_u_flat, rtol=0, atol=0
        )

    def test_layer2_terminal_routes_through_autograd(
        self, fd_plugin: SGAPlugin, ad_plugin: SGAPlugin
    ) -> None:
        fd_ux = fd_plugin._data_dict["u_x"]
        ad_ux = ad_plugin._data_dict["u_x"]
        assert fd_ux.shape == ad_ux.shape

        max_abs_diff = (fd_ux - ad_ux).abs().max().item()
        assert max_abs_diff > 1e-3, (
            f"Layer 2 appears not to route through AutogradProvider: "
            f"FD u_x and AD u_x differ by only {max_abs_diff:.3e}"
        )

    def test_layer2_lhs_target_routes_through_autograd(
        self, fd_plugin: SGAPlugin, ad_plugin: SGAPlugin
    ) -> None:
        assert fd_plugin._y is not None and ad_plugin._y is not None
        max_abs_diff = (fd_plugin._y - ad_plugin._y).abs().max().item()
        assert max_abs_diff > 1e-3, (
            f"LHS target appears not to use AD provider when use_autograd=True: "
            f"max diff {max_abs_diff:.3e}"
        )

    def test_autograd_layer2_values_are_detached(self, ad_plugin: SGAPlugin) -> None:
        assert ad_plugin._y is not None
        assert ad_plugin._y.requires_grad is False
        assert ad_plugin._data_dict["u_x"].requires_grad is False

    def test_layer3_tree_d_on_raw_u_invariant(
        self, fd_plugin: SGAPlugin, ad_plugin: SGAPlugin
    ) -> None:
        tree = Tree(
            root=Node(
                name="d",
                arity=2,
                children=[Node(name="u", arity=0), Node(name="x", arity=0)],
            )
        )
        fd_result = execute_tree(tree, fd_plugin._data_dict, fd_plugin._diff_ctx)
        ad_result = execute_tree(tree, ad_plugin._data_dict, ad_plugin._diff_ctx)
        torch.testing.assert_close(fd_result, ad_result, rtol=0, atol=0)

    def test_layer3_tree_d_on_u_x_differs_due_to_leaf_source(
        self, fd_plugin: SGAPlugin, ad_plugin: SGAPlugin
    ) -> None:
        tree = Tree(
            root=Node(
                name="d",
                arity=2,
                children=[Node(name="u_x", arity=0), Node(name="x", arity=0)],
            )
        )
        fd_result = execute_tree(tree, fd_plugin._data_dict, fd_plugin._diff_ctx)
        ad_result = execute_tree(tree, ad_plugin._data_dict, ad_plugin._diff_ctx)
        max_abs_diff = (fd_result - ad_result).abs().max().item()
        assert max_abs_diff > 1e-3, (
            f"Expected d(u_x, x) to differ between FD and AD modes "
            f"(leaf u_x source differs); got max diff {max_abs_diff:.3e}"
        )

    def test_use_autograd_does_not_replace_context_provider(
        self, ad_components: PlatformComponents, ad_plugin: SGAPlugin
    ) -> None:
        assert isinstance(
            ad_components.context.derivative_provider, FiniteDiffProvider
        ), (
            "use_autograd=True should not replace the platform-shared "
            "context.derivative_provider"
        )
        assert isinstance(ad_plugin._autograd_provider, AutogradProvider)
        assert (
            ad_plugin._autograd_provider
            is not ad_components.context.derivative_provider
        )

    def test_runner_result_actual_uses_autograd_target(
        self, dataset: PDEDataset
    ) -> None:
        components = _build_components(dataset)
        plugin = SGAPlugin(_config(use_autograd=True, dataset=dataset))
        runner = ExperimentRunner(plugin, max_iterations=1, batch_size=4)

        result = runner.run(components)

        assert plugin._y is not None
        torch.testing.assert_close(result.actual, plugin._y, rtol=0, atol=0)
        assert result.final_eval.residuals is not None
        torch.testing.assert_close(
            result.predicted,
            result.actual + result.final_eval.residuals,
            rtol=0,
            atol=0,
        )
        fd_target = components.evaluator.lhs_target
        assert isinstance(fd_target, torch.Tensor)
        assert not torch.allclose(result.actual, fd_target)


@pytest.mark.integration
class TestAutogradConfigValidation:

    def test_field_model_coord_mismatch_rejected(self, dataset: PDEDataset) -> None:
        bad_model = FieldModel(
            coord_names=["x", "y"],
            field_names=["u"],
            hidden_sizes=[4],
        )
        cfg = SGAConfig(use_autograd=True, field_model=bad_model)
        plugin = SGAPlugin(cfg)
        components = _build_components(dataset)
        with pytest.raises(ValueError, match="coord_names"):
            plugin.prepare(components)

    def test_field_model_field_mismatch_rejected(self, dataset: PDEDataset) -> None:
        bad_model = FieldModel(
            coord_names=["x", "t"],
            field_names=["v"],
            hidden_sizes=[4],
        )
        cfg = SGAConfig(use_autograd=True, field_model=bad_model)
        plugin = SGAPlugin(cfg)
        components = _build_components(dataset)
        with pytest.raises(ValueError, match="field_names"):
            plugin.prepare(components)

    def test_use_autograd_false_leaves_provider_untouched(
        self, dataset: PDEDataset
    ) -> None:
        components = _build_components(dataset)
        plugin = SGAPlugin(SGAConfig(use_autograd=False, seed=_SEED))
        plugin.prepare(components)
        assert plugin._autograd_provider is None


@pytest.mark.integration
class TestHtmlAutogradDomainWarning:

    def _run_sga_and_render(
        self,
        dataset: PDEDataset,
        use_autograd: bool,
        tmp_path: Path,
    ) -> ReportResult:
        from kd.viz import VizEngine

        components = _build_components(dataset)
        plugin = SGAPlugin(_config(use_autograd=use_autograd, dataset=dataset))
        runner = ExperimentRunner(plugin, max_iterations=1, batch_size=4)
        result = runner.run(components)
        engine = VizEngine(output_dir=tmp_path)
        return engine.render_all(result, dataset=dataset)

    def test_autograd_true_emits_domain_warning(
        self, dataset: PDEDataset, tmp_path: Path
    ) -> None:
        report = self._run_sga_and_render(dataset, use_autograd=True, tmp_path=tmp_path)
        joined = "\n".join(report.warnings).lower()
        assert "autograd" in joined, (
            f"Expected autograd domain note in HTML warnings; got: {report.warnings}"
        )
        assert "domain" in joined

        html = report.report.read_text() if report.report is not None else ""
        assert "Domain note: this run fitted derivatives in an autograd" in html

    def test_autograd_false_omits_domain_warning(
        self, dataset: PDEDataset, tmp_path: Path
    ) -> None:
        report = self._run_sga_and_render(
            dataset, use_autograd=False, tmp_path=tmp_path
        )
        joined = "\n".join(report.warnings).lower()
        assert "domain note:" not in joined, (
            f"Did not expect autograd domain note for FD-only run; "
            f"got: {report.warnings}"
        )
