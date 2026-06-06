
from __future__ import annotations

import logging
from typing import Any

import pytest
import torch
import torch.nn as nn

from kd.core.evaluator import EvaluationResult
from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd.search.dlga import DLGAConfig, DLGAPlugin
from kd.search.protocol import PlatformComponents, SearchAlgorithm


class _ExactQuadraticModel(nn.Module):

    def forward(self, *, x: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"u": 1.0 + x * x + t * t}


def _make_components(plugin: DLGAPlugin | None = None) -> PlatformComponents:
    from kd.core.platform.builder import PlatformBuilder

    x = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    u = 1.0 + xg * xg + tg * tg
    dataset = PDEDataset(
        name="dlga-plugin-test",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo("x", x), "t": AxisInfo("t", t)},
        axis_order=["x", "t"],
        fields={"u": FieldData("u", u)},
        lhs_field="u",
        lhs_axis="t",
    )
    if plugin is None:


        from kd.models.field_model import FieldModel

        cheap_model = FieldModel(
            coord_names=["x", "t"],
            field_names=["u"],
            hidden_sizes=[8],
            activation="tanh",
        ).to(dtype=torch.float64)
        plugin = DLGAPlugin(surrogate_model=cheap_model)
    return PlatformBuilder(dataset, plugin.derivative_requirements).build()


class TestDLGAPluginProtocol:
    @pytest.mark.smoke
    def test_importable_and_search_algorithm(self) -> None:
        plugin = DLGAPlugin()
        assert isinstance(plugin, SearchAlgorithm)

    @pytest.mark.unit
    def test_adaptive_and_auto_modes_raise_on_prepare(self) -> None:
        for mode in ("adaptive", "auto"):
            with pytest.raises(NotImplementedError, match="mode='constant'"):
                DLGAPlugin(
                    DLGAConfig(mode=mode, pop_size=4),
                    surrogate_model=_ExactQuadraticModel(),
                )


class TestDLGAPluginLifecycle:
    @pytest.mark.unit
    def test_prepare_then_propose_returns_full_population(self) -> None:
        components = _make_components()
        plugin = DLGAPlugin(
            DLGAConfig(pop_size=6, seed=7),
            surrogate_model=_ExactQuadraticModel(),
        )

        plugin.prepare(components)
        candidates = plugin.propose(1)

        assert len(candidates) == 6
        assert all(isinstance(expr, str) and expr for expr in candidates)

    @pytest.mark.unit
    def test_evaluate_const_u_tt_blocked_by_nmse_fallback_guard(self) -> None:
        plugin = DLGAPlugin(
            DLGAConfig(pop_size=4, seed=3, epsilon=0.0),
            surrogate_model=_ExactQuadraticModel(),
        )
        components = _make_components(plugin)
        plugin.prepare(components)

        result = plugin.evaluate(["div(u, u)"])[0]

        assert result.is_valid is True
        assert result.lhs_name == "u_t", (
            "var(u_tt)=0 should trigger H2 guard, forcing selection of u_t"
        )






        assert result.coefficients is not None
        assert result.coefficients.numel() == 1

        assert float(result.coefficients[0].item()) == pytest.approx(1.260, abs=5e-3)
        assert result.nmse == pytest.approx(1.107, abs=5e-3)

    @pytest.mark.unit
    def test_update_and_between_iterations_preserve_population_size(self) -> None:
        components = _make_components()
        plugin = DLGAPlugin(
            DLGAConfig(pop_size=8, seed=11, epsilon=0.0),
            surrogate_model=_ExactQuadraticModel(),
        )

        plugin.prepare(components)
        candidates = plugin.propose(1)
        results = plugin.evaluate(candidates)
        plugin.update(results)
        plugin.between_iterations()

        assert len(plugin.state["population"]) == 8

    @pytest.mark.unit
    def test_state_restore_does_not_reinitialize_population(self) -> None:
        components = _make_components()
        config = DLGAConfig(pop_size=5, seed=13)
        first = DLGAPlugin(config, surrogate_model=_ExactQuadraticModel())
        first.prepare(components)
        first.propose(1)
        saved_state = first.state

        restored = DLGAPlugin(config, surrogate_model=_ExactQuadraticModel())
        restored.state = saved_state
        restored.prepare(components)

        assert restored.state["population"] == saved_state["population"]

    @pytest.mark.unit
    def test_fresh_reprepare_clears_stale_best(self) -> None:
        components = _make_components()
        plugin = DLGAPlugin(
            DLGAConfig(pop_size=5, seed=13),
            surrogate_model=_ExactQuadraticModel(),
        )
        plugin.prepare(components)



        plugin._best_score = -1.0e9
        plugin._best_expression = "STALE_FROM_PRIOR"
        plugin._best_lhs_name = "u_tt"
        plugin._best_genome = plugin._population[0]


        plugin.prepare(components)

        assert plugin.state["best_score"] == float("inf")
        assert plugin.state["best_expression"] == ""
        assert plugin.state["best_lhs_name"] is None
        assert plugin.state["best_genome"] is None

    @pytest.mark.unit
    def test_checkpoint_restore_preserves_best(self) -> None:
        components = _make_components()
        config = DLGAConfig(pop_size=5, seed=13)
        first = DLGAPlugin(config, surrogate_model=_ExactQuadraticModel())
        first.prepare(components)
        first.propose(1)
        saved = first.state
        saved["best_score"] = -42.0
        saved["best_expression"] = "RESTORED_EXPR"

        restored = DLGAPlugin(config, surrogate_model=_ExactQuadraticModel())
        restored.state = saved
        restored.prepare(components)

        assert restored.state["best_score"] == -42.0
        assert restored.state["best_expression"] == "RESTORED_EXPR"

    @pytest.mark.unit
    def test_state_reset_with_empty_dict_clears_best(self) -> None:
        components = _make_components()
        plugin = DLGAPlugin(
            DLGAConfig(pop_size=5, seed=13),
            surrogate_model=_ExactQuadraticModel(),
        )
        plugin.prepare(components)
        plugin._best_score = -1.0e9
        plugin._best_expression = "STALE"
        plugin._best_lhs_name = "u_tt"
        plugin._best_genome = plugin._population[0]

        plugin.state = {}

        assert plugin.state["best_score"] == float("inf")
        assert plugin.state["best_expression"] == ""
        assert plugin.state["best_lhs_name"] is None
        assert plugin.state["best_genome"] is None
        assert plugin.state["population"] is None

    @pytest.mark.unit
    def test_build_final_result_after_state_restore_re_evaluates_best_expression(
        self,
    ) -> None:
        config = DLGAConfig(pop_size=4, seed=17, epsilon=0.0)
        first = DLGAPlugin(config, surrogate_model=_ExactQuadraticModel())
        components = _make_components(first)
        first.prepare(components)
        results = first.evaluate(["div(u, u)"])
        first.update(results)
        saved_state = first.state

        restored = DLGAPlugin(config, surrogate_model=_ExactQuadraticModel())
        restored.state = saved_state
        restored.prepare(components)

        final = restored.build_final_result()

        assert final.is_valid is True
        assert final.error_message == ""
        assert final.expression == "div(u, u)"
        assert final.lhs_name == "u_t", (
            "var(u_tt)=0 should trigger H2 guard, forcing selection of u_t"
        )





        assert final.nmse == pytest.approx(1.107, abs=5e-3)

        assert final.aic == pytest.approx(1.107, abs=5e-3)
        assert final.complexity == 1
        assert final.coefficients is not None
        assert final.coefficients.numel() == 1
        assert float(final.coefficients[0].item()) == pytest.approx(1.260, abs=5e-3)
        assert final.selected_indices == [0]
        assert final.terms == ["div(u, u)"]
        assert final.residuals is not None


        assert final.residuals.numel() == 30
        assert torch.isfinite(final.residuals).all()

    @pytest.mark.unit
    def test_lhs_tie_break_prefers_u_tt_for_v1_parity(self) -> None:
        components = _make_components()
        plugin = DLGAPlugin(
            DLGAConfig(pop_size=4, seed=0, epsilon=0.0),
            surrogate_model=_ExactQuadraticModel(),
        )
        plugin.prepare(components)



        class _StubEvaluator:
            def evaluate_expression(self, expr: str) -> Any:
                return EvaluationResult(
                    mse=0.5,
                    nmse=0.5,
                    r2=0.5,
                    aic=0.5,
                    complexity=1,
                    coefficients=torch.tensor([1.0], dtype=torch.float64),
                    is_valid=True,
                    error_message="",
                    selected_indices=[0],
                    residuals=None,
                    terms=["u"],
                    expression=expr,
                    lhs_name=None,
                )

        plugin._evaluators = {
            "u_t": _StubEvaluator(),
            "u_tt": _StubEvaluator(),
        }

        result = plugin.evaluate(["u"])[0]

        assert result.is_valid is True
        assert result.lhs_name == "u_tt", (
            f"Expected u_tt on tie (the predecessor parity), got {result.lhs_name!r}. "
            "Plugin selects first dict insertion (u_t) on equal mse — must "
            "use deterministic tie-break preferring u_tt."
        )

    @pytest.mark.unit
    def test_fitness_uses_genome_gene_count_for_length_penalty(self) -> None:
        components = _make_components()
        plugin = DLGAPlugin(
            DLGAConfig(pop_size=2, seed=0, epsilon=0.1, lhs_auto_select=False),
            surrogate_model=_ExactQuadraticModel(),
        )
        plugin.prepare(components)
        plugin._population = [[[0, 1, 2]]]

        class _StubEvaluator:
            def evaluate_expression(self, expr: str) -> Any:
                return EvaluationResult(
                    mse=0.25,
                    nmse=0.25,
                    r2=0.0,
                    aic=0.25,
                    complexity=1,
                    coefficients=torch.tensor([1.0], dtype=torch.float64),
                    is_valid=True,
                    error_message="",
                    selected_indices=[0],
                    residuals=torch.zeros(3, dtype=torch.float64),
                    terms=[expr],
                    expression=expr,
                    lhs_name=None,
                )

        plugin._evaluators = {"u_t": _StubEvaluator()}

        result = plugin.evaluate(plugin.propose(1))[0]

        assert result.aic == pytest.approx(0.55)

    @pytest.mark.unit
    def test_evaluators_lhs_targets_have_no_autograd_graph(self) -> None:
        components = _make_components()
        plugin = DLGAPlugin(
            DLGAConfig(pop_size=4, seed=0, epsilon=0.0),
            surrogate_model=_ExactQuadraticModel(),
        )
        plugin.prepare(components)




        assert set(plugin._evaluators) == {"u_t", "u_tt"}, (
            "fixture expects dual-LHS path; both evaluators must exist so "
            "the test covers both Evaluator construction sites in "
            "_build_evaluators"
        )

        for lhs_name, evaluator in plugin._evaluators.items():
            lhs = evaluator._lhs
            lhs_flat = evaluator._lhs_flat
            assert lhs.grad_fn is None, (
                f"{lhs_name}._lhs has grad_fn={lhs.grad_fn!r}; LHS targets "
                "must be detached so the GA loop does not retain the "
                "AutogradProvider create_graph=True chain (spec line 311)"
            )
            assert lhs.requires_grad is False, (
                f"{lhs_name}._lhs.requires_grad=True; LHS targets must not "
                "track gradients — the GA evaluates pop_size × n_generations "
                "candidates and any retained graph will leak surrogate "
                "forward + multi-order derivative buffers"
            )
            assert lhs_flat.grad_fn is None, (
                f"{lhs_name}._lhs_flat has grad_fn={lhs_flat.grad_fn!r}; "
                "``.flatten()`` is a view, so the flat target inherits the "
                "graph from ``_lhs`` unless the source tensor is detached "
                "before construction"
            )
            assert lhs_flat.requires_grad is False, (
                f"{lhs_name}._lhs_flat.requires_grad=True; flat view must "
                "inherit detachment from the source tensor"
            )


class TestExpressionBloatWarningLength:

    @staticmethod
    def _valid_result(nmse: float, complexity: int) -> EvaluationResult:
        return EvaluationResult(
            mse=nmse,
            nmse=nmse,
            r2=0.0,
            aic=nmse,
            complexity=complexity,
            coefficients=torch.tensor([1.0], dtype=torch.float64),
            is_valid=True,
            error_message="",
            selected_indices=[0],
            residuals=torch.zeros(3, dtype=torch.float64),
            terms=["u_xx"],
            expression="u_xx",
            lhs_name="u_t",
        )

    @staticmethod
    def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
        return [
            r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING and r.name == "kd.search.dlga.plugin"
        ]

    @pytest.mark.unit
    def test_genome_branch_uses_gene_count_not_complexity(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        plugin = DLGAPlugin(DLGAConfig(seed=0))
        plugin._best_genome = [[0, 1, 2], [0, 1, 2]]
        result = self._valid_result(nmse=0.5, complexity=2)

        with caplog.at_level(logging.WARNING, logger="kd.search.dlga.plugin"):
            plugin._maybe_warn_expression_bloat(result)

        warnings = self._warnings(caplog)
        assert len(warnings) == 1, f"expected one bloat warning; got {warnings!r}"
        assert "length=6" in warnings[0], (
            f"threshold must use genome gene-count (6), not complexity (2); "
            f"got {warnings[0]!r}"
        )

    @pytest.mark.unit
    def test_genome_gene_count_below_threshold_ignores_high_complexity(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        plugin = DLGAPlugin(DLGAConfig(seed=0))
        plugin._best_genome = [[0, 1]]
        result = self._valid_result(nmse=0.5, complexity=10)

        with caplog.at_level(logging.WARNING, logger="kd.search.dlga.plugin"):
            plugin._maybe_warn_expression_bloat(result)

        assert self._warnings(caplog) == [], (
            "gene-count 2 <= 5 must suppress the warning regardless of complexity"
        )

    @pytest.mark.unit
    def test_none_genome_falls_back_to_complexity(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        plugin = DLGAPlugin(DLGAConfig(seed=0))
        assert plugin._best_genome is None
        result = self._valid_result(nmse=0.5, complexity=6)

        with caplog.at_level(logging.WARNING, logger="kd.search.dlga.plugin"):
            plugin._maybe_warn_expression_bloat(result)

        warnings = self._warnings(caplog)
        assert len(warnings) == 1, f"expected one bloat warning; got {warnings!r}"
        assert "length=6" in warnings[0], (
            f"None-genome branch must use result.complexity (6); got {warnings[0]!r}"
        )
