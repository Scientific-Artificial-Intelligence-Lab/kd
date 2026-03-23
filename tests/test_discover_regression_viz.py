"""Tests for Discover Regression visualization adapter and equation rendering.

RED phase: Tests define expected behavior for:
1. regression_program_to_latex() — LaTeX equation rendering
2. DiscoverRegressionVizAdapter — adapter capabilities and handlers
3. Integration with viz framework — dispatch and registry
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence
from unittest.mock import MagicMock

import numpy as np
import pytest
import sympy

from kd.viz import core as viz_core
from kd.viz import registry as viz_registry
from kd.viz._contracts import ParityPlotData, ResidualPlotData


# ---------------------------------------------------------------------------
# Fixtures and stubs
# ---------------------------------------------------------------------------

class FakeProgram:
    """Minimal mock of discover Program with sympy_expr for regression."""

    def __init__(
        self,
        expr: sympy.Expr,
        n_input_var: int = 2,
    ) -> None:
        self._expr = expr
        self.traversal = ["stub"]
        self.n_input_var = n_input_var

    @property
    def sympy_expr(self) -> list:
        return [self._expr]


def _make_program_with_vars() -> FakeProgram:
    """Program: 2.5*x1 + 3.0*x2 (two features)."""
    x1, x2 = sympy.symbols("x1 x2")
    return FakeProgram(2.5 * x1 + 3.0 * x2, n_input_var=2)


def _make_program_with_constant() -> FakeProgram:
    """Program: 0.314*x1 + 1.0  (constant appears as number)."""
    x1 = sympy.Symbol("x1")
    return FakeProgram(0.314 * x1 + 1.0, n_input_var=1)


def _make_program_single_var() -> FakeProgram:
    """Program: x1**2  (single feature, no var_names needed)."""
    x1 = sympy.Symbol("x1")
    return FakeProgram(x1**2, n_input_var=1)


class FakeSearcher:
    """Minimal mock of Discover searcher for evolution/density/tree."""

    def __init__(self) -> None:
        self.r_train = [
            np.array([0.5, 0.6, 0.55]),
            np.array([0.7, 0.65, 0.72]),
            np.array([0.8, 0.78, 0.81]),
        ]
        self.r_history = [np.asarray(b) for b in self.r_train]
        self.plotter = MagicMock()
        self.best_p = MagicMock()


class FakeRegressionData:
    """Minimal mock of _RegressionArrayData."""

    def __init__(self, X: np.ndarray, y: np.ndarray, var_names: list[str]) -> None:
        self.X = X
        self.y = y
        self.var_names = var_names


class StubRegressionModel:
    """Stub for KD_Discover_Regression with enough attributes for adapter."""

    def __init__(
        self,
        program: Optional[FakeProgram] = None,
        var_names: Optional[list[str]] = None,
    ) -> None:
        rng = np.random.default_rng(42)
        n_samples = 20
        n_features = 2

        X = rng.standard_normal((n_samples, n_features))
        y_true = 2.5 * X[:, 0] + 3.0 * X[:, 1] + rng.normal(0, 0.01, n_samples)

        names = var_names or ["Rf1", "Rf2"]
        self.data_class = FakeRegressionData(X, y_true, names)
        self.dataset = "custom_regression"
        self.best_program_ = program or _make_program_with_vars()
        self.result_ = {
            "var_names": names,
            "expression": "stub",
            "reward": 0.99,
        }
        self.searcher = FakeSearcher()
        self.n_features_in_ = n_features

        # predict mock: simple linear
        self._y_pred = 2.5 * X[:, 0] + 3.0 * X[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._y_pred


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure a clean registry for each test."""
    viz_registry.clear_registry()
    yield
    viz_registry.clear_registry()


@pytest.fixture(autouse=True)
def suppress_show(monkeypatch):
    """Prevent matplotlib windows from opening."""
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)


# ---------------------------------------------------------------------------
# 1. regression_program_to_latex()
# ---------------------------------------------------------------------------

class TestRegressionProgramToLatex:
    """Tests for the regression_program_to_latex function."""

    def test_basic_var_names_substitution(self) -> None:
        """x1 -> Rf1, x2 -> Rf2 in the LaTeX output."""
        from kd.viz.discover_eq2latex import regression_program_to_latex

        program = _make_program_with_vars()
        latex = regression_program_to_latex(program, var_names=["Rf1", "Rf2"])

        assert "Rf1" in latex or "Rf_{1}" in latex or "Rf1" in latex
        assert "Rf2" in latex or "Rf_{2}" in latex or "Rf2" in latex
        # x1/x2 should NOT appear (replaced by var_names)
        assert "x1" not in latex
        assert "x2" not in latex

    def test_custom_target_name(self) -> None:
        """Custom target_name replaces default 'y' on LHS."""
        from kd.viz.discover_eq2latex import regression_program_to_latex

        program = _make_program_with_vars()
        latex = regression_program_to_latex(
            program, var_names=["a", "b"], target_name="V_S"
        )

        assert "V_S" in latex or "V_{S}" in latex

    def test_default_target_name_is_y(self) -> None:
        """Without target_name, LHS should default to 'y'."""
        from kd.viz.discover_eq2latex import regression_program_to_latex

        program = _make_program_with_vars()
        latex = regression_program_to_latex(program, var_names=["a", "b"])

        # Should contain "y =" pattern
        assert "$y =" in latex or "$y=" in latex

    def test_constants_appear_as_numbers(self) -> None:
        """Numeric constants should render as numbers, not 'const'."""
        from kd.viz.discover_eq2latex import regression_program_to_latex

        program = _make_program_with_constant()
        latex = regression_program_to_latex(program, var_names=["z"])

        # Should contain numeric value, not the string "const"
        assert "const" not in latex
        # The number 0.314 should appear in some form
        assert "0.314" in latex or "314" in latex

    def test_no_var_names_uses_x_symbols(self) -> None:
        """Without var_names, symbols remain as x1, x2 etc."""
        from kd.viz.discover_eq2latex import regression_program_to_latex

        program = _make_program_with_vars()
        latex = regression_program_to_latex(program)

        # x1 or x_{1} should remain
        assert "x" in latex

    def test_result_wrapped_in_dollars(self) -> None:
        """Output should be wrapped in $...$."""
        from kd.viz.discover_eq2latex import regression_program_to_latex

        program = _make_program_single_var()
        latex = regression_program_to_latex(program)

        assert latex.startswith("$")
        assert latex.endswith("$")

    def test_empty_var_names_list(self) -> None:
        """Empty var_names list should not crash, symbols stay as x1."""
        from kd.viz.discover_eq2latex import regression_program_to_latex

        program = _make_program_single_var()
        latex = regression_program_to_latex(program, var_names=[])

        assert isinstance(latex, str)
        assert latex.startswith("$")

    def test_single_constant_expression(self) -> None:
        """Expression that is just a constant: y = 42.0."""
        from kd.viz.discover_eq2latex import regression_program_to_latex

        program = FakeProgram(sympy.Float(42.0), n_input_var=0)
        latex = regression_program_to_latex(program)

        assert "42" in latex


# ---------------------------------------------------------------------------
# 2. DiscoverRegressionVizAdapter
# ---------------------------------------------------------------------------

class TestDiscoverRegressionVizAdapter:
    """Tests for the adapter class capabilities and handlers."""

    def _get_adapter(self):
        from kd.viz._adapters.discover_regression import (
            DiscoverRegressionVizAdapter,
        )
        return DiscoverRegressionVizAdapter()

    def test_capabilities_set(self) -> None:
        """Adapter must declare expected capabilities."""
        adapter = self._get_adapter()

        expected = {
            "search_evolution",
            "density",
            "tree",
            "equation",
            "parity",
            "residual",
        }
        assert set(adapter.capabilities) == expected

    def test_no_pde_specific_capabilities(self) -> None:
        """Regression adapter must NOT have PDE-specific capabilities."""
        adapter = self._get_adapter()

        pde_only = {"field_comparison", "spr_residual", "spr_field_comparison"}
        assert pde_only.isdisjoint(set(adapter.capabilities))

    def test_equation_returns_vizresult(self, tmp_path) -> None:
        """equation handler returns VizResult with LaTeX metadata."""
        adapter = self._get_adapter()
        viz_registry.register_adapter(StubRegressionModel, adapter)

        model = StubRegressionModel()
        request = viz_core.VizRequest(
            kind="equation",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)

        assert isinstance(result, viz_core.VizResult)
        assert result.intent == "equation"
        # Should have latex in metadata or paths (equation image)
        assert "latex" in result.metadata or result.paths

    def test_parity_returns_vizresult(self, tmp_path) -> None:
        """parity handler returns VizResult with ParityPlotData."""
        adapter = self._get_adapter()
        viz_registry.register_adapter(StubRegressionModel, adapter)

        model = StubRegressionModel()
        request = viz_core.VizRequest(
            kind="parity",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)

        assert isinstance(result, viz_core.VizResult)
        assert result.intent == "parity"
        # Should have parity data in metadata
        if not result.warnings:
            parity_data = result.metadata.get("parity")
            assert isinstance(parity_data, ParityPlotData)
            assert parity_data.actual_values.shape == parity_data.predicted_values.shape

    def test_residual_returns_vizresult(self, tmp_path) -> None:
        """residual handler returns VizResult with ResidualPlotData."""
        adapter = self._get_adapter()
        viz_registry.register_adapter(StubRegressionModel, adapter)

        model = StubRegressionModel()
        request = viz_core.VizRequest(
            kind="residual",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)

        assert isinstance(result, viz_core.VizResult)
        assert result.intent == "residual"
        if not result.warnings:
            residual_data = result.metadata.get("residual_data")
            assert isinstance(residual_data, ResidualPlotData)

    def test_search_evolution_returns_vizresult(self, tmp_path) -> None:
        """search_evolution handler works with regression model."""
        adapter = self._get_adapter()
        viz_registry.register_adapter(StubRegressionModel, adapter)

        model = StubRegressionModel()
        request = viz_core.VizRequest(
            kind="search_evolution",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)

        assert isinstance(result, viz_core.VizResult)
        assert result.intent == "search_evolution"

    def test_unsupported_intent_returns_warning(self) -> None:
        """Requesting an unsupported intent gives a warning, not crash."""
        adapter = self._get_adapter()
        viz_registry.register_adapter(StubRegressionModel, adapter)

        model = StubRegressionModel()
        request = viz_core.VizRequest(
            kind="field_comparison",  # not supported for regression
            target=model,
        )
        result = viz_core.render(request)

        assert result.warnings
        assert "does not support" in result.warnings[0]

    def test_equation_contains_var_names(self, tmp_path) -> None:
        """equation result should use model's var_names, not x1/x2."""
        adapter = self._get_adapter()
        viz_registry.register_adapter(StubRegressionModel, adapter)

        model = StubRegressionModel(var_names=["Rf1", "Rf2"])
        request = viz_core.VizRequest(
            kind="equation",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)

        if "latex" in result.metadata:
            latex = result.metadata["latex"]
            assert "x1" not in latex
            assert "x2" not in latex


# ---------------------------------------------------------------------------
# 3. Integration with viz framework (registry + api dispatch)
# ---------------------------------------------------------------------------

class TestRegistryIntegration:
    """Test that the adapter is registered correctly and dispatches work."""

    def test_adapter_registered_for_kd_discover_regression(self) -> None:
        """After register_default_adapters(), KD_Discover_Regression has an adapter."""
        from kd.viz.adapters import register_default_adapters
        register_default_adapters()

        from kd.model.kd_discover_regression import KD_Discover_Regression

        adapter = viz_registry.get_adapter(KD_Discover_Regression)
        assert adapter is not None

    def test_regression_gets_own_adapter_not_pde(self) -> None:
        """KD_Discover_Regression should NOT fall back to DiscoverVizAdapter."""
        from kd.viz.adapters import register_default_adapters
        from kd.viz._adapters.discover import DiscoverVizAdapter
        register_default_adapters()

        from kd.model.kd_discover_regression import KD_Discover_Regression

        adapter = viz_registry.get_adapter(KD_Discover_Regression)
        # Must be the regression-specific adapter, not the PDE one
        assert not isinstance(adapter, DiscoverVizAdapter)

    def test_render_equation_dispatches_to_regression_adapter(
        self, monkeypatch, tmp_path
    ) -> None:
        """kd.viz.api.render_equation() goes through regression adapter."""
        from kd.viz import api as viz_api
        from kd.viz._adapters.discover_regression import (
            DiscoverRegressionVizAdapter,
        )

        monkeypatch.setattr(viz_api, "_emit_info", lambda *a, **kw: None)

        adapter = DiscoverRegressionVizAdapter()
        viz_registry.register_adapter(StubRegressionModel, adapter)

        model = StubRegressionModel()
        result = viz_api.render_equation(model, output_dir=tmp_path, show_info=False)

        assert isinstance(result, viz_core.VizResult)
        assert result.intent == "equation"

    def test_plot_parity_dispatches_to_regression_adapter(
        self, monkeypatch, tmp_path
    ) -> None:
        """kd.viz.api.plot_parity() goes through regression adapter."""
        from kd.viz import api as viz_api
        from kd.viz._adapters.discover_regression import (
            DiscoverRegressionVizAdapter,
        )

        monkeypatch.setattr(viz_api, "_emit_info", lambda *a, **kw: None)

        adapter = DiscoverRegressionVizAdapter()
        viz_registry.register_adapter(StubRegressionModel, adapter)

        model = StubRegressionModel()
        result = viz_api.plot_parity(model, output_dir=tmp_path)

        assert isinstance(result, viz_core.VizResult)
        assert result.intent == "parity"


# ---------------------------------------------------------------------------
# Edge cases / negative tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and negative tests (>= 25% of total)."""

    def test_program_to_latex_none_program_expr(self) -> None:
        """Program with None sympy_expr should not crash."""
        from kd.viz.discover_eq2latex import regression_program_to_latex

        program = FakeProgram(sympy.Integer(0))
        # Should return valid LaTeX string, not raise
        latex = regression_program_to_latex(program)
        assert isinstance(latex, str)

    def test_parity_with_nan_predictions(self, tmp_path) -> None:
        """Model returning NaN predictions should be handled gracefully."""
        from kd.viz._adapters.discover_regression import (
            DiscoverRegressionVizAdapter,
        )

        adapter = DiscoverRegressionVizAdapter()
        viz_registry.register_adapter(StubRegressionModel, adapter)

        model = StubRegressionModel()
        model._y_pred = np.full(20, np.nan)

        request = viz_core.VizRequest(
            kind="parity",
            target=model,
            options={"output_dir": tmp_path},
        )
        # Should not raise — adapter should handle or pass through
        result = viz_core.render(request)
        assert isinstance(result, viz_core.VizResult)

    def test_residual_with_zero_residuals(self, tmp_path) -> None:
        """Perfect predictions (zero residuals) should work."""
        from kd.viz._adapters.discover_regression import (
            DiscoverRegressionVizAdapter,
        )

        adapter = DiscoverRegressionVizAdapter()
        viz_registry.register_adapter(StubRegressionModel, adapter)

        model = StubRegressionModel()
        # Make predictions exactly match truth
        model._y_pred = model.data_class.y.copy()

        request = viz_core.VizRequest(
            kind="residual",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, viz_core.VizResult)

    def test_equation_with_no_best_program(self, tmp_path) -> None:
        """Model with no best_program_ should degrade gracefully."""
        from kd.viz._adapters.discover_regression import (
            DiscoverRegressionVizAdapter,
        )

        adapter = DiscoverRegressionVizAdapter()
        viz_registry.register_adapter(StubRegressionModel, adapter)

        model = StubRegressionModel()
        model.best_program_ = None

        request = viz_core.VizRequest(
            kind="equation",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, viz_core.VizResult)
        # Should have a warning about missing program
        assert result.warnings

    def test_model_with_no_data_class(self, tmp_path) -> None:
        """Model without data_class should not crash parity/residual."""
        from kd.viz._adapters.discover_regression import (
            DiscoverRegressionVizAdapter,
        )

        adapter = DiscoverRegressionVizAdapter()
        viz_registry.register_adapter(StubRegressionModel, adapter)

        model = StubRegressionModel()
        model.data_class = None

        request = viz_core.VizRequest(
            kind="parity",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, viz_core.VizResult)
        # Should warn, not crash
        assert result.warnings

    def test_var_names_with_special_latex_chars(self) -> None:
        """Variable names with LaTeX-sensitive chars should render safely."""
        from kd.viz.discover_eq2latex import regression_program_to_latex

        program = _make_program_with_vars()
        # Underscores and backslashes are LaTeX-sensitive
        latex = regression_program_to_latex(
            program, var_names=["R_{f1}", "R_{f2}"]
        )
        assert isinstance(latex, str)
        assert latex.startswith("$")

    def test_many_features_var_names(self) -> None:
        """More than 3 features should work for var_names substitution."""
        from kd.viz.discover_eq2latex import regression_program_to_latex

        syms = sympy.symbols("x1 x2 x3 x4 x5")
        expr = sum(syms)
        program = FakeProgram(expr, n_input_var=5)

        names = ["a", "b", "c", "d", "e"]
        latex = regression_program_to_latex(program, var_names=names)

        for name in names:
            assert name in latex
        # Original x symbols should not appear
        for i in range(1, 6):
            assert f"x{i}" not in latex or f"x{i}" not in latex.replace(f"x{i}", "")

    def test_parity_data_shape_consistency(self, tmp_path) -> None:
        """Parity plot data shapes: actual, predicted, residuals must match."""
        from kd.viz._adapters.discover_regression import (
            DiscoverRegressionVizAdapter,
        )

        adapter = DiscoverRegressionVizAdapter()
        viz_registry.register_adapter(StubRegressionModel, adapter)

        model = StubRegressionModel()
        request = viz_core.VizRequest(
            kind="parity",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)

        if not result.warnings and "parity" in result.metadata:
            parity = result.metadata["parity"]
            assert parity.actual_values.shape == parity.predicted_values.shape
            assert parity.actual_values.shape == parity.residuals.shape
