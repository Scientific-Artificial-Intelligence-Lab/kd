"""Tests for EqGPT visualization adapter (Phase 1-3).

Covers:
- _eqgpt_to_latex() token-to-LaTeX conversion
- EqGPTVizAdapter protocol compliance and render dispatch
- equation rendering intent
- reward_ranking bar chart intent
- reward_evolution line chart intent
- parity plot intent (Phase 3)
- ParityPlotData integration
- result_ storage integration
- adapter registration and discovery
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from kd.viz import RewardEvolutionData
from kd.viz._adapters.eqgpt import EqGPTVizAdapter, _eqgpt_to_latex
from kd.viz._contracts import ParityPlotData
from kd.viz import core as viz_core
from kd.viz import registry as viz_registry
from kd.viz.core import VizRequest, VizResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_CAPABILITIES = {"equation", "reward_ranking", "reward_evolution", "parity"}

# Default parity data for happy-path tests.
# Values represent LHS (actual) and RHS (predicted) from an EqGPT equation fit.
# Independently chosen; NOT from any implementation output.
_DEFAULT_PARITY_LHS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
_DEFAULT_PARITY_RHS = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

# Representative token -> LaTeX mappings derived from the spec and vocabulary.
# These are independently determined from the task description, NOT from code.
SIMPLE_DERIVATIVE_CASES = [
    ("ut", "u_t"),
    ("ux", "u_x"),
    ("uxx", "u_{xx}"),
    ("uxxx", "u_{xxx}"),
    ("uxxxx", "u_{xxxx}"),
    ("uxxxxx", "u_{xxxxx}"),
    ("uy", "u_y"),
    ("uyy", "u_{yy}"),
    ("uyyy", "u_{yyy}"),
    ("uz", "u_z"),
    ("uzz", "u_{zz}"),
    ("uxt", "u_{xt}"),
    ("utt", "u_{tt}"),
    ("uxy", "u_{xy}"),
    ("uxxt", "u_{xxt}"),
    ("uxxtt", "u_{xxtt}"),
    ("uyyt", "u_{yyt}"),
]

POWER_CASES = [
    ("u^2", "u^{2}"),
    ("u^3", "u^{3}"),
    ("ut^2", "u_t^{2}"),
    ("ut^3", "u_t^{3}"),
    ("ux^2", "u_x^{2}"),
    ("uy^2", "u_y^{2}"),
]

COMPOSITE_CASES = [
    ("(uux)x", "(u u_x)_x"),
    ("(uux)t", "(u u_x)_t"),
    ("(uux)xx", "(u u_x)_{xx}"),
    ("(u^4)xx", "(u^{4})_{xx}"),
    ("(u^3)xx", "(u^{3})_{xx}"),
    ("(1/u)xx", "(1/u)_{xx}"),
    ("(u^-2*ux)x", "(u^{-2} u_x)_x"),
    ("(u(u^2)xx)xx", "(u (u^{2})_{xx})_{xx}"),
]

FUNCTION_CASES = [
    ("Laplace(u)", "\\nabla^2 u"),
    ("BiLaplace(u)", "\\nabla^4 u"),
    ("Laplace(utt)", "\\nabla^2 u_{tt}"),
    ("sin(u)", "\\sin(u)"),
    ("sinh(u)", "\\sinh(u)"),
    ("sqrt(u)", "\\sqrt{u}"),
    ("exp(x)", "\\exp(x)"),
    ("exp(-y)", "\\exp(-y)"),
    ("sqrt(x)", "\\sqrt{x}"),
]

COORDINATE_CASES = [
    ("x", "x"),
    ("y", "y"),
    ("t", "t"),
    ("x^2", "x^{2}"),
    ("y^2", "y^{2}"),
    ("x^4", "x^{4}"),
    ("(x+y)", "(x+y)"),
]

OPERATOR_CASES = [
    ("+", "+"),
    ("*", "\\cdot"),
    ("/", "/"),
]

SPECIAL_CASES = [
    ("sinx", "\\sin(x)"),
    ("sint", "\\sin(t)"),
    ("(uxx+ux/x)^2", "(u_{xx}+u_x/x)^{2}"),
]


# ---------------------------------------------------------------------------
# Mock / Stub helpers
# ---------------------------------------------------------------------------
class StubEqGPT:
    """Lightweight mock of KD_EqGPT for testing viz adapter.

    Does NOT import torch or any heavy dependencies.
    """

    def __init__(
        self,
        equations: Optional[List[str]] = None,
        rewards: Optional[List[float]] = None,
        best_equation: Optional[str] = None,
        best_reward: Optional[float] = None,
        reward_history: Optional[List[List[float]]] = None,
        parity_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        if equations is None:
            equations = [
                "ut+u*ux+uxx",
                "ut+u^2*ux",
                "ut+ux+uxx",
                "ut+Laplace(u)",
                "ut+u*ux",
            ]
        if rewards is None:
            rewards = [0.95, 0.88, 0.82, 0.75, 0.70]
        if best_equation is None:
            best_equation = equations[0]
        if best_reward is None:
            best_reward = rewards[0]
        if reward_history is None:
            reward_history = [
                [0.75, 0.72, 0.70, 0.68, 0.66],
                [0.83, 0.80, 0.78, 0.76, 0.74],
                [0.95, 0.90, 0.88, 0.85, 0.82],
            ]

        self.result_: Dict[str, Any] = {
            "equations": equations,
            "rewards": rewards,
            "best_equation": best_equation,
            "best_reward": best_reward,
            "reward_history": reward_history,
        }
        if parity_data is not None:
            self.result_["parity_data"] = parity_data


class StubEqGPTNoResult:
    """Mock EqGPT model with no result_ attribute."""
    pass


class StubEqGPTEmptyResult:
    """Mock EqGPT model with empty equations list."""

    def __init__(self) -> None:
        self.result_: Dict[str, Any] = {
            "equations": [],
            "rewards": [],
            "best_equation": "",
            "best_reward": 0.0,
            "reward_history": [],
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear viz registry before and after every test (class or function)."""
    viz_registry.clear_registry()
    yield
    viz_registry.clear_registry()


@pytest.fixture(autouse=True)
def suppress_show(monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)


def _register_eqgpt() -> EqGPTVizAdapter:
    adapter = EqGPTVizAdapter()
    viz_registry.register_adapter(StubEqGPT, adapter)
    return adapter


# ===================================================================
# A. _eqgpt_to_latex() -- Token -> LaTeX conversion
# ===================================================================
class TestEqGPTToLatex:
    """Token-to-LaTeX conversion tests."""

    # --- Smoke ---
    def test_smoke_basic_conversion(self) -> None:
        """Basic call returns a string without raising."""
        result = _eqgpt_to_latex("ut")
        assert isinstance(result, str)
        assert len(result) > 0

    # --- Happy path: simple derivatives ---
    @pytest.mark.parametrize("token,expected", SIMPLE_DERIVATIVE_CASES)
    def test_simple_derivatives(self, token: str, expected: str) -> None:
        """Single derivative tokens map to subscript notation."""
        result = _eqgpt_to_latex(token)
        assert result == expected

    # --- Happy path: powers ---
    @pytest.mark.parametrize("token,expected", POWER_CASES)
    def test_power_tokens(self, token: str, expected: str) -> None:
        """Power tokens use braces for exponents."""
        result = _eqgpt_to_latex(token)
        assert result == expected

    # --- Happy path: composite terms ---
    @pytest.mark.parametrize("token,expected", COMPOSITE_CASES)
    def test_composite_terms(self, token: str, expected: str) -> None:
        """Composite derivative tokens expand inner products."""
        result = _eqgpt_to_latex(token)
        assert result == expected

    # --- Happy path: function tokens ---
    @pytest.mark.parametrize("token,expected", FUNCTION_CASES)
    def test_function_tokens(self, token: str, expected: str) -> None:
        """Mathematical function tokens use LaTeX commands."""
        result = _eqgpt_to_latex(token)
        assert result == expected

    # --- Happy path: coordinates ---
    @pytest.mark.parametrize("token,expected", COORDINATE_CASES)
    def test_coordinate_tokens(self, token: str, expected: str) -> None:
        """Coordinate tokens are preserved or minimally transformed."""
        result = _eqgpt_to_latex(token)
        assert result == expected

    # --- Happy path: operators ---
    @pytest.mark.parametrize("token,expected", OPERATOR_CASES)
    def test_operator_tokens(self, token: str, expected: str) -> None:
        """Operators map to LaTeX equivalents."""
        result = _eqgpt_to_latex(token)
        assert result == expected

    # --- Happy path: special tokens ---
    @pytest.mark.parametrize("token,expected", SPECIAL_CASES)
    def test_special_tokens(self, token: str, expected: str) -> None:
        """Special combined tokens map correctly."""
        result = _eqgpt_to_latex(token)
        assert result == expected

    # --- Happy path: full equation strings ---
    # _format_results() joins tokens with "".join(), and operators (+, *, /)
    # are themselves vocabulary tokens. So the input to _eqgpt_to_latex is a
    # concatenated string like "ut+u*ux+uxx" where operators naturally
    # separate the other tokens. The function splits on these delimiters.
    @pytest.mark.parametrize(
        "eq,expected",
        [
            ("ut+u*ux+uxx", "u_t+u \\cdot u_x+u_{xx}"),
            ("ut+(uux)x+uxx", "u_t+(u u_x)_x+u_{xx}"),
            ("Laplace(u)+u^2*ux", "\\nabla^2 u+u^{2} \\cdot u_x"),
            ("ut+sin(u)*ux+uxxxx", "u_t+\\sin(u) \\cdot u_x+u_{xxxx}"),
        ],
        ids=[
            "burgers-like",
            "composite-derivative",
            "laplacian-with-power",
            "function-with-high-deriv",
        ],
    )
    def test_full_equation_string(self, eq: str, expected: str) -> None:
        """Concatenated equation string is precisely converted to LaTeX."""
        result = _eqgpt_to_latex(eq)
        assert result == expected

    # --- Edge: empty string ---
    def test_empty_string(self) -> None:
        """Empty string returns empty string (no crash)."""
        result = _eqgpt_to_latex("")
        assert result == ""

    # --- Edge: single variable token ---
    def test_single_variable(self) -> None:
        """Single variable 'u' maps to 'u'."""
        result = _eqgpt_to_latex("u")
        assert result == "u"

    # --- Edge: unknown token passthrough ---
    def test_unknown_token_passthrough(self) -> None:
        """Unknown tokens are passed through as-is, not silently dropped."""
        result = _eqgpt_to_latex("UNKNOWN_TOKEN_XYZ")
        assert "UNKNOWN_TOKEN_XYZ" in result

    # --- Edge: control tokens ---
    def test_pad_token_handling(self) -> None:
        """<pad> tokens should be stripped to empty string."""
        result = _eqgpt_to_latex("<pad>")
        assert result == ""

    def test_start_marker_stripped(self) -> None:
        """'S' (start) marker should be stripped to empty string."""
        assert _eqgpt_to_latex("S") == ""

    def test_end_marker_stripped(self) -> None:
        """'E' (end) marker should be stripped to empty string."""
        assert _eqgpt_to_latex("E") == ""


# ===================================================================
# B. EqGPTVizAdapter -- Protocol compliance
# ===================================================================
class TestEqGPTAdapterProtocol:
    """Adapter protocol compliance tests."""

    # --- Smoke ---
    def test_smoke_has_capabilities_and_render(self) -> None:
        """Adapter exposes capabilities and render method."""
        adapter = EqGPTVizAdapter()
        assert hasattr(adapter, "capabilities")
        assert hasattr(adapter, "render")
        assert callable(adapter.render)

    def test_correct_capabilities(self) -> None:
        """Adapter declares exactly the expected capabilities."""
        adapter = EqGPTVizAdapter()
        assert set(adapter.capabilities) == EXPECTED_CAPABILITIES

    # --- Edge: unsupported kind returns warning ---
    def test_unsupported_kind_returns_warning(self, tmp_path: Path) -> None:
        """Requesting an unsupported kind via core.render returns VizResult with warning."""
        _register_eqgpt()
        model = StubEqGPT()
        request = VizRequest(
            kind="nonexistent_kind",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.warnings
        assert "nonexistent_kind" in result.warnings[0].lower() or \
               "does not support" in result.warnings[0].lower()


# ===================================================================
# C. equation rendering intent
# ===================================================================
class TestEquationRendering:
    """Tests for the 'equation' intent."""

    # --- Smoke ---
    def test_smoke_equation_renders(self, tmp_path: Path) -> None:
        """Equation intent returns a VizResult (may have paths or warnings)."""
        _register_eqgpt()
        model = StubEqGPT(best_equation="ut+u*ux")
        request = VizRequest(
            kind="equation",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.intent == "equation"

    # --- Happy path ---
    def test_equation_produces_output_file(self, tmp_path: Path) -> None:
        """Equation rendering produces a file on disk."""
        _register_eqgpt()
        model = StubEqGPT(best_equation="ut+uxx")
        request = VizRequest(
            kind="equation",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert result.paths, "Expected at least one output path"
        for p in result.paths:
            assert p.exists(), f"Output file {p} should exist"

    def test_equation_metadata_contains_latex(self, tmp_path: Path) -> None:
        """Equation result metadata includes 'latex' key."""
        _register_eqgpt()
        model = StubEqGPT(best_equation="ut+u*ux+uxx")
        request = VizRequest(
            kind="equation",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert "latex" in result.metadata
        latex = result.metadata["latex"]
        assert isinstance(latex, str)
        assert len(latex) > 0

    def test_equation_reads_from_result_(self, tmp_path: Path) -> None:
        """Adapter reads best_equation from model.result_ dict."""
        _register_eqgpt()
        model = StubEqGPT(best_equation="Laplace(u)+u*ux")
        request = VizRequest(
            kind="equation",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        # The LaTeX should contain the nabla symbol from Laplace(u)
        assert "latex" in result.metadata, "metadata must contain 'latex' key"
        assert "\\nabla" in result.metadata["latex"]

    # --- Edge: model has no result_ ---
    def test_equation_no_result_returns_warning(self, tmp_path: Path) -> None:
        """Missing result_ attribute produces a warning, not a crash."""
        adapter = EqGPTVizAdapter()
        viz_registry.register_adapter(StubEqGPTNoResult, adapter)
        model = StubEqGPTNoResult()
        request = VizRequest(
            kind="equation",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        # Either warns or produces empty result, but must not crash
        assert result.warnings or not result.has_content

    # --- Edge: empty best_equation ---
    def test_equation_empty_best_eq(self, tmp_path: Path) -> None:
        """Empty best_equation string produces graceful result."""
        adapter = EqGPTVizAdapter()
        viz_registry.register_adapter(StubEqGPTEmptyResult, adapter)
        model = StubEqGPTEmptyResult()
        request = VizRequest(
            kind="equation",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        # Should not crash; warning or minimal output is acceptable


# ===================================================================
# D. reward_ranking bar chart intent
# ===================================================================
class TestRewardRanking:
    """Tests for the 'reward_ranking' bar chart intent."""

    # --- Smoke ---
    def test_smoke_reward_ranking(self, tmp_path: Path) -> None:
        """reward_ranking intent returns VizResult."""
        _register_eqgpt()
        model = StubEqGPT()
        request = VizRequest(
            kind="reward_ranking",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.intent == "reward_ranking"

    # --- Happy path ---
    def test_reward_ranking_produces_chart(self, tmp_path: Path) -> None:
        """reward_ranking produces an output file."""
        _register_eqgpt()
        model = StubEqGPT()
        request = VizRequest(
            kind="reward_ranking",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert result.paths, "Expected at least one output path for bar chart"
        for p in result.paths:
            assert p.exists()
            assert p.suffix == ".png"

    def test_reward_ranking_correct_bar_count(self, tmp_path: Path) -> None:
        """Bar chart has correct number of bars (5 equations -> 5 bars)."""
        _register_eqgpt()
        num_equations = 5
        equations = [f"eq_{i}" for i in range(num_equations)]
        rewards = [0.9 - i * 0.1 for i in range(num_equations)]
        model = StubEqGPT(equations=equations, rewards=rewards)
        request = VizRequest(
            kind="reward_ranking",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        # Metadata should report the count of bars rendered
        assert "n_bars" in result.metadata, "metadata must contain 'n_bars' key"
        assert result.metadata["n_bars"] == num_equations

    def test_reward_ranking_caps_at_ten(self, tmp_path: Path) -> None:
        """With more than 10 equations, only top 10 are shown."""
        _register_eqgpt()
        num_equations = 15
        equations = [f"eq_{i}" for i in range(num_equations)]
        rewards = [1.0 - i * 0.05 for i in range(num_equations)]
        model = StubEqGPT(equations=equations, rewards=rewards)
        request = VizRequest(
            kind="reward_ranking",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        max_bars = 10
        assert "n_bars" in result.metadata, "metadata must contain 'n_bars' key"
        assert result.metadata["n_bars"] <= max_bars

    def test_reward_ranking_fewer_than_ten(self, tmp_path: Path) -> None:
        """With 3 equations, shows all 3 (no padding or crash)."""
        _register_eqgpt()
        equations = ["ut+uxx", "ut+ux", "ut"]
        rewards = [0.9, 0.7, 0.5]
        model = StubEqGPT(equations=equations, rewards=rewards)
        request = VizRequest(
            kind="reward_ranking",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert result.paths, "Should produce chart even with <10 equations"
        assert "n_bars" in result.metadata, "metadata must contain 'n_bars' key"
        assert result.metadata["n_bars"] == len(equations)

    def test_reward_ranking_dynamic_yaxis(self, tmp_path: Path) -> None:
        """Y-axis range adapts to actual reward values (not hardcoded 0-1)."""
        _register_eqgpt()
        # Use rewards well outside [0, 1] to verify dynamic range
        equations = ["eq_a", "eq_b", "eq_c"]
        rewards = [50.0, 30.0, 10.0]
        model = StubEqGPT(equations=equations, rewards=rewards)
        request = VizRequest(
            kind="reward_ranking",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        # The chart must be produced without error even with large rewards
        assert result.paths, (
            "Dynamic y-axis should handle rewards outside [0, 1]"
        )

    # --- Edge: empty results ---
    def test_reward_ranking_empty_results(self, tmp_path: Path) -> None:
        """Empty equations list produces warning, no crash."""
        adapter = EqGPTVizAdapter()
        viz_registry.register_adapter(StubEqGPTEmptyResult, adapter)
        model = StubEqGPTEmptyResult()
        request = VizRequest(
            kind="reward_ranking",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        # Should warn or produce empty output, not crash
        assert result.warnings or not result.has_content

    # --- Edge: no result_ attribute ---
    def test_reward_ranking_no_result_(self, tmp_path: Path) -> None:
        """Model missing result_ produces warning, not crash."""
        adapter = EqGPTVizAdapter()
        viz_registry.register_adapter(StubEqGPTNoResult, adapter)
        model = StubEqGPTNoResult()
        request = VizRequest(
            kind="reward_ranking",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.warnings or not result.has_content

    # --- Edge: single equation ---
    def test_reward_ranking_single_equation(self, tmp_path: Path) -> None:
        """Single equation produces a valid bar chart."""
        _register_eqgpt()
        model = StubEqGPT(
            equations=["ut+uxx"],
            rewards=[0.99],
            best_equation="ut+uxx",
            best_reward=0.99,
        )
        request = VizRequest(
            kind="reward_ranking",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert result.paths, "Single-equation chart should still produce output"

    # --- Edge: negative rewards ---
    def test_reward_ranking_negative_rewards(self, tmp_path: Path) -> None:
        """Negative reward values do not crash the chart."""
        _register_eqgpt()
        equations = ["eq_a", "eq_b"]
        rewards = [-0.5, -1.2]
        model = StubEqGPT(equations=equations, rewards=rewards)
        request = VizRequest(
            kind="reward_ranking",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        # Must not crash; chart or warning is acceptable
        assert isinstance(result, VizResult)

    # --- Edge: NaN / Inf rewards ---
    def test_reward_ranking_nan_reward(self, tmp_path: Path) -> None:
        """NaN in rewards produces warning about non-finite values."""
        _register_eqgpt()
        equations = ["eq_a", "eq_b"]
        rewards = [0.9, float("nan")]
        model = StubEqGPT(equations=equations, rewards=rewards)
        request = VizRequest(
            kind="reward_ranking",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.warnings, "NaN reward should produce a warning"

    def test_reward_ranking_inf_reward(self, tmp_path: Path) -> None:
        """Inf in rewards produces warning about non-finite values."""
        _register_eqgpt()
        equations = ["eq_a", "eq_b"]
        rewards = [0.9, float("inf")]
        model = StubEqGPT(equations=equations, rewards=rewards)
        request = VizRequest(
            kind="reward_ranking",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.warnings, "Inf reward should produce a warning"

    def test_reward_ranking_all_nan(self, tmp_path: Path) -> None:
        """All-NaN rewards produces warning, no crash."""
        _register_eqgpt()
        equations = ["eq_a", "eq_b", "eq_c"]
        rewards = [float("nan"), float("nan"), float("nan")]
        model = StubEqGPT(equations=equations, rewards=rewards)
        request = VizRequest(
            kind="reward_ranking",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.warnings, "All-NaN rewards should produce a warning"

    def test_reward_ranking_all_inf(self, tmp_path: Path) -> None:
        """All +inf rewards produces warning, no crash."""
        _register_eqgpt()
        equations = ["eq_a", "eq_b", "eq_c"]
        rewards = [float("inf"), float("inf"), float("inf")]
        model = StubEqGPT(equations=equations, rewards=rewards)
        request = VizRequest(
            kind="reward_ranking",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.warnings, "All-inf rewards should produce a warning"

    def test_reward_ranking_neg_inf(self, tmp_path: Path) -> None:
        """Negative infinity in rewards produces warning, no crash."""
        _register_eqgpt()
        equations = ["eq_a", "eq_b", "eq_c"]
        rewards = [0.9, float("-inf"), 0.5]
        model = StubEqGPT(equations=equations, rewards=rewards)
        request = VizRequest(
            kind="reward_ranking",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.warnings, "-inf reward should produce a warning"

    def test_reward_ranking_all_same(self, tmp_path: Path) -> None:
        """All identical rewards produces a valid chart (no crash)."""
        _register_eqgpt()
        equations = ["eq_a", "eq_b", "eq_c"]
        rewards = [0.5, 0.5, 0.5]
        model = StubEqGPT(equations=equations, rewards=rewards)
        request = VizRequest(
            kind="reward_ranking",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.paths, "Identical rewards should still produce a chart"

    def test_reward_ranking_all_zero(self, tmp_path: Path) -> None:
        """All-zero rewards produces valid chart, no division by zero."""
        _register_eqgpt()
        equations = ["eq_a", "eq_b", "eq_c"]
        rewards = [0.0, 0.0, 0.0]
        model = StubEqGPT(equations=equations, rewards=rewards)
        request = VizRequest(
            kind="reward_ranking",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.paths, "All-zero rewards should still produce a chart"

    def test_reward_ranking_length_mismatch(self, tmp_path: Path) -> None:
        """Mismatched equations/rewards lengths produces warning, no crash."""
        _register_eqgpt()
        equations = ["eq_a", "eq_b", "eq_c"]
        rewards = [0.9, 0.8]  # 3 equations but only 2 rewards
        model = StubEqGPT(equations=equations, rewards=rewards)
        request = VizRequest(
            kind="reward_ranking",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.warnings, "Length mismatch should produce a warning"


# ===================================================================
# E. reward_evolution line chart intent
# ===================================================================
class TestRewardEvolution:
    """Tests for the 'reward_evolution' line chart intent."""

    def test_smoke_reward_evolution(self, tmp_path: Path) -> None:
        """reward_evolution intent returns a VizResult."""
        _register_eqgpt()
        model = StubEqGPT()
        request = VizRequest(
            kind="reward_evolution",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.intent == "reward_evolution"

    def test_reward_evolution_produces_chart_and_contract(
        self, tmp_path: Path
    ) -> None:
        """reward_evolution writes a chart and returns RewardEvolutionData."""
        _register_eqgpt()
        model = StubEqGPT(
            reward_history=[
                [0.70, 0.65, 0.60, 0.55, 0.50],
                [0.80, 0.76, 0.72, 0.68, 0.64],
                [0.90, 0.87, 0.84, 0.81, 0.78],
            ]
        )
        request = VizRequest(
            kind="reward_evolution",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert result.paths
        assert result.paths[0].exists()
        assert "reward_evolution" in result.metadata
        data = result.metadata["reward_evolution"]
        assert isinstance(data, RewardEvolutionData)
        assert data.steps.tolist() == [1, 2, 3]
        assert np.allclose(data.best_reward, [0.70, 0.80, 0.90])

    def test_reward_evolution_missing_history_warns(
        self, tmp_path: Path
    ) -> None:
        """Missing reward_history produces a warning, not a crash."""
        _register_eqgpt()
        model = StubEqGPT()
        model.result_.pop("reward_history")
        request = VizRequest(
            kind="reward_evolution",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert result.warnings

    def test_reward_evolution_empty_history_warns(
        self, tmp_path: Path
    ) -> None:
        """Empty reward history produces a warning, not a chart."""
        _register_eqgpt()
        model = StubEqGPT(reward_history=[])
        request = VizRequest(
            kind="reward_evolution",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert result.warnings

    def test_reward_evolution_filters_nonfinite_values(
        self, tmp_path: Path
    ) -> None:
        """NaN/Inf entries are filtered while finite epochs still plot."""
        _register_eqgpt()
        model = StubEqGPT(
            reward_history=[
                [0.70, 0.60, 0.50, 0.40, 0.30],
                [float("nan"), float("inf"), 0.72, 0.62, 0.52],
                [0.90, 0.80, 0.70, 0.60, 0.50],
            ]
        )
        request = VizRequest(
            kind="reward_evolution",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert result.paths
        assert result.warnings
        data = result.metadata["reward_evolution"]
        assert isinstance(data, RewardEvolutionData)
        assert np.isfinite(data.best_reward[[0, 2]]).all()


# ===================================================================
# F. result_ storage integration
# ===================================================================
class TestResultStorage:
    """Test adapter reads model.result_ correctly."""

    def test_adapter_reads_equations_from_result_(self) -> None:
        """Adapter accesses model.result_['equations']."""
        model = StubEqGPT(
            equations=["ut+uxx", "ut+ux"],
            rewards=[0.9, 0.8],
        )
        assert hasattr(model, "result_")
        assert model.result_["equations"] == ["ut+uxx", "ut+ux"]
        assert model.result_["rewards"] == [0.9, 0.8]

    def test_adapter_reads_best_equation(self) -> None:
        """Adapter accesses model.result_['best_equation']."""
        model = StubEqGPT(best_equation="ut+Laplace(u)")
        assert model.result_["best_equation"] == "ut+Laplace(u)"

    def test_adapter_reads_best_reward(self) -> None:
        """Adapter accesses model.result_['best_reward']."""
        model = StubEqGPT(best_reward=0.95)
        assert model.result_["best_reward"] == pytest.approx(0.95)

    def test_result_dict_has_all_keys(self) -> None:
        """result_ dict contains all required keys."""
        model = StubEqGPT()
        required_keys = {
            "equations",
            "rewards",
            "best_equation",
            "best_reward",
            "reward_history",
        }
        assert required_keys.issubset(model.result_.keys())


# ===================================================================
# G. Registration and discovery
# ===================================================================
class TestRegistration:
    """Test adapter registration and capability discovery."""

    def test_list_capabilities_after_registration(self) -> None:
        """list_capabilities returns adapter capabilities after registration."""
        _register_eqgpt()
        model = StubEqGPT()
        caps = viz_core.list_capabilities(model)
        assert set(caps) == EXPECTED_CAPABILITIES

    def test_render_dispatches_to_adapter(self, tmp_path: Path) -> None:
        """render() dispatches to the EqGPT adapter for equation kind."""
        _register_eqgpt()
        model = StubEqGPT(best_equation="ut+uxx")
        request = VizRequest(
            kind="equation",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        # Should not get "No visualization adapter registered" warning
        if result.warnings:
            for w in result.warnings:
                assert "No visualization adapter registered" not in w

    def test_unregistered_model_returns_no_adapter_warning(self, tmp_path: Path) -> None:
        """Model class without registered adapter gets appropriate warning."""
        # Do NOT register any adapter
        model = StubEqGPTNoResult()
        request = VizRequest(
            kind="equation",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert result.warnings
        assert "No visualization adapter registered" in result.warnings[0]

    def test_capabilities_empty_for_unregistered(self) -> None:
        """list_capabilities returns empty for unregistered model."""
        model = StubEqGPTNoResult()
        caps = viz_core.list_capabilities(model)
        assert len(list(caps)) == 0

    def test_adapter_does_not_interfere_with_other_adapters(self) -> None:
        """Registering EqGPT adapter does not break other adapters."""
        _register_eqgpt()

        # Register a dummy adapter for a different model
        class DummyModel:
            pass

        class DummyAdapter:
            capabilities = {"dummy_cap"}

            def render(self, request, ctx):
                return VizResult(intent="dummy_cap")

        viz_registry.register_adapter(DummyModel, DummyAdapter())

        # Both should work independently
        eqgpt_caps = viz_core.list_capabilities(StubEqGPT())
        dummy_caps = viz_core.list_capabilities(DummyModel())
        assert set(eqgpt_caps) == EXPECTED_CAPABILITIES
        assert set(dummy_caps) == {"dummy_cap"}


# ===================================================================
# H. Parity plot intent (Phase 3)
# ===================================================================
class TestParity:
    """Tests for the 'parity' scatter plot intent.

    Parity reads ``model.result_["parity_data"]`` (a dict with "lhs" and "rhs"
    numpy arrays), constructs a ParityPlotData, renders LHS-vs-RHS scatter +
    y=x reference line, and returns residual statistics in metadata.
    """

    # ------------------------------------------------------------------
    # A. Adapter capability
    # ------------------------------------------------------------------
    def test_parity_in_capabilities(self) -> None:
        """'parity' is declared in adapter capabilities."""
        adapter = EqGPTVizAdapter()
        assert "parity" in adapter.capabilities

    # ------------------------------------------------------------------
    # B. Smoke tests
    # ------------------------------------------------------------------
    def test_parity_smoke(self, tmp_path: Path) -> None:
        """Parity intent returns a successful VizResult with content."""
        _register_eqgpt()
        model = StubEqGPT(
            parity_data={
                "lhs": _DEFAULT_PARITY_LHS,
                "rhs": _DEFAULT_PARITY_RHS,
            }
        )
        request = VizRequest(
            kind="parity",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.intent == "parity"
        # Smoke: parity with valid data must produce content (chart)
        assert result.has_content, "Parity with valid data should produce content"

    # ------------------------------------------------------------------
    # C. Happy path
    # ------------------------------------------------------------------
    def test_parity_produces_chart(self, tmp_path: Path) -> None:
        """Parity intent writes a PNG chart to disk."""
        _register_eqgpt()
        model = StubEqGPT(
            parity_data={
                "lhs": _DEFAULT_PARITY_LHS,
                "rhs": _DEFAULT_PARITY_RHS,
            }
        )
        request = VizRequest(
            kind="parity",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert result.paths, "Expected at least one output path for parity chart"
        for p in result.paths:
            assert p.exists(), f"Output file {p} should exist"
            assert p.suffix == ".png"

    def test_parity_metadata_has_parity_data(self, tmp_path: Path) -> None:
        """Parity result metadata contains a ParityPlotData instance."""
        _register_eqgpt()
        model = StubEqGPT(
            parity_data={
                "lhs": _DEFAULT_PARITY_LHS,
                "rhs": _DEFAULT_PARITY_RHS,
            }
        )
        request = VizRequest(
            kind="parity",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert "parity" in result.metadata, "metadata must contain 'parity' key"
        parity = result.metadata["parity"]
        assert isinstance(parity, ParityPlotData)

    def test_parity_metadata_has_residual_stats(self, tmp_path: Path) -> None:
        """Parity metadata includes rmse, mean_residual, max_abs_residual."""
        _register_eqgpt()
        lhs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rhs = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        model = StubEqGPT(parity_data={"lhs": lhs, "rhs": rhs})
        request = VizRequest(
            kind="parity",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert "summary" in result.metadata, "metadata must contain 'summary' key"
        summary = result.metadata["summary"]
        assert "rmse" in summary
        assert "mean_residual" in summary
        assert "max_abs_residual" in summary
        # Verify residual stats are finite and sensible
        assert np.isfinite(summary["rmse"])
        assert np.isfinite(summary["mean_residual"])
        assert np.isfinite(summary["max_abs_residual"])

    def test_parity_residual_stats_numerically_correct(
        self, tmp_path: Path
    ) -> None:
        """Residual statistics match independently computed values."""
        _register_eqgpt()
        lhs = np.array([1.0, 2.0, 3.0])
        rhs = np.array([1.5, 2.5, 3.5])
        # residuals = actual - predicted = lhs - rhs = [-0.5, -0.5, -0.5]
        expected_mean = -0.5
        expected_rmse = 0.5
        expected_max_abs = 0.5
        model = StubEqGPT(parity_data={"lhs": lhs, "rhs": rhs})
        request = VizRequest(
            kind="parity",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        summary = result.metadata["summary"]
        assert np.isclose(summary["mean_residual"], expected_mean, atol=1e-10)
        assert np.isclose(summary["rmse"], expected_rmse, atol=1e-10)
        assert np.isclose(
            summary["max_abs_residual"], expected_max_abs, atol=1e-10
        )

    def test_parity_large_dataset(self, tmp_path: Path) -> None:
        """Parity handles a larger dataset (100 points) without issues."""
        _register_eqgpt()
        rng = np.random.RandomState(42)
        lhs = rng.randn(100)
        rhs = lhs + 0.1 * rng.randn(100)  # close to perfect fit
        model = StubEqGPT(parity_data={"lhs": lhs, "rhs": rhs})
        request = VizRequest(
            kind="parity",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert result.paths
        parity = result.metadata["parity"]
        assert isinstance(parity, ParityPlotData)
        assert parity.actual_values.shape == (100,)
        assert parity.predicted_values.shape == (100,)

    # ------------------------------------------------------------------
    # D. Edge cases -- missing / invalid data
    # ------------------------------------------------------------------
    def test_parity_no_result_(self, tmp_path: Path) -> None:
        """Model without result_ attribute produces warning about result_."""
        adapter = EqGPTVizAdapter()
        viz_registry.register_adapter(StubEqGPTNoResult, adapter)
        model = StubEqGPTNoResult()
        request = VizRequest(
            kind="parity",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.warnings
        # Must warn specifically about result_, not "does not support"
        combined = " ".join(result.warnings).lower()
        assert "result_" in combined or "result" in combined

    def test_parity_no_parity_data(self, tmp_path: Path) -> None:
        """result_ exists but no 'parity_data' key produces warning about parity data."""
        _register_eqgpt()
        model = StubEqGPT()  # no parity_data
        assert "parity_data" not in model.result_
        request = VizRequest(
            kind="parity",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.warnings, "Missing parity_data should produce a warning"
        # Must warn specifically about parity *data*, not "does not support intent"
        combined = " ".join(result.warnings).lower()
        assert "parity_data" in combined or "parity data" in combined

    def test_parity_empty_arrays(self, tmp_path: Path) -> None:
        """parity_data with empty lhs/rhs arrays produces warning about empty."""
        _register_eqgpt()
        model = StubEqGPT(
            parity_data={
                "lhs": np.array([]),
                "rhs": np.array([]),
            }
        )
        request = VizRequest(
            kind="parity",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.warnings, "Empty parity arrays should produce a warning"
        combined = " ".join(result.warnings).lower()
        assert "empty" in combined or "no " in combined

    def test_parity_nan_in_data(self, tmp_path: Path) -> None:
        """parity_data containing NaN values is handled gracefully.

        The adapter must either filter NaNs and produce a chart, or warn about
        non-finite data. It must not crash and must not silently produce broken output.
        """
        _register_eqgpt()
        lhs = np.array([1.0, np.nan, 3.0, 4.0])
        rhs = np.array([1.1, 2.0, np.nan, 4.1])
        model = StubEqGPT(parity_data={"lhs": lhs, "rhs": rhs})
        request = VizRequest(
            kind="parity",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.intent == "parity"
        # Must produce a chart (possibly with filtered finite points) or a
        # meaningful warning -- but not an "unsupported intent" message.
        if result.warnings:
            combined = " ".join(result.warnings).lower()
            assert "does not support" not in combined, (
                "NaN handling must be done by parity handler, not fall through "
                "to unsupported intent"
            )
        # If it produces a chart, ParityPlotData should exist in metadata
        if result.paths:
            assert "parity" in result.metadata
            parity = result.metadata["parity"]
            assert isinstance(parity, ParityPlotData)

    def test_parity_inf_in_data(self, tmp_path: Path) -> None:
        """parity_data containing Inf values is handled gracefully."""
        _register_eqgpt()
        lhs = np.array([1.0, np.inf, 3.0])
        rhs = np.array([1.1, 2.0, -np.inf])
        model = StubEqGPT(parity_data={"lhs": lhs, "rhs": rhs})
        request = VizRequest(
            kind="parity",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.intent == "parity"
        # Must produce either a chart or a meaningful warning --
        # must not fall through to "unsupported intent".
        if result.warnings:
            combined = " ".join(result.warnings).lower()
            assert "does not support" not in combined, (
                "Inf handling must be done by parity handler, not fall through "
                "to unsupported intent"
            )

    def test_parity_all_nan(self, tmp_path: Path) -> None:
        """All-NaN parity data produces warning about non-finite, no chart."""
        _register_eqgpt()
        lhs = np.array([np.nan, np.nan, np.nan])
        rhs = np.array([np.nan, np.nan, np.nan])
        model = StubEqGPT(parity_data={"lhs": lhs, "rhs": rhs})
        request = VizRequest(
            kind="parity",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.intent == "parity"
        assert result.warnings, "All-NaN parity data should produce a warning"
        combined = " ".join(result.warnings).lower()
        assert "nan" in combined or "finite" in combined or "no " in combined

    def test_parity_shape_mismatch(self, tmp_path: Path) -> None:
        """lhs and rhs with different lengths produces warning about shape."""
        _register_eqgpt()
        lhs = np.array([1.0, 2.0, 3.0])
        rhs = np.array([1.1, 1.9])  # shorter than lhs
        model = StubEqGPT(parity_data={"lhs": lhs, "rhs": rhs})
        request = VizRequest(
            kind="parity",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.intent == "parity"
        assert result.warnings, "Shape mismatch should produce a warning"
        combined = " ".join(result.warnings).lower()
        assert "shape" in combined or "mismatch" in combined or "length" in combined

    def test_parity_single_point(self, tmp_path: Path) -> None:
        """Parity with a single data point produces a chart."""
        _register_eqgpt()
        lhs = np.array([3.14])
        rhs = np.array([3.15])
        model = StubEqGPT(parity_data={"lhs": lhs, "rhs": rhs})
        request = VizRequest(
            kind="parity",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.intent == "parity"
        # Single data point should still produce a chart file
        assert result.paths, "Single-point parity should produce a chart"

    def test_parity_parity_data_none_value(self, tmp_path: Path) -> None:
        """parity_data key exists but value is None produces warning about parity."""
        _register_eqgpt()
        model = StubEqGPT()
        model.result_["parity_data"] = None
        request = VizRequest(
            kind="parity",
            target=model,
            options={"output_dir": tmp_path},
        )
        result = viz_core.render(request)
        assert isinstance(result, VizResult)
        assert result.intent == "parity"
        assert result.warnings, "None parity_data should produce a warning"
        # Must warn about parity data specifically, not "unsupported intent"
        combined = " ".join(result.warnings).lower()
        assert "does not support" not in combined, (
            "None parity_data must be handled by parity handler"
        )
        assert "parity" in combined


# ===================================================================
# I. ParityPlotData integration (Phase 3)
# ===================================================================
class TestParityDataIntegration:
    """Tests verifying ParityPlotData creation from parity_data arrays.

    These test the contract between EqGPT's parity output and the
    ParityPlotData data class, independently of the adapter rendering.
    """

    def test_parity_data_from_actual_predicted(self) -> None:
        """ParityPlotData.from_actual_predicted works with typical lhs/rhs."""
        lhs = _DEFAULT_PARITY_LHS
        rhs = _DEFAULT_PARITY_RHS
        parity = ParityPlotData.from_actual_predicted(lhs, rhs)
        assert isinstance(parity, ParityPlotData)
        np.testing.assert_array_equal(parity.actual_values, lhs)
        np.testing.assert_array_equal(parity.predicted_values, rhs)

    def test_parity_data_residuals_correct(self) -> None:
        """Residuals equal actual - predicted (lhs - rhs)."""
        lhs = np.array([10.0, 20.0, 30.0])
        rhs = np.array([11.0, 19.0, 31.0])
        parity = ParityPlotData.from_actual_predicted(lhs, rhs)
        expected_residuals = lhs - rhs  # [-1.0, 1.0, -1.0]
        np.testing.assert_allclose(
            parity.residuals, expected_residuals, atol=1e-12
        )

    def test_parity_data_arrays_are_1d(self) -> None:
        """ParityPlotData reshapes inputs to 1D."""
        # 2D inputs should be flattened to 1D
        lhs_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        rhs_2d = np.array([[1.1, 2.1], [3.1, 4.1]])
        parity = ParityPlotData.from_actual_predicted(lhs_2d, rhs_2d)
        assert parity.actual_values.ndim == 1
        assert parity.predicted_values.ndim == 1
        assert parity.residuals.ndim == 1
        assert parity.actual_values.shape == (4,)

    def test_parity_data_shape_mismatch_raises(self) -> None:
        """ParityPlotData raises ValueError on shape mismatch."""
        lhs = np.array([1.0, 2.0, 3.0])
        rhs = np.array([1.0, 2.0])
        # from_actual_predicted computes residuals = actual - predicted,
        # which triggers numpy broadcast error for incompatible shapes.
        # Either numpy raises or __post_init__ raises -- both ValueError.
        with pytest.raises(ValueError):
            ParityPlotData.from_actual_predicted(lhs, rhs)

    def test_parity_data_metadata_passthrough(self) -> None:
        """Metadata dict is preserved through factory method."""
        meta = {"source": "eqgpt", "case_count": 5}
        parity = ParityPlotData.from_actual_predicted(
            _DEFAULT_PARITY_LHS,
            _DEFAULT_PARITY_RHS,
            metadata=meta,
        )
        assert parity.metadata == meta

    def test_result_contains_parity_data(self) -> None:
        """StubEqGPT with parity_data stores it in result_ dict."""
        parity_dict = {
            "lhs": _DEFAULT_PARITY_LHS,
            "rhs": _DEFAULT_PARITY_RHS,
        }
        model = StubEqGPT(parity_data=parity_dict)
        assert "parity_data" in model.result_
        pd = model.result_["parity_data"]
        np.testing.assert_array_equal(pd["lhs"], _DEFAULT_PARITY_LHS)
        np.testing.assert_array_equal(pd["rhs"], _DEFAULT_PARITY_RHS)

    def test_parity_data_lhs_rhs_are_1d_numpy(self) -> None:
        """parity_data lhs and rhs stored in result_ are 1D numpy arrays."""
        parity_dict = {
            "lhs": _DEFAULT_PARITY_LHS,
            "rhs": _DEFAULT_PARITY_RHS,
        }
        model = StubEqGPT(parity_data=parity_dict)
        pd = model.result_["parity_data"]
        lhs = np.asarray(pd["lhs"])
        rhs = np.asarray(pd["rhs"])
        assert lhs.ndim == 1
        assert rhs.ndim == 1
        assert lhs.shape == rhs.shape
