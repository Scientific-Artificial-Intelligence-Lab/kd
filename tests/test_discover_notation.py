"""TDD tests for DSCV LaTeX dual-mode notation rendering (RED phase).

Tests cover:
1. _SubscriptLatexPrinter unit tests (symbol + derivative rendering)
2. _discover_term_node_to_latex() with notation parameter
3. discover_program_to_latex() with notation parameter

All tests are expected to FAIL until the implementation is written.
"""

import pytest
import sympy
from sympy import Derivative, Symbol, Function
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_node(sympy_str: str):
    """Create a mock term node whose to_sympy_string() returns the given string."""
    return SimpleNamespace(to_sympy_string=lambda: sympy_str)


def _make_mock_program(weights, sympy_strings):
    """Create a mock program object with w and STRidge.terms."""
    nodes = [_make_mock_node(s) for s in sympy_strings]
    stridge = SimpleNamespace(terms=nodes)
    return SimpleNamespace(w=weights, STRidge=stridge)


def _import_subscript_printer():
    """Lazy import of _SubscriptLatexPrinter; raises if not implemented."""
    from kd.viz.discover_eq2latex import _SubscriptLatexPrinter
    return _SubscriptLatexPrinter


def _import_term_to_latex():
    """Import _discover_term_node_to_latex."""
    from kd.viz.discover_eq2latex import _discover_term_node_to_latex
    return _discover_term_node_to_latex


def _import_program_to_latex():
    """Import discover_program_to_latex."""
    from kd.viz.discover_eq2latex import discover_program_to_latex
    return discover_program_to_latex


# ===========================================================================
# 1. _SubscriptLatexPrinter unit tests
# ===========================================================================

class TestSubscriptLatexPrinter:
    """Unit tests for the subscript-mode LaTeX printer."""

    def _render(self, expr) -> str:
        """Render a SymPy expression using the subscript printer."""
        cls = _import_subscript_printer()
        printer = cls()
        return printer.doprint(expr)

    # --- Derivative rendering ---

    def test_first_order_derivative_u_x(self):
        """Derivative(u1, x1) -> u_{x}"""
        u1 = Function("u1")(Symbol("x1"), Symbol("x2"))
        expr = Derivative(u1, Symbol("x1"))
        result = self._render(expr)
        assert result == "u_{x}"

    def test_second_order_derivative_u_xx(self):
        """Derivative(u1, x1, x1) -> u_{xx}"""
        u1 = Function("u1")(Symbol("x1"), Symbol("x2"))
        expr = Derivative(u1, Symbol("x1"), Symbol("x1"))
        result = self._render(expr)
        assert result == "u_{xx}"

    def test_third_order_derivative_u_xxx(self):
        """Derivative(u1, x1, x1, x1) -> u_{xxx}"""
        u1 = Function("u1")(Symbol("x1"), Symbol("x2"))
        expr = Derivative(u1, Symbol("x1"), Symbol("x1"), Symbol("x1"))
        result = self._render(expr)
        assert result == "u_{xxx}"

    def test_derivative_wrt_x2(self):
        """Derivative(u1, x2) -> u_{y}"""
        u1 = Function("u1")(Symbol("x1"), Symbol("x2"))
        expr = Derivative(u1, Symbol("x2"))
        result = self._render(expr)
        assert result == "u_{y}"

    def test_mixed_partial_derivative(self):
        """Derivative(u1, x1, x2) -> u_{xy}"""
        u1 = Function("u1")(Symbol("x1"), Symbol("x2"))
        expr = Derivative(u1, Symbol("x1"), Symbol("x2"))
        result = self._render(expr)
        assert result == "u_{xy}"

    # --- Symbol rendering ---

    def test_symbol_u1_display(self):
        """Symbol u1 renders as 'u'."""
        expr = Symbol("u1")
        result = self._render(expr)
        assert result == "u"

    def test_symbol_x1_display(self):
        """Symbol x1 renders as 'x'."""
        expr = Symbol("x1")
        result = self._render(expr)
        assert result == "x"

    def test_symbol_x2_display(self):
        """Symbol x2 renders as 'y'."""
        expr = Symbol("x2")
        result = self._render(expr)
        assert result == "y"

    def test_symbol_x3_display(self):
        """Symbol x3 renders as 'z'."""
        expr = Symbol("x3")
        result = self._render(expr)
        assert result == "z"

    def test_unknown_symbol_unchanged(self):
        """Unknown symbol 'c' stays as 'c'."""
        expr = Symbol("c")
        result = self._render(expr)
        assert result == "c"

    # --- Non-derivative expression ---

    def test_product_u1_x1(self):
        """u1 * x1 uses simplified symbol names: 'u x'."""
        expr = Symbol("u1") * Symbol("x1")
        result = self._render(expr)
        assert result == "u x"


# ===========================================================================
# 2. _discover_term_node_to_latex() with notation parameter
# ===========================================================================

class TestDiscoverTermNodeToLatex:
    """Integration tests for _discover_term_node_to_latex with notation."""

    def test_subscript_mode(self):
        """notation='subscript' uses subscript-style output."""
        fn = _import_term_to_latex()
        node = _make_mock_node("Derivative(u1, x1, x1)")
        result = fn(node, notation="subscript")
        assert result == "u_{xx}"

    def test_leibniz_mode(self):
        """notation='leibniz' uses SymPy default (Leibniz-style) output."""
        fn = _import_term_to_latex()
        node = _make_mock_node("Derivative(u1, x1, x1)")
        result = fn(node, notation="leibniz")
        # SymPy default: \frac{\partial^2}{\partial x_1^2} u_1 (or similar)
        assert "\\frac" in result
        assert "\\partial" in result

    def test_default_is_subscript(self):
        """Default notation should be 'subscript'."""
        fn = _import_term_to_latex()
        node = _make_mock_node("Derivative(u1, x1)")
        result = fn(node)
        assert result == "u_{x}"

    def test_subscript_non_derivative(self):
        """Non-derivative expression in subscript mode."""
        fn = _import_term_to_latex()
        node = _make_mock_node("u1*x1")
        result = fn(node, notation="subscript")
        assert result == "u x"


# ===========================================================================
# 3. discover_program_to_latex() with notation parameter
# ===========================================================================

class TestDiscoverProgramToLatex:
    """Integration tests for discover_program_to_latex with notation."""

    def test_subscript_mode_equation(self):
        """Full equation in subscript mode."""
        fn = _import_program_to_latex()
        program = _make_mock_program(
            weights=[1.0, -0.5],
            sympy_strings=["Derivative(u1, x1, x1)", "u1"],
        )
        result = fn(program, notation="subscript")
        assert "u_{xx}" in result
        assert "$" in result  # wrapped in $...$

    def test_leibniz_mode_equation(self):
        """Full equation in leibniz mode uses Leibniz-style derivatives."""
        fn = _import_program_to_latex()
        program = _make_mock_program(
            weights=[1.0],
            sympy_strings=["Derivative(u1, x1, x1)"],
        )
        result = fn(program, notation="leibniz")
        assert "\\frac" in result
        assert "\\partial" in result

    def test_default_notation_is_subscript(self):
        """Default notation is subscript."""
        fn = _import_program_to_latex()
        program = _make_mock_program(
            weights=[1.0],
            sympy_strings=["Derivative(u1, x1)"],
        )
        result = fn(program)
        assert "u_{x}" in result

    def test_lhs_axis_still_works(self):
        """lhs_axis parameter is independent of notation."""
        fn = _import_program_to_latex()
        program = _make_mock_program(
            weights=[1.0],
            sympy_strings=["Derivative(u1, x1)"],
        )
        result = fn(program, notation="subscript", lhs_axis="y")
        assert "$u_{y} =" in result

    def test_lhs_default_is_u_t(self):
        """Default LHS is u_t."""
        fn = _import_program_to_latex()
        program = _make_mock_program(
            weights=[1.0],
            sympy_strings=["u1"],
        )
        result = fn(program, notation="subscript")
        assert result.startswith("$u_t")
