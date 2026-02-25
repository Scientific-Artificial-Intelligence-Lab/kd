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


# ===========================================================================
# Helper: build real Node trees from stridge.py
# ===========================================================================

def _make_token(name: str, arity: int):
    """Create a minimal Token-like object for Node construction."""
    return SimpleNamespace(name=name, arity=arity)


def _make_leaf(name: str):
    """Create a leaf Node (arity=0) e.g. u1, x1."""
    from kd.model.discover.stridge import Node
    return Node(_make_token(name, 0))


def _make_node(op_name: str, arity: int, children: list):
    """Create an internal Node with given children."""
    from kd.model.discover.stridge import Node
    node = Node(_make_token(op_name, arity))
    node.children = children
    return node


# Convenience builders for common expression patterns
def _n2(child):
    """n2(child) -> Pow(child, 2)"""
    return _make_node("n2", 1, [child])


def _n3(child):
    """n3(child) -> Pow(child, 3)"""
    return _make_node("n3", 1, [child])


def _diff(func_child, var_child):
    """diff(func, var) -> Derivative(func, var)"""
    return _make_node("diff", 2, [func_child, var_child])


def _diff2(func_child, var_child):
    """diff2(func, var) -> Derivative(func, var, var)"""
    return _make_node("diff2", 2, [func_child, var_child])


def _mul(left, right):
    """mul(left, right) -> Mul(left, right)"""
    return _make_node("mul", 2, [left, right])


def _add(left, right):
    """add(left, right) -> Add(left, right)"""
    return _make_node("add", 2, [left, right])


# ===========================================================================
# 4. Non-Symbol derivative rendering (BUG: _print_Derivative crashes)
# ===========================================================================

class TestNonSymbolDerivative:
    """Tests for derivatives of non-Symbol expressions (the main bug).

    When func in Derivative(func, x) is not a Symbol (e.g., Pow(u1, 2)),
    _SubscriptLatexPrinter._print_Derivative crashes because func.func
    is a SymPy class (e.g., <class Pow>) without a meaningful .name.
    """

    # --- Core crash cases (must fix) ---

    @pytest.mark.unit
    def test_diff_n2_u1_subscript(self):
        """diff(n2(u1), x1) should produce valid LaTeX without 'Error'.

        This is the primary crash case: Derivative(Pow(u1, 2), x1).
        Expected output like '{u^{2}}_{x}' or similar.
        """
        fn = _import_term_to_latex()
        node = _diff(_n2(_make_leaf("u1")), _make_leaf("x1"))
        result = fn(node, notation="subscript")
        assert "Error" not in result, f"Got error output: {result}"
        # Must contain subscript x for the derivative variable
        assert "_{" in result and "x" in result

    @pytest.mark.unit
    def test_diff_mul_u1_u1_subscript(self):
        """diff(mul(u1, u1), x1) should produce valid LaTeX without 'Error'.

        Derivative(Mul(u1, u1), x1) -> something like '{u \\cdot u}_{x}'.
        """
        fn = _import_term_to_latex()
        node = _diff(
            _mul(_make_leaf("u1"), _make_leaf("u1")),
            _make_leaf("x1"),
        )
        result = fn(node, notation="subscript")
        assert "Error" not in result, f"Got error output: {result}"
        assert "_{" in result

    @pytest.mark.unit
    def test_diff2_n3_u1_subscript(self):
        """diff2(n3(u1), x1) should produce valid LaTeX without 'Error'.

        Derivative(Pow(u1, 3), x1, x1) -> second-order derivative.
        """
        fn = _import_term_to_latex()
        node = _diff2(_n3(_make_leaf("u1")), _make_leaf("x1"))
        result = fn(node, notation="subscript")
        assert "Error" not in result, f"Got error output: {result}"
        # Second-order: should have xx in subscript
        assert "xx" in result

    # --- Leibniz mode (same expressions) ---

    @pytest.mark.unit
    def test_diff_n2_u1_leibniz(self):
        """diff(n2(u1), x1) in leibniz mode should also not crash."""
        fn = _import_term_to_latex()
        node = _diff(_n2(_make_leaf("u1")), _make_leaf("x1"))
        result = fn(node, notation="leibniz")
        assert "Error" not in result, f"Got error output: {result}"

    @pytest.mark.unit
    def test_diff_mul_u1_u1_leibniz(self):
        """diff(mul(u1, u1), x1) in leibniz mode should not crash."""
        fn = _import_term_to_latex()
        node = _diff(
            _mul(_make_leaf("u1"), _make_leaf("u1")),
            _make_leaf("x1"),
        )
        result = fn(node, notation="leibniz")
        assert "Error" not in result, f"Got error output: {result}"

    # --- Non-regression: simple derivatives still work ---

    @pytest.mark.unit
    def test_simple_diff_u1_x1_still_works(self):
        """diff(u1, x1) -> u_{x} must not regress."""
        fn = _import_term_to_latex()
        node = _diff(_make_leaf("u1"), _make_leaf("x1"))
        result = fn(node, notation="subscript")
        assert result == "u_{x}"

    @pytest.mark.unit
    def test_simple_diff2_u1_x1_still_works(self):
        """diff2(u1, x1) -> u_{xx} must not regress."""
        fn = _import_term_to_latex()
        node = _diff2(_make_leaf("u1"), _make_leaf("x1"))
        result = fn(node, notation="subscript")
        assert result == "u_{xx}"

    # --- Additional edge cases ---

    @pytest.mark.unit
    def test_nested_diff_diff_u1_x1_x1(self):
        """diff(diff(u1, x1), x1) -> nested derivative.

        This creates Derivative(Derivative(u1, x1), x1), which SymPy may
        simplify to Derivative(u1, x1, x1). Either way should produce
        valid output.
        """
        fn = _import_term_to_latex()
        inner = _diff(_make_leaf("u1"), _make_leaf("x1"))
        outer = _diff(inner, _make_leaf("x1"))
        result = fn(outer, notation="subscript")
        assert "Error" not in result, f"Got error output: {result}"
        assert "x" in result

    @pytest.mark.unit
    def test_diff_n2_u1_x2(self):
        """diff(n2(u1), x2) -> derivative wrt y.

        Derivative(Pow(u1, 2), x2) should produce subscript with y.
        """
        fn = _import_term_to_latex()
        node = _diff(_n2(_make_leaf("u1")), _make_leaf("x2"))
        result = fn(node, notation="subscript")
        assert "Error" not in result, f"Got error output: {result}"
        assert "y" in result

    @pytest.mark.unit
    def test_diff_add_n2_u1_u1(self):
        """diff(add(n2(u1), u1), x1) -> complex expression derivative.

        Derivative(Add(Pow(u1, 2), u1), x1).
        """
        fn = _import_term_to_latex()
        node = _diff(
            _add(_n2(_make_leaf("u1")), _make_leaf("u1")),
            _make_leaf("x1"),
        )
        result = fn(node, notation="subscript")
        assert "Error" not in result, f"Got error output: {result}"
        assert "x" in result


# ===========================================================================
# 5. _SubscriptLatexPrinter direct tests for non-Symbol func
# ===========================================================================

class TestSubscriptPrinterNonSymbolFunc:
    """Direct unit tests on _SubscriptLatexPrinter for non-Symbol derivatives.

    These test the printer directly with SymPy expressions (bypassing Node),
    to isolate the bug in _print_Derivative.
    """

    def _render(self, expr) -> str:
        cls = _import_subscript_printer()
        return cls().doprint(expr)

    @pytest.mark.unit
    def test_derivative_of_pow(self):
        """Derivative(Pow(u1, 2), x1) should not crash.

        This is the exact SymPy expression that triggers the bug.
        """
        u1 = Symbol("u1")
        x1 = Symbol("x1")
        expr = Derivative(u1**2, x1)
        result = self._render(expr)
        # Should produce something like '{u^{2}}_{x}' or '(u^{2})_{x}'
        assert "x" in result
        assert "2" in result  # the exponent should be visible

    @pytest.mark.unit
    def test_derivative_of_mul(self):
        """Derivative(u1 * u1, x1) should not crash."""
        u1 = Symbol("u1")
        x1 = Symbol("x1")
        expr = Derivative(u1 * u1, x1)
        result = self._render(expr)
        assert "x" in result

    @pytest.mark.unit
    def test_derivative_of_add(self):
        """Derivative(u1 + u1**2, x1) should not crash."""
        u1 = Symbol("u1")
        x1 = Symbol("x1")
        expr = Derivative(u1 + u1**2, x1)
        result = self._render(expr)
        assert "x" in result

    @pytest.mark.unit
    def test_second_derivative_of_pow(self):
        """Derivative(Pow(u1, 3), x1, x1) should not crash."""
        u1 = Symbol("u1")
        x1 = Symbol("x1")
        expr = Derivative(u1**3, x1, x1)
        result = self._render(expr)
        assert "xx" in result
        assert "3" in result


# ===========================================================================
# 6. First-order compound derivative expansion
# ===========================================================================

class TestExpandFirstOrderDerivatives:
    """Tests for expanding first-order compound derivatives via doit().

    e.g. Derivative(u1**2, x1) → 2*u1*Derivative(u1, x1)
    The numeric factor should be absorbed into the coefficient.
    """

    # --- Core expansion helper ---

    @pytest.mark.unit
    def test_expand_pow2_first_order(self):
        """Derivative(u1**2, x1) expands to (2, u1*Derivative(u1, x1))."""
        from kd.viz.discover_eq2latex import _expand_first_order_compound_derivatives
        u1 = Symbol("u1")
        x1 = Symbol("x1")
        expr = Derivative(u1**2, x1)
        factor, expanded = _expand_first_order_compound_derivatives(expr)
        # factor should be 2 (from chain rule)
        assert float(factor) == pytest.approx(2.0)
        # expanded should not contain Pow — it's u1*Derivative(u1, x1)
        assert isinstance(expanded, sympy.Mul) or "Derivative" in str(expanded)
        assert "u1" in str(expanded)

    @pytest.mark.unit
    def test_expand_pow3_first_order(self):
        """Derivative(u1**3, x1) expands to (3, u1**2 * Derivative(u1, x1))."""
        from kd.viz.discover_eq2latex import _expand_first_order_compound_derivatives
        u1 = Symbol("u1")
        x1 = Symbol("x1")
        expr = Derivative(u1**3, x1)
        factor, expanded = _expand_first_order_compound_derivatives(expr)
        assert float(factor) == pytest.approx(3.0)

    @pytest.mark.unit
    def test_no_expand_simple_derivative(self):
        """Derivative(u1, x1) should not be expanded (already simple)."""
        from kd.viz.discover_eq2latex import _expand_first_order_compound_derivatives
        u1 = Symbol("u1")
        x1 = Symbol("x1")
        expr = Derivative(u1, x1)
        factor, expanded = _expand_first_order_compound_derivatives(expr)
        assert float(factor) == pytest.approx(1.0)
        # Expression unchanged
        assert expanded == expr

    @pytest.mark.unit
    def test_no_expand_second_order_compound(self):
        """Derivative(u1**2, x1, x1) should NOT be expanded (second order)."""
        from kd.viz.discover_eq2latex import _expand_first_order_compound_derivatives
        u1 = Symbol("u1")
        x1 = Symbol("x1")
        expr = Derivative(u1**2, x1, x1)
        factor, expanded = _expand_first_order_compound_derivatives(expr)
        assert float(factor) == pytest.approx(1.0)
        # Expression unchanged — still has Pow inside Derivative
        assert expanded == expr

    @pytest.mark.unit
    def test_no_expand_non_derivative(self):
        """Non-derivative expression (u1**2) should pass through unchanged."""
        from kd.viz.discover_eq2latex import _expand_first_order_compound_derivatives
        u1 = Symbol("u1")
        expr = u1**2
        factor, expanded = _expand_first_order_compound_derivatives(expr)
        assert float(factor) == pytest.approx(1.0)
        assert expanded == expr

    # --- Integration: term-level with expand ---

    @pytest.mark.unit
    def test_term_expand_diff_n2_u1(self):
        """diff(n2(u1), x1) with expand → LaTeX contains 'u' and 'u_{x}'."""
        from kd.viz.discover_eq2latex import _discover_term_node_to_latex_expanded
        node = _diff(_n2(_make_leaf("u1")), _make_leaf("x1"))
        factor, latex = _discover_term_node_to_latex_expanded(node)
        assert float(factor) == pytest.approx(2.0)
        assert "u_{x}" in latex
        assert "u" in latex
        # Should NOT have parenthesized form
        assert "(u^{2})" not in latex

    @pytest.mark.unit
    def test_term_expand_simple_diff(self):
        """diff(u1, x1) with expand → factor=1, latex='u_{x}'."""
        from kd.viz.discover_eq2latex import _discover_term_node_to_latex_expanded
        node = _diff(_make_leaf("u1"), _make_leaf("x1"))
        factor, latex = _discover_term_node_to_latex_expanded(node)
        assert float(factor) == pytest.approx(1.0)
        assert latex == "u_{x}"

    @pytest.mark.unit
    def test_term_expand_diff2_n2_keeps_parens(self):
        """diff2(n2(u1), x1) with expand → factor=1, keeps (u^{2})_{xx}."""
        from kd.viz.discover_eq2latex import _discover_term_node_to_latex_expanded
        node = _diff2(_n2(_make_leaf("u1")), _make_leaf("x1"))
        factor, latex = _discover_term_node_to_latex_expanded(node)
        assert float(factor) == pytest.approx(1.0)
        assert "xx" in latex

    # --- Integration: program-level with expand_derivatives ---

    @pytest.mark.unit
    def test_program_expand_absorbs_factor(self):
        """discover_program_to_latex with expand_derivatives absorbs factor.

        Program: -0.5 * diff(n2(u1), x1) should become ~ -1.0 * u u_x.
        _format_full_latex_term omits coefficient when |coeff| ≈ 1.
        """
        fn = _import_program_to_latex()
        program = _make_mock_program(
            weights=[-0.5],
            sympy_strings=["Derivative(Pow(u1, 2), x1)"],
        )
        result = fn(program, notation="subscript", expand_derivatives=True)
        # Factor 2 absorbed: -0.5 * 2 = -1.0, displayed as "- u u_{x}"
        assert "u_{x}" in result
        assert "u" in result
        # Should NOT contain parenthesized derivative
        assert "(u^{2})_{x}" not in result

    @pytest.mark.unit
    def test_program_expand_false_keeps_parens(self):
        """expand_derivatives=False preserves current behavior."""
        fn = _import_program_to_latex()
        program = _make_mock_program(
            weights=[-0.5],
            sympy_strings=["Derivative(Pow(u1, 2), x1)"],
        )
        result = fn(program, notation="subscript", expand_derivatives=False)
        # Should keep the parenthesized form
        assert "(u^{2})" in result
        assert "0.5" in result


# ===========================================================================
# 7. Multi-dataset expression pattern rendering (Phase 1)
# ===========================================================================

def _diff3(func_child, var_child):
    """diff3(func, var) -> Derivative(func, var, var, var)"""
    return _make_node("diff3", 2, [func_child, var_child])


def _div(left, right):
    """div(left, right) -> Mul(left, Pow(right, -1))"""
    return _make_node("div", 2, [left, right])


class TestMultiDatasetRendering:
    """Unit tests for expression patterns from all DSCV datasets.

    Each test constructs a mock Node tree matching a real HOF expression
    and verifies the LaTeX output.
    """

    # --- KdV ---

    @pytest.mark.unit
    def test_kdv_diff3_u_xxx(self):
        """KdV: diff3(u1,x1) -> u_{xxx}"""
        fn = _import_term_to_latex()
        node = _diff3(_make_leaf("u1"), _make_leaf("x1"))
        result = fn(node, notation="subscript")
        assert result == "u_{xxx}"

    @pytest.mark.unit
    def test_kdv_mul_u_diff_u_x(self):
        """KdV: mul(u1,diff(u1,x1)) -> u u_{x} (same as Burgers)"""
        fn = _import_term_to_latex()
        node = _mul(_make_leaf("u1"), _diff(_make_leaf("u1"), _make_leaf("x1")))
        result = fn(node, notation="subscript")
        assert result == "u u_{x}"

    # --- Chafee-Infante ---

    @pytest.mark.unit
    def test_chafee_n3_u(self):
        """Chafee-Infante: n3(u1) -> u^{3}"""
        fn = _import_term_to_latex()
        node = _n3(_make_leaf("u1"))
        result = fn(node, notation="subscript")
        assert result == "u^{3}"

    # --- PDE_compound ---

    @pytest.mark.unit
    def test_compound_diff2_n2_u(self):
        r"""PDE_compound: diff2(n2(u1),x1) -> (u^{2})_{xx} (2nd-order, no expand)"""
        fn = _import_term_to_latex()
        node = _diff2(_n2(_make_leaf("u1")), _make_leaf("x1"))
        result = fn(node, notation="subscript")
        assert result == "(u^{2})_{xx}"

    @pytest.mark.unit
    def test_compound_diff2_mul_u_u(self):
        r"""PDE_compound: diff2(mul(u1,u1),x1) -> (u^{2})_{xx} (equivalent mul form)"""
        fn = _import_term_to_latex()
        node = _diff2(_mul(_make_leaf("u1"), _make_leaf("u1")), _make_leaf("x1"))
        result = fn(node, notation="subscript")
        assert result == "(u^{2})_{xx}"

    # --- PDE_divide ---

    @pytest.mark.unit
    def test_divide_div_diff_u_x(self):
        r"""PDE_divide: div(diff(u1,x1),x1) -> \frac{u_{x}}{x}"""
        fn = _import_term_to_latex()
        node = _div(_diff(_make_leaf("u1"), _make_leaf("x1")), _make_leaf("x1"))
        result = fn(node, notation="subscript")
        assert result == r"\frac{u_{x}}{x}"

    # --- Fisher ---

    @pytest.mark.unit
    def test_fisher_n2_diff_u_x(self):
        r"""Fisher: n2(diff(u1,x1)) -> \left(u_{x}\right)^{2} (or u_{x}^{2})"""
        fn = _import_term_to_latex()
        node = _n2(_diff(_make_leaf("u1"), _make_leaf("x1")))
        result = fn(node, notation="subscript")
        assert "Error" not in result
        # Must contain u_x and exponent 2
        assert "u_{x}" in result
        assert "2" in result

    @pytest.mark.unit
    def test_fisher_mul_u_diff2_u(self):
        """Fisher: mul(u1,diff2(u1,x1)) -> u u_{xx}"""
        fn = _import_term_to_latex()
        node = _mul(_make_leaf("u1"), _diff2(_make_leaf("u1"), _make_leaf("x1")))
        result = fn(node, notation="subscript")
        assert result == "u u_{xx}"


# ===========================================================================
# 8. Multi-dataset Program-level rendering (Phase 2)
# ===========================================================================

class TestMultiDatasetProgramRendering:
    """Integration tests for complete Program-level rendering of each dataset."""

    # --- KdV ---

    @pytest.mark.integration
    def test_kdv_best(self):
        r"""KdV best: u_t = -1.0006 u u_{x} - 0.0025 u_{xxx}"""
        fn = _import_program_to_latex()
        program = _make_mock_program(
            weights=[-1.0006, -0.0025],
            sympy_strings=[
                "Mul(u1, Derivative(u1, x1))",
                "Derivative(u1, x1, x1, x1)",
            ],
        )
        result = fn(program, notation="subscript")
        assert "u u_{x}" in result
        assert "u_{xxx}" in result
        assert "1.0006" in result
        assert "0.0025" in result
        assert result.startswith("$u_t =")

    # --- Chafee-Infante ---

    @pytest.mark.integration
    def test_chafee_best(self):
        r"""Chafee-Infante best: u_t = -1.0008 u + 1.0002 u_{xx} + 1.0004 u^{3}"""
        fn = _import_program_to_latex()
        program = _make_mock_program(
            weights=[-1.0008, 1.0002, 1.0004],
            sympy_strings=["u1", "Derivative(u1, x1, x1)", "Pow(u1, 3)"],
        )
        result = fn(program, notation="subscript")
        assert "1.0008" in result
        assert "u_{xx}" in result
        assert "u^{3}" in result
        assert result.startswith("$u_t =")

    # --- PDE_compound ---

    @pytest.mark.integration
    def test_compound_best_expand_true(self):
        r"""PDE_compound best: expand=True still shows (u^{2})_{xx} (2nd-order)"""
        fn = _import_program_to_latex()
        program = _make_mock_program(
            weights=[0.5002],
            sympy_strings=["Derivative(Pow(u1, 2), x1, x1)"],
        )
        result = fn(program, notation="subscript", expand_derivatives=True)
        assert "(u^{2})_{xx}" in result
        assert "0.5002" in result

    @pytest.mark.integration
    def test_compound_best_expand_false(self):
        r"""PDE_compound best: expand=False also shows (u^{2})_{xx}"""
        fn = _import_program_to_latex()
        program = _make_mock_program(
            weights=[0.5002],
            sympy_strings=["Derivative(Pow(u1, 2), x1, x1)"],
        )
        result = fn(program, notation="subscript", expand_derivatives=False)
        assert "(u^{2})_{xx}" in result
        assert "0.5002" in result

    # --- PDE_divide ---

    @pytest.mark.integration
    def test_divide_best(self):
        r"""PDE_divide best: contains \frac{u_{x}}{x} and u_{xx}"""
        fn = _import_program_to_latex()
        program = _make_mock_program(
            weights=[-0.9974, 0.2496],
            sympy_strings=[
                "Mul(Derivative(u1, x1), Pow(x1, -1))",
                "Derivative(u1, x1, x1)",
            ],
        )
        result = fn(program, notation="subscript")
        assert r"\frac{u_{x}}{x}" in result
        assert "u_{xx}" in result
        assert result.startswith("$u_t =")


# ===========================================================================
# 9. ref_lib HOF CSV batch rendering (Phase 3)
# ===========================================================================

# Token arity lookup for reconstructing Node trees from traversal strings.
_ARITY_MAP = {
    # Binary operators
    "add": 2, "mul": 2, "sub": 2, "div": 2,
    "diff": 2, "diff2": 2, "diff3": 2, "diff4": 2,
    "Diff": 2, "Diff2": 2, "Diff_3": 2, "Diff2_3": 2,
    # Unary operators
    "n2": 1, "n3": 1, "n4": 1, "n5": 1,
    "sin": 1, "cos": 1, "exp": 1, "log": 1, "sqrt": 1, "abs": 1,
    "sigmoid": 1, "expneg": 1, "logabs": 1,
    "lap": 1, "lap_3": 1,
}


def _build_node_from_traversal(tokens: list):
    """Recursively build a Node tree from a pre-order list of token name strings.

    Consumes tokens from the front of *tokens* (in-place).
    """
    name = tokens.pop(0)
    arity = _ARITY_MAP.get(name, 0)  # default 0 = leaf
    node = _make_node(name, arity, []) if arity > 0 else _make_leaf(name)
    for _ in range(arity):
        node.children.append(_build_node_from_traversal(tokens))
    return node


def _collect_hof_success_entries():
    """Collect (dataset, expression, traversal) tuples from all ref_lib HOF CSVs.

    Only rows with success=True are included. Returns one CSV per dataset
    (the first found).
    """
    import csv
    import glob
    import os

    pattern = os.path.join(
        "/Users/hao/PhD/project/kd",
        "ref_lib/DISCOVER/dso/log/MODE1/**/dso_*_0_hof.csv",
    )
    csv_files = glob.glob(pattern, recursive=True)

    # Deduplicate: one CSV per dataset name
    seen: dict = {}
    for fp in csv_files:
        key = os.path.basename(fp).replace("dso_", "").replace("_0_hof.csv", "")
        if key not in seen:
            seen[key] = fp

    entries = []
    for dataset, filepath in sorted(seen.items()):
        with open(filepath) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("success") != "True":
                    continue
                entries.append((dataset, row["expression"], row["traversal"]))
    return entries


_HOF_ENTRIES = _collect_hof_success_entries()


class TestRefLibHOFRendering:
    """Batch rendering test: every success=True HOF expression must render
    without errors or exceptions.
    """

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "dataset,expression,traversal",
        _HOF_ENTRIES,
        ids=[f"{ds}:{expr[:50]}" for ds, expr, _ in _HOF_ENTRIES],
    )
    def test_hof_renders_without_error(self, dataset, expression, traversal):
        """Each HOF success entry must render to valid LaTeX."""
        fn = _import_term_to_latex()

        # The HOF expression format is like "-1.0006 * mul(u1,diff(u1,x1)) + ..."
        # We parse individual terms from the traversal, building the whole tree.
        tokens = traversal.split(",")
        root_node = _build_node_from_traversal(tokens)

        result = fn(root_node, notation="subscript")
        assert "Error" not in result, (
            f"[{dataset}] Rendering failed for '{expression}': {result}"
        )
