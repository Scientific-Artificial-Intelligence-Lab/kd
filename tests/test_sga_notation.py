"""TDD tests for SGA LaTeX dual-mode notation rendering (RED phase).

Tests cover:
1. _combine_node unit tests (subscript vs leibniz for derivative nodes)
2. _render_tree integration tests (tree traversal with notation)
3. sga_equation_to_latex integration tests (full equation with notation)

All tests are expected to FAIL until the implementation is written.
"""

import pytest
from types import SimpleNamespace

from kd.model.sga.sgapde.equation import (
    _combine_node,
    _merge_subscript,
    _render_tree,
    sga_equation_to_latex,
    SGAEquationDetails,
    SGAEquationTerm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_node(name: str, child_num: int = 0, child_st=None):
    """Create a mock tree node matching the Node interface."""
    return SimpleNamespace(name=name, child_num=child_num, child_st=child_st, var=None)


def _make_tree_d_u_x():
    """Build a minimal tree: d(u, x) -- first-order derivative."""
    root = _make_mock_node('d', child_num=2, child_st=0)
    leaf_u = _make_mock_node('u')
    leaf_x = _make_mock_node('x')
    tree_obj = SimpleNamespace(tree=[[root], [leaf_u, leaf_x]])
    return tree_obj


def _make_tree_d2_u_x():
    """Build a minimal tree: d^2(u, x) -- second-order derivative."""
    root = _make_mock_node('d^2', child_num=2, child_st=0)
    leaf_u = _make_mock_node('u')
    leaf_x = _make_mock_node('x')
    tree_obj = SimpleNamespace(tree=[[root], [leaf_u, leaf_x]])
    return tree_obj


def _make_tree_d_d_u_x_x():
    """Build a nested tree: d(d(u, x), x) -- nested first-order derivatives."""
    root = _make_mock_node('d', child_num=2, child_st=0)
    inner_d = _make_mock_node('d', child_num=2, child_st=0)
    leaf_x_outer = _make_mock_node('x')
    leaf_u = _make_mock_node('u')
    leaf_x_inner = _make_mock_node('x')
    tree_obj = SimpleNamespace(tree=[[root], [inner_d, leaf_x_outer], [leaf_u, leaf_x_inner]])
    return tree_obj


def _make_tree_d2_d_u_x_y():
    """Build a nested tree: d^2(d(u, x), y) -- mixed partial derivative."""
    root = _make_mock_node('d^2', child_num=2, child_st=0)
    inner_d = _make_mock_node('d', child_num=2, child_st=0)
    leaf_y = _make_mock_node('y')
    leaf_u = _make_mock_node('u')
    leaf_x = _make_mock_node('x')
    tree_obj = SimpleNamespace(tree=[[root], [inner_d, leaf_y], [leaf_u, leaf_x]])
    return tree_obj


def _make_tree_mul_u_x():
    """Build a minimal tree: u * x -- non-derivative."""
    root = _make_mock_node('*', child_num=2, child_st=0)
    leaf_u = _make_mock_node('u')
    leaf_x = _make_mock_node('x')
    tree_obj = SimpleNamespace(tree=[[root], [leaf_u, leaf_x]])
    return tree_obj


# ===========================================================================
# 1. _combine_node unit tests
# ===========================================================================

class TestCombineNode:
    """Unit tests for _combine_node with notation parameter."""

    # --- subscript mode ---

    def test_d_subscript(self):
        """d node with subscript notation: d(u, x) -> u_{x}"""
        result = _combine_node('d', ['u', 'x'], notation='subscript')
        assert result == 'u_{x}'

    def test_d2_subscript(self):
        """d^2 node with subscript notation: d^2(u, x) -> u_{xx}"""
        result = _combine_node('d^2', ['u', 'x'], notation='subscript')
        assert result == 'u_{xx}'

    def test_d_subscript_wrt_t(self):
        """d node subscript with respect to t: d(u, t) -> u_{t}"""
        result = _combine_node('d', ['u', 't'], notation='subscript')
        assert result == 'u_{t}'

    # --- leibniz mode ---

    def test_d_leibniz(self):
        r"""d node with leibniz notation: d(u, x) -> \frac{\partial u}{\partial x}"""
        result = _combine_node('d', ['u', 'x'], notation='leibniz')
        assert result == r'\frac{\partial u}{\partial x}'

    def test_d2_leibniz(self):
        r"""d^2 node with leibniz notation: standard Leibniz second derivative."""
        result = _combine_node('d^2', ['u', 'x'], notation='leibniz')
        assert result == r'\frac{\partial^{2} u}{\partial x^{2}}'

    # --- default notation ---

    def test_default_notation_is_subscript(self):
        """Default notation should be subscript."""
        result = _combine_node('d', ['u', 'x'])
        assert result == 'u_{x}'

    # --- non-derivative nodes unaffected ---

    def test_plus_unaffected_by_notation(self):
        """+ node rendering is unchanged regardless of notation."""
        sub = _combine_node('+', ['a', 'b'], notation='subscript')
        lei = _combine_node('+', ['a', 'b'], notation='leibniz')
        assert sub == lei == '(a + b)'

    def test_mul_unaffected_by_notation(self):
        """* node rendering is unchanged regardless of notation."""
        sub = _combine_node('*', ['u', 'x'], notation='subscript')
        lei = _combine_node('*', ['u', 'x'], notation='leibniz')
        assert sub == lei == r'u \cdot x'

    def test_pow2_unaffected_by_notation(self):
        """^2 node rendering is unchanged regardless of notation."""
        sub = _combine_node('^2', ['u'], notation='subscript')
        lei = _combine_node('^2', ['u'], notation='leibniz')
        assert sub == lei == 'u^{2}'

    def test_div_unaffected_by_notation(self):
        r"""/ node rendering is unchanged regardless of notation."""
        sub = _combine_node('/', ['a', 'b'], notation='subscript')
        lei = _combine_node('/', ['a', 'b'], notation='leibniz')
        assert sub == lei == r'\frac{a}{b}'


# ===========================================================================
# 1b. _merge_subscript unit tests
# ===========================================================================

class TestMergeSubscript:
    """Unit tests for _merge_subscript helper."""

    def test_merge_single_to_single(self):
        """u_{x} + 'x' -> u_{xx}"""
        assert _merge_subscript('u_{x}', 'x') == 'u_{xx}'

    def test_merge_single_to_double(self):
        """u_{xx} + 'x' -> u_{xxx}"""
        assert _merge_subscript('u_{xx}', 'x') == 'u_{xxx}'

    def test_merge_different_vars(self):
        """u_{x} + 'yy' -> u_{xyy}"""
        assert _merge_subscript('u_{x}', 'yy') == 'u_{xyy}'

    def test_no_subscript_returns_none(self):
        """Plain 'u' has no subscript to merge into."""
        assert _merge_subscript('u', 'x') is None

    def test_no_brace_returns_none(self):
        """Expression without closing brace returns None."""
        assert _merge_subscript('abc', 'x') is None


class TestCombineNodeNestedDerivatives:
    """Unit tests for _combine_node with nested derivative subscripts."""

    def test_d_on_subscripted_operand(self):
        """d(u_{x}, x) should merge to u_{xx}, not u_{x}_{x}"""
        result = _combine_node('d', ['u_{x}', 'x'], notation='subscript')
        assert result == 'u_{xx}'
        assert '_{' not in result.replace('u_{xx}', '')  # no double subscript

    def test_d2_on_subscripted_operand(self):
        """d^2(u_{x}, y) should merge to u_{xyy}"""
        result = _combine_node('d^2', ['u_{x}', 'y'], notation='subscript')
        assert result == 'u_{xyy}'

    def test_d_on_plain_operand_unchanged(self):
        """d(u, x) still produces u_{x} as before."""
        result = _combine_node('d', ['u', 'x'], notation='subscript')
        assert result == 'u_{x}'

    def test_leibniz_not_affected(self):
        r"""Leibniz notation should NOT merge subscripts."""
        result = _combine_node('d', ['u_{x}', 'x'], notation='leibniz')
        assert r'\frac{\partial u_{x}}{\partial x}' == result

    def test_d_on_squared_operand(self):
        """d(u^{2}, x) -- operand ends with } but is NOT a subscript."""
        result = _combine_node('d', ['u^{2}', 'x'], notation='subscript')
        assert '_{' not in result or result.count('_{') == 1
        # Should NOT corrupt the ^{2}: must produce u^{2}_{x} or similar valid form
        assert '^{2}' in result

    def test_d_on_subscripted_and_squared_operand(self):
        """d(u_{x}^{2}, x) -- must not corrupt the superscript."""
        result = _combine_node('d', ['u_{x}^{2}', 'x'], notation='subscript')
        # Must not produce broken LaTeX like u_{x}^{2x}
        assert '}^{2' not in result.replace('u_{x}^{2}', '') or 'u_{xx}^{2}' in result

    def test_d_on_frac_operand(self):
        r"""d(\frac{u}{x}, y) -- frac ends with } but is not a subscript."""
        result = _combine_node('d', [r'\frac{u}{x}', 'y'], notation='subscript')
        assert r'\frac{u}{x}' in result  # frac preserved
        assert '_{y}' in result  # subscript appended

    def test_d_on_parenthesized_operand(self):
        """d((u + x), y) -- parenthesized operand, no merge."""
        result = _combine_node('d', ['(u + x)', 'y'], notation='subscript')
        assert result == '(u + x)_{y}'


# ===========================================================================
# 2. _render_tree integration tests
# ===========================================================================

class TestRenderTree:
    """Integration tests for _render_tree with notation parameter."""

    def test_d_tree_subscript(self):
        """d(u, x) tree in subscript mode -> u_{x}"""
        tree = _make_tree_d_u_x()
        result = _render_tree(tree, notation='subscript')
        assert result == 'u_{x}'

    def test_d_tree_leibniz(self):
        r"""d(u, x) tree in leibniz mode -> Leibniz fraction."""
        tree = _make_tree_d_u_x()
        result = _render_tree(tree, notation='leibniz')
        assert result == r'\frac{\partial u}{\partial x}'

    def test_d2_tree_subscript(self):
        """d^2(u, x) tree in subscript mode -> u_{xx}"""
        tree = _make_tree_d2_u_x()
        result = _render_tree(tree, notation='subscript')
        assert result == 'u_{xx}'

    def test_mul_tree_unaffected(self):
        r"""Non-derivative tree is unaffected by notation."""
        tree = _make_tree_mul_u_x()
        sub = _render_tree(tree, notation='subscript')
        lei = _render_tree(tree, notation='leibniz')
        assert sub == lei == r'u \cdot x'

    def test_default_notation_is_subscript(self):
        """Default notation for _render_tree should be subscript."""
        tree = _make_tree_d_u_x()
        result = _render_tree(tree)
        assert result == 'u_{x}'

    def test_nested_d_d_u_x_x_subscript(self):
        """d(d(u, x), x) tree should produce u_{xx}, not u_{x}_{x}."""
        tree = _make_tree_d_d_u_x_x()
        result = _render_tree(tree, notation='subscript')
        assert result == 'u_{xx}'

    def test_nested_d2_d_u_x_y_subscript(self):
        """d^2(d(u, x), y) tree should produce u_{xyy}."""
        tree = _make_tree_d2_d_u_x_y()
        result = _render_tree(tree, notation='subscript')
        assert result == 'u_{xyy}'

    def test_triple_nested_d_subscript(self):
        """d(d(d(u, x), y), z) tree should produce u_{xyz}."""
        # Build 4-level tree: d -> d -> d -> u, with axis leaves x, y, z
        root = _make_mock_node('d', child_num=2, child_st=0)
        mid = _make_mock_node('d', child_num=2, child_st=0)
        leaf_z = _make_mock_node('z')
        inner = _make_mock_node('d', child_num=2, child_st=0)
        leaf_y = _make_mock_node('y')
        leaf_u = _make_mock_node('u')
        leaf_x = _make_mock_node('x')
        tree_obj = SimpleNamespace(tree=[
            [root], [mid, leaf_z], [inner, leaf_y], [leaf_u, leaf_x],
        ])
        result = _render_tree(tree_obj, notation='subscript')
        assert result == 'u_{xyz}'


# ===========================================================================
# 3. sga_equation_to_latex integration tests
# ===========================================================================

class TestSgaEquationToLatexNotation:
    """Integration tests for sga_equation_to_latex with notation."""

    def _make_details_with_tree(self, tree_obj, coeff=1.0):
        """Create SGAEquationDetails containing one generated term with a tree."""
        term = SGAEquationTerm(
            label='d(u, x)',
            source='generated',
            coefficient=coeff,
            tree=tree_obj,
        )
        return SGAEquationDetails(lhs='u_t', terms=[term])

    def test_subscript_equation(self):
        """Full equation in subscript mode contains subscript derivative."""
        details = self._make_details_with_tree(_make_tree_d_u_x())
        result = sga_equation_to_latex(details, notation='subscript')
        assert 'u_{x}' in result

    def test_leibniz_equation(self):
        r"""Full equation in leibniz mode contains Leibniz derivative."""
        details = self._make_details_with_tree(_make_tree_d_u_x())
        result = sga_equation_to_latex(details, notation='leibniz')
        assert r'\frac{\partial u}{\partial x}' in result

    def test_default_notation_is_subscript(self):
        """Default notation for sga_equation_to_latex should be subscript."""
        details = self._make_details_with_tree(_make_tree_d_u_x())
        result = sga_equation_to_latex(details)
        assert 'u_{x}' in result

    def test_invalid_notation_raises(self):
        """Invalid notation value should raise ValueError."""
        details = self._make_details_with_tree(_make_tree_d_u_x())
        with pytest.raises(ValueError, match='notation'):
            sga_equation_to_latex(details, notation='invalid')

    def test_multiple_terms_subscript(self):
        """Equation with multiple terms renders all in subscript."""
        t1 = SGAEquationTerm(
            label='d(u,x)', source='generated', coefficient=1.0,
            tree=_make_tree_d_u_x(),
        )
        t2 = SGAEquationTerm(
            label='d^2(u,x)', source='generated', coefficient=-0.5,
            tree=_make_tree_d2_u_x(),
        )
        details = SGAEquationDetails(lhs='u_t', terms=[t1, t2])
        result = sga_equation_to_latex(details, notation='subscript')
        assert 'u_{x}' in result
        assert 'u_{xx}' in result

    def test_include_coefficients_still_works(self):
        """include_coefficients parameter is orthogonal to notation."""
        details = self._make_details_with_tree(_make_tree_d_u_x(), coeff=3.14)
        with_coeff = sga_equation_to_latex(
            details, notation='subscript', include_coefficients=True,
        )
        without_coeff = sga_equation_to_latex(
            details, notation='subscript', include_coefficients=False,
        )
        assert '3.14' in with_coeff
        assert '3.14' not in without_coeff
