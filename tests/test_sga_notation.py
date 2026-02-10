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
