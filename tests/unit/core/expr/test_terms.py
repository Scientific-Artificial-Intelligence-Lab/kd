
from __future__ import annotations

import pytest

from kd.core.expr import FunctionRegistry, split_terms






@pytest.fixture
def registry() -> FunctionRegistry:
    return FunctionRegistry.create_default()







@pytest.mark.smoke
class TestSplitTermsSmoke:

    def test_split_terms_exists(self) -> None:
        from kd.core.expr import split_terms

        assert callable(split_terms)

    def test_split_terms_returns_list(self, registry: FunctionRegistry) -> None:
        result = split_terms("u", registry)
        assert isinstance(result, list)
        assert all(isinstance(item, str) for item in result)

    def test_single_variable(self, registry: FunctionRegistry) -> None:
        result = split_terms("u", registry)
        assert result == ["u"]







@pytest.mark.unit
class TestSingleTerm:

    def test_single_variable(self, registry: FunctionRegistry) -> None:
        assert split_terms("u", registry) == ["u"]
        assert split_terms("v", registry) == ["v"]
        assert split_terms("u_x", registry) == ["u_x"]
        assert split_terms("u_xx", registry) == ["u_xx"]

    def test_mul_expression(self, registry: FunctionRegistry) -> None:
        result = split_terms("mul(u, v)", registry)
        assert result == ["mul(u, v)"]

    def test_mul_with_derivative(self, registry: FunctionRegistry) -> None:
        result = split_terms("mul(u, u_x)", registry)
        assert result == ["mul(u, u_x)"]

    def test_div_expression(self, registry: FunctionRegistry) -> None:
        result = split_terms("div(u, v)", registry)
        assert result == ["div(u, v)"]

    def test_unary_function(self, registry: FunctionRegistry) -> None:
        assert split_terms("sin(u)", registry) == ["sin(u)"]
        assert split_terms("cos(u)", registry) == ["cos(u)"]
        assert split_terms("neg(u)", registry) == ["neg(u)"]
        assert split_terms("n2(u)", registry) == ["n2(u)"]

    def test_nested_non_add_sub(self, registry: FunctionRegistry) -> None:
        result = split_terms("mul(sin(u), cos(v))", registry)
        assert result == ["mul(sin(u), cos(v))"]

    def test_numeric_constant(self, registry: FunctionRegistry) -> None:
        assert split_terms("1.5", registry) == ["1.5"]
        assert split_terms("0", registry) == ["0"]







@pytest.mark.unit
class TestAddSplitting:

    def test_simple_add(self, registry: FunctionRegistry) -> None:
        result = split_terms("add(a, b)", registry)
        assert result == ["a", "b"]

    def test_add_variables(self, registry: FunctionRegistry) -> None:
        result = split_terms("add(u, v)", registry)
        assert result == ["u", "v"]

    def test_add_with_derivatives(self, registry: FunctionRegistry) -> None:
        result = split_terms("add(u_x, u_xx)", registry)
        assert result == ["u_x", "u_xx"]

    def test_nested_add_right(self, registry: FunctionRegistry) -> None:
        result = split_terms("add(a, add(b, c))", registry)
        assert result == ["a", "b", "c"]

    def test_nested_add_left(self, registry: FunctionRegistry) -> None:
        result = split_terms("add(add(a, b), c)", registry)
        assert result == ["a", "b", "c"]

    def test_deeply_nested_add(self, registry: FunctionRegistry) -> None:

        result = split_terms("add(add(a, add(b, c)), add(d, e))", registry)
        assert result == ["a", "b", "c", "d", "e"]

    def test_add_with_complex_terms(self, registry: FunctionRegistry) -> None:

        result = split_terms("add(mul(u, u_x), u_xx)", registry)
        assert result == ["mul(u, u_x)", "u_xx"]

    def test_add_preserves_inner_add_in_non_top(
        self, registry: FunctionRegistry
    ) -> None:

        result = split_terms("mul(add(a, b), c)", registry)
        assert result == ["mul(add(a, b), c)"]







@pytest.mark.unit
class TestSubHandling:

    def test_simple_sub(self, registry: FunctionRegistry) -> None:
        result = split_terms("sub(a, b)", registry)
        assert result == ["a", "neg(b)"]

    def test_sub_with_variables(self, registry: FunctionRegistry) -> None:
        result = split_terms("sub(u, v)", registry)
        assert result == ["u", "neg(v)"]

    def test_sub_with_complex_second_arg(self, registry: FunctionRegistry) -> None:

        result = split_terms("sub(a, mul(b, c))", registry)
        assert result == ["a", "neg(mul(b, c))"]

    def test_sub_first_arg_is_add(self, registry: FunctionRegistry) -> None:

        result = split_terms("sub(add(a, b), c)", registry)
        assert result == ["a", "b", "neg(c)"]

    def test_sub_second_arg_is_add(self, registry: FunctionRegistry) -> None:
        result = split_terms("sub(a, add(b, c))", registry)

        assert result == ["a", "neg(b)", "neg(c)"]

    def test_sub_second_arg_is_sub(self, registry: FunctionRegistry) -> None:
        result = split_terms("sub(a, sub(b, c))", registry)

        assert result == ["a", "neg(b)", "c"]







@pytest.mark.unit
class TestMixedAddSub:

    def test_add_then_sub(self, registry: FunctionRegistry) -> None:


        result = split_terms("add(a, sub(b, c))", registry)
        assert result == ["a", "b", "neg(c)"]

    def test_sub_then_add(self, registry: FunctionRegistry) -> None:
        result = split_terms("sub(a, add(b, c))", registry)
        assert result == ["a", "neg(b)", "neg(c)"]

    def test_complex_mixed(self, registry: FunctionRegistry) -> None:




        result = split_terms("add(sub(a, b), sub(c, d))", registry)
        assert result == ["a", "neg(b)", "c", "neg(d)"]







@pytest.mark.unit
class TestBurgersExamples:

    def test_burgers_rhs_split(self, registry: FunctionRegistry) -> None:
        result = split_terms("add(mul(u, u_x), u_xx)", registry)
        assert result == ["mul(u, u_x)", "u_xx"]

    def test_burgers_full_equation_split(self, registry: FunctionRegistry) -> None:

        result = split_terms("add(neg(mul(u, u_x)), mul(nu, u_xx))", registry)
        assert result == ["neg(mul(u, u_x))", "mul(nu, u_xx)"]







@pytest.mark.unit
class TestEdgeCases:

    def test_empty_string_raises(self, registry: FunctionRegistry) -> None:
        with pytest.raises(ValueError):
            split_terms("", registry)

    def test_whitespace_only_raises(self, registry: FunctionRegistry) -> None:
        with pytest.raises(ValueError):
            split_terms(" ", registry)

    def test_invalid_syntax_raises(self, registry: FunctionRegistry) -> None:
        with pytest.raises(ValueError):
            split_terms("add(a, )", registry)

        with pytest.raises(ValueError):
            split_terms("add(a", registry)

    def test_triple_add_argument_not_supported(
        self, registry: FunctionRegistry
    ) -> None:



        with pytest.raises((ValueError, TypeError)):
            split_terms("add(a, b, c)", registry)

    def test_unknown_function_preserves(self, registry: FunctionRegistry) -> None:
        result = split_terms("unknown_func(a, b)", registry)
        assert result == ["unknown_func(a, b)"]

    def test_very_deep_nesting(self, registry: FunctionRegistry) -> None:

        expr = "add(add(add(add(a, b), c), d), e)"
        result = split_terms(expr, registry)
        assert result == ["a", "b", "c", "d", "e"]







@pytest.mark.unit
class TestNegSimplification:

    def test_double_neg_preserved(self, registry: FunctionRegistry) -> None:

        result = split_terms("sub(a, neg(b))", registry)

        assert result == ["a", "neg(neg(b))"]

    def test_existing_neg_in_expression(self, registry: FunctionRegistry) -> None:
        result = split_terms("add(neg(a), b)", registry)
        assert result == ["neg(a)", "b"]







@pytest.mark.unit
class TestInfixRejection:

    def test_infix_add_rejected(self, registry: FunctionRegistry) -> None:
        with pytest.raises(ValueError, match="(?i)infix|BinOp|operator"):
            split_terms("u + u_xx", registry)

    def test_infix_sub_rejected(self, registry: FunctionRegistry) -> None:
        with pytest.raises(ValueError, match="(?i)infix|BinOp|operator"):
            split_terms("u - u_xx", registry)

    def test_infix_mul_rejected(self, registry: FunctionRegistry) -> None:
        with pytest.raises(ValueError, match="(?i)infix|BinOp|operator"):
            split_terms("u * u_xx", registry)

    def test_funcall_add_still_works(self, registry: FunctionRegistry) -> None:
        result = split_terms("add(u, u_xx)", registry)
        assert result == ["u", "u_xx"]

    def test_infix_nested_in_funcall_rejected(self, registry: FunctionRegistry) -> None:
        with pytest.raises(ValueError, match="(?i)infix|BinOp|operator"):
            split_terms("add(u + v, u_xx)", registry)
