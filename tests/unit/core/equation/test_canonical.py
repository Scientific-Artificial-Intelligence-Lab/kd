
from __future__ import annotations

import pytest

from kd.core.equation.canonical import canonicalize_expression


class TestCommutativeSort:

    def test_add_args_sorted(self) -> None:
        assert canonicalize_expression("add(b, a)") == "add(a,b)"

    def test_mul_args_sorted(self) -> None:
        assert canonicalize_expression("mul(z, a)") == "mul(a,z)"

    def test_already_canonical_is_fixed_point(self) -> None:
        canonical = "add(a,b)"
        assert canonicalize_expression(canonical) == canonical

    def test_double_canonicalize_is_idempotent(self) -> None:
        once = canonicalize_expression("add(b, a)")
        twice = canonicalize_expression(once)
        assert once == twice == "add(a,b)"


class TestNonCommutativePreserved:

    def test_sub_order_preserved(self) -> None:
        assert canonicalize_expression("sub(a, b)") == "sub(a,b)"
        assert canonicalize_expression("sub(b, a)") == "sub(b,a)"

    def test_div_order_preserved(self) -> None:
        assert canonicalize_expression("div(a, b)") == "div(a,b)"
        assert canonicalize_expression("div(b, a)") == "div(b,a)"

    def test_unknown_op_treated_non_commutative(self) -> None:

        assert canonicalize_expression("foo(b, a)") == "foo(b,a)"


class TestNestedBottomUp:

    def test_nested_add_inside_add(self) -> None:
        assert canonicalize_expression("add(add(c, b), a)") == "add(a,add(b,c))"

    def test_name_sorts_against_call_by_string(self) -> None:


        assert canonicalize_expression("add(mul(b, a), c)") == "add(c,mul(a,b))"

    def test_unary_arg_untouched(self) -> None:
        assert canonicalize_expression("add(diff2_x(u), u)") == "add(diff2_x(u),u)"

    def test_burgers_expression(self) -> None:
        result = canonicalize_expression("sub(diff2_x(u), mul(diff_x(u), u))")
        assert result == "sub(diff2_x(u),mul(diff_x(u),u))"

    def test_swapped_mul_inside_sub_converges(self) -> None:
        a = canonicalize_expression("sub(diff2_x(u), mul(diff_x(u), u))")
        b = canonicalize_expression("sub(diff2_x(u), mul(u, diff_x(u)))")
        assert a == b


class TestOutputFormat:

    def test_no_spaces_in_output(self) -> None:
        result = canonicalize_expression("add( mul(b, a), sub(c, d) )")
        assert " " not in result
        assert result == "add(mul(a,b),sub(c,d))"

    def test_strips_outer_whitespace(self) -> None:
        assert canonicalize_expression(" add(a, b) ") == "add(a,b)"

    def test_internal_whitespace_normalized(self) -> None:
        assert canonicalize_expression("add( a, b )") == "add(a,b)"

    def test_bare_name_passes_through(self) -> None:
        assert canonicalize_expression("u") == "u"


class TestErrorContract:

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            canonicalize_expression("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            canonicalize_expression(" ")

    def test_numeric_constants_rejected(self) -> None:
        with pytest.raises(ValueError, match="constants"):
            canonicalize_expression("add(1, 2)")

    def test_nested_constant_rejected(self) -> None:
        with pytest.raises(ValueError, match="constants"):
            canonicalize_expression("mul(2, u)")

    def test_infix_operators_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported IR syntax"):
            canonicalize_expression("a + b")

    def test_keyword_arguments_rejected(self) -> None:
        with pytest.raises(ValueError, match="Keyword arguments"):
            canonicalize_expression("add(a, b=1)")

    def test_attribute_callee_rejected(self) -> None:
        with pytest.raises(ValueError, match="bare token names"):
            canonicalize_expression("np.add(b, a)")

    def test_syntax_error_wrapped(self) -> None:
        with pytest.raises(ValueError, match="Invalid IR syntax"):
            canonicalize_expression("add(a,")


class TestPublicApi:

    def test_importable_from_package_root(self) -> None:
        from kd.core.equation import canonicalize_expression as from_root

        assert from_root is canonicalize_expression
