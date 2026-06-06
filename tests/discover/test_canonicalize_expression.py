
from __future__ import annotations

import pytest

from kd.search.discover.utils.canonicalize import canonicalize_expression


class TestCanonicalizeCommutative:

    def test_add_args_sorted_lexicographically(self) -> None:
        assert canonicalize_expression("add(b, a)") == "add(a,b)"

    def test_mul_args_sorted_lexicographically(self) -> None:
        assert canonicalize_expression("mul(z, a)") == "mul(a,z)"

    def test_already_sorted_is_idempotent(self) -> None:
        canonical = "add(a,b)"
        assert canonicalize_expression(canonical) == canonical

    def test_double_canonicalize_is_idempotent(self) -> None:
        once = canonicalize_expression("add(b, a)")
        twice = canonicalize_expression(once)
        assert once == twice == "add(a,b)"


class TestCanonicalizeNonCommutative:

    def test_sub_argument_order_preserved(self) -> None:
        assert canonicalize_expression("sub(a, b)") == "sub(a,b)"
        assert canonicalize_expression("sub(b, a)") == "sub(b,a)"

    def test_div_argument_order_preserved(self) -> None:
        assert canonicalize_expression("div(a, b)") == "div(a,b)"
        assert canonicalize_expression("div(b, a)") == "div(b,a)"


class TestCanonicalizeNested:

    def test_nested_add_inside_add(self) -> None:



        result = canonicalize_expression("add(add(c, b), a)")
        assert result == "add(a,add(b,c))"

    def test_mul_inside_add_canonicalizes_inner(self) -> None:


        result = canonicalize_expression("add(mul(b, a), c)")
        assert result == "add(c,mul(a,b))"

    def test_diff_args_preserved_in_unary_op(self) -> None:

        result = canonicalize_expression("add(diff2_x(u), u)")
        assert result == "add(diff2_x(u),u)"

    def test_burgers_pde_canonicalization(self) -> None:






        result = canonicalize_expression("sub(diff2_x(u), mul(diff_x(u), u))")
        assert result == "sub(diff2_x(u),mul(diff_x(u),u))"

    def test_swapped_mul_args_inside_sub_are_canonicalized(self) -> None:



        a = canonicalize_expression("sub(diff2_x(u), mul(diff_x(u), u))")
        b = canonicalize_expression("sub(diff2_x(u), mul(u, diff_x(u)))")
        assert a == b


class TestCanonicalizeWhitespaceAndIdempotency:

    def test_strips_leading_trailing_whitespace(self) -> None:
        assert canonicalize_expression(" add(a, b) ") == "add(a,b)"

    def test_internal_whitespace_in_args(self) -> None:

        assert canonicalize_expression("add( a, b )") == "add(a,b)"


class TestCanonicalizeErrors:

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            canonicalize_expression("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            canonicalize_expression(" ")

    def test_numeric_constants_rejected(self) -> None:
        with pytest.raises(ValueError, match="constants"):
            canonicalize_expression("add(1, 2)")

    def test_infix_operators_rejected(self) -> None:

        with pytest.raises(ValueError, match="Unsupported IR syntax"):
            canonicalize_expression("a + b")

    def test_keyword_arguments_rejected(self) -> None:
        with pytest.raises(ValueError, match="Keyword arguments"):
            canonicalize_expression("add(a, b=1)")

    def test_syntax_error_propagated(self) -> None:
        with pytest.raises(ValueError, match="Invalid IR syntax"):
            canonicalize_expression("add(a,")
