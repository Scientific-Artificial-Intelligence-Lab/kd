
from __future__ import annotations

import pytest

from kd.core.expr.term_key import structure_term_key






@pytest.mark.smoke
def test_plain_terms_key_to_themselves() -> None:
    assert structure_term_key("u") == "u"
    assert structure_term_key("diff2_x(u)") == "diff2_x(u)"







@pytest.mark.parametrize(
    "term",
    [
        "mul(2.0, u)",
        "mul(u, 2.0)",
        "mul(-3, u)",
        "div(u, 2.0)",
        "mul(0.5, mul(2.0, u))",
    ],
)
def test_scalar_factors_are_stripped(term: str) -> None:
    assert structure_term_key(term) == structure_term_key("u")


def test_div_numerator_is_not_stripped() -> None:
    assert structure_term_key("div(2.0, u)") != structure_term_key("u")


def test_non_scalar_mul_survives() -> None:
    assert structure_term_key("mul(u, diff_x(u))") != structure_term_key("u")







@pytest.mark.parametrize(
    "term",
    ["neg(u)", "neg(neg(u))", "neg(neg(neg(u)))", "neg(mul(2.0, u))"],
)
def test_neg_folds_at_any_depth(term: str) -> None:
    assert structure_term_key(term) == structure_term_key("u")


def test_sign_blindness_is_the_contract() -> None:
    assert structure_term_key("neg(diff2_x(u))") == structure_term_key("diff2_x(u)")


def test_inner_neg_in_nonscalar_product_is_preserved() -> None:
    assert structure_term_key("mul(neg(u), diff_x(u))") != structure_term_key(
        "mul(u, diff_x(u))"
    )







def test_commutative_mul_siblings_unify() -> None:
    assert structure_term_key("mul(u, diff_x(u))") == structure_term_key(
        "mul(diff_x(u), u)"
    )


def test_commutative_add_siblings_unify() -> None:
    assert structure_term_key("add(u, n3(u))") == structure_term_key("add(n3(u), u)")


def test_nested_commutative_siblings_unify() -> None:
    assert structure_term_key("mul(diff_x(mul(u, x)), u)") == structure_term_key(
        "mul(u, diff_x(mul(x, u)))"
    )


def test_non_commutative_argument_order_is_preserved() -> None:
    assert structure_term_key("sub(u, x)") != structure_term_key("sub(x, u)")
    assert structure_term_key("div(u, x)") != structure_term_key("div(x, u)")


def test_whitespace_is_not_identity() -> None:
    assert structure_term_key("mul(diff_x(u), u)") == structure_term_key(
        "mul(diff_x(u),u)"
    )







def test_embedded_constant_falls_back_without_raising() -> None:
    key = structure_term_key("add(1.0, u)")
    assert key == "add(1.0,u)"

    assert key == structure_term_key("add(1.0,u)")

    assert key != structure_term_key("add(u, 1.0)")


def test_syntax_error_falls_back_to_whitespace_stripped_input() -> None:
    assert structure_term_key("mul(u,") == "mul(u,"
    assert structure_term_key("mul(u, ") == "mul(u,"
    assert structure_term_key("") == ""


def test_leading_whitespace_still_canonicalizes() -> None:
    assert structure_term_key(" mul(u, diff_x(u))") == structure_term_key(
        "mul(diff_x(u), u)"
    )
    assert structure_term_key("\tmul(u, diff_x(u))") == structure_term_key(
        "mul(diff_x(u), u)"
    )


def test_null_byte_falls_back_instead_of_raising() -> None:
    assert structure_term_key("mul(u,\x00)") == "mul(u,\x00)"







@pytest.mark.parametrize(
    "term",
    [
        "u",
        "neg(mul(2.0, diff_x(u)))",
        "mul(u, diff_x(u))",
        "add(1.0, u)",
        "mul(u,",
    ],
)
def test_key_is_idempotent(term: str) -> None:
    once = structure_term_key(term)
    assert structure_term_key(once) == once
