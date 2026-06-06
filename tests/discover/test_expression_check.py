
from __future__ import annotations

import pytest




import kd.search.discover.tokens.prior
from kd.search.discover.runners.expression_check import (
    check_diffusion_sign_consistency,
    parse_to_sympy,
)






@pytest.mark.unit
class TestParseToSympy:
    def test_atomic_variable(self) -> None:
        expr = parse_to_sympy("u")
        assert str(expr) == "u"

    def test_add(self) -> None:
        expr = parse_to_sympy("add(u, diff2_x(u))")

        assert "diff2_x" in str(expr)
        assert "u" in str(expr)

    def test_sub(self) -> None:
        expr = parse_to_sympy("sub(diff2_x(u), diff2_y(u))")
        assert "diff2_x" in str(expr)
        assert "diff2_y" in str(expr)
        assert "-" in str(expr)

    def test_n3_power(self) -> None:
        expr = parse_to_sympy("n3(u)")
        assert "u**3" in str(expr).replace(" ", "")

    def test_neg_unary(self) -> None:
        expr = parse_to_sympy("neg(u)")
        assert str(expr) == "-u"

    def test_unsupported_term_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_to_sympy("sqrt(u)")







@pytest.mark.unit
class TestSignConsistency:
    def test_correct_form_passes(self) -> None:
        ok, reason = check_diffusion_sign_consistency(
            "add(add(diff2_x(u), diff2_y(u)), sub(u, n3(u)))"
        )
        assert ok, reason

    def test_paper_ic_cheating_form_fails(self) -> None:
        ok, reason = check_diffusion_sign_consistency(
            "sub(sub(diff2_x(u), diff2_y(u)), add(u, n3(u)))"
        )
        assert not ok
        assert "diff2_x" in reason and "diff2_y" in reason

    def test_diff2_y_minus_diff2_x_also_fails(self) -> None:
        ok, reason = check_diffusion_sign_consistency(
            "sub(diff2_y(u), diff2_x(u))"
        )
        assert not ok

    def test_only_diff2_x_passes(self) -> None:
        ok, _ = check_diffusion_sign_consistency("diff2_x(u)")
        assert ok

    def test_only_diff2_y_passes(self) -> None:
        ok, _ = check_diffusion_sign_consistency("diff2_y(u)")
        assert ok

    def test_no_laplacian_passes(self) -> None:
        ok, _ = check_diffusion_sign_consistency("mul(u, diff_x(u))")
        assert ok

    def test_negated_correct_form_passes(self) -> None:
        ok, _ = check_diffusion_sign_consistency(
            "neg(add(diff2_x(u), diff2_y(u)))"
        )
        assert ok

    def test_v10_seed1_form_fails(self) -> None:
        ok, reason = check_diffusion_sign_consistency(
            "sub(add(diff2_y(u), sub(diff2_x(u), n3(u))), u)"
        )
        assert ok, reason

    def test_v10_seed2_form_fails(self) -> None:
        expr = (
            "sub(diff2_y(u), sub(sub(sub(n3(u), u), "
            "add(diff2_x(u), x)), mul(y, div(t, t))))"
        )
        ok, _ = check_diffusion_sign_consistency(expr)
        assert ok

    def test_invalid_expression_returns_false(self) -> None:
        ok, reason = check_diffusion_sign_consistency("not_a_term(u)")
        assert not ok
        assert "parse" in reason.lower() or "unsupported" in reason.lower()
