
from __future__ import annotations

import pytest

from kd.core.equation import (
    EquationAttrs,
    Evolution,
    LhsSpec,
    Scalar,
    make_evolution,
    residual_program,
)

_LHS = LhsSpec(field="u", axis="t", order=1)


class TestResidualProgramEvolution:
    @pytest.mark.unit
    def test_single_term_renders_sub_of_scaled_term_and_lhs(self) -> None:
        eq = make_evolution(_LHS, [("diff2_x(u)", Scalar(0.1))])

        assert residual_program(eq) == "sub(mul(0.1, diff2_x(u)), u_t)"

    @pytest.mark.unit
    def test_multi_term_renders_added_rhs_before_lhs_subtraction(self) -> None:
        eq = make_evolution(
            _LHS, [("u_x", Scalar(-1.0)), ("diff2_x(u)", Scalar(0.1))]
        )

        assert residual_program(eq) == (
            "sub(add(mul(-1.0, u_x), mul(0.1, diff2_x(u))), u_t)"
        )

    @pytest.mark.unit
    def test_second_order_lhs_uses_render_lhs_label_token(self) -> None:
        wave_lhs = LhsSpec(field="u", axis="t", order=2)
        eq = make_evolution(wave_lhs, [("diff2_x(u)", Scalar(1.0))])

        program = residual_program(eq)


        from kd.core.equation.rendering import render_lhs_label

        assert program.endswith(f", {render_lhs_label(wave_lhs)})")

    @pytest.mark.unit
    def test_non_finite_coefficient_raises(self) -> None:


        eq = Evolution(
            lhs_spec=_LHS,
            terms=(("u_x", Scalar(float("nan"))),),
            attrs=EquationAttrs(),
        )

        with pytest.raises(ValueError, match="non-finite"):
            residual_program(eq)


class TestResidualProgramGuards:

    @pytest.mark.unit
    def test_empty_terms_raise_value_error(self) -> None:


        eq = Evolution(lhs_spec=_LHS, terms=(), attrs=EquationAttrs())

        with pytest.raises(ValueError, match="no terms"):
            residual_program(eq)
