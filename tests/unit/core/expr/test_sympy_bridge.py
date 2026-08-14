
from __future__ import annotations

import pytest
import sympy
import torch

from kd.core.expr.sympy_bridge import (
    FormattedEquation,
    are_equivalent,
    format_pde,
    from_sympy,
    symbolic_diff,
    to_latex,
    to_sympy,
    to_unicode,
)






class TestToSympyBasic:

    @pytest.mark.smoke
    def test_single_variable(self) -> None:
        result = to_sympy("u")
        assert result == sympy.Symbol("u")

    def test_add(self) -> None:
        result = to_sympy("add(u, u_x)")
        assert result == sympy.Symbol("u") + sympy.Symbol("u_x")

    def test_mul(self) -> None:
        result = to_sympy("mul(u, u_x)")
        assert result == sympy.Symbol("u") * sympy.Symbol("u_x")

    def test_sub(self) -> None:
        result = to_sympy("sub(u, x)")
        assert result == sympy.Symbol("u") - sympy.Symbol("x")

    def test_div(self) -> None:
        result = to_sympy("div(u, x)")
        assert result == sympy.Symbol("u") / sympy.Symbol("x")

    def test_n2(self) -> None:
        result = to_sympy("n2(u)")
        assert result == sympy.Symbol("u") ** 2

    def test_n3(self) -> None:
        result = to_sympy("n3(u)")
        assert result == sympy.Symbol("u") ** 3







class TestToSympyDerivatives:

    def test_diff_x_simple(self) -> None:
        result = to_sympy("diff_x(u)")
        assert result == sympy.Symbol("u_x")

    def test_diff2_x_simple(self) -> None:
        result = to_sympy("diff2_x(u)")
        assert result == sympy.Symbol("u_xx")

    def test_diff_t(self) -> None:
        result = to_sympy("diff_t(u)")
        assert result == sympy.Symbol("u_t")

    def test_diff2_t(self) -> None:
        result = to_sympy("diff2_t(u)")
        assert result == sympy.Symbol("u_tt")

    def test_nested_diff_produces_compound_derivative_symbol(self) -> None:
        result = to_sympy("diff_y(u_x)")
        assert result == sympy.Symbol("u_x_y")

    def test_nested_higher_diff_produces_compound_symbol(self) -> None:
        result = to_sympy("diff2_y(u_xx)")
        assert result == sympy.Symbol("u_xx_yy")

    def test_nested_same_axis_diff_preserves_fd_sequence(self) -> None:
        result = to_sympy("diff_x(diff_x(u))")
        assert result == sympy.Symbol("u_x_x")

    def test_linear_diff_distributes_over_add(self) -> None:
        result = to_sympy("diff_x(add(u_x, diff_x(u)))")
        assert result == 2 * sympy.Symbol("u_x_x")

    def test_linear_diff_extracts_numeric_scalar(self) -> None:
        result = to_sympy("diff_x(mul(2, u_x))")
        assert result == 2 * sympy.Symbol("u_x_x")

    def test_nonlinear_product_diff_remains_open_form(self) -> None:
        u = sympy.Symbol("u")
        u_x = sympy.Symbol("u_x")
        diff_x = sympy.Function("diff_x")

        result = to_sympy("diff_x(mul(u, u_x))")

        assert result == diff_x(u * u_x)

    def test_function_diff_remains_round_trippable_open_form(self) -> None:
        u = sympy.Symbol("u")
        diff_x = sympy.Function("diff_x")

        result = to_sympy("diff_x(sin(u))")

        assert result == diff_x(sympy.sin(u))

    def test_diff_of_integer_literal_returns_zero(self) -> None:
        assert to_sympy("diff_x(0)") == sympy.Integer(0)

    def test_diff_of_float_literal_returns_zero(self) -> None:
        assert to_sympy("diff_x(2.5)") == sympy.Integer(0)







class TestToSympySpecialOperators:

    def test_lap_simple(self) -> None:
        u = sympy.Symbol("u")
        lap = sympy.Function("lap")

        result = to_sympy("lap(u)")

        assert result == lap(u)

    def test_lap_nested_expression(self) -> None:
        u = sympy.Symbol("u")
        lap = sympy.Function("lap")

        result = to_sympy("lap(n2(u))")

        assert result == lap(u**2)







class TestToSympyNested:

    def test_add_mul(self) -> None:
        result = to_sympy("add(mul(u, u_x), u_xx)")
        u = sympy.Symbol("u")
        u_x = sympy.Symbol("u_x")
        u_xx = sympy.Symbol("u_xx")
        expected = u * u_x + u_xx
        assert sympy.expand(result - expected) == 0

    def test_mul_with_nested_add(self) -> None:
        result = to_sympy("mul(u, add(x, t))")
        u, x, t = sympy.symbols("u x t")
        expected = u * (x + t)
        assert sympy.expand(result - expected) == 0

    def test_deeply_nested(self) -> None:
        result = to_sympy("div(n3(u), add(x, t))")
        u, x, t = sympy.symbols("u x t")
        expected = u**3 / (x + t)

        assert sympy.simplify(result - expected) == 0







class TestToSympyNumericLiterals:

    def test_integer_coefficient(self) -> None:
        result = to_sympy("mul(2, u)")
        assert result == 2 * sympy.Symbol("u")

    def test_float_coefficient(self) -> None:
        result = to_sympy("mul(0.1, u_xx)")
        u_xx = sympy.Symbol("u_xx")
        expected = sympy.Float(0.1) * u_xx
        assert abs(float(sympy.simplify(result - expected))) < 1e-15

    def test_negative_coefficient(self) -> None:
        result = to_sympy("mul(-1.0, u)")
        u = sympy.Symbol("u")
        expected = sympy.Float(-1.0) * u
        assert abs(float(sympy.simplify(result - expected))) < 1e-15







class TestToSympyErrors:

    def test_strict_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            to_sympy("@@@invalid!!!", strict=True)

    def test_nonstrict_invalid_returns_symbol_fallback(self) -> None:
        code = "@@@invalid!!!"
        result = to_sympy(code, strict=False)
        assert isinstance(result, sympy.Basic)

    def test_strict_is_default(self) -> None:
        with pytest.raises(ValueError):
            to_sympy("@@@invalid!!!")

    def test_empty_string_raises_strict(self) -> None:
        with pytest.raises(ValueError):
            to_sympy("", strict=True)

    def test_empty_string_nonstrict_fallback(self) -> None:
        result = to_sympy("", strict=False)
        assert isinstance(result, sympy.Basic)







class TestToLatex:

    @pytest.mark.smoke
    def test_returns_string(self) -> None:
        result = to_latex("add(u, u_x)")
        assert isinstance(result, str)

    def test_subscript_rendering(self) -> None:
        result = to_latex("u_x")
        assert "u_{x}" in result or "u_x" in result

    def test_add_latex(self) -> None:
        result = to_latex("add(u, u_x)")
        assert "+" in result

    def test_numeric_coefficient_preserved(self) -> None:
        result = to_latex("mul(0.1, u_xx)")
        assert "0.1" in result

    def test_strict_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            to_latex("@@@invalid!!!", strict=True)

    def test_nonstrict_invalid_returns_string(self) -> None:
        result = to_latex("@@@invalid!!!", strict=False)
        assert isinstance(result, str)


class TestCompositeDerivativeSpelling:

    def test_latex_composite_diff_renders_subscript(self) -> None:
        latex = to_latex("diff_x(mul(u, u))")
        assert "diff" not in latex
        assert "operatorname" not in latex
        assert "_{x}" in latex

    def test_latex_higher_order_merges_order_into_subscript(self) -> None:
        latex = to_latex("diff2_x(mul(u, u_x))")
        assert "diff" not in latex
        assert "_{xx}" in latex

    def test_latex_lap_renders_nabla(self) -> None:
        latex = to_latex("lap(u)")
        assert "operatorname" not in latex
        assert "nabla" in latex

    def test_latex_power_of_composite_diff(self) -> None:
        latex = to_latex("n2(diff_x(n2(u)))")
        assert "diff" not in latex

    def test_unicode_composite_diff_renders_subscript(self) -> None:
        text = to_unicode("diff_x(mul(u, u))")
        assert "diff" not in text
        assert "ₓ" in text

    def test_unicode_lap_renders_nabla(self) -> None:
        text = to_unicode("lap(u)")
        assert "lap" not in text
        assert "∇" in text

    def test_unicode_power_of_lap_parenthesizes_base(self) -> None:

        text = to_unicode("n2(lap(u))")
        assert "(∇²(u))" in text

    def test_str_single_line_fallback_no_ir_name(self) -> None:


        flat = str(to_sympy("diff_x(mul(u, u))"))
        assert "diff" not in flat
        assert "\n" not in flat

    def test_stock_sympy_latex_on_public_rhs_is_clean(self) -> None:


        result = format_pde(["diff_x(mul(u, u))"], torch.tensor([-0.5]))
        assert "operatorname" not in sympy.latex(result.rhs)

    def test_wrong_arity_lap_rejected_in_strict_mode(self) -> None:


        with pytest.raises(ValueError):
            to_latex("lap(u, u_x)")

    def test_format_pde_composite_diff_clean_on_both_renderings(self) -> None:
        result = format_pde(
            ["diff2_x(diff_x(u))", "diff_x(mul(u, u))"],
            torch.tensor([-0.0025, -0.5]),
            lhs="u_t",
        )
        assert "diff" not in result.latex
        assert "diff" not in result.unicode







class TestFormatPde:

    @pytest.mark.smoke
    def test_returns_formatted_equation(self) -> None:
        result = format_pde(
            ["mul(u, u_x)", "u_xx"],
            torch.tensor([-1.0, 0.1]),
        )
        assert isinstance(result, FormattedEquation)

    def test_has_required_fields(self) -> None:
        result = format_pde(
            ["mul(u, u_x)", "u_xx"],
            torch.tensor([-1.0, 0.1]),
        )
        assert hasattr(result, "latex")
        assert hasattr(result, "sympy_expr")
        assert hasattr(result, "lhs")
        assert hasattr(result, "rhs")

    def test_lhs_default_is_u_t(self) -> None:
        result = format_pde(
            ["u"],
            torch.tensor([1.0]),
        )
        assert result.lhs == "u_t"

    def test_custom_lhs(self) -> None:
        result = format_pde(
            ["u"],
            torch.tensor([1.0]),
            lhs="v_t",
        )
        assert result.lhs == "v_t"

    def test_latex_contains_lhs(self) -> None:
        result = format_pde(
            ["u"],
            torch.tensor([1.0]),
            lhs="u_t",
        )

        assert "u" in result.latex

    def test_burgers_equation(self) -> None:
        result = format_pde(
            ["mul(u, u_x)", "u_xx"],
            torch.tensor([-1.0, 0.1]),
        )
        latex = result.latex
        assert isinstance(latex, str)
        assert len(latex) > 0

        assert "mul(" not in latex
        assert "add(" not in latex

    def test_single_term(self) -> None:
        result = format_pde(
            ["u_xx"],
            torch.tensor([1.0]),
        )
        assert isinstance(result.latex, str)
        assert result.rhs is not None

    def test_zero_coefficient_excluded(self) -> None:
        result = format_pde(
            ["u", "u_x", "u_xx"],
            torch.tensor([1.0, 0.0, 0.5]),
        )

        rhs_symbols = result.rhs.free_symbols
        assert sympy.Symbol("u_x") not in rhs_symbols

        assert sympy.Symbol("u") in rhs_symbols
        assert sympy.Symbol("u_xx") in rhs_symbols

    def test_empty_terms_list(self) -> None:
        result = format_pde(
            [],
            torch.tensor([]),
        )
        assert isinstance(result, FormattedEquation)
        assert isinstance(result.latex, str)

    def test_lap_term_does_not_crash(self) -> None:
        result = format_pde(
            ["lap(u)"],
            torch.tensor([1.0]),
            lhs="u_t",
        )
        lap = sympy.Function("lap")
        u = sympy.Symbol("u")

        assert isinstance(result.latex, str)
        assert result.rhs == lap(u)







class TestFormatPdeProperties:

    def test_sympy_expr_is_equation(self) -> None:
        result = format_pde(
            ["u", "u_xx"],
            torch.tensor([2.0, -0.5]),
        )

        assert isinstance(result.sympy_expr, sympy.Eq)

    def test_rhs_matches_manual_construction(self) -> None:
        result = format_pde(
            ["u", "u_xx"],
            torch.tensor([2.0, -0.5]),
        )
        u = sympy.Symbol("u")
        u_xx = sympy.Symbol("u_xx")
        expected_rhs = 2.0 * u + (-0.5) * u_xx
        diff = sympy.simplify(result.rhs - expected_rhs)
        assert abs(float(diff)) < 1e-12

    def test_coefficient_sign_preserved(self) -> None:
        result = format_pde(
            ["u"],
            torch.tensor([-3.0]),
        )

        rhs_float = float(result.rhs.subs(sympy.Symbol("u"), 1.0))
        assert rhs_float < 0

    def test_selected_indices(self) -> None:
        result = format_pde(
            ["u", "u_x", "u_xx"],
            torch.tensor([1.0, 2.0, 3.0]),
            selected_indices=[0, 2],
        )
        rhs_symbols = result.rhs.free_symbols
        assert sympy.Symbol("u_x") not in rhs_symbols
        assert sympy.Symbol("u") in rhs_symbols
        assert sympy.Symbol("u_xx") in rhs_symbols

    def test_selected_indices_empty_list(self) -> None:
        result = format_pde(
            ["u", "u_x"],
            torch.tensor([1.0, 2.0]),
            selected_indices=[],
        )

        assert result.rhs == 0

    def test_negative_selected_index_rejected(self) -> None:
        with pytest.raises(ValueError, match="-1"):
            format_pde(
                ["u", "u_x"],
                torch.tensor([1.0, 2.0]),
                selected_indices=[-1],
            )

    def test_out_of_range_selected_index_rejected(self) -> None:
        with pytest.raises(ValueError, match="selected_indices"):
            format_pde(
                ["u", "u_x"],
                torch.tensor([1.0, 2.0]),
                selected_indices=[0, 5],
            )







class TestAreEquivalent:

    @pytest.mark.smoke
    def test_identical(self) -> None:
        assert are_equivalent("add(u, x)", "add(u, x)") is True

    def test_commutativity_add(self) -> None:
        assert are_equivalent("add(u, x)", "add(x, u)") is True

    def test_commutativity_mul(self) -> None:
        assert are_equivalent("mul(u, x)", "mul(x, u)") is True

    def test_distributivity(self) -> None:
        assert (
            are_equivalent(
                "mul(u, add(x, t))",
                "add(mul(u, x), mul(u, t))",
            )
            is True
        )

    def test_non_equivalent(self) -> None:
        assert are_equivalent("add(u, x)", "mul(u, x)") is False

    def test_sub_not_commutative(self) -> None:
        assert are_equivalent("sub(u, x)", "sub(x, u)") is False

    def test_lap_identical_expressions(self) -> None:
        assert are_equivalent("lap(u)", "lap(u)") is True

    def test_lap_different_arguments_not_equivalent(self) -> None:
        assert are_equivalent("lap(u)", "lap(v)") is False







class TestSymbolicDiff:

    @pytest.mark.smoke
    def test_returns_string(self) -> None:
        result = symbolic_diff("n2(u)", "u")
        assert isinstance(result, str)

    def test_n2_derivative(self) -> None:
        result = symbolic_diff("n2(u)", "u")

        result_sympy = to_sympy(result, strict=False)
        u = sympy.Symbol("u")
        expected = 2 * u
        assert sympy.expand(result_sympy - expected) == 0

    def test_n3_derivative(self) -> None:
        result = symbolic_diff("n3(u)", "u")
        result_sympy = to_sympy(result, strict=False)
        u = sympy.Symbol("u")
        expected = 3 * u**2
        assert sympy.expand(result_sympy - expected) == 0

    def test_diff_wrt_other_variable_is_zero(self) -> None:
        result = symbolic_diff("u", "x")
        result_sympy = to_sympy(result, strict=False)
        assert result_sympy == 0

    def test_mul_derivative(self) -> None:
        result = symbolic_diff("mul(u, x)", "u")
        result_sympy = to_sympy(result, strict=False)
        x = sympy.Symbol("x")
        assert sympy.expand(result_sympy - x) == 0







class TestSympyBridgeNegative:

    def test_to_sympy_unbalanced_parens_strict_raises(self) -> None:
        with pytest.raises(ValueError):
            to_sympy("add(u, x", strict=True)

    def test_to_sympy_unknown_function_strict_raises(self) -> None:
        with pytest.raises(ValueError):
            to_sympy("unknown_func(u, x)", strict=True)

    def test_format_pde_mismatched_length_raises(self) -> None:
        with pytest.raises((ValueError, RuntimeError)):
            format_pde(
                ["u", "u_x", "u_xx"],
                torch.tensor([1.0, 2.0]),
            )

    def test_are_equivalent_invalid_input(self) -> None:
        try:
            result = are_equivalent("@@@", "add(u, x)")
            assert result is False
        except (ValueError, SyntaxError):
            pass

    def test_symbolic_diff_empty_var_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            symbolic_diff("n2(u)", "")

    def test_format_pde_nan_coefficient(self) -> None:
        try:
            result = format_pde(
                ["u"],
                torch.tensor([float("nan")]),
            )

            assert isinstance(result.latex, str)
        except (ValueError, RuntimeError):
            pass

    def test_format_pde_inf_coefficient(self) -> None:
        try:
            result = format_pde(
                ["u"],
                torch.tensor([float("inf")]),
            )
            assert isinstance(result.latex, str)
        except (ValueError, RuntimeError):
            pass







class TestSympyToIrTrigExpLog:

    def test_sympy_to_ir_sin(self) -> None:
        from kd.core.expr.sympy_bridge import _sympy_to_ir

        x = sympy.Symbol("x")
        result = _sympy_to_ir(sympy.sin(x))

        assert result == "sin(x)"

    def test_sympy_to_ir_cos(self) -> None:
        from kd.core.expr.sympy_bridge import _sympy_to_ir

        x = sympy.Symbol("x")
        result = _sympy_to_ir(sympy.cos(x))
        assert result == "cos(x)"

    def test_sympy_to_ir_exp(self) -> None:
        from kd.core.expr.sympy_bridge import _sympy_to_ir

        x = sympy.Symbol("x")
        result = _sympy_to_ir(sympy.exp(x))
        assert result == "exp(x)"

    def test_sympy_to_ir_log(self) -> None:
        from kd.core.expr.sympy_bridge import _sympy_to_ir

        x = sympy.Symbol("x")
        result = _sympy_to_ir(sympy.log(x))
        assert result == "log(x)"


class TestSympyToIrLap:

    def test_sympy_to_ir_lap(self) -> None:
        from kd.core.expr.sympy_bridge import _sympy_to_ir

        lap_func = sympy.Function("lap")
        u = sympy.Symbol("u")

        result = _sympy_to_ir(lap_func(u))

        assert result == "lap(u)"


class TestSympyToIrOpenFormDiff:

    def test_sympy_to_ir_diff_sin_round_trip(self) -> None:
        from kd.core.expr.sympy_bridge import _sympy_to_ir

        u = sympy.Symbol("u")
        diff_x = sympy.Function("diff_x")

        result = _sympy_to_ir(diff_x(sympy.sin(u)))

        assert result == "diff_x(sin(u))"
        assert to_sympy(result) == diff_x(sympy.sin(u))

    def test_sympy_to_ir_diff_lap_round_trip(self) -> None:
        from kd.core.expr.sympy_bridge import _sympy_to_ir

        u = sympy.Symbol("u")
        diff_x = sympy.Function("diff_x")
        lap = sympy.Function("lap")

        result = _sympy_to_ir(diff_x(lap(u)))

        assert result == "diff_x(lap(u))"
        assert to_sympy(result) == diff_x(lap(u))







class TestSymbolicDiffWithTrig:

    def test_symbolic_diff_with_sin(self) -> None:
        result = symbolic_diff("sin(u)", "u")

        result_sympy = to_sympy(result, strict=False)
        u = sympy.Symbol("u")
        expected = sympy.cos(u)
        assert sympy.expand(result_sympy - expected) == 0

    def test_symbolic_diff_with_exp(self) -> None:
        result = symbolic_diff("exp(u)", "u")
        result_sympy = to_sympy(result, strict=False)
        u = sympy.Symbol("u")
        expected = sympy.exp(u)
        assert sympy.expand(result_sympy - expected) == 0

    def test_symbolic_diff_mul_sin(self) -> None:
        result = symbolic_diff("mul(u, sin(u))", "u")
        result_sympy = to_sympy(result, strict=False)
        u = sympy.Symbol("u")
        expected = sympy.sin(u) + u * sympy.cos(u)
        assert sympy.expand(result_sympy - expected) == 0


class TestSymbolicDiffWithLap:

    def test_symbolic_diff_mul_v_lap_u_wrt_v(self) -> None:
        result = symbolic_diff("mul(v, lap(u))", "v")

        assert result == "lap(u)"







class TestDerivativeSymbolNameNonSymbol:

    def test_derivative_symbol_name_preserves_function(self) -> None:
        from kd.core.expr.sympy_bridge import _derivative_symbol_name

        u = sympy.Symbol("u")
        expr = sympy.sin(u)
        result = _derivative_symbol_name(expr, "x", 1)


        assert result != "d1_x", (
            "_derivative_symbol_name fell back to generic 'd1_x' "
            "for sin(u), losing expression structure"
        )

        assert isinstance(result, str)
        assert len(result) > 0







class TestFromSympy:

    @pytest.mark.smoke
    def test_symbol(self) -> None:
        assert from_sympy(sympy.Symbol("u")) == "u"

    def test_integer(self) -> None:
        assert from_sympy(sympy.Integer(1)) == "1"

    def test_negative_integer(self) -> None:
        result = from_sympy(sympy.Integer(-3))
        assert to_sympy(result) == sympy.Integer(-3)

    def test_float(self) -> None:
        result = from_sympy(sympy.Float(0.1))
        assert abs(float(to_sympy(result)) - 0.1) < 1e-12

    def test_rational(self) -> None:
        result = from_sympy(sympy.Rational(1, 2))
        assert sympy.simplify(to_sympy(result) - sympy.Rational(1, 2)) == 0

    def test_add(self) -> None:
        u, u_x = sympy.symbols("u u_x")
        result = from_sympy(u + u_x)
        assert sympy.expand(to_sympy(result) - (u + u_x)) == 0

    def test_mul(self) -> None:
        u, u_x = sympy.symbols("u u_x")
        result = from_sympy(u * u_x)
        assert sympy.expand(to_sympy(result) - (u * u_x)) == 0

    def test_integer_power(self) -> None:
        u = sympy.Symbol("u")
        result = from_sympy(u**2)
        assert sympy.expand(to_sympy(result) - u**2) == 0

    def test_sin_via_round_trip(self) -> None:
        u = sympy.Symbol("u")
        result = from_sympy(sympy.sin(u))
        assert to_sympy(result) == sympy.sin(u)

    def test_cos_via_round_trip(self) -> None:
        u = sympy.Symbol("u")
        result = from_sympy(sympy.cos(u))
        assert to_sympy(result) == sympy.cos(u)

    def test_exp_via_round_trip(self) -> None:
        u = sympy.Symbol("u")
        result = from_sympy(sympy.exp(u))
        assert to_sympy(result) == sympy.exp(u)

    def test_log_via_round_trip(self) -> None:
        u = sympy.Symbol("u")
        result = from_sympy(sympy.log(u))
        assert to_sympy(result) == sympy.log(u)

    def test_lap_via_round_trip(self) -> None:
        u = sympy.Symbol("u")
        lap = sympy.Function("lap")
        result = from_sympy(lap(u))
        assert to_sympy(result) == lap(u)

    def test_diff_single_arg_round_trip(self) -> None:
        u = sympy.Symbol("u")
        diff_x = sympy.Function("diff_x")
        result = from_sympy(diff_x(u))
        assert to_sympy(result) == to_sympy("diff_x(u)")

    def test_nested_compound_round_trip(self) -> None:
        u, u_x, u_xx = sympy.symbols("u u_x u_xx")
        result = from_sympy(u * u_x + u_xx)
        assert sympy.expand(to_sympy(result) - (u * u_x + u_xx)) == 0


class TestFromSympyRoundTripProperty:

    @pytest.mark.parametrize(
        "ir",
        [
            "u",
            "1",
            "add(u, u_x)",
            "add(mul(u, u_x), u_xx)",
            "mul(0.1, u_xx)",
            "n2(u_x)",
            "n3(u)",
            "div(u, x)",
            "sin(u)",
            "cos(u)",
            "exp(u)",
            "log(u)",
            "lap(u)",
            "diff_x(u)",
        ],
    )
    def test_round_trip_is_equivalent(self, ir: str) -> None:
        assert are_equivalent(from_sympy(to_sympy(ir)), ir)


class TestFromSympyHardFailures:

    def test_sqrt_raises(self) -> None:
        with pytest.raises(ValueError):
            from_sympy(sympy.sqrt(sympy.Symbol("u")))

    def test_rational_power_raises(self) -> None:
        with pytest.raises(ValueError):
            from_sympy(sympy.Symbol("c0") ** sympy.Rational(1, 2))

    def test_abs_raises(self) -> None:
        with pytest.raises(ValueError):
            from_sympy(sympy.Abs(sympy.Symbol("u")))

    def test_piecewise_raises(self) -> None:
        u = sympy.Symbol("u")
        pw = sympy.Piecewise((u, u > 0), (sympy.Integer(0), True))
        with pytest.raises(ValueError):
            from_sympy(pw)

    def test_min_raises(self) -> None:
        a, b = sympy.symbols("a b")
        with pytest.raises(ValueError):
            from_sympy(sympy.Min(a, b))

    def test_max_raises(self) -> None:
        a, b = sympy.symbols("a b")
        with pytest.raises(ValueError):
            from_sympy(sympy.Max(a, b))

    def test_unknown_function_raises(self) -> None:
        weird = sympy.Function("weird")
        with pytest.raises(ValueError):
            from_sympy(weird(sympy.Symbol("u")))

    def test_symbol_power_symbol_raises(self) -> None:
        x = sympy.Symbol("x")
        with pytest.raises(ValueError):
            from_sympy(x**x)


class TestOneConstantColumn:

    def test_one_folds_to_integer_in_display(self) -> None:
        assert to_latex("one") == "1"
        assert to_latex("mul(0.5, one)") == "0.5"
        assert to_sympy("mul(2, one)") == sympy.Integer(2)
