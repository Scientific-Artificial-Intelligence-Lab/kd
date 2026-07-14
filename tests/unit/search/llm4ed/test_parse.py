
from __future__ import annotations

import ast

import pytest

from kd.search.llm4ed.parse import (
    InexpressibleTermError,
    Llm4edParseError,
    ParsedEquation,
    UndefinedOperandError,
    UndefinedOperatorError,
    parse_equation,
)


OPERANDS = ("x", "u_x", "u_xx", "u_xxx", "u")









IR_TRUTH_TABLE: list[tuple[str, list[str]]] = [

    ("u_xx", ["u_xx"]),
    ("u_xxx", ["u_xxx"]),

    ("u^2", ["n2(u)"]),
    ("u^3", ["n3(u)"]),

    ("u^4", ["n2(n2(u))"]),
    ("u^5", ["mul(n2(n2(u)), u)"]),




    ("u^-2", ["recip(n2(u))"]),
    ("u/x", ["mul(u, recip(x))"]),
    ("u/x^2", ["mul(u, recip(n2(x)))"]),
    ("1/u_x^2", ["recip(n2(u_x))"]),
    ("u_x/u_xx", ["mul(u_x, recip(u_xx))"]),

    ("u/(u+x)", ["mul(u, recip(add(u, x)))"]),

    ("x u_x", ["mul(u_x, x)"]),
    ("2u", ["mul(2.0, u)"]),
    ("xu", ["mul(u, x)"]),

    ("2*u_xx - 0.5*u*u_x", ["mul(2.0, u_xx)", "mul(mul(-0.5, u), u_x)"]),
    ("u/3", ["mul(0.3333333333333333, u)"]),

    ("(u+u_x)^2", ["n2(u)", "n2(u_x)", "mul(mul(2.0, u), u_x)"]),

    ("(u*u_x)^2", ["mul(n2(u), n2(u_x))"]),

    ("u*u_x*u_xx", ["mul(mul(u, u_x), u_xx)"]),
    ("u^2 u_x", ["mul(u_x, n2(u))"]),

    ("0.1*u_xx - u*u_x", ["mul(0.1, u_xx)", "mul(mul(-1.0, u), u_x)"]),
    ("u_xx - u + u^3", ["u_xx", "n3(u)", "mul(-1.0, u)"]),
    (
        "u*u_xx + u_x^2 + u + u^2",
        ["u", "n2(u)", "n2(u_x)", "mul(u, u_xx)"],
    ),
]


ERROR_TRUTH_TABLE: list[tuple[str, type[Exception]]] = [

    ("u_xxxx", UndefinedOperandError),
    ("v*u", UndefinedOperandError),
    ("u_t + u", UndefinedOperandError),

    ("u^6", UndefinedOperatorError),
    ("u^1", UndefinedOperatorError),

    ("v^6", UndefinedOperatorError),

    ("sin(u)", InexpressibleTermError),
    ("sqrt(u)", InexpressibleTermError),
    ("u_x + 1", InexpressibleTermError),
    ("u - u", InexpressibleTermError),
    ("1/x/x/x/x/x/x", InexpressibleTermError),

    ("u_x +* u", Llm4edParseError),
    ("", Llm4edParseError),
    (" ", Llm4edParseError),
]


class TestIRTruthTable:

    @pytest.mark.parametrize(("equation", "expected_ir"), IR_TRUTH_TABLE)
    def test_ir_conversion(self, equation: str, expected_ir: list[str]) -> None:
        parsed = parse_equation(equation, OPERANDS)
        assert [term.ir for term in parsed.terms] == expected_ir

    @pytest.mark.parametrize(("equation", "expected_ir"), IR_TRUTH_TABLE)
    def test_ir_is_valid_python_expression(
        self, equation: str, expected_ir: list[str]
    ) -> None:
        del expected_ir
        parsed = parse_equation(equation, OPERANDS)
        for term in parsed.terms:
            ast.parse(term.ir, mode="eval")


class TestErrorTruthTable:

    @pytest.mark.parametrize(("equation", "error"), ERROR_TRUTH_TABLE)
    def test_raises(self, equation: str, error: type[Exception]) -> None:
        with pytest.raises(error):
            parse_equation(equation, OPERANDS)

    def test_all_errors_are_llm4ed_parse_errors(self) -> None:
        for equation, _ in ERROR_TRUTH_TABLE:
            with pytest.raises(Llm4edParseError):
                parse_equation(equation, OPERANDS)


class TestParsedEquationStructure:

    def test_result_carries_equation_and_terms(self) -> None:
        parsed = parse_equation("u^2 + u*u_x", OPERANDS)
        assert isinstance(parsed, ParsedEquation)
        assert parsed.equation == "u^2 + u*u_x"
        assert len(parsed.terms) == 2

    def test_term_str_uses_edl_caret_convention(self) -> None:
        parsed = parse_equation("u^2 + u*u_x", OPERANDS)
        assert [term.term_str for term in parsed.terms] == ["u^2", "u*u_x"]

    def test_single_term_equation(self) -> None:
        parsed = parse_equation("u*u_x", OPERANDS)
        assert len(parsed.terms) == 1
        assert parsed.terms[0].ir == "mul(u, u_x)"

    def test_sympy_expr_is_expanded(self) -> None:
        import sympy

        parsed = parse_equation("(u+u_x)^2", OPERANDS)
        u, u_x = sympy.Symbol("u"), sympy.Symbol("u_x")
        assert parsed.sympy_expr == sympy.expand(u**2 + 2 * u * u_x + u_x**2)

    def test_terms_follow_sympy_args_order(self) -> None:
        parsed = parse_equation("u_xx - u + u^3", OPERANDS)
        assert [t.term_str for t in parsed.terms] == ["u_xx", "u^3", "-u"]

    def test_coefficient_factor_not_stripped(self) -> None:
        parsed = parse_equation("(u+u_x)^2", OPERANDS)
        cross = [t for t in parsed.terms if "2.0" in t.ir]
        assert len(cross) == 1, "expanded cross term must keep its 2.0 factor"
