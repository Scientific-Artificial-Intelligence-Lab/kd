
from __future__ import annotations

from typing import Final, assert_never

from kd.core.equation.projection import active_law
from kd.core.equation.types import (
    Equation,
    Evolution,
    Homogeneous,
    LhsSpec,
    Regression,
    Scalar,
)
from kd.core.expr.naming import build_derivative_name
from kd.core.expr.sympy_bridge import FormattedEquation, format_pde

HOMOGENEOUS_LHS_LABEL: Final[str] = "0"







DEFAULT_LHS_LABEL: Final[str] = "u_t"


def render_homogeneous_label(eq: Homogeneous) -> str:
    return " + ".join(term_ir for term_ir, _coefficient in eq.terms) + " = 0"


def render_lhs_label(lhs_spec: LhsSpec) -> str:
    if lhs_spec.order == 1:
        return f"{lhs_spec.field}_{lhs_spec.axis}"
    return build_derivative_name(lhs_spec.field, lhs_spec.axis, lhs_spec.order)


def scalar_equation_terms(equation: Equation) -> tuple[list[str], list[float]]:


    if equation.active_indices == ():
        return [], []
    law = active_law(equation)
    terms: list[str] = []
    coefficients: list[float] = []
    for term, coefficient in law.terms:
        if not isinstance(coefficient, Scalar):
            raise NotImplementedError("Visualization requires scalar coefficients")
        terms.append(term)
        coefficients.append(coefficient.value)
    return terms, coefficients


def format_equation(equation: Equation) -> FormattedEquation:
    terms, coefficients = scalar_equation_terms(equation)
    match equation:
        case Evolution():
            lhs = render_lhs_label(equation.lhs_spec)
        case Regression():
            lhs = equation.lhs_spec.field
        case Homogeneous():
            lhs = HOMOGENEOUS_LHS_LABEL
        case _:
            assert_never(equation)
    return format_pde(terms, coefficients, lhs=lhs)


def equation_text(equation: Equation) -> str:
    formatted = format_equation(equation)
    return f"{formatted.lhs} = {formatted.rhs}"
