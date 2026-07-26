
from __future__ import annotations

import math
from typing import assert_never

from kd.core.equation.rendering import render_lhs_label
from kd.core.equation.types import Equation, Evolution, Homogeneous, Scalar, Term


def _rendered_scaled_terms(terms: tuple[Term, ...]) -> list[str]:
    rendered_terms: list[str] = []
    for term_ir, coefficient in terms:
        if not isinstance(coefficient, Scalar):
            raise NotImplementedError(
                f"{type(coefficient).__name__} residual coefficients "
                "are reserved"
            )





        if not math.isfinite(coefficient.value):
            raise ValueError(
                f"residual_program cannot render a non-finite coefficient "
                f"({coefficient.value!r}) for term {term_ir!r}"
            )
        rendered_terms.append(f"mul({coefficient.value!r}, {term_ir})")
    return rendered_terms


def _folded_sum(rendered_terms: list[str]) -> str:
    if not rendered_terms:
        raise ValueError(
            "residual_program cannot render an equation with no terms"
        )
    total = rendered_terms[-1]
    for rendered_term in reversed(rendered_terms[:-1]):
        total = f"add({rendered_term}, {total})"
    return total


def residual_program(eq: Equation) -> str:
    match eq:
        case Evolution():
            rhs = _folded_sum(_rendered_scaled_terms(eq.terms))
            return f"sub({rhs}, {render_lhs_label(eq.lhs_spec)})"
        case Homogeneous():
            return _folded_sum(_rendered_scaled_terms(eq.terms))
    assert_never(eq)
