
from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from kd.core.equation.types import (
    Equation,
    Evolution,
    Homogeneous,
    LhsSpec,
    Term,
    fold_terms,
)


@dataclass(frozen=True)
class RegressionForm:

    lhs_spec: LhsSpec
    term_irs: tuple[str, ...]


@dataclass(frozen=True)
class PivotRegressionForm:

    pivot_ir: str
    rhs_irs: tuple[str, ...]


def lower_to_regression(eq: Equation) -> RegressionForm | PivotRegressionForm:
    match eq:
        case Evolution():
            term_irs: list[str] = []
            folded_term_irs: list[str] = fold_terms(eq.terms, term_irs, _append_term_ir)
            return RegressionForm(
                lhs_spec=eq.lhs_spec,
                term_irs=tuple(folded_term_irs),
            )
        case Homogeneous():
            pivot_ir, _pivot_coefficient = eq.terms[0]
            return PivotRegressionForm(
                pivot_ir=pivot_ir,
                rhs_irs=tuple(term_ir for term_ir, _coefficient in eq.terms[1:]),
            )
    assert_never(eq)


def _append_term_ir(term_irs: list[str], term: Term) -> list[str]:
    term_ir, _coefficient = term
    term_irs.append(term_ir)
    return term_irs
