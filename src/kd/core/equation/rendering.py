
from __future__ import annotations

from typing import Final

from kd.core.equation.types import Homogeneous, LhsSpec
from kd.core.expr.naming import build_derivative_name

HOMOGENEOUS_LHS_LABEL: Final[str] = "0"







DEFAULT_LHS_LABEL: Final[str] = "u_t"


def render_homogeneous_label(eq: Homogeneous) -> str:
    return " + ".join(term_ir for term_ir, _coefficient in eq.terms) + " = 0"


def render_lhs_label(lhs_spec: LhsSpec) -> str:
    if lhs_spec.order == 1:
        return f"{lhs_spec.field}_{lhs_spec.axis}"
    return build_derivative_name(lhs_spec.field, lhs_spec.axis, lhs_spec.order)
