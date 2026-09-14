
from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from kd.core.equation import (
    Evolution,
    Homogeneous,
    Regression,
    render_lhs_label,
)
from kd.core.equation.rendering import scalar_equation_terms

if TYPE_CHECKING:
    from kd.search.result import ExperimentResult

_NO_SKETCH_SOLUTION = "No certified sketch solution is available"


def _equation_data(
    result: ExperimentResult,
) -> tuple[list[str] | None, list[float] | None, list[int] | None]:
    equation = result.equation
    if equation is None:
        if result.config.get("sketch") is not None:
            return None, None, None
        fit = result.final_eval
        coefficients = (
            None
            if fit.coefficients is None
            else [float(c) for c in fit.coefficients.detach().cpu()]
        )
        return fit.terms, coefficients, fit.selected_indices
    terms, values = scalar_equation_terms(equation)
    return terms, values, None


def _equation_lhs(result: ExperimentResult) -> str:
    match result.equation:
        case Evolution() as equation:
            return render_lhs_label(equation.lhs_spec)
        case Homogeneous():
            return "0"
        case Regression() as equation:
            return equation.lhs_spec.field
        case None:
            return result.lhs_label
    assert_never(result.equation)


def _has_reduced_target(result: ExperimentResult) -> bool:
    sketch = result.config.get("sketch")
    return sketch is not None and bool(
        sketch["pinned"] and (sketch["anchored"] or sketch["holes"])
    )


def _fit_target_label(result: ExperimentResult) -> str:
    if _has_reduced_target(result):
        return f"{result.lhs_label} minus fixed terms (reduced target)"
    return result.lhs_label


def _sketch_fit_note(result: ExperimentResult) -> str | None:
    sketch = result.config.get("sketch")
    if sketch is None:
        return None
    if _has_reduced_target(result):
        return (
            f"Sketch search fit uses the reduced target: {result.lhs_label} minus "
            "fixed terms. Metrics and residuals describe this fit, not "
            "full-equation quality; full-equation verification is separate."
        )
    if not sketch["anchored"] and not sketch["holes"]:
        return (
            "Closed sketch: search-fit metrics and residuals do not measure "
            "the published fixed equation; full-equation verification is separate."
        )
    return (
        "Sketch metrics and residuals describe the native search fit; "
        "full-equation verification is separate."
    )
