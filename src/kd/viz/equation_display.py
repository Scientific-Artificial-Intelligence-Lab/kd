
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kd.core.equation.rendering import format_equation
from kd.core.expr.sympy_bridge import format_pde, to_latex, to_sympy
from kd.viz._result_data import _NO_SKETCH_SOLUTION, _equation_data, _equation_lhs

if TYPE_CHECKING:
    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)








RENDER_ERRORS = (ValueError, IndexError, TypeError)

STRUCTURE_ONLY_NOTE = "structure only: no LHS, coefficients not applied"
UNRENDERABLE_NOTE = "expression not renderable as display math"
UNRENDERABLE_MARKER = "unrenderable"


@dataclass(frozen=True)
class EquationDisplay:

    text: str
    note: str | None
    is_math: bool = True

    @property
    def degraded(self) -> bool:
        return self.note is not None


def expression_display(code: str) -> EquationDisplay:
    try:
        parts = code.split("=", maxsplit=1)
        return EquationDisplay(
            " = ".join(str(to_sympy(part.strip(), strict=True)) for part in parts), None
        )
    except RENDER_ERRORS:
        logger.exception("Expression is not renderable as display math: %s", code)
        return EquationDisplay(UNRENDERABLE_MARKER, UNRENDERABLE_NOTE, is_math=False)


def _fitted_latex(result: ExperimentResult) -> str | None:
    try:
        if result.equation is not None:
            return format_equation(result.equation).latex
        terms, coefficients, selected = _equation_data(result)
        if terms is None or coefficients is None:
            return None
        return format_pde(
            terms,
            coefficients,
            lhs=_equation_lhs(result),
            selected_indices=selected,
        ).latex
    except RENDER_ERRORS:
        logger.exception("Failed to format equation as full PDE")
        return None


def _structure_latex(code: str, label: str) -> str | None:
    try:
        return to_latex(code, strict=True)
    except RENDER_ERRORS:




        logger.exception(
            "Equation for %s is not renderable as display math; raw IR: %s",
            label,
            code,
        )
        return None


def latex_display(result: ExperimentResult, *, label: str) -> EquationDisplay:
    if result.equation is None and result.config.get("sketch") is not None:
        return EquationDisplay("unavailable", _NO_SKETCH_SOLUTION, is_math=False)
    fitted = _fitted_latex(result)
    if fitted is not None:
        return EquationDisplay(text=fitted, note=None)

    structure = _structure_latex(result.best_expression, label)
    if structure is not None:
        return EquationDisplay(text=structure, note=STRUCTURE_ONLY_NOTE)

    return EquationDisplay(
        text=UNRENDERABLE_MARKER, note=UNRENDERABLE_NOTE, is_math=False
    )


__all__ = [
    "RENDER_ERRORS",
    "STRUCTURE_ONLY_NOTE",
    "UNRENDERABLE_MARKER",
    "UNRENDERABLE_NOTE",
    "EquationDisplay",
    "expression_display",
    "latex_display",
]
