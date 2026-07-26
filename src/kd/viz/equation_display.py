
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kd.core.expr.sympy_bridge import format_pde, to_latex

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


def _fitted_latex(result: ExperimentResult) -> str | None:
    final_eval = result.final_eval
    if final_eval.terms is None or final_eval.coefficients is None:
        return None
    try:
        return format_pde(
            final_eval.terms,
            final_eval.coefficients,
            lhs=result.lhs_label,
            selected_indices=final_eval.selected_indices,
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
    "latex_display",
]
