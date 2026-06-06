
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kd.core.expr.sympy_bridge import format_pde, to_latex

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_EQUATION_FONT_SIZE = 16


def _ensure_math_mode(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("$") and stripped.endswith("$") and len(stripped) >= 2:
        return text
    return f"${text}$"


def _equation_latex(result: ExperimentResult) -> str:
    final_eval = result.final_eval
    if final_eval.terms is not None and final_eval.coefficients is not None:
        return format_pde(
            final_eval.terms,
            final_eval.coefficients,
            lhs=result.lhs_label,
            selected_indices=final_eval.selected_indices,
        ).latex
    return to_latex(result.best_expression, strict=False)


def plot_equation(
    result: ExperimentResult,
    ax: Axes,
) -> list[str]:
    warnings: list[str] = []
    expr = result.best_expression

    if not expr:
        warnings.append("Empty expression; skipping equation plot")
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No expression",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return warnings

    try:
        latex_str = _ensure_math_mode(_equation_latex(result))
    except ValueError:
        logger.exception("Failed to format equation as LaTeX")
        warnings.append("Failed to format full PDE; falling back to expression LaTeX")
        latex_str = _ensure_math_mode(to_latex(expr, strict=False))

    ax.text(
        0.5,
        0.5,
        latex_str,
        size=_EQUATION_FONT_SIZE,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.axis("off")
    ax.set_title("Best Expression")

    return warnings
