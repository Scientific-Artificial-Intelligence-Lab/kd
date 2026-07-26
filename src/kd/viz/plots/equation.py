
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kd.viz.equation_display import latex_display

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_EQUATION_FONT_SIZE = 16


_MARKER_FONT_SIZE = 12
_TITLE = "Best Expression"


def _ensure_math_mode(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("$") and stripped.endswith("$") and len(stripped) >= 2:
        return text
    return f"${text}$"


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

    display = latex_display(result, label=result.algorithm_name)
    if display.note is not None:
        warnings.append(f"Equation panel shows a degraded rendering: {display.note}")

    ax.text(
        0.5,
        0.5,
        _ensure_math_mode(display.text) if display.is_math else display.text,
        size=_EQUATION_FONT_SIZE if display.is_math else _MARKER_FONT_SIZE,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.axis("off")
    ax.set_title(_TITLE if display.note is None else f"{_TITLE}\n({display.note})")

    return warnings
