
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kd.viz.axes import _get_axes
from kd.viz.equation_display import expression_display
from kd.viz.style import (
    FIGSIZE_SINGLE,
    INK_MUTED,
    draw_table,
    style_context,
    table_height,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.result import ExperimentResult


def plot_pareto_table(
    result: ExperimentResult,
    ax: Axes | None = None,
    *,
    style: dict[str, Any] | None = None,
) -> list[str]:
    entries = result.pareto_front()
    ax = _get_axes(ax, style, figsize=pareto_table_figsize(len(entries)), fill=True)
    displays = [expression_display(entry.expression) for entry in entries]
    with style_context(style):
        ax.axis("off")
        if not entries:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color=INK_MUTED)
            return ["Pareto front table has no data"]
        rows = [
            [
                str(entry.complexity),
                f"{entry.loss:.6g}",
                "" if entry.scale is None else f"{entry.scale:.6g}",
                display.text,
            ]
            for entry, display in zip(entries, displays, strict=True)
        ]
        draw_table(
            ax, rows, ["Complexity", "Loss", "Scale", "Expression"], text_columns=(3,)
        )
        ax.set_title("Pareto Front")
    return [display.note for display in displays if display.note is not None]


def pareto_table_figsize(n_entries: int) -> tuple[float, float]:
    return FIGSIZE_SINGLE[0], table_height(max(n_entries, 1))


__all__ = ["plot_pareto_table"]
