
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kd.viz.axes import _get_axes
from kd.viz.equation_display import expression_display
from kd.viz.style import style_context

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.result import ExperimentResult


def plot_pareto_table(
    result: ExperimentResult,
    ax: Axes | None = None,
    *,
    style: dict[str, Any] | None = None,
) -> list[str]:
    ax = _get_axes(ax, style, figsize=(12, 5))
    entries = result.pareto_front()
    displays = [expression_display(entry.expression) for entry in entries]
    with style_context(style):
        ax.axis("off")
        if not entries:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
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
        table = ax.table(
            cellText=rows,
            colLabels=["Complexity", "Loss", "Scale", "Expression"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.auto_set_column_width(range(4))
        ax.set_title("Pareto Front")
    return [display.note for display in displays if display.note is not None]


__all__ = ["plot_pareto_table"]
