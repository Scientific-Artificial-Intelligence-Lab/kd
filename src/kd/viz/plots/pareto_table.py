
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kd.viz.style import style_context

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.result import ExperimentResult


def plot_pareto_table(
    result: ExperimentResult,
    ax: Axes,
    *,
    style: dict[str, Any] | None = None,
) -> list[str]:
    entries = result.pareto_front()
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
                entry.expression,
            ]
            for entry in entries
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
    return []


__all__ = ["plot_pareto_table"]
