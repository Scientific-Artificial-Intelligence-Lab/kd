
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from kd.viz.style import style_context

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _get_axes(
    ax: Axes | None,
    style: dict[str, Any] | None,
    *,
    figsize: tuple[float, float] = (7, 4),
) -> Axes:
    if ax is None:
        with style_context(style):
            _, ax = plt.subplots(figsize=figsize, layout="constrained")
    return ax


def integer_ticks(ax: Axes, axis: Literal["x", "y"] = "x") -> None:
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(MaxNLocator(integer=True))
