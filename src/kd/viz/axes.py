
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from matplotlib.ticker import MaxNLocator

from kd.viz.style import style_context

if TYPE_CHECKING:
    from matplotlib.axes import Axes




FULL_FIGURE = (0.0, 0.0, 1.0, 1.0)


def _get_axes(
    ax: Axes | None,
    style: dict[str, Any] | None,
    *,
    figsize: tuple[float, float] = (7, 4),
    fill: bool = False,
) -> Axes:
    if ax is None:
        import matplotlib.pyplot as plt

        with style_context(style):
            if fill:
                ax = plt.figure(figsize=figsize).add_axes(FULL_FIGURE)
            else:
                _, ax = plt.subplots(figsize=figsize, layout="constrained")
    return ax


def integer_ticks(ax: Axes, axis: Literal["x", "y"] = "x") -> None:
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(MaxNLocator(integer=True))


def value_bar_scale(ax: Axes, heights: Sequence[float], *, headroom: float) -> None:
    if heights and all(height > 0 for height in heights):
        ax.set_yscale("log")
        ax.set_ylim(min(heights) / 10.0, max(heights) * 10.0)
    else:
        ax.margins(y=headroom)
