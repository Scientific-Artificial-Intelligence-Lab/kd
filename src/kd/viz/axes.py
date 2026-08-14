
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def integer_ticks(ax: Axes, axis: Literal["x", "y"] = "x") -> None:
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(MaxNLocator(integer=True))
