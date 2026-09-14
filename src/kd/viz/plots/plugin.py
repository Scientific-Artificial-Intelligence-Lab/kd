
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from kd.viz.style import style_context

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.viz.extension import VizExtension


def plot_plugin(
    algorithm: VizExtension,
    name: str,
    ax: Axes | None = None,
    *,
    style: dict[str, Any] | None = None,
) -> list[str] | None:
    info = {plot.name: plot for plot in algorithm.list_plots()}[name]
    with style_context(style):
        if ax is None:
            _, ax = plt.subplots(
                figsize=(8, 4),
                subplot_kw={"projection": info.projection},
                layout="constrained",
            )
        ax.set_title(info.title)
        notes = algorithm.render_plot(name, ax)
    return notes
