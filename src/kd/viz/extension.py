
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from matplotlib.axes import Axes


@dataclass
class PlotInfo:

    name: str
    title: str
    description: str = ""


@runtime_checkable
class VizExtension(Protocol):

    def list_plots(self) -> list[PlotInfo]:
        ...

    def render_plot(self, name: str, ax: Axes) -> None:
        ...

    def get_plot_data(self, name: str) -> Any:
        ...
