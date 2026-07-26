
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from matplotlib.axes import Axes

if TYPE_CHECKING:
    from kd.search.result import ExperimentResult


@dataclass
class PlotInfo:

    name: str
    title: str
    description: str = ""
    projection: str | None = None


@runtime_checkable
class VizExtension(Protocol):

    def list_plots(self) -> list[PlotInfo]:
        ...

    def render_plot(self, name: str, ax: Axes) -> None:
        ...

    def get_plot_data(self, name: str) -> Any:
        ...


@runtime_checkable
class HomogeneousVizExtension(Protocol):

    def list_homogeneous_plots(self) -> list[PlotInfo]:
        ...

    def render_homogeneous_plot(
        self, name: str, ax: Axes, result: ExperimentResult
    ) -> None:
        ...
