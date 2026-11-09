
from typing import TYPE_CHECKING

from kd.viz.equation_display import EquationDisplay, latex_display
from kd.viz.extension import PlotInfo, VizExtension
from kd.viz.integration_assembly import build_integration_result
from kd.viz.report import FigureSpec, ReportResult
from kd.viz.style import style_context

if TYPE_CHECKING:
    from kd.viz.engine import VizEngine


def __getattr__(name: str) -> object:

    if name == "VizEngine":
        from kd.viz.engine import VizEngine

        globals()[name] = VizEngine
        return VizEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"VizEngine"})


__all__ = [
    "EquationDisplay",
    "FigureSpec",
    "PlotInfo",
    "ReportResult",
    "VizEngine",
    "VizExtension",
    "build_integration_result",
    "latex_display",
    "style_context",
]
