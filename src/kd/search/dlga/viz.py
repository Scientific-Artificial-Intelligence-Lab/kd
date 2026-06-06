
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from kd.viz.extension import PlotInfo

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.recorder import VizRecorder


logger = logging.getLogger(__name__)













_PLOT_METRIC: dict[str, str] = {
    "fitness_spread": "gen_mean_fitness",
    "population_diversity": "n_unique",
    "complexity_evolution": "gen_mean_complexity",
}



_PLOT_INFOS: tuple[PlotInfo, ...] = (
    PlotInfo(
        name="fitness_spread",
        title="Fitness Spread",
        description=(
            "Per-generation MEAN GA fitness (NMSE + epsilon*length; lower is "
            "better) over valid individuals. Complements the platform "
            "'convergence' plot, which draws the single global best: this "
            "shows how the whole population converges, not just the running "
            "optimum (DLGA's GA has no elitism, so the per-generation best "
            "fluctuates). No-valid / all-cross-LHS-guarded generations appear "
            "as gaps (their +inf sentinel is masked)."
        ),
    ),
    PlotInfo(
        name="population_diversity",
        title="Population Diversity",
        description=(
            "Distinct candidate-expression count per generation over the whole "
            "population. A collapsing count signals premature convergence."
        ),
    ),
    PlotInfo(
        name="complexity_evolution",
        title="Complexity Evolution",
        description=(
            "Mean genome complexity of valid individuals per generation — a "
            "bloat monitor (DLGA's NMSE-era epsilon can tolerate noise terms)."
        ),
    ),
)



_NO_DATA_TEXT = "No data"

_X_LABEL = "Generation"


_LINE_MARKER = "."
_LINE_MARKER_SIZE = 3


def list_plot_infos() -> list[PlotInfo]:
    return [
        PlotInfo(name=info.name, title=info.title, description=info.description)
        for info in _PLOT_INFOS
    ]


def render(name: str, ax: Axes, recorder: VizRecorder | None) -> None:
    _check_known_name(name)
    metric = _PLOT_METRIC[name]
    series = _safe_get_series(recorder, metric)
    title = _plot_title(name)

    ax.set_xlabel(_X_LABEL)
    ax.set_ylabel(metric)
    ax.set_title(title)

    if not series:
        ax.text(
            0.5,
            0.5,
            f"{_NO_DATA_TEXT} ({_no_data_reason(recorder)})",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return







    cleaned = [
        float("nan")
        if value is None or (isinstance(value, float) and not math.isfinite(value))
        else value
        for value in series
    ]

    ax.plot(
        range(len(series)),
        cleaned,
        marker=_LINE_MARKER,
        markersize=_LINE_MARKER_SIZE,
    )


def get_data(name: str, recorder: VizRecorder | None) -> dict[str, Any]:
    _check_known_name(name)
    metric = _PLOT_METRIC[name]
    series = _safe_get_series(recorder, metric)
    return {
        "x": list(range(len(series))),
        "y": [_sanitize_y(value) for value in series],
        "xlabel": _X_LABEL,
        "ylabel": metric,
        "title": _plot_title(name),
    }





def _check_known_name(name: str) -> None:
    if name not in _PLOT_METRIC:
        available = ", ".join(info.name for info in _PLOT_INFOS)
        raise ValueError(f"Unknown plot name: {name!r}. Available: {available}")


def _safe_get_series(recorder: VizRecorder | None, metric: str) -> list[Any]:
    if recorder is None:
        return []
    return recorder.get(metric)


def _sanitize_y(value: Any) -> float | None:
    if value is None:



        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, int):
        return float(value)
    logger.warning(
        "dlga.viz._sanitize_y: dropping unsupported recorder payload "
        "type=%s value=%r — VizRecorder should only store scalar numeric "
        "samples for plotted metrics.",
        type(value).__name__,
        value,
    )
    return None


def _no_data_reason(recorder: VizRecorder | None) -> str:
    if recorder is None:
        return "no recorder"
    return "empty"


def _plot_title(name: str) -> str:
    for info in _PLOT_INFOS:
        if info.name == name:
            return info.title
    return name


__all__ = [
    "get_data",
    "list_plot_infos",
    "render",
]
