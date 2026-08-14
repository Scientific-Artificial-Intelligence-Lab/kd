
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any





from kd.search.dlga.viz import (
    render_surrogate,
    surrogate_data,
)
from kd.viz.axes import integer_ticks
from kd.viz.extension import PlotInfo

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.recorder import VizRecorder


logger = logging.getLogger(__name__)
















GEN_BEST_AIC_KEY = "gen_best_aic"
GEN_MEAN_AIC_KEY = "gen_mean_aic"
GEN_BEST_NMSE_KEY = "gen_best_nmse"
N_VALID_KEY = "n_valid"
N_UNIQUE_KEY = "n_unique"
GEN_MEAN_COMPLEXITY_KEY = "gen_mean_complexity"




POP_MEAN_AIC_KEY = "pop_mean_aic"
LOGGED_METRICS: tuple[str, ...] = (
    GEN_BEST_AIC_KEY,
    GEN_MEAN_AIC_KEY,
    GEN_BEST_NMSE_KEY,
    N_VALID_KEY,
    N_UNIQUE_KEY,
    GEN_MEAN_COMPLEXITY_KEY,
    POP_MEAN_AIC_KEY,
)





BEST_AIC_KEY = "best_aic"












_PLOT_METRIC: dict[str, str] = {
    "population_diversity": N_UNIQUE_KEY,
    "complexity_evolution": GEN_MEAN_COMPLEXITY_KEY,
    "fitness_spread": POP_MEAN_AIC_KEY,
}




_COUNT_METRICS: frozenset[str] = frozenset({N_UNIQUE_KEY})



_PLOT_INFOS: tuple[PlotInfo, ...] = (
    PlotInfo(
        name="population_diversity",
        title="Offspring Diversity",
        description=(
            "Distinct candidate-expression count among this generation's "
            "evaluated offspring (the staged batch, post-dedup; failed "
            "evaluations count too). A collapsing count signals premature "
            "convergence / loss of search diversity. NOT a population "
            "statistic — the count can exceed the population size."
        ),
    ),
    PlotInfo(
        name="complexity_evolution",
        title="Complexity Evolution",
        description=(
            "Mean number of STRidge-selected (non-zero) terms across this "
            "generation's valid offspring — a sparsity / bloat monitor for "
            "the evolving PDE candidates. All-invalid generations appear as "
            "gaps (nothing was measured)."
        ),
    ),
    PlotInfo(
        name="fitness_spread",
        title="Fitness Spread",
        description=(
            "Per-generation MEAN AIC over the surviving population's finite "
            "scores (lower is better). The platform 'convergence' plot draws "
            "the single best AIC; this curve shows the level of the whole "
            "surviving population — under elitist truncation it declines "
            "alongside the best curve, and the distance between the two is "
            "how far the population trails its optimum. Generations with no "
            "finite-scored survivors (e.g. an all-invalid start) appear as "
            "gaps (their +inf sentinel is masked)."
        ),
    ),
    PlotInfo(
        name="surrogate_training",
        title="Surrogate Training Curve",
        description=(
            "FieldModel surrogate training loss "
            "vs epoch (log-y): train and — when a validation split exists — "
            "validation MSE, with the best-validation epoch marked. The autograd "
            "derivatives (u_x, u_t) the GA reads from come from this network. "
            "Absent (No data panel) for finite-diff mode or when the surrogate "
            "was pre-trained (user-supplied field_model), not trained by the "
            "platform builder."
        ),
    ),
)





_SURROGATE_PLOT_NAME = "surrogate_training"



_NO_DATA_TEXT = "No data"

_X_LABEL = "Generation"


_LINE_MARKER = "."
_LINE_MARKER_SIZE = 3


def list_plot_infos() -> list[PlotInfo]:
    return [
        PlotInfo(name=info.name, title=info.title, description=info.description)
        for info in _PLOT_INFOS
    ]


def render(name: str, ax: Axes, recorder: VizRecorder | None) -> list[str]:
    _check_known_name(name)




    if name == _SURROGATE_PLOT_NAME:
        return render_surrogate(ax, recorder)
    metric = _PLOT_METRIC[name]
    series = _safe_get_series(recorder, metric)
    title = _plot_title(name)

    ax.set_xlabel(_X_LABEL)
    ax.set_ylabel(metric)
    integer_ticks(ax)
    if metric in _COUNT_METRICS:
        integer_ticks(ax, "y")
    ax.set_title(title)

    if not series:
        reason = _no_data_reason(recorder)
        ax.text(
            0.5,
            0.5,
            f"{_NO_DATA_TEXT} ({reason})",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return [f"plugin plot '{name}': {_NO_DATA_TEXT} ({reason})"]







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
    return []


def get_data(name: str, recorder: VizRecorder | None) -> dict[str, Any]:
    _check_known_name(name)
    if name == _SURROGATE_PLOT_NAME:
        return surrogate_data(recorder)
    metric = _PLOT_METRIC[name]
    series = _safe_get_series(recorder, metric)
    return {
        "x": list(range(len(series))),
        "y": [_sanitize_y(value) for value in series],
        "xlabel": _X_LABEL,
        "ylabel": metric,
        "title": _plot_title(name),
    }





_KNOWN_PLOT_NAMES: frozenset[str] = frozenset(info.name for info in _PLOT_INFOS)


def _check_known_name(name: str) -> None:
    if name not in _KNOWN_PLOT_NAMES:
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
        "sga.viz._sanitize_y: dropping unsupported recorder payload "
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
    "BEST_AIC_KEY",
    "GEN_BEST_AIC_KEY",
    "GEN_BEST_NMSE_KEY",
    "GEN_MEAN_AIC_KEY",
    "GEN_MEAN_COMPLEXITY_KEY",
    "LOGGED_METRICS",
    "N_UNIQUE_KEY",
    "N_VALID_KEY",
    "POP_MEAN_AIC_KEY",
    "get_data",
    "list_plot_infos",
    "render",
]
