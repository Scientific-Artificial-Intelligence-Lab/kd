
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kd.core.jsonsafe import sanitize_float
from kd.viz.extension import PlotInfo
from kd.viz.gap_notes import (
    GapVocabulary,
    annotate_gaps,
    band_measured_flags,
    measured_flags,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.recorder import VizRecorder








POOL_BEST_KEY = "pool_best"
POOL_MEDIAN_KEY = "pool_median"
POOL_WORST_KEY = "pool_worst"
N_INVALID_KEY = "n_invalid"
N_LLM_CALLS_KEY = "n_llm_calls"
N_VALID_KEY = "n_valid"
LOGGED_METRICS: tuple[str, ...] = (
    POOL_BEST_KEY,
    POOL_MEDIAN_KEY,
    POOL_WORST_KEY,
    N_INVALID_KEY,
    N_LLM_CALLS_KEY,
    N_VALID_KEY,
)


_PLOT_INFOS: tuple[PlotInfo, ...] = (
    PlotInfo(
        name="pool_reward_spread",
        title="Pool Reward Spread",
        description=(
            "Elite-pool reward band (best / median / worst) per round -- pool "
            "quality and diversity as the LLM proposer evolves."
        ),
    ),
    PlotInfo(
        name="invalid_count",
        title="Invalid Count",
        description=(
            "Number of invalid scoring events per round (n_invalid): parse / "
            "solve failures across all resamples. Rising counts flag the LLM "
            "proposing more unusable equations."
        ),
    ),
    PlotInfo(
        name="llm_calls",
        title="Cumulative LLM Calls",
        description=(
            "Total provider completions issued through each round (running sum "
            "of per-round n_llm_calls) -- the search's whole-run budget spend."
        ),
    ),
)

_PLOT_NAMES: tuple[str, ...] = tuple(info.name for info in _PLOT_INFOS)
_SPREAD_METRICS: tuple[str, ...] = (POOL_BEST_KEY, POOL_MEDIAN_KEY, POOL_WORST_KEY)
_YLABEL: dict[str, str] = {
    "pool_reward_spread": "reward",
    "invalid_count": "invalid events",
    "llm_calls": "cumulative calls",
}
_NO_DATA_TEXT = "No data"
_X_LABEL = "Round"
_LINE_MARKER = "."
_LINE_MARKER_SIZE = 3




_SINGLE_SOURCE: dict[str, str] = {
    "invalid_count": N_INVALID_KEY,
    "llm_calls": N_LLM_CALLS_KEY,
}






_POOL_GAPS = GapVocabulary(
    unit_plural="rounds", missing="had no pool", nothing="no pool"
)
_PANEL_GAPS: dict[str, GapVocabulary] = {
    "invalid_count": GapVocabulary(
        unit_plural="rounds",
        missing="recorded no invalid count",
        nothing="no invalid count",
    ),
    "llm_calls": GapVocabulary(
        unit_plural="rounds",
        missing="recorded no call count",
        nothing="no call count",
    ),
}


def list_plot_infos() -> list[PlotInfo]:
    return [
        PlotInfo(name=info.name, title=info.title, description=info.description)
        for info in _PLOT_INFOS
    ]


def render(name: str, ax: Axes, recorder: VizRecorder | None) -> list[str]:
    _check_known_name(name)
    ax.set_xlabel(_X_LABEL)
    ax.set_ylabel(_YLABEL[name])
    ax.set_title(_plot_title(name))

    if name == "pool_reward_spread":
        series_by_metric = {
            metric: _safe_get_series(recorder, metric) for metric in _SPREAD_METRICS
        }
        max_len = max((len(series) for series in series_by_metric.values()), default=0)
        if max_len == 0:
            return _draw_no_data(ax, recorder, name)
        x = list(range(max_len))
        for metric in _SPREAD_METRICS:
            ax.plot(
                x,
                _sanitized(_pad_series(series_by_metric[metric], max_len)),
                marker=_LINE_MARKER,
                markersize=_LINE_MARKER_SIZE,
                label=metric,
            )
        ax.legend()
        annotate_gaps(
            ax, band_measured_flags(series_by_metric.values(), max_len), _POOL_GAPS
        )
        return []

    series = _derived_series(name, recorder)
    if not series:
        return _draw_no_data(ax, recorder, name)
    ax.plot(
        range(len(series)),
        _sanitized(series),
        marker=_LINE_MARKER,
        markersize=_LINE_MARKER_SIZE,
    )
    annotate_gaps(ax, measured_flags(_gap_source(name, recorder)), _PANEL_GAPS[name])
    return []


def get_data(name: str, recorder: VizRecorder | None) -> dict[str, Any]:
    _check_known_name(name)
    if name == "pool_reward_spread":
        series_by_metric = {
            metric: _safe_get_series(recorder, metric) for metric in _SPREAD_METRICS
        }
        max_len = max((len(series) for series in series_by_metric.values()), default=0)
        return {
            "x": list(range(max_len)),
            "y": {
                metric: _sanitized(_pad_series(series_by_metric[metric], max_len))
                for metric in _SPREAD_METRICS
            },
            "xlabel": _X_LABEL,
            "ylabel": _YLABEL[name],
            "title": _plot_title(name),
        }

    series = _derived_series(name, recorder)
    return {
        "x": list(range(len(series))),
        "y": _sanitized(series),
        "xlabel": _X_LABEL,
        "ylabel": _YLABEL[name],
        "title": _plot_title(name),
    }


def _derived_series(name: str, recorder: VizRecorder | None) -> list[float | None]:
    if name == "invalid_count":
        return [
            float(value)
            for value in _safe_get_series(recorder, _SINGLE_SOURCE["invalid_count"])
        ]

    n_llm_calls = _safe_get_series(recorder, _SINGLE_SOURCE["llm_calls"])
    cumulative: list[float | None] = []
    running = 0.0
    for value in n_llm_calls:
        running += float(value)
        cumulative.append(running)
    return cumulative


def _gap_source(name: str, recorder: VizRecorder | None) -> list[Any]:
    return _safe_get_series(recorder, _SINGLE_SOURCE[name])


def _sanitized(series: list[Any]) -> list[Any]:
    return [None if value is None else sanitize_float(float(value)) for value in series]


def _check_known_name(name: str) -> None:
    if name not in _PLOT_NAMES:
        available = ", ".join(_PLOT_NAMES)
        raise ValueError(f"Unknown plot name: {name!r}. Available: {available}")


def _safe_get_series(recorder: VizRecorder | None, metric: str) -> list[Any]:
    if recorder is None or not recorder.enabled:
        return []
    return recorder.get(metric)


def _pad_series(series: list[Any], length: int) -> list[Any]:
    if len(series) >= length:
        return list(series)
    return [*series, *([None] * (length - len(series)))]


def _draw_no_data(ax: Axes, recorder: VizRecorder | None, name: str) -> list[str]:
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


def _no_data_reason(recorder: VizRecorder | None) -> str:
    if recorder is None:
        return "no recorder"
    if not recorder.enabled:
        return "disabled"
    return "empty"


def _plot_title(name: str) -> str:
    for info in _PLOT_INFOS:
        if info.name == name:
            return info.title
    return name


__all__ = [
    "LOGGED_METRICS",
    "N_INVALID_KEY",
    "N_LLM_CALLS_KEY",
    "N_VALID_KEY",
    "POOL_BEST_KEY",
    "POOL_MEDIAN_KEY",
    "POOL_WORST_KEY",
    "get_data",
    "list_plot_infos",
    "render",
]
