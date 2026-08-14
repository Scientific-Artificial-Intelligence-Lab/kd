
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from kd.viz.axes import integer_ticks
from kd.viz.extension import PlotInfo
from kd.viz.gap_notes import (
    NO_MEASUREMENT,
    GapVocabulary,
    all_gap_note,
    annotate_gaps,
    append_subtitle,
    band_measured_flags,
    gap_phrase,
    is_measured,
    measured_flags,
    partial_gap_note,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.recorder import VizRecorder







POOL_BEST_KEY = "pool_best"
POOL_MEDIAN_KEY = "pool_median"
POOL_WORST_KEY = "pool_worst"
FINETUNE_LOSS_KEY = "finetune_loss"
LOGGED_METRICS: tuple[str, ...] = (
    POOL_BEST_KEY,
    POOL_MEDIAN_KEY,
    POOL_WORST_KEY,
    FINETUNE_LOSS_KEY,
)



_PLOT_METRIC: dict[str, str] = {
    "reward_convergence": POOL_BEST_KEY,
    "finetune_loss": FINETUNE_LOSS_KEY,
}
_SPREAD_METRICS: tuple[str, ...] = (POOL_BEST_KEY, POOL_MEDIAN_KEY, POOL_WORST_KEY)



_YLABEL: dict[str, str] = {
    "reward_convergence": POOL_BEST_KEY,
    "finetune_loss": FINETUNE_LOSS_KEY,
    "pool_reward_spread": "reward",
}

















_PLOT_INFOS: tuple[PlotInfo, ...] = (
    PlotInfo(
        name="reward_convergence",
        title="Reward Convergence",
        description="Best top-k pool reward per epoch (the search's primary signal).",
    ),
    PlotInfo(
        name="pool_reward_spread",
        title="Pool Reward Spread",
        description=(
            "Top-k pool reward band (best / median / worst) per epoch -- pool "
            "quality and diversity as the GPT proposer evolves."
        ),
    ),
    PlotInfo(
        name="finetune_loss",
        title="Fine-tune Loss",
        description=(
            "Mean cross-entropy loss of the per-epoch GPT fine-tune on the top-k pool."
        ),
    ),
)

_PLOT_NAMES: tuple[str, ...] = tuple(info.name for info in _PLOT_INFOS)
_NO_DATA_TEXT = "No data"
_X_LABEL = "Epoch"
_LINE_MARKER = "."
_LINE_MARKER_SIZE = 3






_POOL_GAPS = GapVocabulary(
    unit_plural="epochs", missing="had no pool", nothing="no pool"
)
_FINETUNE_GAPS = GapVocabulary(
    unit_plural="epochs",
    missing="skipped the fine-tune",
    nothing="no fine-tune",
)
_PANEL_GAPS: dict[str, GapVocabulary] = {
    "reward_convergence": _POOL_GAPS,
    "pool_reward_spread": _POOL_GAPS,
    "finetune_loss": _FINETUNE_GAPS,
}






_PER_CASE_PLOT_NAME = "per_case_reward"
_PER_CASE_TITLE = "Per-case Reward"
PER_CASE_PLOT_INFO = PlotInfo(
    name=_PER_CASE_PLOT_NAME,
    title=_PER_CASE_TITLE,
    description=(
        "Reward of the discovered structure fitted to each wave-breaking case "
        "on its own -- one bar per experiment. A uniformly high band shows a "
        "single discovered law explaining every case."
    ),
)
_PER_CASE_XLABEL = "case"
_PER_CASE_YLABEL = "reward"
_PER_CASE_LABEL_FONTSIZE = 6





_PER_CASE_SCOPE_CAPTION = (
    "score = mean reward over the scored cases; R^2/MSE = primary case only"
)
_PER_CASE_SCOPE_WARNING = (
    f"plugin plot '{_PER_CASE_PLOT_NAME}': headline score is the mean reward "
    "over the scored cases (n/a cases excluded); R^2/MSE metrics are "
    "primary-case only"
)

logger = logging.getLogger(__name__)


def list_plot_infos() -> list[PlotInfo]:
    return [
        PlotInfo(name=info.name, title=info.title, description=info.description)
        for info in _PLOT_INFOS
    ]


def render(name: str, ax: Axes, recorder: VizRecorder | None) -> list[str]:
    _check_known_name(name)
    ax.set_xlabel(_X_LABEL)
    ax.set_ylabel(_YLABEL[name])
    integer_ticks(ax)
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
                _pad_series(series_by_metric[metric], max_len),
                marker=_LINE_MARKER,
                markersize=_LINE_MARKER_SIZE,
                label=metric,
            )
        ax.legend()
        annotate_gaps(
            ax, band_measured_flags(series_by_metric.values(), max_len), _POOL_GAPS
        )
        return []

    metric = _PLOT_METRIC[name]
    series = _safe_get_series(recorder, metric)
    if not series:
        return _draw_no_data(ax, recorder, name)
    ax.plot(
        range(len(series)),
        series,
        marker=_LINE_MARKER,
        markersize=_LINE_MARKER_SIZE,
    )
    if name == "reward_convergence":


        _annotate_reward_convergence(ax, series)
    else:
        annotate_gaps(ax, measured_flags(series), _PANEL_GAPS[name])
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
                metric: [
                    _sanitize_y(value)
                    for value in _pad_series(series_by_metric[metric], max_len)
                ]
                for metric in _SPREAD_METRICS
            },
            "xlabel": _X_LABEL,
            "ylabel": _YLABEL[name],
            "title": _plot_title(name),
        }

    metric = _PLOT_METRIC[name]
    series = _safe_get_series(recorder, metric)
    return {
        "x": list(range(len(series))),
        "y": [_sanitize_y(value) for value in series],
        "xlabel": _X_LABEL,
        "ylabel": _YLABEL[name],
        "title": _plot_title(name),
    }


def render_per_case_reward(ax: Axes, per_case: dict[str, float]) -> list[str]:
    ax.set_xlabel(_PER_CASE_XLABEL)
    ax.set_ylabel(_PER_CASE_YLABEL)
    ax.set_title(_PER_CASE_TITLE)
    names = list(per_case)
    if not names or not any(is_measured(per_case[name]) for name in names):
        reason = "no candidate" if not names else "no survivor cases"
        ax.text(
            0.5,
            0.5,
            f"{_NO_DATA_TEXT} ({reason})",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return [
            f"plugin plot '{PER_CASE_PLOT_INFO.name}': "
            f"{_NO_DATA_TEXT} ({reason})"
        ]
    positions = list(range(len(names)))




    heights = [
        per_case[name] if is_measured(per_case[name]) else NO_MEASUREMENT
        for name in names
    ]
    ax.bar(positions, heights)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [_short_case(name) for name in names],
        rotation=90,
        fontsize=_PER_CASE_LABEL_FONTSIZE,
    )
    for position, name in enumerate(names):
        if not is_measured(per_case[name]):








            ax.text(
                position,
                0.02,
                "n/a",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=_PER_CASE_LABEL_FONTSIZE,
            )
    append_subtitle(ax, _PER_CASE_SCOPE_CAPTION)
    return [_PER_CASE_SCOPE_WARNING]


def per_case_data(per_case: dict[str, float]) -> dict[str, Any]:
    names = list(per_case)
    return {
        "x": names,
        "y": [_sanitize_y(per_case[name]) for name in names],
        "xlabel": _PER_CASE_XLABEL,
        "ylabel": _PER_CASE_YLABEL,
        "title": _PER_CASE_TITLE,
    }


def _short_case(name: str) -> str:
    return name[2:] if name.startswith("N_") else name


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
        "eqgpt.viz._sanitize_y: dropping unsupported recorder payload "
        "type=%s value=%r.",
        type(value).__name__,
        value,
    )
    return None


def _annotate_reward_convergence(ax: Axes, series: list[Any]) -> None:
    from kd.viz.plots.convergence import flat_value

    measured = [index for index, value in enumerate(series) if is_measured(value)]
    gaps = len(series) - len(measured)
    if not measured:
        note = all_gap_note(len(series), _POOL_GAPS)
    else:
        constant = flat_value(series)
        if constant is None:
            if gaps == 0:
                return
            note = partial_gap_note(gaps, len(series), _POOL_GAPS)
        elif gaps:
            note = (
                f"best reward constant at {constant:.4g} across the "
                f"{len(measured)} measured epochs "
                f"({gap_phrase(gaps, len(series), _POOL_GAPS)})"
            )
        else:
            note = (
                f"best reward constant at {constant:.4g} (reached at the first epoch)"
            )
    append_subtitle(ax, note)


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
    "FINETUNE_LOSS_KEY",
    "LOGGED_METRICS",
    "PER_CASE_PLOT_INFO",
    "POOL_BEST_KEY",
    "POOL_MEDIAN_KEY",
    "POOL_WORST_KEY",
    "get_data",
    "list_plot_infos",
    "per_case_data",
    "render",
    "render_per_case_reward",
]
