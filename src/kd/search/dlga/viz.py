
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from kd.search.dlga.surrogate_log import (
    _SURROGATE_BEST_EPOCH_KEY,
    _SURROGATE_EPOCH_KEY,
    _SURROGATE_TRAIN_LOSS_KEY,
    _SURROGATE_VAL_LOSS_KEY,
)
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
    PlotInfo(
        name="surrogate_training",
        title="Surrogate Training Curve",
        description=(
            "NN_1 surrogate (Xu 2020 §2.B) training loss vs epoch (log-y): "
            "train and — when a validation split exists — validation MSE, with "
            "the best-validation epoch marked. This is the 'deep learning' "
            "evidence of the DLGA pipeline — the autograd-derivative source the "
            "GA reads from. Absent (No data panel) when the surrogate was "
            "pre-trained or not trained by the platform builder."
        ),
    ),
)



_NO_DATA_TEXT = "No data"

_X_LABEL = "Generation"


_LINE_MARKER = "."
_LINE_MARKER_SIZE = 3










_SURROGATE_PLOT_NAME = "surrogate_training"

_SURROGATE_X_LABEL = "Epoch"
_SURROGATE_Y_LABEL = "MSE loss"
_SURROGATE_TRAIN_LABEL = "train"
_SURROGATE_VAL_LABEL = "val"
_SURROGATE_BEST_EPOCH_LABEL = "best epoch"


def list_plot_infos() -> list[PlotInfo]:
    return [
        PlotInfo(name=info.name, title=info.title, description=info.description)
        for info in _PLOT_INFOS
    ]


def render(name: str, ax: Axes, recorder: VizRecorder | None) -> None:
    _check_known_name(name)



    if name == _SURROGATE_PLOT_NAME:
        _render_surrogate(ax, recorder)
        return
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
    if name == _SURROGATE_PLOT_NAME:
        return _surrogate_data(recorder)
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


def _render_surrogate(ax: Axes, recorder: VizRecorder | None) -> None:
    epochs = _last_logged_list(recorder, _SURROGATE_EPOCH_KEY)
    train = _last_logged_list(recorder, _SURROGATE_TRAIN_LOSS_KEY)

    ax.set_xlabel(_SURROGATE_X_LABEL)
    ax.set_ylabel(_SURROGATE_Y_LABEL)
    ax.set_title(_plot_title(_SURROGATE_PLOT_NAME))

    if not epochs or not train:
        ax.text(
            0.5,
            0.5,
            f"{_NO_DATA_TEXT} ({_no_data_reason(recorder)})",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return

    train_x, train_y = _aligned_xy(epochs, train, _SURROGATE_TRAIN_LOSS_KEY)
    ax.set_yscale("log")
    ax.plot(
        train_x,
        _mask_for_log(train_y),
        marker=_LINE_MARKER,
        markersize=_LINE_MARKER_SIZE,
        label=_SURROGATE_TRAIN_LABEL,
    )
    val = _last_logged_list(recorder, _SURROGATE_VAL_LOSS_KEY)
    if val:
        val_x, val_y = _aligned_xy(epochs, val, _SURROGATE_VAL_LOSS_KEY)
        ax.plot(
            val_x,
            _mask_for_log(val_y),
            marker=_LINE_MARKER,
            markersize=_LINE_MARKER_SIZE,
            label=_SURROGATE_VAL_LABEL,
        )
    best_epoch = _last_logged_scalar(recorder, _SURROGATE_BEST_EPOCH_KEY)
    if isinstance(best_epoch, (int, float)) and not isinstance(best_epoch, bool):
        ax.axvline(float(best_epoch), label=_SURROGATE_BEST_EPOCH_LABEL)
    ax.legend()


def _surrogate_data(recorder: VizRecorder | None) -> dict[str, Any]:
    epochs = _last_logged_list(recorder, _SURROGATE_EPOCH_KEY)
    train = _last_logged_list(recorder, _SURROGATE_TRAIN_LOSS_KEY)
    val = _last_logged_list(recorder, _SURROGATE_VAL_LOSS_KEY)
    best_epoch = _last_logged_scalar(recorder, _SURROGATE_BEST_EPOCH_KEY)


    train_x, train_y = _aligned_xy(epochs, train, _SURROGATE_TRAIN_LOSS_KEY)
    val_y = _aligned_xy(epochs, val, _SURROGATE_VAL_LOSS_KEY)[1] if val else None
    return {
        "x": [_sanitize_x(value) for value in train_x],
        "y_train": [_sanitize_y(value) for value in train_y],
        "y_val": [_sanitize_y(value) for value in val_y] if val_y is not None else None,
        "best_epoch": _sanitize_y(best_epoch),
        "xlabel": _SURROGATE_X_LABEL,
        "ylabel": _SURROGATE_Y_LABEL,
        "title": _plot_title(_SURROGATE_PLOT_NAME),
    }


def _aligned_xy(
    epochs: list[Any], series: list[Any], series_key: str
) -> tuple[list[Any], list[Any]]:
    if len(epochs) == len(series):
        return epochs, series
    common = min(len(epochs), len(series))
    logger.warning(
        "dlga.viz: surrogate '%s' length %d != epoch length %d; truncating to "
        "%d (drifted recorder).",
        series_key,
        len(series),
        len(epochs),
        common,
    )
    return epochs[:common], series[:common]


def _last_logged_list(recorder: VizRecorder | None, key: str) -> list[Any]:
    if recorder is None:
        return []
    series = recorder.get(key)
    if not series:
        return []
    last = series[-1]
    if not isinstance(last, list):
        return []
    return list(last)


def _last_logged_scalar(recorder: VizRecorder | None, key: str) -> Any:
    if recorder is None:
        return None
    series = recorder.get(key)
    if not series:
        return None
    return series[-1]


def _mask_for_log(series: list[Any]) -> np.ndarray:
    cleaned = [
        float("nan")
        if value is None
        or not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0.0
        else float(value)
        for value in series
    ]
    return np.asarray(cleaned, dtype=float)


def _sanitize_x(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    return None


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
