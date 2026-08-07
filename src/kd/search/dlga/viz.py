
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from kd.search.surrogate_log import (
    SURROGATE_BEST_EPOCH_KEY,
    SURROGATE_EPOCH_KEY,
    SURROGATE_TRAIN_LOSS_KEY,
    SURROGATE_VAL_LOSS_KEY,
)
from kd.viz.extension import PlotInfo

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.recorder import VizRecorder


logger = logging.getLogger(__name__)











GEN_BEST_FITNESS_KEY = "gen_best_fitness"
GEN_MEAN_FITNESS_KEY = "gen_mean_fitness"
GEN_BEST_NMSE_KEY = "gen_best_nmse"
N_VALID_KEY = "n_valid"
N_UNIQUE_KEY = "n_unique"
GEN_MEAN_COMPLEXITY_KEY = "gen_mean_complexity"
LHS_UT_KEY = "lhs_ut"
LHS_UTT_KEY = "lhs_utt"
LOGGED_METRICS: tuple[str, ...] = (
    GEN_BEST_FITNESS_KEY,
    GEN_MEAN_FITNESS_KEY,
    GEN_BEST_NMSE_KEY,
    N_VALID_KEY,
    N_UNIQUE_KEY,
    GEN_MEAN_COMPLEXITY_KEY,
    LHS_UT_KEY,
    LHS_UTT_KEY,
)













_PLOT_METRIC: dict[str, str] = {
    "fitness_spread": GEN_MEAN_FITNESS_KEY,
    "population_diversity": N_UNIQUE_KEY,
    "complexity_evolution": GEN_MEAN_COMPLEXITY_KEY,
}



_PLOT_INFOS: tuple[PlotInfo, ...] = (
    PlotInfo(
        name="fitness_spread",
        title="Fitness Spread",
        description=(
            "Per-generation MEAN GA fitness (NMSE + epsilon*length; lower is "
            "better) over valid individuals, on a LOG y-axis: single "
            "blown-up individuals (svd_null_space normalization) push the "
            "mean into the thousands and a linear axis flattens the range "
            "the search actually works in. Complements the platform "
            "'convergence' plot, which draws the single global best (DLGA's "
            "GA has no elitism, so the per-generation best fluctuates). A "
            "generation appears as a gap whenever its mean is non-finite — "
            "one inf-fitness individual suffices, not only all-invalid "
            "generations."
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
            "Mean retained-term count (EvaluationResult.complexity — terms "
            "kept in the final fitted equation) of valid individuals per "
            "generation. NOT the genome gene count that the epsilon fitness "
            "penalty acts on: module-internal bloat (more genes per term) "
            "moves the penalty but not this curve. All-invalid generations "
            "appear as gaps (nothing was measured)."
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




_LOG_Y_PLOTS: frozenset[str] = frozenset({"fitness_spread"})

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


def render(name: str, ax: Axes, recorder: VizRecorder | None) -> list[str]:
    _check_known_name(name)



    if name == _SURROGATE_PLOT_NAME:
        return render_surrogate(ax, recorder)
    metric = _PLOT_METRIC[name]
    series = _safe_get_series(recorder, metric)
    title = _plot_title(name)

    ax.set_xlabel(_X_LABEL)
    ax.set_ylabel(metric)
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







    if name in _LOG_Y_PLOTS:



        ax.set_yscale("log")
        display: Any = _mask_for_log(series)
    else:
        display = [
            float("nan")
            if value is None or (isinstance(value, float) and not math.isfinite(value))
            else value
            for value in series
        ]

    ax.plot(
        range(len(series)),
        display,
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


def render_surrogate(ax: Axes, recorder: VizRecorder | None) -> list[str]:
    warnings: list[str] = []
    epochs = _last_logged_list(recorder, SURROGATE_EPOCH_KEY)
    train = _last_logged_list(recorder, SURROGATE_TRAIN_LOSS_KEY)

    ax.set_xlabel(_SURROGATE_X_LABEL)
    ax.set_ylabel(_SURROGATE_Y_LABEL)
    ax.set_title(_plot_title(_SURROGATE_PLOT_NAME))

    if not epochs or not train:






        reason = (
            "no recorder"
            if recorder is None
            else "no surrogate training logged for this run"
        )
        ax.text(
            0.5,
            0.5,
            f"{_NO_DATA_TEXT} ({reason})",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return [
            f"plugin plot '{_SURROGATE_PLOT_NAME}': {_NO_DATA_TEXT} ({reason})"
        ]

    train_x, train_y = _aligned_xy(
        epochs, train, SURROGATE_TRAIN_LOSS_KEY, warnings
    )
    ax.set_yscale("log")
    ax.plot(
        train_x,
        _mask_for_log(train_y),
        marker=_LINE_MARKER,
        markersize=_LINE_MARKER_SIZE,
        label=_SURROGATE_TRAIN_LABEL,
    )
    val = _last_logged_list(recorder, SURROGATE_VAL_LOSS_KEY)
    if val:
        val_x, val_y = _aligned_xy(
            epochs, val, SURROGATE_VAL_LOSS_KEY, warnings
        )
        ax.plot(
            val_x,
            _mask_for_log(val_y),
            marker=_LINE_MARKER,
            markersize=_LINE_MARKER_SIZE,
            label=_SURROGATE_VAL_LABEL,
        )
    best_epoch = _last_logged_scalar(recorder, SURROGATE_BEST_EPOCH_KEY)
    if isinstance(best_epoch, (int, float)) and not isinstance(best_epoch, bool):
        ax.axvline(float(best_epoch), label=_SURROGATE_BEST_EPOCH_LABEL)
    ax.legend()
    return warnings


def surrogate_data(recorder: VizRecorder | None) -> dict[str, Any]:
    epochs = _last_logged_list(recorder, SURROGATE_EPOCH_KEY)
    train = _last_logged_list(recorder, SURROGATE_TRAIN_LOSS_KEY)
    val = _last_logged_list(recorder, SURROGATE_VAL_LOSS_KEY)
    best_epoch = _last_logged_scalar(recorder, SURROGATE_BEST_EPOCH_KEY)


    train_x, train_y = _aligned_xy(epochs, train, SURROGATE_TRAIN_LOSS_KEY)
    val_y = (
        _val_aligned_to_x(val, len(train_x), SURROGATE_VAL_LOSS_KEY) if val else None
    )
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
    epochs: list[Any],
    series: list[Any],
    series_key: str,
    warnings: list[str] | None = None,
) -> tuple[list[Any], list[Any]]:
    if len(epochs) == len(series):
        return epochs, series
    common = min(len(epochs), len(series))



    note = (
        f"plugin plot '{_SURROGATE_PLOT_NAME}': surrogate '{series_key}' "
        f"length {len(series)} != epoch length {len(epochs)}; truncated to "
        f"{common} (drifted recorder)"
    )
    logger.warning(note)
    if warnings is not None:
        warnings.append(note)
    return epochs[:common], series[:common]


def _val_aligned_to_x(series: list[Any], width: int, series_key: str) -> list[Any]:
    if len(series) == width:
        return series
    logger.warning(
        "dlga.viz: surrogate '%s' length %d != exported x length %d; %s to %d "
        "(drifted recorder).",
        series_key,
        len(series),
        width,
        "truncating" if len(series) > width else "tail-padding with None",
        width,
    )
    if len(series) > width:
        return series[:width]
    return [*series, *([None] * (width - len(series)))]


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
    "GEN_BEST_FITNESS_KEY",
    "GEN_BEST_NMSE_KEY",
    "GEN_MEAN_COMPLEXITY_KEY",
    "GEN_MEAN_FITNESS_KEY",
    "LHS_UTT_KEY",
    "LHS_UT_KEY",
    "LOGGED_METRICS",
    "N_UNIQUE_KEY",
    "N_VALID_KEY",
    "get_data",
    "list_plot_infos",
    "render",
]
