
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
    "reward_convergence": "reward",
    "entropy_loss_decay": "entropy_loss",
    "baseline_ewma": "baseline",
}





_PLOT_INFOS: tuple[PlotInfo, ...] = (
    PlotInfo(
        name="reward_convergence",
        title="Reward Convergence",
        description=(
            "Per-iteration reward signal from the policy-gradient training loop."
        ),
    ),
    PlotInfo(
        name="entropy_loss_decay",
        title="Entropy Loss Decay",
        description=("Entropy regularizer loss component over training iterations."),
    ),
    PlotInfo(
        name="baseline_ewma",
        title="Reward Baseline",
        description=(
            "Scalar reward baseline used by the RSPG policy-gradient "
            "estimator. NOTE: the trailing ``_ewma`` in the plot name is a "
            "legacy artifact retained for backward compatibility with the "
            "Stage 3 spec — the actual series semantics depend on "
            "``RSPGStrategy.baseline``: ``'R_e'`` (default) is a "
            "risk-seeking quantile, ``'ewma_R'`` is an exponential moving "
            "average, ``'combined'`` sums both. Read the configured "
            "baseline mode before interpreting the curve; the human-readable "
            "plot title (``Reward Baseline``) is mode-neutral on purpose."
        ),
    ),
)




_NO_DATA_TEXT = "No data"

_X_LABEL = "Iteration"




_LINE_MARKER = "."
_LINE_MARKER_SIZE = 3


def list_plot_infos() -> list[PlotInfo]:
    return [
        PlotInfo(
            name=info.name,
            title=info.title,
            description=info.description,
        )
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

    ax.plot(
        range(len(series)),
        series,
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


def _safe_get_series(
    recorder: VizRecorder | None,
    metric: str,
) -> list[Any]:
    if recorder is None:
        return []
    return recorder.get(metric)


def _sanitize_y(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, int):
        return float(value)
    logger.warning(
        "discover.viz._sanitize_y: dropping unsupported recorder payload "
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
