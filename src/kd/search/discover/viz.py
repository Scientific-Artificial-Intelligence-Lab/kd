
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from kd.viz.axes import integer_ticks
from kd.viz.extension import PlotInfo

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.recorder import VizRecorder


logger = logging.getLogger(__name__)





_PLOT_METRIC: dict[str, str] = {
    "reward_convergence": "reward",
    "reward_full_mean": "reward_full",
    "entropy_loss_decay": "entropy_loss",
    "baseline_ewma": "baseline",
}






_PLOT_INFOS: tuple[PlotInfo, ...] = (
    PlotInfo(
        name="reward_convergence",
        title="Reward (Mean of Top ε)",
        description=(
            "Per-iteration mean reward of the top-ε risk-seeking subset the "
            "policy gradient trains on — kd's analogue of what the reference "
            "implementation calls 'Mean of Top ε' (r_avg_sub; kd selects the "
            "subset after invalid-row filtering, delta, so the subsets "
            "can differ for larger batches), NOT the batch reward. Under "
            "the default batch=16, ε=0.05 the subset is a single sample, so "
            "this curve equals the per-iteration maximum, and with the "
            "default 'R_e' baseline it also coincides with the Reward "
            "Baseline plot by construction. The whole-batch signal is the "
            "'Reward (Full-Batch Mean)' plot."
        ),
    ),
    PlotInfo(
        name="reward_full_mean",
        title="Reward (Full-Batch Mean)",
        description=(
            "Per-iteration mean reward over the FULL sampled batch (invalid "
            "expressions carry the invalid-reward fill) — the reference "
            "implementation's r_avg_full. Unlike the top-ε curve this one "
            "moves with the whole policy, so the two panels together show "
            "elite vs population progress."
        ),
    ),
    PlotInfo(
        name="entropy_loss_decay",
        title="Entropy Loss",
        description=(
            "Entropy regularizer loss component (= -weight x mean policy "
            "entropy, typically negative). Direction-neutral title on "
            "purpose: the curve RISES toward zero as the policy's entropy "
            "collapses — it does not 'decay' for a healthy run."
        ),
    ),
    PlotInfo(
        name="baseline_ewma",
        title="Reward Baseline",
        description=(
            "Scalar reward baseline used by the RSPG policy-gradient "
            "estimator. NOTE: the trailing ``_ewma`` in the plot name is a "
            "legacy artifact retained for backward compatibility with an "
            "earlier plot-naming scheme — the actual series semantics depend on "
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


def render(name: str, ax: Axes, recorder: VizRecorder | None) -> list[str]:
    _check_known_name(name)
    metric = _PLOT_METRIC[name]
    series = _safe_get_series(recorder, metric)
    title = _plot_title(name)

    ax.set_xlabel(_X_LABEL)
    ax.set_ylabel(metric)
    integer_ticks(ax)
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

    ax.plot(
        range(len(series)),
        series,
        marker=_LINE_MARKER,
        markersize=_LINE_MARKER_SIZE,
    )
    return []


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
    if value is None:




        return None
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
