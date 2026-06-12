
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from kd.viz.extension import PlotInfo

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.recorder import VizRecorder


logger = logging.getLogger(__name__)




_PARETO_COMPLEXITY_KEY = "pareto_complexity"
_PARETO_LOSS_KEY = "pareto_loss"
_PARETO_NMSE_KEY = "pareto_nmse"
_SELECTED_COMPLEXITY_KEY = "selected_complexity"
_SELECTED_LOSS_KEY = "selected_loss"
_SELECTED_NMSE_KEY = "selected_nmse"



_COMPLEXITY_LABEL = "complexity"
_PYSR_LOSS_LABEL = "pysr_loss"
_KD_NMSE_LABEL = "kd_nmse"


class _PlotSpec:

    __slots__ = (
        "name",
        "title",
        "description",
        "x_key",
        "y_key",
        "xlabel",
        "ylabel",
        "selected_x_key",
        "selected_y_key",
        "is_line",
    )

    def __init__(
        self,
        *,
        name: str,
        title: str,
        description: str,
        x_key: str,
        y_key: str,
        xlabel: str,
        ylabel: str,
        selected_x_key: str,
        selected_y_key: str,
        is_line: bool,
    ) -> None:
        self.name = name
        self.title = title
        self.description = description
        self.x_key = x_key
        self.y_key = y_key
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.selected_x_key = selected_x_key
        self.selected_y_key = selected_y_key
        self.is_line = is_line





_PLOT_SPECS: tuple[_PlotSpec, ...] = (
    _PlotSpec(
        name="pareto_front",
        title="Pareto Front",
        description=(
            "PySR's own complexity/loss Pareto front; the selected model is "
            "highlighted."
        ),
        x_key=_PARETO_COMPLEXITY_KEY,
        y_key=_PARETO_LOSS_KEY,
        xlabel=_COMPLEXITY_LABEL,
        ylabel=_PYSR_LOSS_LABEL,
        selected_x_key=_SELECTED_COMPLEXITY_KEY,
        selected_y_key=_SELECTED_LOSS_KEY,
        is_line=True,
    ),
    _PlotSpec(
        name="kd_audit_path",
        title="kd Audit Path",
        description=(
            "kd's independent NMSE re-score per Pareto complexity (the audit "
            "substrate); the selected model is highlighted."
        ),
        x_key=_PARETO_COMPLEXITY_KEY,
        y_key=_PARETO_NMSE_KEY,
        xlabel=_COMPLEXITY_LABEL,
        ylabel=_KD_NMSE_LABEL,
        selected_x_key=_SELECTED_COMPLEXITY_KEY,
        selected_y_key=_SELECTED_NMSE_KEY,
        is_line=True,
    ),
    _PlotSpec(
        name="score_agreement",
        title="Score Agreement",
        description=(
            "PySR loss vs kd NMSE per Pareto point -- do the two rankings agree?"
        ),
        x_key=_PARETO_LOSS_KEY,
        y_key=_PARETO_NMSE_KEY,
        xlabel=_PYSR_LOSS_LABEL,
        ylabel=_KD_NMSE_LABEL,
        selected_x_key=_SELECTED_LOSS_KEY,
        selected_y_key=_SELECTED_NMSE_KEY,
        is_line=False,
    ),
)

_PLOT_BY_NAME: dict[str, _PlotSpec] = {spec.name: spec for spec in _PLOT_SPECS}



_NO_DATA_TEXT = "No data"




_LINE_MARKER = "o"
_LINE_MARKER_SIZE = 4
_SCATTER_MARKER = "o"
_SCATTER_MARKER_SIZE = 6
_SELECTED_MARKER = "*"
_SELECTED_MARKER_SIZE = 14


def list_plot_infos() -> list[PlotInfo]:
    return [
        PlotInfo(name=spec.name, title=spec.title, description=spec.description)
        for spec in _PLOT_SPECS
    ]


def render(name: str, ax: Axes, recorder: VizRecorder | None) -> None:
    spec = _require_spec(name)
    x_values, y_values = _finite_xy_pairs(
        _last_logged_list(recorder, spec.x_key),
        _last_logged_list(recorder, spec.y_key),
    )

    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel(spec.ylabel)
    ax.set_title(spec.title)

    if not x_values or not y_values:
        ax.text(
            0.5,
            0.5,
            f"{_NO_DATA_TEXT} ({_no_data_reason(recorder)})",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return

    if spec.is_line:
        ax.plot(x_values, y_values, marker=_LINE_MARKER, markersize=_LINE_MARKER_SIZE)
    else:
        ax.scatter(x_values, y_values, marker=_SCATTER_MARKER, s=_SCATTER_MARKER_SIZE)
    _draw_selected(ax, spec, recorder)


def get_data(name: str, recorder: VizRecorder | None) -> dict[str, Any]:
    spec = _require_spec(name)
    x_values, y_values = _finite_xy_pairs(
        _last_logged_list(recorder, spec.x_key),
        _last_logged_list(recorder, spec.y_key),
    )
    data: dict[str, Any] = {
        "x": x_values,
        "y": y_values,
        "xlabel": spec.xlabel,
        "ylabel": spec.ylabel,
        "title": spec.title,
    }
    selected_x, selected_y = _selected_coordinates(recorder, spec)
    if selected_x is not None and selected_y is not None:
        data["selected_x"] = selected_x
        data["selected_y"] = selected_y
    return data





def _require_spec(name: str) -> _PlotSpec:
    spec = _PLOT_BY_NAME.get(name)
    if spec is None:
        available = ", ".join(spec.name for spec in _PLOT_SPECS)
        raise ValueError(f"Unknown plot name: {name!r}. Available: {available}")
    return spec


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


def _finite_xy_pairs(xs: list[Any], ys: list[Any]) -> tuple[list[Any], list[Any]]:
    out_x: list[Any] = []
    out_y: list[Any] = []
    for x_value, y_value in zip(xs, ys, strict=False):
        if _is_finite(x_value) and _is_finite(y_value):
            out_x.append(x_value)
            out_y.append(y_value)
    return out_x, out_y


def _selected_coordinates(
    recorder: VizRecorder | None,
    spec: _PlotSpec,
) -> tuple[Any, Any]:
    selected_x = _last_logged_scalar(recorder, spec.selected_x_key)
    selected_y = _last_logged_scalar(recorder, spec.selected_y_key)
    if not _is_finite(selected_x) or not _is_finite(selected_y):
        return None, None
    return selected_x, selected_y


def _last_logged_scalar(recorder: VizRecorder | None, key: str) -> Any:
    if recorder is None:
        return None
    series = recorder.get(key)
    if not series:
        return None
    return series[-1]


def _is_finite(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return not (math.isnan(value) or math.isinf(value))
    logger.warning(
        "pysr.viz._is_finite: dropping unsupported recorder payload "
        "type=%s value=%r -- VizRecorder should only store scalar numeric "
        "samples for plotted Pareto axes.",
        type(value).__name__,
        value,
    )
    return False


def _draw_selected(ax: Axes, spec: _PlotSpec, recorder: VizRecorder | None) -> None:
    selected_x, selected_y = _selected_coordinates(recorder, spec)
    if selected_x is None or selected_y is None:
        return
    ax.scatter(
        [selected_x],
        [selected_y],
        marker=_SELECTED_MARKER,
        s=_SELECTED_MARKER_SIZE**2,
        zorder=5,
    )


def _no_data_reason(recorder: VizRecorder | None) -> str:
    if recorder is None:
        return "no recorder"
    return "empty"


__all__ = [
    "get_data",
    "list_plot_infos",
    "render",
]
