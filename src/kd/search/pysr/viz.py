
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kd.search.series_keys import (
    PARETO_COMPLEXITY_KEY,
    PARETO_EXPRESSIONS_KEY,
    PARETO_LOSS_KEY,
    PARETO_NMSE_KEY,
    PARETO_SCALE_KEY,
    SELECTED_COMPLEXITY_KEY,
    SELECTED_LOSS_KEY,
    SELECTED_NMSE_KEY,
)
from kd.viz.axes import integer_ticks
from kd.viz.plugin_panels import (
    PanelSet,
    draw_no_data,
    last_logged_list,
    last_logged_scalar,
    no_data_reason,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.recorder import VizRecorder


logger = logging.getLogger(__name__)















LOGGED_METRICS: tuple[str, ...] = (
    PARETO_COMPLEXITY_KEY,
    PARETO_LOSS_KEY,
    PARETO_NMSE_KEY,
    PARETO_EXPRESSIONS_KEY,
    PARETO_SCALE_KEY,
    SELECTED_COMPLEXITY_KEY,
    SELECTED_LOSS_KEY,
    SELECTED_NMSE_KEY,
)



_COMPLEXITY_LABEL = "complexity"
_PYSR_LOSS_LABEL = "pysr_loss"
_KD_NMSE_LABEL = "kd_nmse"





_LINE_MARKER = "o"
_LINE_MARKER_SIZE = 4
_SCATTER_MARKER = "o"
_SCATTER_MARKER_SIZE = 6
_SELECTED_MARKER = "*"
_SELECTED_MARKER_SIZE = 14


@dataclass(frozen=True)
class _ParetoPanel:

    name: str
    title: str
    description: str
    x_key: str
    y_key: str
    xlabel: str
    ylabel: str
    selected_x_key: str
    selected_y_key: str
    is_line: bool

    def render(self, ax: Axes, recorder: VizRecorder | None) -> list[str]:
        raw_x = last_logged_list(recorder, self.x_key)
        raw_y = last_logged_list(recorder, self.y_key)
        x_values, y_values = _finite_xy_pairs(raw_x, raw_y)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if self.xlabel == _COMPLEXITY_LABEL:


            integer_ticks(ax)
        ax.set_title(self.title)
        if not x_values:
            return draw_no_data(ax, self.name, no_data_reason(recorder))
        if self.is_line:
            ax.plot(
                x_values, y_values, marker=_LINE_MARKER, markersize=_LINE_MARKER_SIZE
            )
        else:
            ax.scatter(
                x_values, y_values, marker=_SCATTER_MARKER, s=_SCATTER_MARKER_SIZE
            )
        _draw_selected(ax, self, recorder)
        n_logged = min(len(raw_x), len(raw_y))
        n_dropped = n_logged - len(x_values)
        if n_dropped > 0:
            return [
                f"plugin plot '{self.name}': {n_dropped} of {n_logged} points "
                "dropped (non-finite coordinate)"
            ]
        return []

    def data(self, recorder: VizRecorder | None) -> dict[str, Any]:
        x_values, y_values = _finite_xy_pairs(
            last_logged_list(recorder, self.x_key),
            last_logged_list(recorder, self.y_key),
        )
        data: dict[str, Any] = {
            "x": x_values,
            "y": y_values,
            "xlabel": self.xlabel,
            "ylabel": self.ylabel,
            "title": self.title,
        }
        selected_x, selected_y = _selected_coordinates(recorder, self)
        if selected_x is not None and selected_y is not None:
            data["selected_x"] = selected_x
            data["selected_y"] = selected_y
        return data





_PANELS = PanelSet(
    (
        _ParetoPanel(
            name="pareto_front",
            title="Pareto Front",
            description=(
                "PySR's own complexity/loss Pareto front; the selected model is "
                "highlighted."
            ),
            x_key=PARETO_COMPLEXITY_KEY,
            y_key=PARETO_LOSS_KEY,
            xlabel=_COMPLEXITY_LABEL,
            ylabel=_PYSR_LOSS_LABEL,
            selected_x_key=SELECTED_COMPLEXITY_KEY,
            selected_y_key=SELECTED_LOSS_KEY,
            is_line=True,
        ),
        _ParetoPanel(
            name="kd_audit_path",
            title="KD Audit Path",
            description=(
                "KD's independent NMSE re-score per Pareto complexity (the audit "
                "substrate); the selected model is highlighted."
            ),
            x_key=PARETO_COMPLEXITY_KEY,
            y_key=PARETO_NMSE_KEY,
            xlabel=_COMPLEXITY_LABEL,
            ylabel=_KD_NMSE_LABEL,
            selected_x_key=SELECTED_COMPLEXITY_KEY,
            selected_y_key=SELECTED_NMSE_KEY,
            is_line=True,
        ),
        _ParetoPanel(
            name="score_agreement",
            title="Score Agreement",
            description=(
                "PySR loss vs KD NMSE per Pareto point -- do the two rankings agree?"
            ),
            x_key=PARETO_LOSS_KEY,
            y_key=PARETO_NMSE_KEY,
            xlabel=_PYSR_LOSS_LABEL,
            ylabel=_KD_NMSE_LABEL,
            selected_x_key=SELECTED_LOSS_KEY,
            selected_y_key=SELECTED_NMSE_KEY,
            is_line=False,
        ),
    )
)

list_plot_infos = _PANELS.list_plot_infos
render = _PANELS.render
get_data = _PANELS.get_data





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
    panel: _ParetoPanel,
) -> tuple[Any, Any]:
    selected_x = last_logged_scalar(recorder, panel.selected_x_key)
    selected_y = last_logged_scalar(recorder, panel.selected_y_key)
    if not _is_finite(selected_x) or not _is_finite(selected_y):
        return None, None
    return selected_x, selected_y


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


def _draw_selected(ax: Axes, panel: _ParetoPanel, recorder: VizRecorder | None) -> None:
    selected_x, selected_y = _selected_coordinates(recorder, panel)
    if selected_x is None or selected_y is None:
        return
    ax.scatter(
        [selected_x],
        [selected_y],
        marker=_SELECTED_MARKER,
        s=_SELECTED_MARKER_SIZE**2,
        zorder=5,
    )


__all__ = [
    "LOGGED_METRICS",
    "PARETO_COMPLEXITY_KEY",
    "PARETO_EXPRESSIONS_KEY",
    "PARETO_LOSS_KEY",
    "PARETO_NMSE_KEY",
    "PARETO_SCALE_KEY",
    "SELECTED_COMPLEXITY_KEY",
    "SELECTED_LOSS_KEY",
    "SELECTED_NMSE_KEY",
    "get_data",
    "list_plot_infos",
    "render",
]
