
from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from kd.core.jsonsafe import finite_or_none
from kd.viz.axes import integer_ticks
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

logger = logging.getLogger(__name__)


NO_DATA_TEXT = "No data"



_LINE_MARKER = "."
_LINE_MARKER_SIZE = 3


class Panel(Protocol):

    @property
    def name(self) -> str: ...

    @property
    def title(self) -> str: ...

    @property
    def description(self) -> str: ...

    def render(self, ax: Axes, recorder: VizRecorder | None) -> list[str]:
        ...

    def data(self, recorder: VizRecorder | None) -> dict[str, Any]:
        ...


class PanelSet:

    def __init__(self, panels: Sequence[Panel]) -> None:
        self._panels: dict[str, Panel] = {panel.name: panel for panel in panels}

    def list_plot_infos(self) -> list[PlotInfo]:
        return [
            PlotInfo(name=panel.name, title=panel.title, description=panel.description)
            for panel in self._panels.values()
        ]

    def render(self, name: str, ax: Axes, recorder: VizRecorder | None) -> list[str]:
        return self._panel(name).render(ax, recorder)

    def get_data(self, name: str, recorder: VizRecorder | None) -> dict[str, Any]:
        return self._panel(name).data(recorder)

    def _panel(self, name: str) -> Panel:
        panel = self._panels.get(name)
        if panel is None:
            available = ", ".join(self._panels)
            raise ValueError(f"Unknown plot name: {name!r}. Available: {available}")
        return panel





def recorder_series(recorder: VizRecorder | None, key: str) -> list[Any]:
    if recorder is None:
        return []
    return recorder.get(key)


def last_logged_list(recorder: VizRecorder | None, key: str) -> list[Any]:
    series = recorder_series(recorder, key)
    if not series:
        return []
    return list(series[-1])


def last_logged_scalar(recorder: VizRecorder | None, key: str) -> Any:
    series = recorder_series(recorder, key)
    if not series:
        return None
    return series[-1]


def no_data_reason(recorder: VizRecorder | None) -> str:
    if recorder is None:
        return "no recorder"
    if not recorder.enabled:
        return "disabled"
    return "empty"





def draw_no_data(ax: Axes, name: str, reason: str) -> list[str]:
    ax.text(
        0.5,
        0.5,
        f"{NO_DATA_TEXT} ({reason})",
        transform=ax.transAxes,
        ha="center",
        va="center",
    )
    return [f"plugin plot '{name}': {NO_DATA_TEXT} ({reason})"]


def sanitize_y(value: Any) -> float | None:
    if value is not None and not isinstance(value, (int, float)):
        logger.warning(
            "plugin viz: dropping unsupported recorder payload type=%s value=%r",
            type(value).__name__,
            value,
        )
    return finite_or_none(value)


def mask_non_finite(series: Sequence[Any]) -> list[Any]:
    return [
        math.nan
        if value is None or (isinstance(value, float) and not math.isfinite(value))
        else value
        for value in series
    ]


def mask_for_log(series: Sequence[Any]) -> np.ndarray:
    return np.asarray(
        [
            math.nan
            if value is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0.0
            else float(value)
            for value in series
        ],
        dtype=float,
    )


def pad_series(series: Sequence[Any], length: int) -> list[Any]:
    return [*series, *([None] * (length - len(series)))]


def series_gap_annotator(vocab: GapVocabulary) -> Callable[[Axes, list[Any]], None]:

    def annotate(ax: Axes, series: list[Any]) -> None:
        annotate_gaps(ax, measured_flags(series), vocab)

    return annotate


def _plot_index_curve(ax: Axes, values: Any, *, label: str | None = None) -> None:
    ax.plot(
        range(len(values)),
        values,
        marker=_LINE_MARKER,
        markersize=_LINE_MARKER_SIZE,
        label=label,
    )





@dataclass(frozen=True)
class SeriesPanel:

    name: str
    title: str
    description: str
    metric: str
    x_label: str
    ylabel: str | None = None
    count_axis: bool = False
    log_y: bool = False
    derive: Callable[[list[Any]], list[Any]] | None = None
    annotate: Callable[[Axes, list[Any]], None] | None = None

    def _series(self, recorder: VizRecorder | None) -> tuple[list[Any], list[Any]]:
        recorded = recorder_series(recorder, self.metric)
        plotted = recorded if self.derive is None else self.derive(recorded)
        return recorded, plotted

    def _ylabel(self) -> str:
        return self.metric if self.ylabel is None else self.ylabel

    def render(self, ax: Axes, recorder: VizRecorder | None) -> list[str]:
        recorded, plotted = self._series(recorder)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self._ylabel())
        integer_ticks(ax)
        if self.count_axis:
            integer_ticks(ax, "y")
        ax.set_title(self.title)
        if not plotted:
            return draw_no_data(ax, self.name, no_data_reason(recorder))
        if self.log_y:
            ax.set_yscale("log")
            _plot_index_curve(ax, mask_for_log(plotted))
        else:
            _plot_index_curve(ax, mask_non_finite(plotted))
        if self.annotate is not None:
            self.annotate(ax, recorded)
        return []

    def data(self, recorder: VizRecorder | None) -> dict[str, Any]:
        _, plotted = self._series(recorder)
        return {
            "x": list(range(len(plotted))),
            "y": [sanitize_y(value) for value in plotted],
            "xlabel": self.x_label,
            "ylabel": self._ylabel(),
            "title": self.title,
        }


@dataclass(frozen=True)
class BandPanel:

    name: str
    title: str
    description: str
    metrics: tuple[str, ...]
    x_label: str
    ylabel: str
    gaps: GapVocabulary

    def _band(self, recorder: VizRecorder | None) -> tuple[dict[str, list[Any]], int]:
        by_metric = {
            metric: recorder_series(recorder, metric) for metric in self.metrics
        }
        width = max((len(series) for series in by_metric.values()), default=0)
        return by_metric, width

    def render(self, ax: Axes, recorder: VizRecorder | None) -> list[str]:
        by_metric, width = self._band(recorder)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.ylabel)
        integer_ticks(ax)
        ax.set_title(self.title)
        if width == 0:
            return draw_no_data(ax, self.name, no_data_reason(recorder))
        for metric in self.metrics:
            padded = pad_series(by_metric[metric], width)
            _plot_index_curve(ax, mask_non_finite(padded), label=metric)
        ax.legend()
        annotate_gaps(ax, band_measured_flags(by_metric.values(), width), self.gaps)
        return []

    def data(self, recorder: VizRecorder | None) -> dict[str, Any]:
        by_metric, width = self._band(recorder)
        return {
            "x": list(range(width)),
            "y": {
                metric: [sanitize_y(value) for value in pad_series(series, width)]
                for metric, series in by_metric.items()
            },
            "xlabel": self.x_label,
            "ylabel": self.ylabel,
            "title": self.title,
        }


_SURROGATE_X_LABEL = "Epoch"
_SURROGATE_Y_LABEL = "MSE loss"
_SURROGATE_TRAIN_LABEL = "train"
_SURROGATE_VAL_LABEL = "val"
_SURROGATE_BEST_EPOCH_LABEL = "best epoch"


@dataclass(frozen=True)
class SurrogateCurvePanel:

    name: str
    title: str
    description: str
    epoch_key: str
    train_key: str
    val_key: str
    best_epoch_key: str

    def render(self, ax: Axes, recorder: VizRecorder | None) -> list[str]:
        warnings: list[str] = []
        epochs = last_logged_list(recorder, self.epoch_key)
        train = last_logged_list(recorder, self.train_key)
        ax.set_xlabel(_SURROGATE_X_LABEL)
        ax.set_ylabel(_SURROGATE_Y_LABEL)
        integer_ticks(ax)
        ax.set_title(self.title)
        if not epochs or not train:



            reason = (
                "no recorder"
                if recorder is None
                else "no surrogate training logged for this run"
            )
            return draw_no_data(ax, self.name, reason)
        ax.set_yscale("log")
        train_x, train_y = self._aligned(epochs, train, self.train_key, warnings)
        ax.plot(
            train_x,
            mask_for_log(train_y),
            marker=_LINE_MARKER,
            markersize=_LINE_MARKER_SIZE,
            label=_SURROGATE_TRAIN_LABEL,
        )
        val = last_logged_list(recorder, self.val_key)
        if val:
            val_x, val_y = self._aligned(epochs, val, self.val_key, warnings)
            ax.plot(
                val_x,
                mask_for_log(val_y),
                marker=_LINE_MARKER,
                markersize=_LINE_MARKER_SIZE,
                label=_SURROGATE_VAL_LABEL,
            )
        best_epoch = last_logged_scalar(recorder, self.best_epoch_key)
        if isinstance(best_epoch, (int, float)) and not isinstance(best_epoch, bool):
            ax.axvline(float(best_epoch), label=_SURROGATE_BEST_EPOCH_LABEL)
        ax.legend()
        return warnings

    def data(self, recorder: VizRecorder | None) -> dict[str, Any]:
        epochs = last_logged_list(recorder, self.epoch_key)
        train = last_logged_list(recorder, self.train_key)
        val = last_logged_list(recorder, self.val_key)
        train_x, train_y = self._aligned(epochs, train, self.train_key, None)
        val_y = self._val_on_x(val, len(train_x)) if val else None
        return {
            "x": [finite_or_none(value) for value in train_x],
            "y_train": [sanitize_y(value) for value in train_y],
            "y_val": None if val_y is None else [sanitize_y(value) for value in val_y],
            "best_epoch": sanitize_y(last_logged_scalar(recorder, self.best_epoch_key)),
            "xlabel": _SURROGATE_X_LABEL,
            "ylabel": _SURROGATE_Y_LABEL,
            "title": self.title,
        }

    def _aligned(
        self,
        epochs: list[Any],
        series: list[Any],
        series_key: str,
        warnings: list[str] | None,
    ) -> tuple[list[Any], list[Any]]:
        if len(epochs) == len(series):
            return epochs, series
        common = min(len(epochs), len(series))
        note = (
            f"plugin plot '{self.name}': surrogate '{series_key}' "
            f"length {len(series)} != epoch length {len(epochs)}; truncated to "
            f"{common} (drifted recorder)"
        )
        logger.warning(note)
        if warnings is not None:
            warnings.append(note)
        return epochs[:common], series[:common]

    def _val_on_x(self, series: list[Any], width: int) -> list[Any]:
        if len(series) == width:
            return series
        logger.warning(
            "plugin plot '%s': surrogate '%s' length %d != exported x length %d; "
            "%s to %d (drifted recorder).",
            self.name,
            self.val_key,
            len(series),
            width,
            "truncating" if len(series) > width else "tail-padding with None",
            width,
        )
        if len(series) > width:
            return series[:width]
        return pad_series(series, width)


__all__ = [
    "NO_DATA_TEXT",
    "BandPanel",
    "Panel",
    "PanelSet",
    "SeriesPanel",
    "SurrogateCurvePanel",
    "draw_no_data",
    "last_logged_list",
    "last_logged_scalar",
    "mask_for_log",
    "mask_non_finite",
    "no_data_reason",
    "pad_series",
    "recorder_series",
    "sanitize_y",
    "series_gap_annotator",
]
