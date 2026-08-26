
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from kd.core.jsonsafe import finite_or_none
from kd.viz.extension import PlotInfo

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.recorder import VizRecorder


logger = logging.getLogger(__name__)




NATIVE_NMSE_KEY = "native_nmse"
REFIT_NMSE_KEY = "refit_nmse"
SUPPORT_SIZE_KEY = "support_size"







STLSQ_HISTORY_KEY = "stlsq_history"
LOGGED_METRICS: tuple[str, ...] = (
    NATIVE_NMSE_KEY,
    REFIT_NMSE_KEY,
    SUPPORT_SIZE_KEY,
    STLSQ_HISTORY_KEY,
)

_AGREEMENT_PLOT_NAME = "native_refit_agreement"
_AGREEMENT_TITLE = "Native vs Refit NMSE"
_XLABEL = "score_source"
_YLABEL = "nmse"
_NATIVE_BAR_LABEL = "native"
_REFIT_BAR_LABEL = "refit"


_NO_DATA_TEXT = "No data"


def list_plot_infos() -> list[PlotInfo]:
    return [
        PlotInfo(
            name=_AGREEMENT_PLOT_NAME,
            title=_AGREEMENT_TITLE,
            description=(
                "pysindy's native-coefficient NMSE vs kd's independent refit "
                "re-score for the one selected STLSQ model -- do the two "
                "scores agree? Support size annotated."
            ),
        )
    ]


def render(name: str, ax: Axes, recorder: VizRecorder | None) -> list[str]:
    _require_known(name)
    native = finite_or_none(_last_logged_scalar(recorder, NATIVE_NMSE_KEY))
    refit = finite_or_none(_last_logged_scalar(recorder, REFIT_NMSE_KEY))
    support = _last_logged_scalar(recorder, SUPPORT_SIZE_KEY)

    ax.set_xlabel(_XLABEL)
    ax.set_ylabel(_YLABEL)
    ax.set_title(_AGREEMENT_TITLE)

    bars = [
        (label, value)
        for label, value in ((_NATIVE_BAR_LABEL, native), (_REFIT_BAR_LABEL, refit))
        if value is not None
    ]
    if not bars:
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

    warnings: list[str] = []
    labels = [label for label, _ in bars]
    heights = [value for _, value in bars]
    ax.bar(labels, heights)
    if all(height > 0 for height in heights):
        ax.set_yscale("log")

    notes: list[str] = []
    if isinstance(support, int) and not isinstance(support, bool):
        notes.append(f"support size: {support}")
    if refit is None and native is not None:
        notes.append("refit invalid")
        warnings.append(f"plugin plot '{name}': refit invalid; refit bar omitted")
    if notes:
        ax.text(
            0.02,
            0.98,
            "\n".join(notes),
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
    return warnings


def get_data(name: str, recorder: VizRecorder | None) -> dict[str, Any]:
    _require_known(name)
    support = _last_logged_scalar(recorder, SUPPORT_SIZE_KEY)
    return {
        NATIVE_NMSE_KEY: finite_or_none(
            _last_logged_scalar(recorder, NATIVE_NMSE_KEY)
        ),
        REFIT_NMSE_KEY: finite_or_none(_last_logged_scalar(recorder, REFIT_NMSE_KEY)),
        SUPPORT_SIZE_KEY: (
            support if isinstance(support, int) and not isinstance(support, bool)
            else None
        ),
        "xlabel": _XLABEL,
        "ylabel": _YLABEL,
        "title": _AGREEMENT_TITLE,
    }





def _require_known(name: str) -> None:
    if name != _AGREEMENT_PLOT_NAME:
        raise ValueError(
            f"Unknown pysindy plot {name!r}; expected {_AGREEMENT_PLOT_NAME!r}"
        )


def _last_logged_scalar(recorder: VizRecorder | None, key: str) -> Any:
    if recorder is None:
        return None
    series = recorder.get(key)
    if not series:
        return None
    return series[-1]


def _no_data_reason(recorder: VizRecorder | None) -> str:
    if recorder is None:
        return "no recorder"
    return "metrics not logged"


__all__ = [
    "LOGGED_METRICS",
    "NATIVE_NMSE_KEY",
    "REFIT_NMSE_KEY",
    "STLSQ_HISTORY_KEY",
    "SUPPORT_SIZE_KEY",
    "get_data",
    "list_plot_infos",
    "render",
]
