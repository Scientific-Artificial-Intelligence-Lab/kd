
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kd.core.jsonsafe import finite_or_none
from kd.viz.plugin_panels import PanelSet, draw_no_data, last_logged_scalar

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.recorder import VizRecorder




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

_XLABEL = "score_source"
_YLABEL = "nmse"
_NATIVE_BAR_LABEL = "native"
_REFIT_BAR_LABEL = "refit"


@dataclass(frozen=True)
class _AgreementPanel:

    name: str
    title: str
    description: str

    def render(self, ax: Axes, recorder: VizRecorder | None) -> list[str]:
        native = finite_or_none(last_logged_scalar(recorder, NATIVE_NMSE_KEY))
        refit = finite_or_none(last_logged_scalar(recorder, REFIT_NMSE_KEY))
        support = last_logged_scalar(recorder, SUPPORT_SIZE_KEY)
        ax.set_xlabel(_XLABEL)
        ax.set_ylabel(_YLABEL)
        ax.set_title(self.title)
        bars = [
            (label, value)
            for label, value in ((_NATIVE_BAR_LABEL, native), (_REFIT_BAR_LABEL, refit))
            if value is not None
        ]
        if not bars:
            reason = "no recorder" if recorder is None else "metrics not logged"
            return draw_no_data(ax, self.name, reason)
        warnings: list[str] = []
        heights = [value for _, value in bars]
        ax.bar([label for label, _ in bars], heights)
        if all(height > 0 for height in heights):
            ax.set_yscale("log")
        notes: list[str] = []
        if isinstance(support, int) and not isinstance(support, bool):
            notes.append(f"support size: {support}")
        if refit is None and native is not None:
            notes.append("refit invalid")
            warnings.append(
                f"plugin plot '{self.name}': refit invalid; refit bar omitted"
            )
        if native is None and refit is not None:
            notes.append("native invalid")
            warnings.append(
                f"plugin plot '{self.name}': native invalid; native bar omitted"
            )
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

    def data(self, recorder: VizRecorder | None) -> dict[str, Any]:
        support = last_logged_scalar(recorder, SUPPORT_SIZE_KEY)
        return {
            NATIVE_NMSE_KEY: finite_or_none(
                last_logged_scalar(recorder, NATIVE_NMSE_KEY)
            ),
            REFIT_NMSE_KEY: finite_or_none(
                last_logged_scalar(recorder, REFIT_NMSE_KEY)
            ),
            SUPPORT_SIZE_KEY: (
                support
                if isinstance(support, int) and not isinstance(support, bool)
                else None
            ),
            "xlabel": _XLABEL,
            "ylabel": _YLABEL,
            "title": self.title,
        }


_PANELS = PanelSet(
    (
        _AgreementPanel(
            name="native_refit_agreement",
            title="Native vs Refit NMSE",
            description=(
                "pysindy's native-coefficient NMSE vs kd's independent refit "
                "re-score for the one selected STLSQ model -- do the two "
                "scores agree? Support size annotated."
            ),
        ),
    )
)

list_plot_infos = _PANELS.list_plot_infos
render = _PANELS.render
get_data = _PANELS.get_data


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
