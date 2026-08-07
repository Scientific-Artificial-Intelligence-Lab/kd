
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import torch

from kd.core.safety import safe_div
from kd.viz.extension import PlotInfo

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_TITLE_PREFIX: Final[str] = "Reproduces EqGPT's steady result"
_NO_DATA: Final[str] = "No data available"
_NON_FINITE: Final[str] = "Non-finite data: plot unavailable"

PLOT_INFOS: Final[tuple[PlotInfo, ...]] = (
    PlotInfo(
        name="residual_domain",
        title=f"{_TITLE_PREFIX}: residual domain",
        description="Scattered x, y points colored by absolute equation residual.",
        projection="3d",
    ),
    PlotInfo(
        name="term_balance",
        title=f"{_TITLE_PREFIX}: term balance",
        description="Per-term column RMS with the free-pivot column highlighted.",
    ),
    PlotInfo(
        name="surrogate_fit",
        title=f"{_TITLE_PREFIX}: surrogate fit",
        description="Observed versus surrogate-predicted field values with R-squared.",
    ),
)


@dataclass(frozen=True)
class SteadyVizData:

    x: np.ndarray | None = None
    y: np.ndarray | None = None
    residual: np.ndarray | None = None
    matrix: np.ndarray | None = None
    terms: tuple[str, ...] = ()
    pivot_index: int = 0
    observed: np.ndarray | None = None
    predicted: np.ndarray | None = None


def render_steady_residual_domain(
    ax: Axes,
    x: np.ndarray | None,
    y: np.ndarray | None,
    residual: np.ndarray | None,
) -> list[str]:
    ax.set_title(f"{_TITLE_PREFIX}: residual domain")
    if x is None or y is None or residual is None:
        return _degrade(ax, _NO_DATA)
    x_arr, y_arr, residual_arr = (
        np.asarray(value).reshape(-1) for value in (x, y, residual)
    )
    if not (x_arr.size == y_arr.size == residual_arr.size) or x_arr.size == 0:
        return _degrade(ax, _NO_DATA)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(residual_arr)
    if not finite.all():
        return _degrade(ax, _NON_FINITE)
    magnitude = np.abs(residual_arr[finite])
    ax_3d = cast(Any, ax)
    ax_3d.scatter(
        x_arr[finite],
        y_arr[finite],
        magnitude,
        c=magnitude,
        s=10,
        rasterized=True,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax_3d.set_zlabel("|residual|")
    return []


def render_steady_term_balance(
    ax: Axes,
    matrix: np.ndarray | None,
    terms: list[str] | tuple[str, ...] | None,
    *,
    pivot_index: int,
) -> list[str]:
    ax.set_title(f"{_TITLE_PREFIX}: term balance")
    if matrix is None or terms is None:
        return _degrade(ax, _NO_DATA)
    values = np.asarray(matrix)



    labels = ["1" if term == "one" else term for term in terms]
    if values.ndim != 2 or values.shape[1] != len(labels) or values.size == 0:
        return _degrade(ax, _NO_DATA)
    if not np.isfinite(values).all():
        return _degrade(ax, _NON_FINITE)
    if not 0 <= pivot_index < len(labels):
        return _degrade(ax, "No data: pivot index is unavailable")
    with np.errstate(over="ignore", invalid="ignore"):
        rms = np.sqrt(np.mean(np.square(values), axis=0))
    if not np.isfinite(rms).all():
        return _degrade(ax, _NON_FINITE)
    colors = [
        "tab:orange" if index == pivot_index else "tab:blue"
        for index in range(len(labels))
    ]
    bars = ax.bar(labels, rms, color=colors)
    bars[pivot_index].set_label("Pivot")
    ax.text(pivot_index, rms[pivot_index], "pivot", ha="center", va="bottom")
    ax.set_ylabel("Column RMS")
    return []


def render_steady_surrogate_fit(
    ax: Axes,
    observed: np.ndarray | None,
    predicted: np.ndarray | None,
) -> list[str]:
    ax.set_title(f"{_TITLE_PREFIX}: surrogate fit")
    if observed is None or predicted is None:
        return _degrade(ax, _NO_DATA)
    observed_arr = np.asarray(observed).reshape(-1)
    predicted_arr = np.asarray(predicted).reshape(-1)
    if observed_arr.size != predicted_arr.size or observed_arr.size == 0:
        return _degrade(ax, _NO_DATA)
    finite = np.isfinite(observed_arr) & np.isfinite(predicted_arr)
    if not finite.all():
        return _degrade(ax, _NON_FINITE)
    observed_finite = observed_arr[finite]
    predicted_finite = predicted_arr[finite]
    ax.scatter(observed_finite, predicted_finite, s=12, rasterized=True)
    lower = float(min(observed_finite.min(), predicted_finite.min()))
    upper = float(max(observed_finite.max(), predicted_finite.max()))
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="black")
    centered = observed_finite - observed_finite.mean()
    denominator = float(np.sum(np.square(centered)))
    if denominator > 0.0:
        numerator = float(np.sum(np.square(observed_finite - predicted_finite)))
        ratio = safe_div(
            torch.tensor(numerator, dtype=torch.float64),
            torch.tensor(denominator, dtype=torch.float64),
        )
        r2_text = f"R² = {1.0 - float(ratio.item()):.4f}"
    else:
        r2_text = "R² undefined (constant observations)"
    ax.text(0.05, 0.95, r2_text, transform=ax.transAxes, va="top")
    ax.set_xlabel("Observed u")
    ax.set_ylabel("Predicted u")
    return []


def render(name: str, ax: Axes, data: SteadyVizData) -> list[str]:
    if name == "residual_domain":
        notes = render_steady_residual_domain(ax, data.x, data.y, data.residual)
    elif name == "term_balance":
        notes = render_steady_term_balance(
            ax, data.matrix, data.terms, pivot_index=data.pivot_index
        )
    elif name == "surrogate_fit":
        notes = render_steady_surrogate_fit(ax, data.observed, data.predicted)
    else:
        raise ValueError(f"Unknown EqGPT steady plot: {name!r}")
    return [f"homogeneous plot '{name}': {note}" for note in notes]


def _degrade(ax: Axes, message: str) -> list[str]:
    text_2d = getattr(ax, "text2D", None)
    if text_2d is not None:
        text_2d(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    return [message]


__all__ = [
    "PLOT_INFOS",
    "SteadyVizData",
    "render",
    "render_steady_residual_domain",
    "render_steady_surrogate_fit",
    "render_steady_term_balance",
]
