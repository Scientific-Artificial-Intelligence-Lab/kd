
from __future__ import annotations

import math
import textwrap
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib.colors import Colormap
from numpy.typing import NDArray

from kd.viz import style as kd_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import QuadMesh
    from matplotlib.colorbar import Colorbar
    from matplotlib.image import AxesImage

    from kd.core.integrator import IntegrationResult

_RESIDUAL_PERCENTILE = 99.0
_WARNING_FONTSIZE = 9
_WARNING_WRAP_WIDTH = 38
_COLORBAR_FRACTION = 0.05
_COLORBAR_PAD = 0.02
_NOTE_GAP_POINTS = 3.0








_RESIDUAL_SIGN = "Predicted - True"




_OBSERVED_LABEL = "Observed"
_DISCOVERED_LABEL = "Discovered law, integrated"

FieldCmap = Literal["sequential", "diverging"]

_FIELD_CMAPS: dict[str, str | Colormap] = {
    "sequential": kd_style.CMAP_SEQ,
    "diverging": kd_style.CMAP_DIV,
}


def _field_cmap(field_cmap: FieldCmap) -> str | Colormap:
    return _FIELD_CMAPS[field_cmap]


def _field_limits(
    reference: NDArray[np.floating],
    comparison: NDArray[np.floating] | None,
    field_cmap: FieldCmap,
) -> tuple[float, float]:
    lo, hi = _reference_limits(reference, comparison)
    if field_cmap == "diverging":
        bound = max(abs(lo), abs(hi))
        return -bound, bound
    return lo, hi


def _pcolormesh_panel(
    ax: Axes,
    t_coords: NDArray[np.floating],
    s_coords: NDArray[np.floating],
    data: NDArray[np.floating],
    title: str,
    *,
    time_axis: str,
    spatial_axis: str,
    residual: bool = False,
    limits: tuple[float, float] | None = None,
    cmap: str | Colormap = kd_style.CMAP_SEQ,
) -> QuadMesh:
    display = np.where(np.isfinite(data), data, np.nan)
    cmap, vmin, vmax = _panel_scale(
        display, residual=residual, limits=limits, cmap=cmap
    )
    mesh = ax.pcolormesh(
        t_coords,
        s_coords,
        display,
        shading="auto",
        rasterized=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    _bare_panel(ax)
    ax.set_xlabel(time_axis)
    ax.set_ylabel(spatial_axis)
    ax.set_title(title)
    return mesh


def _heatmap_panel(
    ax: Axes,
    data: NDArray[np.floating],
    title: str,
    *,
    residual: bool = False,
    extent: tuple[float, float, float, float] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    limits: tuple[float, float] | None = None,
    cmap: str | Colormap = kd_style.CMAP_SEQ,
) -> AxesImage:
    display = np.where(np.isfinite(data), data, np.nan)
    cmap, vmin, vmax = _panel_scale(
        display, residual=residual, limits=limits, cmap=cmap
    )
    im = ax.imshow(
        display,
        aspect="auto",
        origin="lower",
        extent=extent,
        rasterized=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    _bare_panel(ax)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    return im


def _bare_panel(ax: Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _scale_colorbar(
    mappables: Sequence[ScalarMappable],
    axes: Sequence[Axes],
    *,
    data: Sequence[NDArray[np.floating]] | None = None,
    label: str | None = None,
) -> Colorbar:
    if data is None:
        data = [
            np.ma.filled(np.ma.asarray(m.get_array(), dtype=np.float64), np.nan)
            for m in mappables
        ]
    vmin, vmax = mappables[0].get_clim()
    pooled = np.concatenate([np.ravel(np.asarray(d, dtype=np.float64)) for d in data])
    colorbar = axes[0].figure.colorbar(
        mappables[0],
        ax=list(axes),
        fraction=_COLORBAR_FRACTION,
        pad=_COLORBAR_PAD,
        extend=_colorbar_extend(pooled, vmin, vmax),
        label=label,
    )
    kd_style.style_colorbar(colorbar)
    return colorbar


def _warning_panel(ax: Axes, reason: str, label: str = "Predicted") -> None:
    wrapped = textwrap.fill(reason, width=_WARNING_WRAP_WIDTH)
    ax.text(
        0.5,
        0.5,
        wrapped,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=_WARNING_FONTSIZE,
        color=kd_style.WARNING_INK,
    )
    ax.set_title(label)
    ax.set_xticks([])
    ax.set_yticks([])
    _bare_panel(ax)


def _panel_scale(
    display: NDArray[np.floating],
    *,
    residual: bool,
    limits: tuple[float, float] | None,
    cmap: str | Colormap = kd_style.CMAP_SEQ,
) -> tuple[str | Colormap, float, float]:
    if residual:
        if limits is not None:
            return kd_style.CMAP_DIV, limits[0], limits[1]
        vmax = _robust_abs_max(display)
        return kd_style.CMAP_DIV, -vmax, vmax
    vmin, vmax = limits if limits is not None else _shared_value_limits(display)
    return cmap, vmin, vmax


def _residual_limits(data: NDArray[np.floating]) -> tuple[float, float]:
    display = np.where(np.isfinite(data), data, np.nan)
    vmax = _robust_abs_max(display)
    return -vmax, vmax


def _shared_value_limits(data: NDArray[np.floating]) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    lo, hi = 0.0, 0.0
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        center = lo if np.isfinite(lo) else 0.0
        pad = abs(center) * 0.5 or 1.0
        lo, hi = center - pad, center + pad
    return lo, hi


def _has_finite_span(data: NDArray[np.floating]) -> bool:
    finite = data[np.isfinite(data)]
    return bool(finite.size) and float(finite.min()) < float(finite.max())


def _reference_limits(
    reference: NDArray[np.floating],
    comparison: NDArray[np.floating] | None = None,
) -> tuple[float, float]:
    if comparison is None or _has_finite_span(reference):
        return _shared_value_limits(reference)
    return _shared_value_limits(
        np.concatenate([np.ravel(reference), np.ravel(comparison)])
    )


def _colorbar_extend(display: NDArray[np.floating], vmin: float, vmax: float) -> str:
    finite = display[np.isfinite(display)]
    if finite.size == 0:
        return "neither"
    below = bool(finite.min() < vmin)
    above = bool(finite.max() > vmax)
    if below and above:
        return "both"
    if below:
        return "min"
    return "max" if above else "neither"


def _range_note(data: NDArray[np.floating], limits: tuple[float, float]) -> str:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return "\n(no finite values)"
    lo, hi = float(finite.min()), float(finite.max())
    if lo >= limits[0] and hi <= limits[1]:
        return ""
    return f"\nclipped, actual [{lo:.3g}, {hi:.3g}]"


def _panel_range_note(
    ax: Axes, data: NDArray[np.floating], limits: tuple[float, float]
) -> None:
    note = _range_note(data, limits).strip()
    if not note:
        return
    ax.annotate(
        note,
        xy=(0.0, 0.0),
        xycoords=("axes fraction", ax.xaxis.label),
        xytext=(0.0, -_NOTE_GAP_POINTS),
        textcoords="offset points",
        ha="left",
        va="top",
        color=kd_style.INK_MUTED,
        fontsize="small",
    )


def _robust_abs_max(data: NDArray[np.floating]) -> float:
    abs_data = np.abs(data[np.isfinite(data)])
    if abs_data.size == 0:
        return 1.0
    vmax = float(np.percentile(abs_data, _RESIDUAL_PERCENTILE))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = float(abs_data.max()) if abs_data.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    return vmax


def _resolve_shape(
    data: NDArray[np.floating],
    field_shape: tuple[int, ...] | None,
    warnings: list[str],
    *,
    infer_grid: bool = True,
) -> tuple[int, ...] | None:
    n = data.size

    if field_shape is not None:
        expected = 1
        for s in field_shape:
            expected *= s
        if expected != n:
            warnings.append(
                f"field_shape {field_shape} (size {expected}) "
                f"does not match data size {n}"
            )
            return None
        if len(field_shape) < 2:
            warnings.append(
                f"field_shape {field_shape} has fewer than 2 dimensions; "
                "spatial panel skipped"
            )
            return None
        return field_shape


    if infer_grid:
        sqrt_n = int(math.isqrt(n))
        if sqrt_n * sqrt_n == n and sqrt_n > 1:




            warnings.append(
                "No field_shape given; spatial panel rendered on a GUESSED "
                f"square grid {(sqrt_n, sqrt_n)} -- pass "
                "field_shape=dataset.get_shape() to confirm the true layout"
            )
            return (sqrt_n, sqrt_n)




        warnings.append(
            f"No field_shape given and residual count {n} is not a perfect "
            "square; spatial panel skipped -- pass "
            "field_shape=dataset.get_shape() for a real heatmap"
        )

    return None


def _diverged_text(
    integration_result: IntegrationResult | None,
    time_axis: str,
) -> str:
    if integration_result is not None and integration_result.diverged_at_t is not None:
        return f"DIVERGED at {time_axis}={integration_result.diverged_at_t:.3g}"
    return "DIVERGED"


def _diverged_tag(
    diverged: bool,
    integration_result: IntegrationResult | None,
    time_axis: str,
) -> str:
    if not diverged:
        return ""
    return f" ({_diverged_text(integration_result, time_axis)})"
