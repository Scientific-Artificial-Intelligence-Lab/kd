
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from kd.viz.style import style_context

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_HISTOGRAM_BINS = 30
_DEFAULT_DPI = 150
_WARNING_FONTSIZE = 12


def plot_residual(
    result: ExperimentResult,
    *,
    style: dict[str, Any] | None = None,
    field_shape: tuple[int, ...] | None = None,
) -> tuple[Figure, list[str]]:
    warnings: list[str] = []
    residuals = result.final_eval.residuals

    with style_context(style):
        fig, axes_arr = plt.subplots(1, 2, figsize=(12, 5), dpi=_DEFAULT_DPI)
        ax_hist: Axes = axes_arr[0]
        ax_spatial: Axes = axes_arr[1]

        if residuals is None:
            warnings.append("No residuals available; skipping residual plot")
            _empty_panel(ax_hist, "No data", "Residual Distribution")
            _empty_panel(ax_spatial, "No data", "Spatial Residual")
            return fig, warnings

        data = np.array(residuals.detach().cpu().numpy(), dtype=np.float64)


        finite_mask = np.isfinite(data)
        n_nonfinite = int(np.sum(~finite_mask))
        if n_nonfinite > 0:
            warnings.append(f"Filtered {n_nonfinite}/{len(data)} non-finite residuals")
        finite_data = data[finite_mask]


        _render_histogram(ax_hist, finite_data, warnings)


        _render_spatial(ax_spatial, data, field_shape, warnings)

    return fig, warnings


def _render_histogram(
    ax: Axes,
    data: np.ndarray,
    warnings: list[str],
) -> None:
    if len(data) == 0:
        warnings.append("All residuals are non-finite; skipping histogram")
        _empty_panel(ax, "No finite data", "Residual Distribution")
        return

    abs_data = np.abs(data)
    xlim = float(np.percentile(abs_data, 99))
    if not np.isfinite(xlim) or xlim == 0.0:
        xlim = float(np.std(data)) or 1.0

    ax.hist(
        data,
        bins=_HISTOGRAM_BINS,
        range=(-xlim, xlim),
        edgecolor="black",
        alpha=0.7,
    )
    ax.set_xlim(-xlim, xlim)

    mean_val = float(np.mean(data))
    std_val = float(np.std(data))
    n_outliers = int(np.sum(abs_data > xlim))
    annotation = f"mean={mean_val:.4g}\nstd={std_val:.4g}"
    if n_outliers > 0:
        annotation += f"\n{n_outliers} outside |r|>{xlim:.3g}"

    ax.text(
        0.95,
        0.95,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
    )

    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")


def _render_spatial(
    ax: Axes,
    data: np.ndarray,
    field_shape: tuple[int, ...] | None,
    warnings: list[str],
) -> None:
    shape = _resolve_shape(data, field_shape, warnings)

    if shape is None:
        _empty_panel(ax, "No spatial data", "Spatial Residual")
        return

    try:
        reshaped = data.reshape(shape)
    except ValueError:
        warnings.append(f"Cannot reshape residuals to {shape}")
        _empty_panel(ax, "No spatial data", "Spatial Residual")
        return


    if reshaped.ndim > 2:
        from kd.viz.plots._dim_utils import _slice_nd_to_2d

        reshaped = _slice_nd_to_2d(reshaped, (0, 1))

    display = np.where(np.isfinite(reshaped), reshaped, np.nan)




    finite_vals = display[np.isfinite(display)]
    if finite_vals.size > 0:
        vmax = float(np.percentile(np.abs(finite_vals), 99))
        if vmax == 0.0:
            vmax = 1.0
        vmin = -vmax
    else:
        vmin, vmax = -1.0, 1.0

    im = ax.imshow(
        display,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Spatial Residual (predicted - actual)")
    ax.set_xlabel("time index")
    ax.set_ylabel("space index")


def _resolve_shape(
    data: np.ndarray,
    field_shape: tuple[int, ...] | None,
    warnings: list[str],
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
            return None
        return field_shape


    sqrt_n = int(math.isqrt(n))
    if sqrt_n * sqrt_n == n and sqrt_n > 1:
        return (sqrt_n, sqrt_n)

    return None


def _empty_panel(ax: Axes, text: str, title: str) -> None:
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=_WARNING_FONTSIZE,
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
