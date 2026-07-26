
from __future__ import annotations

import logging
import textwrap
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from kd.viz.plots._dim_utils import (
    _imshow_extent_for_spatial_axes,
    _pick_time_steps,
    _slice_nd_to_2d,
)
from kd.viz.plots._field_panels import _RESIDUAL_SIGN, _robust_abs_max
from kd.viz.style import style_context

if TYPE_CHECKING:
    from kd.core.integrator import IntegrationResult
    from kd.data.schema import PDEDataset
    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_DEFAULT_DPI = 150
_WARNING_FONTSIZE = 11
_WARNING_WRAP_WIDTH = 60
_N_TIME_SNAPSHOTS = 3
_COLORBAR_LABEL = f"Error ({_RESIDUAL_SIGN})"


def plot_error_heatmap(
    result: ExperimentResult,
    dataset: PDEDataset,
    integration_result: IntegrationResult,
    *,
    style: dict[str, Any] | None = None,
) -> tuple[Figure, list[str]]:
    warnings: list[str] = []


    if integration_result.predicted_field is None:
        msg = integration_result.warning or "Integration failed"
        warnings.append(msg)
        with style_context(style):
            fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=_DEFAULT_DPI)
            wrapped = textwrap.fill(
                f"Cannot compute error: {msg}", width=_WARNING_WRAP_WIDTH
            )
            ax.text(
                0.5,
                0.5,
                wrapped,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=_WARNING_FONTSIZE,
                color="red",
            )
            ax.set_title("Error Heatmap")
            ax.set_xticks([])
            ax.set_yticks([])
        return fig, warnings


    diverged = not integration_result.success
    if diverged:
        warnings.append(integration_result.warning or "Integration did not succeed")


    field_name = dataset.lhs_field
    true_field = np.array(
        dataset.get_field(field_name).detach().cpu().numpy(), dtype=np.float64
    )
    pred_field = np.array(
        integration_result.predicted_field.detach().cpu().numpy(), dtype=np.float64
    )

    if true_field.shape != pred_field.shape:
        warnings.append(
            f"Shape mismatch: true {true_field.shape} vs predicted {pred_field.shape}"
        )
        with style_context(style):
            fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=_DEFAULT_DPI)
            ax.text(
                0.5,
                0.5,
                "Shape mismatch",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=_WARNING_FONTSIZE,
                color="red",
            )
            ax.set_title("Error Heatmap")
        return fig, warnings



    error = pred_field - true_field
    error = np.where(np.isfinite(error), error, np.nan)


    if dataset.axis_order is None:
        warnings.append("axis_order is None; skipping error heatmap")
        with style_context(style):
            fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=_DEFAULT_DPI)
            ax.text(
                0.5,
                0.5,
                "No axis_order available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=_WARNING_FONTSIZE,
                color="red",
            )
            ax.set_title("Error Heatmap")
            ax.set_xticks([])
            ax.set_yticks([])
        return fig, warnings

    time_axis = dataset.lhs_axis
    spatial_axes = dataset.spatial_axes
    n_spatial = len(spatial_axes)


    if n_spatial == 0:
        warnings.append("No spatial dimensions; skipping error heatmap")
        with style_context(style):
            fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=_DEFAULT_DPI)
            ax.text(
                0.5,
                0.5,
                "No spatial dimensions",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=_WARNING_FONTSIZE,
                color="red",
            )
            ax.set_title("Error Heatmap")
            ax.set_xticks([])
            ax.set_yticks([])
        return fig, warnings

    time_dim = dataset.axis_order.index(time_axis)

    div_tag = _diverged_tag(diverged, integration_result, time_axis)

    with style_context(style):
        if n_spatial <= 1:
            fig = _render_1d_error(
                error,
                dataset,
                time_axis,
                spatial_axes,
                time_dim,
                div_tag=div_tag,
            )
        else:
            fig = _render_2d_error(
                error,
                dataset,
                time_axis,
                spatial_axes,
                time_dim,
                div_tag=div_tag,
            )

    return fig, warnings


def _diverged_tag(
    diverged: bool,
    integration_result: IntegrationResult,
    time_axis: str,
) -> str:
    if not diverged:
        return ""
    if integration_result.diverged_at_t is not None:
        return f" (DIVERGED at {time_axis}={integration_result.diverged_at_t:.3g})"
    return " (DIVERGED)"


def _render_1d_error(
    error: np.ndarray,
    dataset: PDEDataset,
    time_axis: str,
    spatial_axes: list[str],
    time_dim: int,
    *,
    div_tag: str = "",
) -> Figure:
    t_coords = dataset.get_coords(time_axis).detach().cpu().numpy()
    s_name = spatial_axes[0]
    s_coords = dataset.get_coords(s_name).detach().cpu().numpy()


    error_2d = error.T if time_dim == 0 else error

    extent = (
        float(t_coords[0]),
        float(t_coords[-1]),
        float(s_coords[0]),
        float(s_coords[-1]),
    )


    vmax = _robust_abs_max(error_2d)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=_DEFAULT_DPI)
    im = ax.imshow(
        error_2d,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        rasterized=True,
    )
    fig.colorbar(im, ax=ax, label=_COLORBAR_LABEL)
    ax.set_xlabel(time_axis)
    ax.set_ylabel(s_name)
    ax.set_title(f"Error Heatmap{div_tag}")

    return fig


def _render_2d_error(
    error: np.ndarray,
    dataset: PDEDataset,
    time_axis: str,
    spatial_axes: list[str],
    time_dim: int,
    *,
    div_tag: str = "",
) -> Figure:
    n_t = error.shape[time_dim]
    time_indices = _pick_time_steps(n_t, _N_TIME_SNAPSHOTS)
    n_cols = len(time_indices)

    t_coords = dataset.get_coords(time_axis).detach().cpu().numpy()
    extent, xlabel, ylabel = _imshow_extent_for_spatial_axes(dataset, spatial_axes)

    error_slices = []
    for t_idx in time_indices:
        error_slice = np.take(error, t_idx, axis=time_dim)
        if error_slice.ndim > 2:
            error_slice = _slice_nd_to_2d(error_slice, (0, 1))
        error_slices.append(error_slice)

    vmax = _robust_abs_max(np.stack(error_slices, axis=0))

    fig, axes_arr = plt.subplots(
        1,
        n_cols,
        figsize=(5 * n_cols, 4),
        dpi=_DEFAULT_DPI,
        constrained_layout=True,
        squeeze=False,
    )
    mappable: Any | None = None
    for col, (t_idx, error_slice) in enumerate(
        zip(time_indices, error_slices, strict=True)
    ):
        ax = axes_arr[0, col]
        t_val = float(t_coords[t_idx])
        mappable = ax.imshow(
            error_slice,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            rasterized=True,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Error Heatmap{div_tag} ({time_axis}={t_val:.3g})")

    if mappable is not None:
        fig.colorbar(
            mappable,
            ax=list(axes_arr.flat),
            label=_COLORBAR_LABEL,
        )

    return fig
