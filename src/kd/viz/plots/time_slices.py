
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
from kd.viz.style import style_context

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.core.integrator import IntegrationResult
    from kd.data.schema import PDEDataset
    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_DEFAULT_DPI = 150
_WARNING_FONTSIZE = 9
_WARNING_WRAP_WIDTH = 38


def plot_time_slices(
    result: ExperimentResult,
    dataset: PDEDataset,
    integration_result: IntegrationResult,
    *,
    style: dict[str, Any] | None = None,
    n_slices: int = 3,
) -> tuple[Figure, list[str]]:
    warnings: list[str] = []


    field_name = dataset.lhs_field
    true_field = np.array(
        dataset.get_field(field_name).detach().cpu().numpy(), dtype=np.float64
    )


    if dataset.axis_order is None:
        warnings.append("axis_order is None; skipping time slices")
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
            ax.set_title("Time Slices")
            ax.set_xticks([])
            ax.set_yticks([])
        return fig, warnings

    time_axis = dataset.lhs_axis
    spatial_axes = dataset.spatial_axes
    n_spatial = len(spatial_axes)


    if n_spatial == 0:
        warnings.append("No spatial dimensions; skipping time slices")
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
            ax.set_title("Time Slices")
            ax.set_xticks([])
            ax.set_yticks([])
        return fig, warnings

    time_dim = dataset.axis_order.index(time_axis)
    n_t = true_field.shape[time_dim]


    time_indices = _pick_time_steps(n_t, n_slices)
    n_cols = len(time_indices)


    t_coords = dataset.get_coords(time_axis).detach().cpu().numpy()


    pred_field, diverged = _extract_pred(integration_result, true_field.shape, warnings)

    with style_context(style):
        if n_spatial <= 1:
            fig = _render_1d_slices(
                true_field,
                pred_field,
                dataset,
                time_axis,
                spatial_axes,
                time_dim,
                time_indices,
                t_coords,
                n_cols,
                warnings,
                diverged=diverged,
                integration_result=integration_result,
            )
        else:
            fig = _render_2d_slices(
                true_field,
                pred_field,
                dataset,
                time_axis,
                spatial_axes,
                time_dim,
                time_indices,
                t_coords,
                n_cols,
                warnings,
                diverged=diverged,
                integration_result=integration_result,
            )

    return fig, warnings


def _extract_pred(
    integration_result: IntegrationResult,
    true_shape: tuple[int, ...],
    warnings: list[str],
) -> tuple[np.ndarray | None, bool]:
    if integration_result.predicted_field is None:
        msg = integration_result.warning or "Integration failed"
        warnings.append(msg)
        return None, False

    pred = np.array(
        integration_result.predicted_field.detach().cpu().numpy(), dtype=np.float64
    )

    if pred.shape != true_shape:
        warnings.append(f"Shape mismatch: true {true_shape} vs predicted {pred.shape}")
        return None, False

    diverged = not integration_result.success
    if diverged:
        warnings.append(integration_result.warning or "Integration did not succeed")

    return pred, diverged


def _render_1d_slices(
    true_field: np.ndarray,
    pred_field: np.ndarray | None,
    dataset: PDEDataset,
    time_axis: str,
    spatial_axes: list[str],
    time_dim: int,
    time_indices: list[int],
    t_coords: np.ndarray,
    n_cols: int,
    warnings: list[str],
    *,
    diverged: bool = False,
    integration_result: IntegrationResult | None = None,
) -> Figure:
    fig, axes_arr = plt.subplots(
        1,
        n_cols,
        figsize=(5 * n_cols, 4),
        dpi=_DEFAULT_DPI,
        squeeze=False,
    )


    s_name = spatial_axes[0]
    s_coords = dataset.get_coords(s_name).detach().cpu().numpy()

    div_tag = _diverged_tag(diverged, integration_result, time_axis)

    for col, t_idx in enumerate(time_indices):
        ax: Axes = axes_arr[0, col]
        t_val = float(t_coords[t_idx])


        true_slice = np.take(true_field, t_idx, axis=time_dim)
        true_display = np.where(np.isfinite(true_slice), true_slice, np.nan)
        ax.plot(s_coords, true_display, "b-", label="True", linewidth=1.5)

        if pred_field is not None:
            pred_slice = np.take(pred_field, t_idx, axis=time_dim)
            pred_display = np.where(np.isfinite(pred_slice), pred_slice, np.nan)
            pred_label = f"Predicted{div_tag}"
            ax.plot(s_coords, pred_display, "r--", label=pred_label, linewidth=1.5)
        else:
            _add_warning_text(ax, "No prediction")

        title = (
            f"{time_axis} = {t_val:.3g}{div_tag}"
            if div_tag
            else f"{time_axis} = {t_val:.3g}"
        )
        ax.set_title(title)
        ax.set_xlabel(s_name)
        if col == 0:
            ax.set_ylabel(dataset.lhs_field)
            ax.legend()

    fig.tight_layout()
    return fig


def _render_2d_slices(
    true_field: np.ndarray,
    pred_field: np.ndarray | None,
    dataset: PDEDataset,
    time_axis: str,
    spatial_axes: list[str],
    time_dim: int,
    time_indices: list[int],
    t_coords: np.ndarray,
    n_cols: int,
    warnings: list[str],
    *,
    diverged: bool = False,
    integration_result: IntegrationResult | None = None,
) -> Figure:
    n_rows = 2
    fig, axes_arr = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        dpi=_DEFAULT_DPI,
        squeeze=False,
    )

    div_tag = _diverged_tag(diverged, integration_result, time_axis)
    extent, xlabel, ylabel = _imshow_extent_for_spatial_axes(dataset, spatial_axes)

    for col, t_idx in enumerate(time_indices):
        t_val = float(t_coords[t_idx])

        true_slice = np.take(true_field, t_idx, axis=time_dim)
        if true_slice.ndim > 2:
            true_slice = _slice_nd_to_2d(true_slice, (0, 1))
        true_display = np.where(np.isfinite(true_slice), true_slice, np.nan)
        axes_arr[0, col].imshow(
            true_display,
            aspect="auto",
            origin="lower",
            extent=extent,
            rasterized=True,
        )
        axes_arr[0, col].set_title(f"True ({time_axis}={t_val:.3g})")
        axes_arr[0, col].set_xlabel(xlabel)
        axes_arr[0, col].set_ylabel(ylabel)

        if pred_field is not None:
            pred_slice = np.take(pred_field, t_idx, axis=time_dim)
            if pred_slice.ndim > 2:
                pred_slice = _slice_nd_to_2d(pred_slice, (0, 1))
            pred_display = np.where(np.isfinite(pred_slice), pred_slice, np.nan)
            axes_arr[1, col].imshow(
                pred_display,
                aspect="auto",
                origin="lower",
                extent=extent,
                rasterized=True,
            )
            axes_arr[1, col].set_title(
                f"Predicted{div_tag} ({time_axis}={t_val:.3g})"
            )
            axes_arr[1, col].set_xlabel(xlabel)
            axes_arr[1, col].set_ylabel(ylabel)
        else:
            _warning_panel(axes_arr[1, col])

    fig.tight_layout()
    return fig


def _diverged_tag(
    diverged: bool,
    integration_result: IntegrationResult | None,
    time_axis: str,
) -> str:
    if not diverged:
        return ""
    if integration_result is not None and integration_result.diverged_at_t is not None:
        return f" (DIVERGED at {time_axis}={integration_result.diverged_at_t:.3g})"
    return " (DIVERGED)"


def _add_warning_text(ax: Axes, msg: str) -> None:
    ax.text(
        0.5,
        0.95,
        msg,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=_WARNING_FONTSIZE,
        color="red",
    )


def _warning_panel(ax: Axes, msg: str = "No prediction available") -> None:
    wrapped = textwrap.fill(msg, width=_WARNING_WRAP_WIDTH)
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
    ax.set_xticks([])
    ax.set_yticks([])
