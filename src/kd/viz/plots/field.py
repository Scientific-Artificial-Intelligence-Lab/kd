
from __future__ import annotations

import logging
import textwrap
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from kd.viz.plots._dim_utils import (
    _imshow_extent_for_spatial_axes,
    _pick_time_steps,
    _slice_nd_to_2d,
)
from kd.viz.plots._field_panels import (
    _RESIDUAL_SIGN,
    _heatmap_panel,
    _pcolormesh_panel,
    _range_note,
    _reference_limits,
)
from kd.viz.style import style_context

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.core.integrator import IntegrationResult
    from kd.data.schema import PDEDataset
    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_FIELD_FIGSIZE_1D = (15, 4)
_N_TIME_SNAPSHOTS = 3
_DEFAULT_DPI = 150
_WARNING_FONTSIZE = 9
_WARNING_WRAP_WIDTH = 38


def plot_field_comparison(
    result: ExperimentResult,
    dataset: PDEDataset,
    integration_result: IntegrationResult,
    *,
    style: dict[str, Any] | None = None,
) -> tuple[Figure, list[str]]:
    warnings: list[str] = []


    field_name = dataset.lhs_field
    true_field = dataset.get_field(field_name).detach().cpu().numpy().astype(np.float64)


    if dataset.axis_order is None:
        warnings.append("axis_order is None; skipping field comparison")
        with style_context(style):
            fig, ax = plt.subplots(1, 1, figsize=_FIELD_FIGSIZE_1D, dpi=_DEFAULT_DPI)
            ax.text(
                0.5,
                0.5,
                "axis_order unavailable",
                transform=ax.transAxes,
                fontsize=14,
                color="red",
                ha="center",
                va="center",
            )
            ax.set_xticks([])
            ax.set_yticks([])
        return fig, warnings


    time_axis = dataset.lhs_axis
    spatial_axes = dataset.spatial_axes
    n_spatial = len(spatial_axes)


    if n_spatial == 0:
        warnings.append("No spatial dimensions; skipping field comparison")
        with style_context(style):
            fig, ax = plt.subplots(1, 1, figsize=_FIELD_FIGSIZE_1D, dpi=_DEFAULT_DPI)
            ax.text(
                0.5,
                0.5,
                "No spatial dimensions",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        return fig, warnings


    pred_field, diverged = _extract_predicted(
        integration_result, true_field.shape, warnings
    )

    with style_context(style):
        if n_spatial <= 1:
            fig = _render_1d_spatial(
                true_field,
                pred_field,
                dataset,
                time_axis,
                spatial_axes,
                integration_result,
                warnings,
                diverged=diverged,
            )
        else:
            fig = _render_2d_spatial(
                true_field,
                pred_field,
                dataset,
                time_axis,
                spatial_axes,
                integration_result,
                warnings,
                diverged=diverged,
            )

    return fig, warnings


def _extract_predicted(
    integration_result: IntegrationResult,
    true_shape: tuple[int, ...],
    warnings: list[str],
) -> tuple[NDArray[np.floating] | None, bool]:
    if integration_result.predicted_field is None:
        msg = integration_result.warning or "Integration failed"
        warnings.append(msg)
        return None, False

    pred = np.array(
        integration_result.predicted_field.detach().cpu().numpy(),
        dtype=np.float64,
    )

    if pred.shape != true_shape:
        warnings.append(f"Shape mismatch: true {true_shape} vs predicted {pred.shape}")
        return None, False

    diverged = not integration_result.success
    if diverged:
        warnings.append(integration_result.warning or "Integration did not succeed")

    return pred, diverged


def _render_1d_spatial(
    true_field: NDArray[np.floating],
    pred_field: NDArray[np.floating] | None,
    dataset: PDEDataset,
    time_axis: str,
    spatial_axes: list[str],
    integration_result: IntegrationResult,
    warnings: list[str],
    *,
    diverged: bool = False,
) -> Figure:
    fig, axes_arr = plt.subplots(1, 3, figsize=_FIELD_FIGSIZE_1D, dpi=_DEFAULT_DPI)
    axes: list[Axes] = list(axes_arr.flat)


    t_coords = dataset.get_coords(time_axis).detach().cpu().numpy()

    assert (
        dataset.axis_order is not None
    )
    time_dim = dataset.axis_order.index(time_axis)


    s_name = spatial_axes[0]
    s_coords = dataset.get_coords(s_name).detach().cpu().numpy()


    if time_dim == 0:
        true_2d = true_field.T
        pred_2d = pred_field.T if pred_field is not None else None
    else:
        true_2d = true_field
        pred_2d = pred_field




    limits = _reference_limits(true_2d, pred_2d)

    _pcolormesh_panel(
        axes[0],
        t_coords,
        s_coords,
        true_2d,
        "True" + _range_note(true_2d, limits),
        time_axis=time_axis,
        spatial_axis=s_name,
        limits=limits,
    )

    if pred_2d is not None:
        pred_title = _predicted_title(
            diverged,
            integration_result,
            time_axis=time_axis,
        ) + _range_note(pred_2d, limits)
        _pcolormesh_panel(
            axes[1],
            t_coords,
            s_coords,
            pred_2d,
            pred_title,
            time_axis=time_axis,
            spatial_axis=s_name,
            limits=limits,
        )
        residual = pred_2d - true_2d
        _pcolormesh_panel(
            axes[2],
            t_coords,
            s_coords,
            residual,
            f"Residual ({_RESIDUAL_SIGN})",
            time_axis=time_axis,
            spatial_axis=s_name,
            residual=True,
        )
    else:
        _warning_panel(axes[1], integration_result)
        _warning_panel(axes[2], integration_result, label="Residual")

    fig.tight_layout()
    return fig


def _render_2d_spatial(
    true_field: NDArray[np.floating],
    pred_field: NDArray[np.floating] | None,
    dataset: PDEDataset,
    time_axis: str,
    spatial_axes: list[str],
    integration_result: IntegrationResult,
    warnings: list[str],
    *,
    diverged: bool = False,
) -> Figure:
    assert (
        dataset.axis_order is not None
    )
    time_dim = dataset.axis_order.index(time_axis)
    n_t = true_field.shape[time_dim]
    time_indices = _pick_time_steps(n_t, _N_TIME_SNAPSHOTS)
    n_snaps = len(time_indices)


    n_rows = 3
    fig, axes_arr = plt.subplots(
        n_rows,
        n_snaps,
        figsize=(5 * n_snaps, 4 * n_rows),
        dpi=_DEFAULT_DPI,
        squeeze=False,
    )

    t_coords = dataset.get_coords(time_axis).detach().cpu().numpy()
    extent, xlabel, ylabel = _imshow_extent_for_spatial_axes(dataset, spatial_axes)




    limits = _reference_limits(true_field, pred_field)

    for col, t_idx in enumerate(time_indices):

        true_slice = np.take(true_field, t_idx, axis=time_dim)
        t_val = float(t_coords[t_idx])


        if true_slice.ndim > 2:
            true_slice = _slice_nd_to_2d(true_slice, (0, 1))

        _heatmap_panel(
            axes_arr[0, col],
            true_slice,
            f"True ({time_axis}={t_val:.3g})" + _range_note(true_slice, limits),
            extent=extent,
            xlabel=xlabel,
            ylabel=ylabel,
            limits=limits,
        )

        if pred_field is not None:
            pred_slice = np.take(pred_field, t_idx, axis=time_dim)
            if pred_slice.ndim > 2:
                pred_slice = _slice_nd_to_2d(pred_slice, (0, 1))

            pred_title = _predicted_title(
                diverged,
                integration_result,
                time_axis=time_axis,
                t_val=t_val,
            ) + _range_note(pred_slice, limits)
            _heatmap_panel(
                axes_arr[1, col],
                pred_slice,
                pred_title,
                extent=extent,
                xlabel=xlabel,
                ylabel=ylabel,
                limits=limits,
            )

            residual_slice = pred_slice - true_slice
            _heatmap_panel(
                axes_arr[2, col],
                residual_slice,
                f"Residual ({_RESIDUAL_SIGN}, {time_axis}={t_val:.3g})",
                residual=True,
                extent=extent,
                xlabel=xlabel,
                ylabel=ylabel,
            )
        else:
            _warning_panel(axes_arr[1, col], integration_result)
            _warning_panel(axes_arr[2, col], integration_result, label="Residual")

    fig.tight_layout()
    return fig


def _predicted_title(
    diverged: bool,
    integration_result: IntegrationResult,
    *,
    time_axis: str = "t",
    t_val: float | None = None,
) -> str:
    if not diverged:
        if t_val is not None:
            return f"Predicted ({time_axis}={t_val:.3g})"
        return "Predicted"

    div_t = integration_result.diverged_at_t
    tag = f"DIVERGED at {time_axis}={div_t:.3g}" if div_t is not None else "DIVERGED"

    if t_val is not None:
        return f"Predicted ({tag}, {time_axis}={t_val:.3g})"
    return f"Predicted ({tag})"


def _warning_panel(
    ax: Axes,
    integration_result: IntegrationResult,
    label: str = "Predicted",
) -> None:
    msg = integration_result.warning or "Integration failed"
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
    ax.set_title(label)
    ax.set_xticks([])
    ax.set_yticks([])
