
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from kd.data.schema import DataTopology
from kd.viz.plots._dim_utils import (
    _annotate_spatial_slice,
    _imshow_extent_for_spatial_axes,
    _pick_time_steps,
    _slice_nd_to_2d,
)
from kd.viz.plots._field_panels import (
    _DISCOVERED_LABEL,
    _OBSERVED_LABEL,
    FieldCmap,
    _diverged_text,
    _field_cmap,
    _field_limits,
    _heatmap_panel,
    _panel_range_note,
    _pcolormesh_panel,
    _residual_limits,
    _scale_colorbar,
    _warning_panel,
)
from kd.viz.style import (
    CMAP_SEQ,
    DPI,
    PANEL_H,
    PANEL_W,
    WARNING_INK,
    style_context,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.core.integrator import IntegrationResult
    from kd.data.schema import PDEDataset

logger = logging.getLogger(__name__)

_FIELD_FIGSIZE_1D = (3 * PANEL_W, PANEL_H)


_FIELD_SIGN = "discovered - observed"


_HEADING_ROW_RATIO = 0.01
_N_TIME_SNAPSHOTS = 3


def plot_field_comparison(
    dataset: PDEDataset,
    integration_result: IntegrationResult,
    *,
    style: dict[str, Any] | None = None,
    field_cmap: FieldCmap = "sequential",
) -> tuple[Figure, list[str]]:
    warnings: list[str] = []


    field_name = dataset.lhs_field
    true_field = dataset.get_field(field_name).detach().cpu().numpy().astype(np.float64)


    if dataset.axis_order is None:
        warnings.append("axis_order is None; skipping field comparison")
        with style_context(style):
            fig, ax = plt.subplots(1, 1, figsize=_FIELD_FIGSIZE_1D, dpi=DPI)
            ax.text(
                0.5,
                0.5,
                "axis_order unavailable",
                transform=ax.transAxes,
                fontsize=14,
                color=WARNING_INK,
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
            fig, ax = plt.subplots(1, 1, figsize=_FIELD_FIGSIZE_1D, dpi=DPI)
            ax.text(
                0.5,
                0.5,
                "No spatial dimensions",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        return fig, warnings


    pred_field, diverged, no_pred_reason = _extract_predicted(
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
                no_pred_reason=no_pred_reason,
                field_cmap=field_cmap,
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
                no_pred_reason=no_pred_reason,
                field_cmap=field_cmap,
            )

    _annotate_spatial_slice(fig, dataset, warnings)
    return fig, warnings


def _extract_predicted(
    integration_result: IntegrationResult,
    true_shape: tuple[int, ...],
    warnings: list[str],
) -> tuple[NDArray[np.floating] | None, bool, str | None]:
    if integration_result.predicted_field is None:
        msg = integration_result.warning or "Integration failed"
        warnings.append(msg)
        return None, False, msg

    pred = np.array(
        integration_result.predicted_field.detach().cpu().numpy(),
        dtype=np.float64,
    )

    if pred.shape != true_shape:
        msg = f"Shape mismatch: true {true_shape} vs predicted {pred.shape}"
        warnings.append(msg)
        return None, False, msg

    diverged = not integration_result.success
    if diverged:
        warnings.append(integration_result.warning or "Integration did not succeed")

    return pred, diverged, None


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
    no_pred_reason: str | None = None,
    field_cmap: FieldCmap = "sequential",
) -> Figure:
    fig, axes_arr = plt.subplots(
        1, 3, figsize=_FIELD_FIGSIZE_1D, dpi=DPI, layout="constrained"
    )
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




    limits = _field_limits(true_2d, pred_2d, field_cmap)
    cmap = _field_cmap(field_cmap)

    true_mesh = _pcolormesh_panel(
        axes[0],
        t_coords,
        s_coords,
        true_2d,
        _OBSERVED_LABEL,
        limits=limits,
        cmap=cmap,
        time_axis=time_axis,
        spatial_axis=s_name,
    )
    _panel_range_note(axes[0], true_2d, limits)

    if pred_2d is None:
        assert (
            no_pred_reason is not None
        )
        _scale_colorbar([true_mesh], [axes[0]])
        _warning_panel(axes[1], no_pred_reason, label=_DISCOVERED_LABEL)
        _warning_panel(axes[2], no_pred_reason, label="Residual")
        return fig

    pred_title = _discovered_title(diverged, integration_result, time_axis)
    pred_mesh = _pcolormesh_panel(
        axes[1],
        t_coords,
        s_coords,
        pred_2d,
        pred_title,
        limits=limits,
        cmap=cmap,
        time_axis=time_axis,
        spatial_axis=s_name,
    )
    _panel_range_note(axes[1], pred_2d, limits)
    _scale_colorbar([true_mesh, pred_mesh], axes[:2])
    residual = pred_2d - true_2d
    residual_mesh = _pcolormesh_panel(
        axes[2],
        t_coords,
        s_coords,
        residual,
        f"Residual ({_FIELD_SIGN})",
        residual=True,
        time_axis=time_axis,
        spatial_axis=s_name,
    )
    _scale_colorbar([residual_mesh], [axes[2]])
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
    no_pred_reason: str | None = None,
    field_cmap: FieldCmap = "sequential",
) -> Figure:
    assert (
        dataset.axis_order is not None
    )
    time_dim = dataset.axis_order.index(time_axis)
    n_t = true_field.shape[time_dim]
    time_indices = _pick_time_steps(n_t, _N_TIME_SNAPSHOTS)
    n_snaps = len(time_indices)



    n_rows = 3
    fig = plt.figure(
        figsize=(PANEL_W * n_snaps, PANEL_H * n_rows), dpi=DPI, layout="constrained"
    )
    grid = fig.add_gridspec(
        2 * n_rows, n_snaps, height_ratios=[_HEADING_ROW_RATIO, 1.0] * n_rows
    )
    axes_arr = np.array(
        [
            [fig.add_subplot(grid[2 * row + 1, col]) for col in range(n_snaps)]
            for row in range(n_rows)
        ]
    )

    t_coords = dataset.get_coords(time_axis).detach().cpu().numpy()
    extent, xlabel, ylabel = _imshow_extent_for_spatial_axes(dataset, spatial_axes)




    limits = _field_limits(true_field, pred_field, field_cmap)
    cmap = _field_cmap(field_cmap)



    residual_row_limits = (
        _residual_limits(pred_field - true_field) if pred_field is not None else None
    )

    field_images: list[Any] = []
    residual_images: list[Any] = []
    for col, t_idx in enumerate(time_indices):
        when = f"{time_axis}={float(t_coords[t_idx]):.3g}"
        true_slice = _snapshot(true_field, t_idx, time_dim)
        field_images.append(
            _heatmap_panel(
                axes_arr[0, col],
                true_slice,
                when,
                limits=limits,
                cmap=cmap,
                extent=extent,
                xlabel=xlabel,
                ylabel=ylabel,
            )
        )
        _panel_range_note(axes_arr[0, col], true_slice, limits)

        if pred_field is None:
            assert (
                no_pred_reason is not None
            )
            _warning_panel(axes_arr[1, col], no_pred_reason, label=when)
            _warning_panel(axes_arr[2, col], no_pred_reason, label=when)
            continue

        pred_slice = _snapshot(pred_field, t_idx, time_dim)
        field_images.append(
            _heatmap_panel(
                axes_arr[1, col],
                pred_slice,
                when,
                limits=limits,
                cmap=cmap,
                extent=extent,
                xlabel=xlabel,
                ylabel=ylabel,
            )
        )
        _panel_range_note(axes_arr[1, col], pred_slice, limits)

        residual_slice = pred_slice - true_slice
        assert (
            residual_row_limits is not None
        )
        residual_images.append(
            _heatmap_panel(
                axes_arr[2, col],
                residual_slice,
                when,
                residual=True,
                limits=residual_row_limits,
                extent=extent,
                xlabel=xlabel,
                ylabel=ylabel,
            )
        )
        _panel_range_note(axes_arr[2, col], residual_slice, residual_row_limits)

    headings = (
        _OBSERVED_LABEL,
        _discovered_title(diverged, integration_result, time_axis),
        f"Residual ({_FIELD_SIGN})",
    )
    for row, heading in enumerate(headings):
        _row_heading(fig.add_subplot(grid[2 * row,:]), heading)
    field_rows = 1 if pred_field is None else 2
    _scale_colorbar(field_images, list(axes_arr[:field_rows].flat))
    if residual_images:
        _scale_colorbar(residual_images, list(axes_arr[2].flat))
    return fig


def _snapshot(
    field: NDArray[np.floating], t_idx: int, time_dim: int
) -> NDArray[np.floating]:
    snapshot: NDArray[np.floating] = np.take(field, t_idx, axis=time_dim)
    if snapshot.ndim > 2:
        snapshot = _slice_nd_to_2d(snapshot, (0, 1))
    return snapshot


def _discovered_title(
    diverged: bool, integration_result: IntegrationResult, time_axis: str
) -> str:
    if not diverged:
        return _DISCOVERED_LABEL
    return f"{_DISCOVERED_LABEL} ({_diverged_text(integration_result, time_axis)})"


def _row_heading(ax: Axes, heading: str) -> None:
    ax.axis("off")
    ax.set_title(heading, loc="left")


_RAW_FIELD_FIGSIZE = (7.0, 3.2)


def plot_field(
    dataset: PDEDataset,
    ax: Axes | None = None,
    *,
    field: str | None = None,
    style: dict[str, Any] | None = None,
) -> None:
    name = dataset.lhs_field if field is None else field
    values = dataset.get_field(name).detach().cpu().numpy().astype(np.float64)
    spatial = dataset.spatial_axes
    if len(spatial) != 1:
        raise ValueError(
            f"plot_field draws one spatial axis, dataset {dataset.name!r} has "
            f"{len(spatial)} spatial axes {spatial}"
        )
    time_axis, spatial_axis = dataset.lhs_axis, spatial[0]
    t = dataset.get_coords(time_axis).detach().cpu().numpy()
    s = dataset.get_coords(spatial_axis).detach().cpu().numpy()
    title = f"{name}({spatial_axis}, {time_axis})"
    with style_context(style):
        if ax is None:
            _, ax = plt.subplots(figsize=_RAW_FIELD_FIGSIZE, dpi=DPI)
        if dataset.topology is DataTopology.SCATTERED:
            points = ax.scatter(
                t, s, c=values, s=8, cmap=CMAP_SEQ, linewidths=0, rasterized=True
            )
            _scale_colorbar([points], [ax])
            ax.grid(False)
            ax.set_xlabel(time_axis)
            ax.set_ylabel(spatial_axis)
            ax.set_title(title)
            return

        order = dataset.axis_order
        assert order is not None
        grid = np.moveaxis(values, order.index(time_axis), -1)
        mesh = _pcolormesh_panel(
            ax, t, s, grid, title, time_axis=time_axis, spatial_axis=spatial_axis
        )
        _scale_colorbar([mesh], [ax])
