
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from kd.viz._result_data import (
    _fit_target_label,
    _has_reduced_target,
    _sketch_fit_note,
)
from kd.viz.axes import integer_ticks
from kd.viz.plots._dim_utils import _annotate_spatial_slice
from kd.viz.plots._field_panels import (
    _RESIDUAL_SIGN,
    _heatmap_panel,
    _pcolormesh_panel,
    _range_note,
    _reference_limits,
    _resolve_shape,
    _scale_colorbar,
)
from kd.viz.style import (
    CONTEXT,
    DISCOVERED,
    DPI,
    OBSERVED,
    PANEL_H,
    PANEL_W,
    style_context,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable

    from kd.data.schema import PDEDataset
    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_RESIDUAL_FIGSIZE = (3 * PANEL_W, PANEL_H)
_WARNING_FONTSIZE = 12


def plot_pde_residual_field(
    result: ExperimentResult,
    *,
    style: dict[str, Any] | None = None,
    field_shape: tuple[int, ...] | None = None,
    dataset: PDEDataset | None = None,
    infer_grid: bool = True,
) -> tuple[Figure, list[str]]:
    warnings: list[str] = []

    actual = np.array(result.actual.detach().cpu().numpy(), dtype=np.float64)
    predicted = np.array(result.predicted.detach().cpu().numpy(), dtype=np.float64)


    if not np.all(np.isfinite(actual)):
        warnings.append("Actual data contains NaN/Inf values")
    if not np.all(np.isfinite(predicted)):
        warnings.append("Predicted data contains NaN/Inf values")



    lhs_label = "reduced target" if _has_reduced_target(result) else result.lhs_label
    note = _sketch_fit_note(result)
    if note is not None:
        warnings.append(note)


    if dataset is not None and _has_axis_info(dataset):
        with style_context(style):
            fig = _render_axis_aware(
                actual,
                predicted,
                dataset,
                field_shape,
                warnings,
                lhs_label,
                infer_grid=infer_grid,
            )
            if note is not None:
                fig.suptitle(f"Search fit: {_fit_target_label(result)}", fontsize=11)
            _annotate_spatial_slice(fig, dataset, warnings)
        return fig, warnings


    if dataset is None:
        logger.warning(
            "No dataset provided; falling back to 1D layout for PDE residual"
        )
    shape = _resolve_shape(actual, field_shape, warnings, infer_grid=infer_grid)

    with style_context(style):
        fig, axes_arr = plt.subplots(
            1,
            3,
            figsize=_RESIDUAL_FIGSIZE,
            dpi=DPI,
            layout="constrained",
        )
        axes: list[Axes] = list(axes_arr.flat)

        if shape is not None:
            actual_2d = _safe_reshape(actual, shape, warnings, "actual")
            pred_2d = _safe_reshape(predicted, shape, warnings, "predicted")
        else:
            actual_2d = None
            pred_2d = None

        if actual_2d is not None and pred_2d is not None:



            limits = _reference_limits(actual_2d, pred_2d)



            _triptych(
                axes,
                (actual_2d, pred_2d, pred_2d - actual_2d),
                lhs_label,
                limits,
            )
        else:

            _line_fallback(axes[0], actual, f"True ({lhs_label} actual)", OBSERVED)
            _line_fallback(
                axes[1], predicted, f"Predicted ({lhs_label} predicted)", DISCOVERED
            )
            residual_1d = predicted - actual
            _line_fallback(axes[2], residual_1d, f"Residual ({_RESIDUAL_SIGN})")

    if note is not None:
        fig.suptitle(f"Search fit: {_fit_target_label(result)}", fontsize=11)
    return fig, warnings


def _has_axis_info(dataset: PDEDataset) -> bool:
    return (
        dataset.axis_order is not None
        and dataset.axes is not None
        and len(dataset.axis_order) >= 2
    )


def _shape_matches_dataset(
    shape: tuple[int, ...],
    dataset: PDEDataset,
    time_axis: str,
    spatial_axes: list[str],
) -> bool:
    assert dataset.axis_order is not None
    assert dataset.axes is not None


    expected_sizes = []
    for axis_name in dataset.axis_order:
        if axis_name in dataset.axes:
            expected_sizes.append(dataset.axes[axis_name].values.numel())

    expected_shape = tuple(expected_sizes)
    return shape == expected_shape


def _render_axis_aware(
    actual: np.ndarray,
    predicted: np.ndarray,
    dataset: PDEDataset,
    field_shape: tuple[int, ...] | None,
    warnings: list[str],
    lhs_label: str,
    *,
    infer_grid: bool = True,
) -> Figure:
    import contextlib

    assert dataset.axis_order is not None
    time_axis = dataset.lhs_axis
    spatial_axes = dataset.spatial_axes
    n_spatial = len(spatial_axes)





    if field_shape is None:
        with contextlib.suppress(ValueError, AttributeError):
            field_shape = dataset.get_shape()

    shape = _resolve_shape(actual, field_shape, warnings, infer_grid=infer_grid)
    if shape is None:

        fig, axes_arr = plt.subplots(
            1,
            3,
            figsize=_RESIDUAL_FIGSIZE,
            dpi=DPI,
            layout="constrained",
        )
        axes: list[Axes] = list(axes_arr.flat)
        _line_fallback(axes[0], actual, f"True ({lhs_label} actual)", OBSERVED)
        _line_fallback(
            axes[1], predicted, f"Predicted ({lhs_label} predicted)", DISCOVERED
        )
        _line_fallback(axes[2], predicted - actual, f"Residual ({_RESIDUAL_SIGN})")
        return fig

    actual_nd = _safe_reshape(actual, shape, warnings, "actual")
    pred_nd = _safe_reshape(predicted, shape, warnings, "predicted")

    if actual_nd is None or pred_nd is None:
        fig, axes_arr = plt.subplots(
            1,
            3,
            figsize=_RESIDUAL_FIGSIZE,
            dpi=DPI,
            layout="constrained",
        )
        axes = list(axes_arr.flat)
        _line_fallback(axes[0], actual, f"True ({lhs_label} actual)", OBSERVED)
        _line_fallback(
            axes[1], predicted, f"Predicted ({lhs_label} predicted)", DISCOVERED
        )
        _line_fallback(axes[2], predicted - actual, f"Residual ({_RESIDUAL_SIGN})")
        return fig

    residual_nd = pred_nd - actual_nd



    if not _shape_matches_dataset(shape, dataset, time_axis, spatial_axes):
        warnings.append("field_shape does not match dataset axes; using generic render")
        fig, axes_arr = plt.subplots(
            1,
            3,
            figsize=_RESIDUAL_FIGSIZE,
            dpi=DPI,
            layout="constrained",
        )
        axes = list(axes_arr.flat)
        limits = _reference_limits(actual_nd, pred_nd)
        _triptych(axes, (actual_nd, pred_nd, residual_nd), lhs_label, limits)
        return fig

    if n_spatial <= 1 and len(shape) == 2:

        return _render_1d_axis_aware(
            actual_nd,
            pred_nd,
            residual_nd,
            dataset,
            time_axis,
            spatial_axes,
            lhs_label,
        )
    else:

        time_dim = dataset.axis_order.index(time_axis)
        return _render_2d_axis_aware(
            actual_nd,
            pred_nd,
            residual_nd,
            dataset,
            time_axis,
            time_dim,
        )


def _render_1d_axis_aware(
    actual_2d: np.ndarray,
    pred_2d: np.ndarray,
    residual_2d: np.ndarray,
    dataset: PDEDataset,
    time_axis: str,
    spatial_axes: list[str],
    lhs_label: str,
) -> Figure:
    assert dataset.axis_order is not None

    t_coords = dataset.get_coords(time_axis).detach().cpu().numpy()
    s_name = spatial_axes[0]
    s_coords = dataset.get_coords(s_name).detach().cpu().numpy()

    time_dim = dataset.axis_order.index(time_axis)


    if time_dim == 0:
        actual_display = actual_2d.T
        pred_display = pred_2d.T
        residual_display = residual_2d.T
    else:
        actual_display = actual_2d
        pred_display = pred_2d
        residual_display = residual_2d

    fig, axes_arr = plt.subplots(
        1,
        3,
        figsize=_RESIDUAL_FIGSIZE,
        dpi=DPI,
        layout="constrained",
    )
    axes: list[Axes] = list(axes_arr.flat)




    limits = _reference_limits(actual_display, pred_display)

    true_mesh = _pcolormesh_panel(
        axes[0],
        t_coords,
        s_coords,
        actual_display,
        f"True ({lhs_label} actual)" + _range_note(actual_display, limits),
        time_axis=time_axis,
        spatial_axis=s_name,
        limits=limits,
    )
    pred_mesh = _pcolormesh_panel(
        axes[1],
        t_coords,
        s_coords,
        pred_display,
        f"Predicted ({lhs_label} predicted)" + _range_note(pred_display, limits),
        time_axis=time_axis,
        spatial_axis=s_name,
        limits=limits,
    )
    residual_mesh = _pcolormesh_panel(
        axes[2],
        t_coords,
        s_coords,
        residual_display,
        f"Residual ({_RESIDUAL_SIGN})",
        time_axis=time_axis,
        spatial_axis=s_name,
        residual=True,
    )
    _triptych_colorbars(axes, (true_mesh, pred_mesh, residual_mesh))
    return fig


def _render_2d_axis_aware(
    actual_nd: np.ndarray,
    pred_nd: np.ndarray,
    residual_nd: np.ndarray,
    dataset: PDEDataset,
    time_axis: str,
    time_dim: int,
) -> Figure:
    from kd.viz.plots._dim_utils import (
        _imshow_extent_for_spatial_axes,
        _slice_nd_to_2d,
    )




    assert dataset.axis_order is not None
    spatial_axes = [a for a in dataset.axis_order if a != time_axis]
    extent, xlabel, ylabel = _imshow_extent_for_spatial_axes(dataset, spatial_axes)

    n_t = actual_nd.shape[time_dim]



    mid_idx = n_t // 2

    t_coords = dataset.get_coords(time_axis).detach().cpu().numpy()
    t_val = float(t_coords[mid_idx])

    actual_slice = np.take(actual_nd, mid_idx, axis=time_dim)
    pred_slice = np.take(pred_nd, mid_idx, axis=time_dim)
    residual_slice = np.take(residual_nd, mid_idx, axis=time_dim)

    if actual_slice.ndim > 2:
        actual_slice = _slice_nd_to_2d(actual_slice, (0, 1))
        pred_slice = _slice_nd_to_2d(pred_slice, (0, 1))
        residual_slice = _slice_nd_to_2d(residual_slice, (0, 1))

    fig, axes_arr = plt.subplots(
        1,
        3,
        figsize=_RESIDUAL_FIGSIZE,
        dpi=DPI,
        layout="constrained",
    )
    axes: list[Axes] = list(axes_arr.flat)


    limits = _reference_limits(actual_slice, pred_slice)
    true_image = _heatmap_panel(
        axes[0],
        actual_slice,
        f"True ({time_axis}={t_val:.3g})" + _range_note(actual_slice, limits),
        extent=extent,
        xlabel=xlabel,
        ylabel=ylabel,
        limits=limits,
    )
    pred_image = _heatmap_panel(
        axes[1],
        pred_slice,
        f"Predicted ({time_axis}={t_val:.3g})" + _range_note(pred_slice, limits),
        extent=extent,
        xlabel=xlabel,
        ylabel=ylabel,
        limits=limits,
    )
    residual_image = _heatmap_panel(
        axes[2],
        residual_slice,
        f"Residual ({_RESIDUAL_SIGN}, {time_axis}={t_val:.3g})",
        residual=True,
        extent=extent,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    _triptych_colorbars(axes, (true_image, pred_image, residual_image))
    return fig


def _safe_reshape(
    data: np.ndarray,
    shape: tuple[int, ...],
    warnings: list[str],
    label: str,
) -> np.ndarray | None:
    try:
        return data.reshape(shape)
    except ValueError:
        warnings.append(f"Cannot reshape {label} data to {shape}")
        return None


def _triptych(
    axes: list[Axes],
    fields: tuple[np.ndarray, np.ndarray, np.ndarray],
    lhs_label: str,
    limits: tuple[float, float],
) -> None:
    actual, predicted, residual = fields
    mappables = (
        _render_panel(axes[0], actual, f"True ({lhs_label} actual)", limits=limits),
        _render_panel(
            axes[1], predicted, f"Predicted ({lhs_label} predicted)", limits=limits
        ),
        _render_panel(axes[2], residual, f"Residual ({_RESIDUAL_SIGN})", residual=True),
    )
    _triptych_colorbars(axes, mappables)


def _triptych_colorbars(
    axes: Sequence[Axes], mappables: Sequence[ScalarMappable]
) -> None:
    _scale_colorbar(mappables[:2], axes[:2])
    _scale_colorbar(mappables[2:], axes[2:])


def _render_panel(
    ax: Axes,
    data: np.ndarray,
    title: str,
    *,
    residual: bool = False,
    limits: tuple[float, float] | None = None,
) -> ScalarMappable:
    from kd.viz.plots._dim_utils import _slice_nd_to_2d

    render_data = data
    if render_data.ndim > 2:
        render_data = _slice_nd_to_2d(render_data, (0, 1))
    note = "" if residual or limits is None else _range_note(render_data, limits)
    return _heatmap_panel(
        ax, render_data, title + note, residual=residual, limits=limits
    )


def _line_fallback(
    ax: Axes,
    data: np.ndarray,
    title: str,
    color: str = CONTEXT,
) -> None:
    ax.plot(data.ravel(), color=color)
    ax.set_title(title)
    ax.set_xlabel("Index")
    integer_ticks(ax)
