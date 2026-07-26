
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

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

    from kd.data.schema import PDEDataset
    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_RESIDUAL_FIGSIZE = (15, 4)
_DEFAULT_DPI = 150
_WARNING_FONTSIZE = 12


def plot_pde_residual_field(
    result: ExperimentResult,
    *,
    style: dict[str, Any] | None = None,
    field_shape: tuple[int, ...] | None = None,
    dataset: PDEDataset | None = None,
) -> tuple[Figure, list[str]]:
    warnings: list[str] = []

    actual = np.array(result.actual.detach().cpu().numpy(), dtype=np.float64)
    predicted = np.array(result.predicted.detach().cpu().numpy(), dtype=np.float64)


    if not np.all(np.isfinite(actual)):
        warnings.append("Actual data contains NaN/Inf values")
    if not np.all(np.isfinite(predicted)):
        warnings.append("Predicted data contains NaN/Inf values")






    lhs_label = result.lhs_label


    if dataset is not None and _has_axis_info(dataset):
        with style_context(style):
            fig = _render_axis_aware(
                actual, predicted, dataset, field_shape, warnings, lhs_label
            )
        return fig, warnings


    if dataset is None:
        logger.warning(
            "No dataset provided; falling back to 1D layout for PDE residual"
        )
    shape = _resolve_shape(actual, field_shape, warnings)

    with style_context(style):
        fig, axes_arr = plt.subplots(
            1,
            3,
            figsize=_RESIDUAL_FIGSIZE,
            dpi=_DEFAULT_DPI,
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
            _render_panel(
                axes[0], actual_2d, f"True ({lhs_label} actual)", limits=limits
            )
            _render_panel(
                axes[1],
                pred_2d,
                f"Predicted ({lhs_label} predicted)",
                limits=limits,
            )



            residual = pred_2d - actual_2d
            _render_panel(
                axes[2], residual, f"Residual ({_RESIDUAL_SIGN})", residual=True
            )
        else:

            _line_fallback(axes[0], actual, f"True ({lhs_label} actual)")
            _line_fallback(axes[1], predicted, f"Predicted ({lhs_label} predicted)")
            residual_1d = predicted - actual
            _line_fallback(axes[2], residual_1d, f"Residual ({_RESIDUAL_SIGN})")

        fig.tight_layout()

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
) -> Figure:
    import contextlib

    assert dataset.axis_order is not None
    time_axis = dataset.lhs_axis
    spatial_axes = dataset.spatial_axes
    n_spatial = len(spatial_axes)


    if field_shape is None:
        with contextlib.suppress(ValueError, AttributeError):
            field_shape = dataset.get_shape()

    shape = _resolve_shape(actual, field_shape, warnings)
    if shape is None:

        fig, axes_arr = plt.subplots(
            1,
            3,
            figsize=_RESIDUAL_FIGSIZE,
            dpi=_DEFAULT_DPI,
        )
        axes: list[Axes] = list(axes_arr.flat)
        _line_fallback(axes[0], actual, f"True ({lhs_label} actual)")
        _line_fallback(axes[1], predicted, f"Predicted ({lhs_label} predicted)")
        _line_fallback(axes[2], predicted - actual, f"Residual ({_RESIDUAL_SIGN})")
        fig.tight_layout()
        return fig

    actual_nd = _safe_reshape(actual, shape, warnings, "actual")
    pred_nd = _safe_reshape(predicted, shape, warnings, "predicted")

    if actual_nd is None or pred_nd is None:
        fig, axes_arr = plt.subplots(
            1,
            3,
            figsize=_RESIDUAL_FIGSIZE,
            dpi=_DEFAULT_DPI,
        )
        axes = list(axes_arr.flat)
        _line_fallback(axes[0], actual, f"True ({lhs_label} actual)")
        _line_fallback(axes[1], predicted, f"Predicted ({lhs_label} predicted)")
        _line_fallback(axes[2], predicted - actual, f"Residual ({_RESIDUAL_SIGN})")
        fig.tight_layout()
        return fig

    residual_nd = pred_nd - actual_nd



    if not _shape_matches_dataset(shape, dataset, time_axis, spatial_axes):
        warnings.append("field_shape does not match dataset axes; using generic render")
        fig, axes_arr = plt.subplots(
            1,
            3,
            figsize=_RESIDUAL_FIGSIZE,
            dpi=_DEFAULT_DPI,
        )
        axes = list(axes_arr.flat)
        limits = _reference_limits(actual_nd, pred_nd)
        _render_panel(axes[0], actual_nd, f"True ({lhs_label} actual)", limits=limits)
        _render_panel(
            axes[1], pred_nd, f"Predicted ({lhs_label} predicted)", limits=limits
        )
        _render_panel(
            axes[2], residual_nd, f"Residual ({_RESIDUAL_SIGN})", residual=True
        )
        fig.tight_layout()
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
        dpi=_DEFAULT_DPI,
    )
    axes: list[Axes] = list(axes_arr.flat)




    limits = _reference_limits(actual_display, pred_display)

    _pcolormesh_panel(
        axes[0],
        t_coords,
        s_coords,
        actual_display,
        f"True ({lhs_label} actual)" + _range_note(actual_display, limits),
        time_axis=time_axis,
        spatial_axis=s_name,
        limits=limits,
    )
    _pcolormesh_panel(
        axes[1],
        t_coords,
        s_coords,
        pred_display,
        f"Predicted ({lhs_label} predicted)" + _range_note(pred_display, limits),
        time_axis=time_axis,
        spatial_axis=s_name,
        limits=limits,
    )
    _pcolormesh_panel(
        axes[2],
        t_coords,
        s_coords,
        residual_display,
        f"Residual ({_RESIDUAL_SIGN})",
        time_axis=time_axis,
        spatial_axis=s_name,
        residual=True,
    )

    fig.tight_layout()
    return fig


def _render_2d_axis_aware(
    actual_nd: np.ndarray,
    pred_nd: np.ndarray,
    residual_nd: np.ndarray,
    dataset: PDEDataset,
    time_axis: str,
    time_dim: int,
) -> Figure:
    from kd.viz.plots._dim_utils import _slice_nd_to_2d

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
        dpi=_DEFAULT_DPI,
    )
    axes: list[Axes] = list(axes_arr.flat)


    limits = _reference_limits(actual_slice, pred_slice)
    _heatmap_panel(
        axes[0],
        actual_slice,
        f"True ({time_axis}={t_val:.3g})" + _range_note(actual_slice, limits),
        limits=limits,
    )
    _heatmap_panel(
        axes[1],
        pred_slice,
        f"Predicted ({time_axis}={t_val:.3g})" + _range_note(pred_slice, limits),
        limits=limits,
    )
    _heatmap_panel(
        axes[2],
        residual_slice,
        f"Residual ({_RESIDUAL_SIGN}, {time_axis}={t_val:.3g})",
        residual=True,
    )

    fig.tight_layout()
    return fig


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

    warnings.append(f"No field_shape provided and cannot infer 2D shape from size {n}")
    return None


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


def _render_panel(
    ax: Axes,
    data: np.ndarray,
    title: str,
    *,
    residual: bool = False,
    limits: tuple[float, float] | None = None,
) -> None:
    from kd.viz.plots._dim_utils import _slice_nd_to_2d

    render_data = data
    if render_data.ndim > 2:
        render_data = _slice_nd_to_2d(render_data, (0, 1))
    note = "" if residual or limits is None else _range_note(render_data, limits)
    _heatmap_panel(ax, render_data, title + note, residual=residual, limits=limits)


def _line_fallback(
    ax: Axes,
    data: np.ndarray,
    title: str,
) -> None:
    ax.plot(data.ravel())
    ax.set_title(title)
    ax.set_xlabel("Index")
