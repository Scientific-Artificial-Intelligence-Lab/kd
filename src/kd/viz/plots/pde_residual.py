
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

    from kd.data.schema import PDEDataset
    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_RESIDUAL_FIGSIZE = (15, 4)
_DEFAULT_DPI = 150
_WARNING_FONTSIZE = 12
_RESIDUAL_PERCENTILE = 99.0


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


    if dataset is not None and _has_axis_info(dataset):
        with style_context(style):
            fig = _render_axis_aware(actual, predicted, dataset, field_shape, warnings)
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
            _render_panel(axes[0], actual_2d, "True (u_t actual)")
            _render_panel(axes[1], pred_2d, "Predicted (u_t predicted)")
            residual = actual_2d - pred_2d
            _render_panel(axes[2], residual, "Residual", residual=True)
        else:

            _line_fallback(axes[0], actual, "True (u_t actual)")
            _line_fallback(axes[1], predicted, "Predicted (u_t predicted)")
            residual_1d = actual - predicted
            _line_fallback(axes[2], residual_1d, "Residual")

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
        _line_fallback(axes[0], actual, "True (u_t actual)")
        _line_fallback(axes[1], predicted, "Predicted (u_t predicted)")
        _line_fallback(axes[2], actual - predicted, "Residual")
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
        _line_fallback(axes[0], actual, "True (u_t actual)")
        _line_fallback(axes[1], predicted, "Predicted (u_t predicted)")
        _line_fallback(axes[2], actual - predicted, "Residual")
        fig.tight_layout()
        return fig

    residual_nd = actual_nd - pred_nd



    if not _shape_matches_dataset(shape, dataset, time_axis, spatial_axes):
        warnings.append("field_shape does not match dataset axes; using generic render")
        fig, axes_arr = plt.subplots(
            1,
            3,
            figsize=_RESIDUAL_FIGSIZE,
            dpi=_DEFAULT_DPI,
        )
        axes = list(axes_arr.flat)
        _render_panel(axes[0], actual_nd, "True (u_t actual)")
        _render_panel(axes[1], pred_nd, "Predicted (u_t predicted)")
        _render_panel(axes[2], residual_nd, "Residual", residual=True)
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

    _pcolormesh_panel(axes[0], t_coords, s_coords, actual_display, "True (u_t actual)")
    _pcolormesh_panel(
        axes[1], t_coords, s_coords, pred_display, "Predicted (u_t predicted)"
    )
    _pcolormesh_panel(
        axes[2], t_coords, s_coords, residual_display, "Residual", residual=True
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
    from kd.viz.plots._dim_utils import _pick_time_steps, _slice_nd_to_2d

    n_t = actual_nd.shape[time_dim]
    mid_indices = _pick_time_steps(n_t, 1)
    mid_idx = mid_indices[0]

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

    _heatmap_panel(axes[0], actual_slice, f"True (t={t_val:.3g})")
    _heatmap_panel(axes[1], pred_slice, f"Predicted (t={t_val:.3g})")
    _heatmap_panel(axes[2], residual_slice, f"Residual (t={t_val:.3g})", residual=True)

    fig.tight_layout()
    return fig


def _pcolormesh_panel(
    ax: Axes,
    t_coords: np.ndarray,
    s_coords: np.ndarray,
    data: np.ndarray,
    title: str,
    *,
    residual: bool = False,
) -> None:
    display = np.where(np.isfinite(data), data, np.nan)
    if residual:
        vmax = _robust_abs_max(display)
        mesh = ax.pcolormesh(
            t_coords,
            s_coords,
            display,
            shading="auto",
            rasterized=True,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.pcolormesh(t_coords, s_coords, display, shading="auto", rasterized=True)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(title)


def _heatmap_panel(
    ax: Axes,
    data: np.ndarray,
    title: str,
    *,
    residual: bool = False,
) -> None:
    display = np.where(np.isfinite(data), data, np.nan)
    if residual:
        vmax = _robust_abs_max(display)
        im = ax.imshow(
            display,
            aspect="auto",
            origin="lower",
            rasterized=True,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.imshow(display, aspect="auto", origin="lower", rasterized=True)
    ax.set_title(title)


def _robust_abs_max(data: np.ndarray) -> float:
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
) -> None:
    from kd.viz.plots._dim_utils import _slice_nd_to_2d

    render_data = data
    if render_data.ndim > 2:
        render_data = _slice_nd_to_2d(render_data, (0, 1))

    display = np.where(np.isfinite(render_data), render_data, np.nan)
    if residual:
        vmax = _robust_abs_max(display)
        im = ax.imshow(
            display,
            aspect="auto",
            origin="lower",
            rasterized=True,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.imshow(display, aspect="auto", origin="lower", rasterized=True)
    ax.set_title(title)


def _line_fallback(
    ax: Axes,
    data: np.ndarray,
    title: str,
) -> None:
    ax.plot(data.ravel())
    ax.set_title(title)
    ax.set_xlabel("Index")
