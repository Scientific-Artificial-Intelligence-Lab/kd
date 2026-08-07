
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from kd.viz.plots._dim_utils import (
    _imshow_extent_for_spatial_axes,
    _pick_animation_frames,
    _slice_nd_to_2d,
)
from kd.viz.plots._field_panels import (
    _colorbar_extend,
    _range_note,
    _reference_limits,
)
from kd.viz.style import style_context

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.core.integrator import IntegrationResult
    from kd.data.schema import PDEDataset

logger = logging.getLogger(__name__)

_DEFAULT_DPI = 150


def plot_field_animation(
    dataset: PDEDataset,
    integration_result: IntegrationResult,
    *,
    style: dict[str, Any] | None = None,
    max_frames: int = 24,
    fps: int = 8,
) -> tuple[FuncAnimation | None, list[str]]:
    warnings: list[str] = []
    if dataset.axis_order is None:
        warnings.append("axis_order is None; skipping field animation")
        return None, warnings
    spatial_axes = dataset.spatial_axes
    if len(spatial_axes) != 2:
        warnings.append("Field animation requires exactly 2 spatial dimensions")
        return None, warnings
    if fps < 1:
        raise ValueError(f"fps must be >= 1, got {fps}")

    time_axis = dataset.lhs_axis
    time_dim = dataset.axis_order.index(time_axis)
    true_field = np.array(
        dataset.get_field(dataset.lhs_field).detach().cpu().numpy(),
        dtype=np.float64,
    )
    n_t = true_field.shape[time_dim]
    frame_indices = _pick_animation_frames(n_t, max_frames)
    n_shown = len(frame_indices)
    suptitle = None
    if n_shown < n_t:


        suptitle = f"Showing {n_shown} of {n_t} time steps"
        warnings.append(
            f"Animation shows {n_shown} of {n_t} time steps "
            f"(max_frames={max_frames})"
        )
    t_coords = np.asarray(
        dataset.get_coords(time_axis).detach().cpu().numpy(),
        dtype=np.float64,
    )

    pred_field = _extract_prediction_for_animation(
        integration_result,
        true_field.shape,
        warnings,
    )







    vmin, vmax = _reference_limits(true_field, pred_field)
    extent, xlabel, ylabel = _imshow_extent_for_spatial_axes(dataset, spatial_axes)

    with style_context(style):
        fig, axes, images = _init_animation_figure(
            true_field,
            pred_field,
            time_dim,
            frame_indices[0],
            t_coords,
            time_axis,
            dataset.lhs_field,
            extent,
            xlabel,
            ylabel,
            vmin,
            vmax,
            suptitle=suptitle,
        )

    def _update(frame_idx: int) -> list[AxesImage]:
        t_val = float(t_coords[frame_idx])
        true_slice = _display_slice(true_field, frame_idx, time_dim)
        images[0].set_data(true_slice)
        axes[0].set_title(f"True ({time_axis}={t_val:.3g})")
        if pred_field is not None and len(images) > 1:
            pred_slice = _display_slice(pred_field, frame_idx, time_dim)
            images[1].set_data(pred_slice)



            axes[1].set_title(
                f"Predicted ({time_axis}={t_val:.3g})"
                + _range_note(pred_slice, (vmin, vmax))
            )
        return images

    interval_ms = 1000.0 / float(fps)
    animation = FuncAnimation(
        fig,
        _update,
        frames=frame_indices,
        interval=interval_ms,
        blit=False,
    )
    return animation, warnings


def _extract_prediction_for_animation(
    integration_result: IntegrationResult,
    true_shape: tuple[int, ...],
    warnings: list[str],
) -> NDArray[np.floating] | None:
    if not integration_result.success:
        warnings.append(
            integration_result.warning
            or "Integration did not succeed; rendering True field only"
        )
        return None
    if integration_result.predicted_field is None:
        warnings.append("No predicted field available; rendering True field only")
        return None

    pred = np.array(
        integration_result.predicted_field.detach().cpu().numpy(),
        dtype=np.float64,
    )
    if pred.shape != true_shape:
        warnings.append(f"Shape mismatch: true {true_shape} vs predicted {pred.shape}")
        return None
    return pred


def _init_animation_figure(
    true_field: NDArray[np.floating],
    pred_field: NDArray[np.floating] | None,
    time_dim: int,
    frame_idx: int,
    t_coords: NDArray[np.floating],
    time_axis: str,
    field_name: str,
    extent: tuple[float, float, float, float],
    xlabel: str,
    ylabel: str,
    vmin: float,
    vmax: float,
    *,
    suptitle: str | None = None,
) -> tuple[Figure, list[Axes], list[AxesImage]]:
    n_cols = 2 if pred_field is not None else 1
    fig, axes_arr = plt.subplots(
        1,
        n_cols,
        figsize=(5 * n_cols, 4),
        dpi=_DEFAULT_DPI,
        constrained_layout=True,
        squeeze=False,
    )
    axes: list[Axes] = list(axes_arr.flat)
    images: list[AxesImage] = []
    t_val = float(t_coords[frame_idx])

    images.append(
        _add_image(
            axes[0],
            _display_slice(true_field, frame_idx, time_dim),
            f"True ({time_axis}={t_val:.3g})",
            extent,
            xlabel,
            ylabel,
            vmin,
            vmax,
        )
    )
    if pred_field is not None:
        pred_slice = _display_slice(pred_field, frame_idx, time_dim)
        images.append(
            _add_image(
                axes[1],
                pred_slice,
                f"Predicted ({time_axis}={t_val:.3g})"
                + _range_note(pred_slice, (vmin, vmax)),
                extent,
                xlabel,
                ylabel,
                vmin,
                vmax,
            )
        )

    if suptitle is not None:
        fig.suptitle(suptitle)
    extend = (
        _colorbar_extend(pred_field, vmin, vmax)
        if pred_field is not None
        else "neither"
    )
    fig.colorbar(images[0], ax=axes, label=field_name, extend=extend)
    return fig, axes, images


def _add_image(
    ax: Axes,
    data: NDArray[np.floating],
    title: str,
    extent: tuple[float, float, float, float],
    xlabel: str,
    ylabel: str,
    vmin: float,
    vmax: float,
) -> AxesImage:
    image = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return image


def _display_slice(
    field: NDArray[np.floating],
    frame_idx: int,
    time_dim: int,
) -> NDArray[np.floating]:
    frame = np.take(field, frame_idx, axis=time_dim)
    if frame.ndim > 2:
        frame = _slice_nd_to_2d(frame, (0, 1))
    return np.where(np.isfinite(frame), frame, np.nan)
