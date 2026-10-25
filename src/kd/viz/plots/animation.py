
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import (
    AbstractMovieWriter,
    FFMpegWriter,
    FuncAnimation,
    PillowWriter,
    writers,
)
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from kd.viz.plots._dim_utils import (
    _imshow_extent_for_spatial_axes,
    _pick_animation_frames,
    _slice_nd_to_2d,
)
from kd.viz.plots._field_panels import (
    _RESIDUAL_SIGN,
    FieldCmap,
    _bare_panel,
    _field_cmap,
    _field_limits,
    _range_note,
    _residual_limits,
    _scale_colorbar,
)
from kd.viz.style import CMAP_DIV, DPI, INK_MUTED, PANEL_H, PANEL_W, style_context

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap

    from kd.core.integrator import IntegrationResult
    from kd.data.schema import PDEDataset

logger = logging.getLogger(__name__)

_DIFFERENCE_TITLE = f"Difference ({_RESIDUAL_SIGN})"

_MP4_EXTRA_ARGS = ["-crf", "24"]
_MAX_TIME_DECIMALS = 6


def _movie_writer(path: Path, fps: int) -> AbstractMovieWriter:
    suffix = path.suffix.lower()
    if suffix == ".gif":
        return PillowWriter(fps=fps)
    if suffix == ".mp4":
        if not writers.is_available("ffmpeg"):
            raise RuntimeError(
                "Requested MovieWriter (ffmpeg) not available: install ffmpeg "
                "or set rcParams['animation.ffmpeg_path'] to write an .mp4"
            )
        return FFMpegWriter(fps=fps, codec="h264", extra_args=_MP4_EXTRA_ARGS)
    raise ValueError(f"movie path must end in .gif or .mp4, got {path.name!r}")


def _time_decimals(t: NDArray[np.floating]) -> int:
    if t.size < 2:
        return 2
    step = float(np.min(np.diff(t)))
    return min(max(math.ceil(-math.log10(step)), 0), _MAX_TIME_DECIMALS)


def save_field_animation(
    dataset: PDEDataset,
    integration_result: IntegrationResult,
    output_path: str | Path,
    *,
    fps: int = 8,
    max_frames: int = 24,
    field_cmap: FieldCmap = "sequential",
    difference: bool = True,
) -> list[str]:
    if not integration_result.success:
        raise ValueError(
            f"Cannot animate failed integration: {integration_result.warning}"
        )
    animation, notes = plot_field_animation(
        dataset,
        integration_result,
        fps=fps,
        max_frames=max_frames,
        field_cmap=field_cmap,
        difference=difference,
    )
    if animation is None:
        raise ValueError("; ".join(notes))
    figure = plt.gcf()
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(path, writer=PillowWriter(fps=fps))
    finally:
        plt.close(figure)
    return notes


@dataclass(frozen=True)
class _Movie:

    true_field: NDArray[np.floating]
    pred_field: NDArray[np.floating] | None
    difference: NDArray[np.floating] | None
    time_dim: int
    t_coords: NDArray[np.floating]
    time_axis: str
    field_name: str
    extent: tuple[float, float, float, float]
    xlabel: str
    ylabel: str
    limits: tuple[float, float]
    cmap: str | Colormap
    difference_limits: tuple[float, float] | None


def plot_field_animation(
    dataset: PDEDataset,
    integration_result: IntegrationResult,
    *,
    style: dict[str, Any] | None = None,
    max_frames: int = 24,
    fps: int = 8,
    field_cmap: FieldCmap = "sequential",
    difference: bool = True,
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

    movie = _prepare_movie(
        dataset, integration_result, warnings, field_cmap, difference=difference
    )
    n_t = movie.true_field.shape[movie.time_dim]
    frame_indices = _pick_animation_frames(n_t, max_frames)
    n_shown = len(frame_indices)
    suptitle = None
    if n_shown < n_t:


        suptitle = f"Showing {n_shown} of {n_t} time steps"
        warnings.append(
            f"Animation shows {n_shown} of {n_t} time steps (max_frames={max_frames})"
        )

    with style_context(style):
        fig, axes, images = _init_animation_figure(
            movie, frame_indices[0], suptitle=suptitle
        )

    def _update(frame_idx: int) -> list[AxesImage]:
        _draw_frame(movie, axes, images, frame_idx)
        return images

    animation = FuncAnimation(
        fig,
        _update,
        frames=frame_indices,
        interval=1000.0 / float(fps),
        blit=False,
    )
    return animation, warnings


def _prepare_movie(
    dataset: PDEDataset,
    integration_result: IntegrationResult,
    warnings: list[str],
    field_cmap: FieldCmap,
    *,
    difference: bool,
) -> _Movie:
    assert dataset.axis_order is not None
    time_axis = dataset.lhs_axis
    true_field = np.array(
        dataset.get_field(dataset.lhs_field).detach().cpu().numpy(),
        dtype=np.float64,
    )
    pred_field = _extract_prediction_for_animation(
        integration_result,
        true_field.shape,
        warnings,
    )
    diff = pred_field - true_field if difference and pred_field is not None else None
    extent, xlabel, ylabel = _imshow_extent_for_spatial_axes(
        dataset, dataset.spatial_axes
    )







    return _Movie(
        true_field=true_field,
        pred_field=pred_field,
        difference=diff,
        time_dim=dataset.axis_order.index(time_axis),
        t_coords=np.asarray(
            dataset.get_coords(time_axis).detach().cpu().numpy(), dtype=np.float64
        ),
        time_axis=time_axis,
        field_name=dataset.lhs_field,
        extent=extent,
        xlabel=xlabel,
        ylabel=ylabel,
        limits=_field_limits(true_field, pred_field, field_cmap),
        cmap=_field_cmap(field_cmap),
        difference_limits=None if diff is None else _residual_limits(diff),
    )


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


def _panel_titles(movie: _Movie, frame_idx: int) -> list[str]:
    when = f"{movie.time_axis}={float(movie.t_coords[frame_idx]):.3g}"
    titles = [f"True ({when})"]
    if movie.pred_field is not None:
        pred_slice = _display_slice(movie.pred_field, frame_idx, movie.time_dim)
        titles.append(f"Predicted ({when})" + _range_note(pred_slice, movie.limits))
    if movie.difference is not None:
        titles.append(_DIFFERENCE_TITLE)
    return titles


def _panel_fields(movie: _Movie) -> list[NDArray[np.floating]]:
    fields = [movie.true_field]
    if movie.pred_field is not None:
        fields.append(movie.pred_field)
    if movie.difference is not None:
        fields.append(movie.difference)
    return fields


def _draw_frame(
    movie: _Movie, axes: list[Axes], images: list[AxesImage], frame_idx: int
) -> None:
    titles = _panel_titles(movie, frame_idx)
    for ax, image, field, title in zip(
        axes, images, _panel_fields(movie), titles, strict=True
    ):
        image.set_data(_display_slice(field, frame_idx, movie.time_dim))
        ax.set_title(title)


def _init_animation_figure(
    movie: _Movie,
    frame_idx: int,
    *,
    suptitle: str | None = None,
) -> tuple[Figure, list[Axes], list[AxesImage]]:
    fields = _panel_fields(movie)
    n_cols = len(fields)
    fig, axes_arr = plt.subplots(
        1,
        n_cols,
        figsize=(PANEL_W * n_cols, PANEL_H + 0.4),
        dpi=DPI,
        layout="constrained",
        squeeze=False,
    )
    axes: list[Axes] = list(axes_arr.flat)
    images: list[AxesImage] = []
    for index, (ax, field) in enumerate(zip(axes, fields, strict=True)):
        is_difference = movie.difference is not None and index == n_cols - 1
        limits = movie.difference_limits if is_difference else movie.limits
        assert limits is not None
        images.append(
            _add_image(
                ax,
                _display_slice(field, frame_idx, movie.time_dim),
                movie,
                CMAP_DIV if is_difference else movie.cmap,
                limits,
            )
        )
    _draw_frame(movie, axes, images, frame_idx)

    if suptitle is not None:
        fig.suptitle(suptitle, color=INK_MUTED, fontsize="medium")
    n_fields = 1 if movie.pred_field is None else 2
    _scale_colorbar(
        images[:n_fields],
        axes[:n_fields],
        data=fields[:n_fields],
        label=movie.field_name,
    )
    if movie.difference is not None:
        _scale_colorbar(images[-1:], axes[-1:], data=[movie.difference])
    return fig, axes, images


def _add_image(
    ax: Axes,
    data: NDArray[np.floating],
    movie: _Movie,
    cmap: str | Colormap,
    limits: tuple[float, float],
) -> AxesImage:
    image = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=movie.extent,
        cmap=cmap,
        vmin=limits[0],
        vmax=limits[1],
        rasterized=True,
    )
    _bare_panel(ax)
    ax.set_xlabel(movie.xlabel)
    ax.set_ylabel(movie.ylabel)
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
