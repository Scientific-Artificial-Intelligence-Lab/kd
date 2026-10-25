
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from kd.viz.plots._dim_utils import _pick_animation_frames, _pick_time_steps
from kd.viz.plots._field_panels import (
    _DISCOVERED_LABEL,
    _OBSERVED_LABEL,
    _range_note,
    _reference_limits,
)
from kd.viz.plots.animation import (
    _extract_prediction_for_animation,
    _movie_writer,
    _time_decimals,
)
from kd.viz.style import (
    DISCOVERED,
    DISCOVERED_DASH,
    DPI,
    FIGSIZE_SINGLE,
    INK_2,
    INK_MUTED,
    LINE_WIDTH,
    OBSERVED,
    SURFACE,
    TRAJECTORY_WIDTH,
    style_context,
)

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    from kd.core.integrator import IntegrationResult
    from kd.data.schema import PDEDataset

logger = logging.getLogger(__name__)


_Y_PAD_FRACTION = 0.08

_OFFSET_FRACTION = 0.6
_WATERFALL_ROW_INCHES = 0.55
_WATERFALL_FRAME_INCHES = 1.4


@dataclass(frozen=True)
class _Profiles:

    x: NDArray[np.floating]
    t: NDArray[np.floating]
    observed: NDArray[np.floating]
    discovered: NDArray[np.floating] | None
    field_name: str
    space_axis: str
    time_axis: str


def _profiles(
    dataset: PDEDataset, integration_result: IntegrationResult, warnings: list[str]
) -> _Profiles:
    spatial = dataset.spatial_axes
    if len(spatial) != 1:
        raise ValueError(
            f"profile plots draw one spatial axis; dataset {dataset.name!r} has "
            f"{len(spatial)} {spatial}. Use save_field_animation for two."
        )
    order = dataset.axis_order
    assert order is not None
    time_axis = dataset.lhs_axis
    observed = np.array(
        dataset.get_field(dataset.lhs_field).detach().cpu().numpy(), dtype=np.float64
    )
    discovered = _extract_prediction_for_animation(
        integration_result, observed.shape, warnings
    )
    time_dim = order.index(time_axis)
    return _Profiles(
        x=dataset.get_coords(spatial[0]).detach().cpu().numpy().astype(np.float64),
        t=dataset.get_coords(time_axis).detach().cpu().numpy().astype(np.float64),
        observed=np.moveaxis(observed, time_dim, -1),
        discovered=None
        if discovered is None
        else np.moveaxis(discovered, time_dim, -1),
        field_name=dataset.lhs_field,
        space_axis=spatial[0],
        time_axis=time_axis,
    )


def _time_label(profiles: _Profiles, index: int) -> str:
    decimals = _time_decimals(profiles.t)
    return f"{profiles.time_axis} = {float(profiles.t[index]):.{decimals}f}"


def _fixed_ylim(observed: NDArray[np.floating]) -> tuple[float, float]:
    lo, hi = _reference_limits(observed)
    pad = (hi - lo) * _Y_PAD_FRACTION
    return lo - pad, hi + pad


def _decimation_note(n_shown: int, n_t: int, max_frames: int) -> str | None:
    if n_shown >= n_t:
        return None
    return f"Animation shows {n_shown} of {n_t} time steps (max_frames={max_frames})"


def plot_profile_animation(
    dataset: PDEDataset,
    integration_result: IntegrationResult,
    *,
    fps: int = 12,
    max_frames: int = 60,
    style: dict[str, Any] | None = None,
) -> tuple[FuncAnimation, list[str]]:
    _, animation, notes = _profile_movie(
        dataset, integration_result, fps=fps, max_frames=max_frames, style=style
    )
    return animation, notes


def save_profile_animation(
    dataset: PDEDataset,
    integration_result: IntegrationResult,
    output_path: str | Path,
    *,
    fps: int = 12,
    max_frames: int = 60,
) -> list[str]:
    if not integration_result.success:
        raise ValueError(
            f"Cannot animate failed integration: {integration_result.warning}"
        )
    path = Path(output_path)
    writer = _movie_writer(path, fps)
    figure, animation, notes = _profile_movie(
        dataset, integration_result, fps=fps, max_frames=max_frames, style=None
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(path, writer=writer)
    finally:
        plt.close(figure)
    return notes


def _profile_movie(
    dataset: PDEDataset,
    integration_result: IntegrationResult,
    *,
    fps: int,
    max_frames: int,
    style: dict[str, Any] | None,
) -> tuple[Figure, FuncAnimation, list[str]]:
    if fps < 1:
        raise ValueError(f"fps must be >= 1, got {fps}")
    notes: list[str] = []
    profiles = _profiles(dataset, integration_result, notes)
    n_t = profiles.t.size
    frames = _pick_animation_frames(n_t, max_frames)
    decimation = _decimation_note(len(frames), n_t, max_frames)
    if decimation is not None:
        notes.append(decimation)
    with style_context(style):
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, dpi=DPI, layout="constrained")
        lines = _draw_profiles(ax, profiles, frames[0])
        if decimation is not None:

            ax.set_title(
                f"Showing {len(frames)} of {n_t} time steps",
                loc="right",
                color=INK_MUTED,
                fontsize="small",
            )
        clip_notes = _clip_notes(profiles, frames, ax.get_ylim(), notes)
        clip_text = _note_text(ax, clip_notes.get(frames[0], ""))

    fields = [f for f in (profiles.observed, profiles.discovered) if f is not None]

    def _update(frame: int) -> list[Artist]:
        for line, field in zip(lines, fields, strict=True):
            line.set_ydata(field[:, frame])
        ax.set_title(_time_label(profiles, frame), loc="left")
        clip_text.set_text(clip_notes.get(frame, ""))
        return [*lines, clip_text]

    animation = FuncAnimation(
        fig, _update, frames=frames, interval=1000.0 / float(fps), blit=False
    )
    return fig, animation, notes


def _clip_notes(
    profiles: _Profiles,
    frames: list[int],
    ylim: tuple[float, float],
    notes: list[str],
) -> dict[int, str]:
    if profiles.discovered is None:
        return {}
    by_frame = {}
    for frame in frames:
        note = _range_note(profiles.discovered[:, frame], ylim).strip()
        if note:
            by_frame[frame] = f"Discovered profile {note}"
    if by_frame:
        notes.append(
            f"Discovered profile outside the fixed y range in {len(by_frame)} of "
            f"{len(frames)} frames"
        )
    return by_frame


def _note_text(ax: Axes, text: str) -> Text:
    return ax.text(
        0.01,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=INK_MUTED,
        fontsize="small",
        bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 2},
    )


def _draw_profiles(ax: Axes, profiles: _Profiles, frame: int) -> list[Line2D]:
    (observed,) = ax.plot(
        profiles.x,
        profiles.observed[:, frame],
        color=OBSERVED,
        linewidth=LINE_WIDTH,
        label=_OBSERVED_LABEL,
    )
    lines = [observed]
    if profiles.discovered is not None:
        (discovered,) = ax.plot(
            profiles.x,
            profiles.discovered[:, frame],
            color=DISCOVERED,
            linestyle=DISCOVERED_DASH,
            linewidth=LINE_WIDTH,
            label=_DISCOVERED_LABEL,
        )
        lines.append(discovered)
    ax.set_xlim(float(profiles.x.min()), float(profiles.x.max()))
    ax.set_ylim(*_fixed_ylim(profiles.observed))
    ax.set_xlabel(profiles.space_axis)
    ax.set_ylabel(profiles.field_name)
    ax.set_title(_time_label(profiles, frame), loc="left")
    ax.figure.legend(loc="outside upper right", ncols=2)
    return lines


def plot_profile_waterfall(
    dataset: PDEDataset,
    integration_result: IntegrationResult,
    *,
    n_profiles: int = 8,
    offset: float | None = None,
    style: dict[str, Any] | None = None,
) -> tuple[Figure, list[str]]:
    if n_profiles < 1:
        raise ValueError(f"n_profiles must be >= 1, got {n_profiles}")
    notes: list[str] = []
    profiles = _profiles(dataset, integration_result, notes)
    indices = _pick_time_steps(profiles.t.size, n_profiles)
    lo, hi = _reference_limits(profiles.observed)
    step = _round_offset(_OFFSET_FRACTION * (hi - lo)) if offset is None else offset
    height = _WATERFALL_FRAME_INCHES + _WATERFALL_ROW_INCHES * len(indices)
    with style_context(style):
        fig, ax = plt.subplots(
            figsize=(FIGSIZE_SINGLE[0], height), dpi=DPI, layout="constrained"
        )
        for row, index in enumerate(indices):
            _draw_waterfall_row(ax, profiles, index, row * step, first=row == 0)
        _finish_waterfall(ax, profiles, step)
    return fig, notes


def _round_offset(value: float) -> float:
    return float(f"{value:.1g}")


def _draw_waterfall_row(
    ax: Axes, profiles: _Profiles, index: int, shift: float, *, first: bool
) -> None:
    ax.plot(
        profiles.x,
        profiles.observed[:, index] + shift,
        color=OBSERVED,
        linewidth=LINE_WIDTH,
        label=_OBSERVED_LABEL if first else None,
    )
    if profiles.discovered is not None:
        ax.plot(
            profiles.x,
            profiles.discovered[:, index] + shift,
            color=DISCOVERED,
            linestyle=DISCOVERED_DASH,
            linewidth=TRAJECTORY_WIDTH,
            label=_DISCOVERED_LABEL if first else None,
        )
    span = float(profiles.x.max() - profiles.x.min())
    ax.text(
        float(profiles.x.min()) - 0.015 * span,
        float(profiles.observed[0, index]) + shift,
        _time_label(profiles, index),
        ha="right",
        va="center",
        color=INK_2,
        fontsize="small",
    )


def _finish_waterfall(ax: Axes, profiles: _Profiles, step: float) -> None:
    ax.set_xlim(float(profiles.x.min()), float(profiles.x.max()))
    ax.set_xlabel(profiles.space_axis)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_title(f"{profiles.field_name} profiles, each raised by {step:g}", loc="left")
    ax.figure.legend(loc="outside upper right", ncols=2)


__all__ = ["plot_profile_animation", "plot_profile_waterfall", "save_profile_animation"]
