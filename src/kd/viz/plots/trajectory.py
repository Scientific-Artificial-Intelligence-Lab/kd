
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from kd.viz.plots._field_panels import _DISCOVERED_LABEL, _OBSERVED_LABEL
from kd.viz.plots.animation import _movie_writer, _time_decimals
from kd.viz.style import (
    AXIS,
    CONTEXT,
    DISCOVERED,
    DISCOVERED_DASH,
    DPI,
    FIGSIZE_SINGLE,
    GRID,
    INK,
    LINE_WIDTH,
    OBSERVED,
    PANEL_H,
    PANEL_W,
    TRAJECTORY_CONTEXT_WIDTH,
    TRAJECTORY_WIDTH,
    style_context,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    from kd.data.trajectory import Trajectory, TrajectorySet

logger = logging.getLogger(__name__)

_START_MARKER_SIZE = 4.5
_STATE_PANEL_INCHES = 1.5
_OVERLAY_FIGSIZE = (2 * PANEL_W, 2 * PANEL_H)
_LIMIT_PAD_FRACTION = 0.04


@dataclass(frozen=True)
class _Pair:

    observed: Trajectory
    discovered: Trajectory | None
    observed_rows: tuple[int, ...]
    discovered_rows: tuple[int, ...]

    def observed_states(self) -> NDArray[np.floating]:
        return self.observed.state[list(self.observed_rows)]

    def discovered_states(self) -> NDArray[np.floating] | None:
        if self.discovered is None:
            return None
        return self.discovered.state[list(self.discovered_rows)]


def _default_states(observed: TrajectorySet) -> tuple[str, ...]:
    if len(observed.state_vars) < 2:
        raise ValueError(
            f"a trajectory overlay needs two or three states; set "
            f"{observed.name!r} has {list(observed.state_vars)}"
        )
    return observed.state_vars[:3]


def _rows(trajectories: TrajectorySet, states: Sequence[str]) -> tuple[int, ...]:
    missing = [state for state in states if state not in trajectories.state_vars]
    if missing:
        raise ValueError(
            f"states {missing} are not in set {trajectories.name!r} "
            f"(state_vars {list(trajectories.state_vars)})"
        )
    return tuple(trajectories.state_vars.index(state) for state in states)


def _pairs(
    observed: TrajectorySet, predicted: TrajectorySet, states: Sequence[str]
) -> tuple[list[_Pair], list[str]]:
    observed_rows = _rows(observed, states)
    discovered_rows = _rows(predicted, states)
    by_id = {trajectory.id: trajectory for trajectory in predicted.trajectories}
    pairs = [
        _Pair(t, by_id.get(t.id), observed_rows, discovered_rows)
        for t in observed.trajectories
    ]
    missing = [pair.observed.id for pair in pairs if pair.discovered is None]
    notes = []
    if missing:
        notes.append(
            f"No discovered-law trajectory for {', '.join(missing)}; "
            "drawn observed only"
        )
    return pairs, notes


def plot_trajectory_overlay(
    observed: TrajectorySet,
    predicted: TrajectorySet,
    *,
    states: Sequence[str] | None = None,
    ax: Axes | None = None,
    style: dict[str, Any] | None = None,
) -> list[str]:
    chosen = tuple(states) if states is not None else _default_states(observed)
    if len(chosen) not in (2, 3):
        raise ValueError(f"states must name two or three variables, got {chosen}")
    pairs, notes = _pairs(observed, predicted, chosen)
    with style_context(style):
        ax = _overlay_axes(ax, three_d=len(chosen) == 3)
        _draw_overlay(ax, pairs)
        _label_state_axes(ax, chosen)
        ax.legend(loc="upper left")
    return notes


def _overlay_axes(ax: Axes | None, *, three_d: bool) -> Axes:
    if ax is None:
        ax = _new_overlay_figure(three_d=three_d).axes[0]
    elif three_d and ax.name != "3d":
        raise ValueError(
            "three states need 3D axes: pass an Axes created with "
            "projection='3d', or ax=None"
        )
    if three_d:
        _clean_3d(cast("Axes3D", ax))
    return ax


def _new_overlay_figure(
    *, three_d: bool, figsize: tuple[float, float] = _OVERLAY_FIGSIZE
) -> Figure:
    fig = plt.figure(figsize=figsize, dpi=DPI, layout="constrained")
    fig.add_subplot(projection="3d" if three_d else None)
    return fig



_ZOOM_3D = 0.88





_MOVIE_ZOOM_3D = 1.0
_MOVIE_AXES_POSITION = (0.0, 0.02, 1.0, 1.0)
_MOVIE_TITLE_XY = (0.012, 0.985)
_MOVIE_LEGEND_XY = (0.012, 0.93)


def _clean_3d(ax: Axes3D) -> None:
    ax.set_box_aspect(None, zoom=_ZOOM_3D)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_fill(False)
        axis.pane.set_edgecolor(GRID)
        axis.line.set_color(AXIS)


_Drawn = tuple["Line2D", NDArray[np.floating], NDArray[np.floating]]


def _draw_overlay(ax: Axes, pairs: list[_Pair]) -> list[_Drawn]:
    drawn: list[_Drawn] = []
    discovered_label: str | None = _DISCOVERED_LABEL
    for index, pair in enumerate(pairs):
        observed = pair.observed_states()
        line = _curve(
            ax,
            observed,
            color=CONTEXT,
            linewidth=TRAJECTORY_CONTEXT_WIDTH,
            label=_OBSERVED_LABEL if index == 0 else None,
        )
        drawn.append((line, pair.observed.t, observed))
        discovered = pair.discovered_states()
        if pair.discovered is not None and discovered is not None:
            line = _curve(
                ax,
                discovered,
                color=DISCOVERED,
                linewidth=TRAJECTORY_WIDTH,
                label=discovered_label,
            )
            discovered_label = None
            drawn.append((line, pair.discovered.t, discovered))
        ax.plot(
            *observed[:, :1],
            marker="o",
            linestyle="None",
            markersize=_START_MARKER_SIZE,
            color=INK,
        )
    return drawn


def _curve(ax: Axes, states: NDArray[np.floating], **kwargs: Any) -> Line2D:
    (line,) = ax.plot(*states, **kwargs)
    return line


def _label_state_axes(ax: Axes, states: Sequence[str]) -> None:
    ax.set_xlabel(states[0])
    ax.set_ylabel(states[1])
    if len(states) == 3:
        cast("Axes3D", ax).set_zlabel(states[2])


def plot_state_panels(
    observed: TrajectorySet,
    predicted: TrajectorySet,
    *,
    t_max: float | None = None,
    style: dict[str, Any] | None = None,
) -> tuple[Figure, list[str]]:
    states = observed.state_vars
    pairs, notes = _pairs(observed, predicted, states)
    t_start = min(float(p.observed.t[0]) for p in pairs)
    t_end = max(float(p.observed.t[-1]) for p in pairs)
    t_stop = t_end if t_max is None else t_max
    if not t_start < t_stop <= t_end:
        raise ValueError(
            f"t_max={t_max} is outside the observed time range ({t_start}, {t_end}]"
        )
    height = max(PANEL_H, _STATE_PANEL_INCHES * len(states) + 0.6)
    with style_context(style):
        fig, axes = plt.subplots(
            len(states),
            1,
            sharex=True,
            squeeze=False,
            figsize=(FIGSIZE_SINGLE[0], height),
            dpi=DPI,
            layout="constrained",
        )
        for row, state in enumerate(states):
            _draw_state_row(axes[row, 0], pairs, row, t_stop)
            axes[row, 0].set_ylabel(state)
        axes[-1, 0].set_xlabel(observed.time_var)
        axes[-1, 0].set_xlim(t_start, t_stop)
        fig.legend(loc="outside upper right", ncols=2)
    return fig, notes


def _draw_state_row(ax: Axes, pairs: list[_Pair], row: int, t_stop: float) -> None:
    discovered_label = _DISCOVERED_LABEL if row == 0 else None
    for index, pair in enumerate(pairs):
        kept = pair.observed.t <= t_stop
        ax.plot(
            pair.observed.t[kept],
            pair.observed_states()[row, kept],
            color=OBSERVED,
            linewidth=LINE_WIDTH,
            label=_OBSERVED_LABEL if index == 0 and row == 0 else None,
        )
        discovered = pair.discovered_states()
        if pair.discovered is None or discovered is None:
            continue
        kept = pair.discovered.t <= t_stop
        ax.plot(
            pair.discovered.t[kept],
            discovered[row, kept],
            color=DISCOVERED,
            linestyle=DISCOVERED_DASH,
            linewidth=LINE_WIDTH,
            label=discovered_label,
        )
        discovered_label = None


def save_trajectory_animation(
    observed: TrajectorySet,
    predicted: TrajectorySet,
    output_path: str | Path,
    *,
    fps: int = 24,
    max_frames: int = 300,
    states: Sequence[str] | None = None,
) -> list[str]:
    if fps < 1:
        raise ValueError(f"fps must be >= 1, got {fps}")
    path = Path(output_path)
    writer = _movie_writer(path, fps)
    chosen = tuple(states) if states is not None else _default_states(observed)
    pairs, notes = _pairs(observed, predicted, chosen)
    three_d = len(chosen) == 3
    with style_context(None):
        figure = _new_overlay_figure(three_d=three_d, figsize=FIGSIZE_SINGLE)
        ax = _overlay_axes(figure.axes[0], three_d=three_d)
        drawn = _draw_overlay(ax, pairs)
        _label_state_axes(ax, chosen)
        title = _movie_title_and_legend(figure, ax, three_d=three_d)
        _fix_limits(ax, drawn, three_d=three_d)
        animation = _growth_animation(
            figure, title, drawn, max_frames, fps, observed.time_var
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(path, writer=writer)
    finally:
        plt.close(figure)
    return notes


def _movie_title_and_legend(figure: Figure, ax: Axes, *, three_d: bool) -> Text:
    if not three_d:
        ax.legend(loc="upper left")
        return ax.set_title(" ", loc="left")
    figure.set_layout_engine("none")
    ax.set_position(_MOVIE_AXES_POSITION)
    cast("Axes3D", ax).set_box_aspect(None, zoom=_MOVIE_ZOOM_3D)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=_MOVIE_LEGEND_XY,
        bbox_transform=figure.transFigure,
        borderaxespad=0.0,
    )
    return figure.text(
        *_MOVIE_TITLE_XY,
        " ",
        ha="left",
        va="top",
        fontsize=plt.rcParams["axes.titlesize"],
        fontweight=plt.rcParams["axes.titleweight"],
        color=plt.rcParams["axes.titlecolor"],
    )


def _fix_limits(ax: Axes, drawn: list[_Drawn], *, three_d: bool) -> None:
    stacked = np.concatenate([states for _, _, states in drawn], axis=1)
    lo, hi = stacked.min(axis=1), stacked.max(axis=1)
    pad = (hi - lo) * _LIMIT_PAD_FRACTION
    low, high = (lo - pad).tolist(), (hi + pad).tolist()
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    if three_d:
        cast("Axes3D", ax).set_zlim(low[2], high[2])


def _growth_animation(
    figure: Figure,
    title: Text,
    drawn: list[_Drawn],
    max_frames: int,
    fps: int,
    time_var: str,
) -> FuncAnimation:
    t_start = min(float(t[0]) for _, t, _ in drawn)
    t_end = max(float(t[-1]) for _, t, _ in drawn)
    longest = max(t.size for _, t, _ in drawn)
    times = np.linspace(t_start, t_end, min(max_frames, longest))
    decimals = _time_decimals(times)
    labels = [f"{time_var} = {t:.{decimals}f}" for t in times]
    lines = [line for line, _, _ in drawn]
    title.set_text(labels[0])

    def _update(frame: int) -> list[Line2D]:
        title.set_text(labels[frame])
        for line, t, states in drawn:
            shown = states[:,: int(np.searchsorted(t, times[frame], side="right"))]
            if states.shape[0] == 3:
                line.set_data_3d(*shown)
            else:
                line.set_data(*shown)
        return lines

    return FuncAnimation(
        figure, _update, frames=len(times), interval=1000.0 / float(fps), blit=False
    )


__all__ = ["plot_state_panels", "plot_trajectory_overlay", "save_trajectory_animation"]
