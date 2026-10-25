
from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import matplotlib
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from numpy.typing import ArrayLike


SERIES = ("#139d96", "#c9583f", "#6a45c0", "#b58a00")

OBSERVED = SERIES[0]
DISCOVERED = SERIES[1]

CONTEXT = "#898781"
BAND = "#c3c2b7"
BAND_ALPHA = 0.25

NO_DATA = "#c4c4c4"


INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#898781"

WARNING_INK = "#a8341f"

GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
TABLE_HEAD = "#eef2f7"
TABLE_EDGE = "#d8dee9"

ALL_TOKENS: frozenset[str] = frozenset(
    {
        *SERIES,
        CONTEXT,
        BAND,
        NO_DATA,
        INK,
        INK_2,
        INK_MUTED,
        WARNING_INK,
        GRID,
        AXIS,
        SURFACE,
        TABLE_HEAD,
        TABLE_EDGE,
    }
)


CMAP_SEQ = "viridis"


CMAP_DIV = LinearSegmentedColormap.from_list(
    "kd_div",
    ["#0b4f4b", "#139d96", "#8fd0cb", "#f0efec", "#e8a08e", "#c9583f", "#6e2415"],
)

LINE_WIDTH = 2.0
CONTEXT_WIDTH = 1.4
REF_LINE_WIDTH = 1.0
MARKER_SIZE = 5.0

DISCOVERED_DASH = (0, (4, 2))

TRAJECTORY_WIDTH = 1.2
TRAJECTORY_CONTEXT_WIDTH = 0.9
COLORBAR_OUTLINE_WIDTH = 0.6

DPI = 150
SAVE_DPI = 200
FIGSIZE_SINGLE = (8.0, 4.5)

PANEL_W = 2.8
PANEL_H = 2.6

DEFAULT_STYLE: dict[str, Any] = {
    "font.family": ["sans-serif"],
    "font.sans-serif": ["IBM Plex Sans", "Source Sans 3", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "figure.titlesize": 12,
    "text.color": INK,
    "axes.titlecolor": INK,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.prop_cycle": cycler(color=list(SERIES)),
    "image.cmap": CMAP_SEQ,
    "lines.linewidth": LINE_WIDTH,
    "lines.markersize": MARKER_SIZE,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "savefig.dpi": SAVE_DPI,
    "savefig.bbox": "tight",
}


@contextmanager
def style_context(
    extra_style: dict[str, Any] | None = None,
) -> Generator[None, None, None]:
    combined: dict[str, Any] = dict(DEFAULT_STYLE)
    if extra_style:
        combined.update(extra_style)
    original = {key: matplotlib.rcParams[key] for key in combined}
    try:
        matplotlib.rcParams.update(combined)
        yield
    finally:
        matplotlib.rcParams.update(original)


def style_colorbar(colorbar: Any) -> None:
    colorbar.outline.set_edgecolor(AXIS)
    colorbar.outline.set_linewidth(COLORBAR_OUTLINE_WIDTH)
    colorbar.ax.tick_params(color=AXIS, labelcolor=INK_2, length=2.5)
    colorbar.ax.grid(False)


def style_table(table: Any) -> None:
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(TABLE_EDGE)
        if row == 0:
            cell.set_facecolor(TABLE_HEAD)


def isolated_point_marks(values: ArrayLike) -> dict[str, Any]:
    finite = np.isfinite(np.asarray(values, dtype=float))
    left = np.concatenate(([False], finite[:-1]))
    right = np.concatenate((finite[1:], [False]))
    isolated = finite & ~left & ~right
    if not isolated.any():
        return {}
    return {"marker": "o", "markersize": MARKER_SIZE, "markevery": isolated.tolist()}
