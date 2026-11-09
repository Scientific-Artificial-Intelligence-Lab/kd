
from __future__ import annotations

import re
from collections.abc import Callable, Collection, Generator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import matplotlib
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.table import Table
    from matplotlib.text import Text


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
TABLE_HEAD = "#eeede8"
TABLE_EDGE = "#d6d5cd"

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

LINE_WIDTH = 1.5
CONTEXT_WIDTH = 1.1
REF_LINE_WIDTH = 0.8
MARKER_SIZE = 6.0


SCATTER_SIZE = 8.0
SCATTER_ALPHA = 0.4

DISCOVERED_DASH = (0, (4, 2))



SERIES_DASHES: tuple[Any, ...] = ("-", (0, (5, 2)), (0, (1, 2)), (0, (5, 2, 1, 2)))

TRAJECTORY_WIDTH = 1.0
TRAJECTORY_CONTEXT_WIDTH = 0.8
COLORBAR_OUTLINE_WIDTH = 0.6

BAR_FILL = 0.55


RING_WIDTH = 1.2



FONTSIZE = 11.0
TITLE_FONTSIZE = 12.0

LABEL_FONTSIZE = 10.5



SMALL_FONTSIZE = 9.5




TABLE_FONTSIZE = 10.0
TABLE_ROW_H = 0.34
TABLE_LINE_H = 0.19
TABLE_PAD = 0.08

DPI = 150
SAVE_DPI = 200
FIGSIZE_SINGLE = (8.0, 4.5)

PANEL_W = 2.8
PANEL_H = 2.6

DEFAULT_STYLE: dict[str, Any] = {
    "font.family": ["sans-serif"],
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "path",
    "font.size": FONTSIZE,
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.titlepad": 8.0,
    "axes.labelsize": FONTSIZE,
    "axes.labelpad": 5.0,
    "xtick.labelsize": LABEL_FONTSIZE,
    "ytick.labelsize": LABEL_FONTSIZE,
    "legend.fontsize": LABEL_FONTSIZE,
    "figure.titlesize": TITLE_FONTSIZE,
    "text.color": INK,
    "axes.titlecolor": INK,
    "axes.labelcolor": INK_2,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.75,
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
    "scatter.edgecolors": "face",
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.borderaxespad": 0.3,


    "figure.constrained_layout.wspace": 0.06,
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


def table_height(n_rows: int) -> float:
    return TABLE_ROW_H * (n_rows + 1)




_LEADING_OPERATORS = frozenset({"+", "-", "\u2212"})


def wrap_text(text: str, fits: Callable[[str], bool]) -> str:
    is_math = len(text) > 1 and text[0] == text[-1] == "$" and text.count("$") == 2
    if is_math:
        pieces = _math_terms(text[1:-1])
    else:
        pieces = []
        for word in _formula_words(text):
            parts = [word] if fits(word) else re.split(r"(?<=_)", word)
            pieces.extend([" " + parts[0], *parts[1:]])

    def render(line: str) -> str:
        return f"${line}$" if is_math else line

    lines: list[str] = []
    for piece in pieces:
        if lines and fits(render(lines[-1] + piece)):
            lines[-1] += piece
        else:
            lines.append(piece.lstrip(" "))
    return "\n".join(render(line) for line in lines)


def _formula_words(text: str) -> list[str]:
    words: list[str] = []
    for word in text.split(" "):
        if words and (word == "=" or words[-1] in _LEADING_OPERATORS):
            words[-1] += " " + word
        else:
            words.append(word)
    return words


def _math_terms(body: str) -> list[str]:
    pieces: list[str] = []
    start = depth = 0
    for index, char in enumerate(body):
        depth += (char in "({[") - (char in ")}]")
        if depth == 0 and index > start and body.startswith((" + ", " - "), index):
            pieces.append(body[start:index])
            start = index
    pieces.append(body[start:])
    return pieces


def draw_table(
    ax: Axes,
    rows: Sequence[Sequence[str]],
    col_labels: Sequence[str],
    *,
    text_columns: Collection[int],
    col_widths: Sequence[float] | None = None,
) -> Table:
    ax.axis("off")
    table = ax.table(
        cellText=[list(row) for row in rows],
        colLabels=list(col_labels),
        cellLoc="center",
        bbox=Bbox.from_bounds(0.0, 0.0, 1.0, 1.0),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_FONTSIZE)
    cells = table.get_celld()
    for (row, col), cell in cells.items():
        if row == 0:



            cell.set(facecolor=TABLE_HEAD, edgecolor=TABLE_HEAD)
            cell.set_text_props(weight="bold")
        else:
            cell.set_edgecolor(TABLE_EDGE)
            cell.visible_edges = "horizontal"
        if col in text_columns:
            cell.get_text().set_horizontalalignment("left")
    figure = cast("Figure", ax.get_figure(root=True))
    figure.draw_without_rendering()
    pad = TABLE_PAD * figure.dpi
    room = ax.get_window_extent().width
    widths = _column_widths(table, room, len(col_labels), text_columns, col_widths, pad)
    lines = [1] * (len(rows) + 1)
    for (row, col), cell in cells.items():
        cell.set_width(widths[col])
        if col in text_columns:

            cell.PAD = pad / widths[col]
            lines[row] = max(
                lines[row], _wrap_cell(cell.get_text(), widths[col] - 2 * pad)
            )
    for (row, _col), cell in cells.items():
        cell.set_height(TABLE_ROW_H + (lines[row] - 1) * TABLE_LINE_H)
    _grow_axes(ax, TABLE_LINE_H * (sum(lines) - len(lines)))
    return table


def _wrap_cell(text: Text, room: float) -> int:
    text.set_text(wrap_to_width(text.get_text(), text, room))
    return text.get_text().count("\n") + 1


def wrap_to_width(string: str, font: Text, room: float) -> str:

    def fits(line: str) -> bool:
        font.set_text(line)
        return bool(font.get_window_extent().width <= room)

    return wrap_text(string, fits)


def _column_widths(
    table: Table,
    room: float,
    n_cols: int,
    text_columns: Collection[int],
    col_widths: Sequence[float] | None,
    pad: float,
) -> list[float]:
    if col_widths is not None:
        return [room * share / sum(col_widths) for share in col_widths]
    natural = [0.0] * n_cols
    floor = [0.0] * n_cols
    for (_row, col), cell in table.get_celld().items():
        text = cell.get_text()
        natural[col] = max(natural[col], text.get_window_extent().width + 2 * pad)
        if col in text_columns:
            floor[col] = max(floor[col], _widest_piece(text) + 2 * pad)
    if sum(natural) <= room:
        return [room * width / sum(natural) for width in natural]
    numbers = sum(w for col, w in enumerate(natural) if col not in text_columns)
    columns = sorted(text_columns)
    cap = _width_cap(
        [floor[col] for col in columns],
        [natural[col] for col in columns],
        room - numbers,
    )
    return [
        min(max(cap, floor[col]), width) if col in text_columns else width
        for col, width in enumerate(natural)
    ]


def _widest_piece(text: Text) -> float:
    string = text.get_text()
    text.set_text(wrap_to_width(string, text, 0.0))
    width = float(text.get_window_extent().width)
    text.set_text(string)
    return width


def _width_cap(floors: list[float], widths: list[float], room: float) -> float:

    def total(cap: float) -> float:
        return sum(min(max(cap, lo), hi) for lo, hi in zip(floors, widths, strict=True))

    points = sorted({*floors, *widths})
    if total(points[0]) >= room:
        return points[0]


    right = next(point for point in points if total(point) >= room)
    left = max(point for point in points if point < right)
    return left + (room - total(left)) * (right - left) / (total(right) - total(left))


def _grow_axes(ax: Axes, extra: float) -> None:
    figure = cast("Figure", ax.get_figure(root=True))
    width, height = figure.get_size_inches()
    spec = ax.get_subplotspec()
    if spec is not None:


        grid = spec.get_gridspec()
        ratios = np.asarray(grid.get_height_ratios(), dtype=float)
        ratios[spec.rowspan.start: spec.rowspan.stop] *= 1.0 + extra / (
            ax.get_position().height * height
        )
        grid.set_height_ratios(ratios)
    figure.set_size_inches(width, height + extra)


def isolated_point_marks(values: ArrayLike) -> dict[str, Any]:
    finite = np.isfinite(np.asarray(values, dtype=float))
    left = np.concatenate(([False], finite[:-1]))
    right = np.concatenate((finite[1:], [False]))
    isolated = finite & ~left & ~right
    if not isolated.any():
        return {}
    return {"marker": "o", "markersize": MARKER_SIZE, "markevery": isolated.tolist()}
