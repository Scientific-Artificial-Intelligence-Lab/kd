
from __future__ import annotations

import logging
import textwrap
from collections import Counter
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from kd.core.expr import structure_term_key
from kd.core.expr.sympy_bridge import to_latex
from kd.viz._result_data import _sketch_fit_note
from kd.viz.axes import integer_ticks
from kd.viz.equation_display import RENDER_ERRORS, UNRENDERABLE_NOTE
from kd.viz.style import style_context

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from kd.search import SearchTrajectoryCandidate
    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_TermKey = tuple[str, str]
_PRESENT = "#2878b5"
_EMPTY = "#bdbdbd"


def _figure(height: float, notes: list[str]) -> tuple[Figure, Axes]:
    if not notes:
        return plt.subplots(figsize=(12, height), layout="constrained")
    wrapped = "\n".join(textwrap.fill(note, width=115) for note in notes)
    note_height = max(0.7, 0.2 * (wrapped.count("\n") + 1))
    figure = plt.figure(figsize=(12, height + note_height), layout="constrained")
    grid = figure.add_gridspec(2, 1, height_ratios=[height, note_height])
    ax = figure.add_subplot(grid[0])
    text_ax = figure.add_subplot(grid[1])
    text_ax.axis("off")
    text_ax.text(0, 0.5, wrapped, va="center", fontsize=10, transform=text_ax.transAxes)
    return figure, ax


def _term_history(
    history: list[list[SearchTrajectoryCandidate]],
) -> list[set[_TermKey]]:
    return [
        {
            (candidate.lhs, structure_term_key(term))
            for candidate in generation
            for term in candidate.terms
        }
        for generation in history
    ]


def _term_labels(keys: list[_TermKey]) -> tuple[list[str], list[str]]:
    labels, notes = [], []
    for index, (lhs, term) in enumerate(keys):
        try:
            labels.append(f"${to_latex(lhs)}$: ${to_latex(term)}$")
        except RENDER_ERRORS:
            logger.exception("Unrenderable trajectory term %s for target %s", term, lhs)
            labels.append(f"unrenderable term {index + 1}")
            notes.append(f"Term {index + 1}: {UNRENDERABLE_NOTE}.")
    return labels, notes


def _draw_presence(
    ax: Axes,
    present: list[set[_TermKey]],
    keys: list[_TermKey],
    labels: list[str],
    empty: list[bool],
) -> None:
    values = np.array([[key in generation for generation in present] for key in keys])
    if not keys:
        values = np.zeros((1, len(present)), dtype=bool)
        labels = ["No active fitted terms"]
    mask = np.broadcast_to(empty, values.shape)
    cmap = ListedColormap(["white", _PRESENT])
    cmap.set_bad(_EMPTY)
    ax.imshow(
        np.ma.array(values, mask=mask),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=1,
    )
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylabel("Fitted term by target")
    ax.legend(
        handles=[
            Patch(facecolor="white", edgecolor="gray", label="Absent"),
            Patch(facecolor=_PRESENT, label="Present"),
            Patch(facecolor=_EMPTY, label="No eligible candidates"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncols=3,
        fontsize=10,
    )


def _selected_rows(
    present: list[set[_TermKey]], max_terms: int | None
) -> tuple[list[_TermKey], list[str], list[str]]:
    counts = Counter(key for generation in present for key in generation)
    keys = sorted(counts, key=lambda key: (-counts[key], key))
    shown = keys if max_terms is None else keys[:max_terms]
    labels, notes = _term_labels(shown)
    if len(shown) < len(keys):
        notes.append(
            f"Showing {len(shown)} of {len(keys)} terms; "
            f"{len(keys) - len(shown)} omitted from this figure."
        )
    return shown, labels, notes


def plot_term_presence(
    result: ExperimentResult,
    *,
    max_terms: int | None = 40,
    style: dict[str, Any] | None = None,
) -> tuple[Figure, list[str]]:
    if max_terms is not None and (
        isinstance(max_terms, bool) or not isinstance(max_terms, int) or max_terms < 1
    ):
        raise ValueError("max_terms must be a positive integer or None")
    history = result.search_trajectory()
    present = _term_history(history)
    shown, labels, notes = _selected_rows(present, max_terms)
    if not shown:
        notes.append(
            "No recorded search trajectory"
            if not history
            else "No eligible candidates"
            if not any(history)
            else "No active fitted terms"
        )
    sketch_note = _sketch_fit_note(result)
    if sketch_note is not None:
        notes.append(sketch_note)
    with style_context(style):
        figure, ax = _figure(max(4.0, 0.27 * len(shown) + 1.8), notes)
        if history:
            _draw_presence(ax, present, shown, labels, [not rows for rows in history])
        else:
            ax.text(
                0.5,
                0.5,
                "No recorded search trajectory",
                ha="center",
                transform=ax.transAxes,
            )
        ax.set_xlabel("Recorded iteration")
        integer_ticks(ax)
        ax.set_title("Active fitted term presence\nTop-k distinct structures")
    return figure, notes


def _draw_score_quantiles(
    ax: Axes, history: list[list[SearchTrajectoryCandidate]]
) -> None:
    quantiles = np.full((5, len(history)), np.nan)
    for iteration, generation in enumerate(history):
        if generation:
            quantiles[:, iteration] = np.quantile(
                [candidate.score for candidate in generation], [0, 0.25, 0.5, 0.75, 1]
            )
    iterations = np.arange(len(history))
    for low, high, label, alpha in [(0, 4, "Min–max", 0.12), (1, 3, "IQR", 0.3)]:
        ax.fill_between(
            iterations,
            quantiles[low],
            quantiles[high],
            color=_PRESENT,
            alpha=alpha,
            label=label,
        )
    ax.plot(
        iterations,
        quantiles[2],
        color=_PRESENT,
        marker=".",
        markersize=3,
        label="Median",
    )
    ax.legend()


def plot_search_score_distribution(
    result: ExperimentResult,
    *,
    style: dict[str, Any] | None = None,
) -> tuple[Figure, list[str]]:
    history = result.search_trajectory()
    notes = []
    if not any(history):
        notes.append(
            "No recorded search trajectory" if not history else "No eligible candidates"
        )
    sketch_note = _sketch_fit_note(result)
    if sketch_note is not None:
        notes.append(sketch_note)
    with style_context(style):
        figure, ax = _figure(4.5, notes)
        if any(history):
            _draw_score_quantiles(ax, history)
        else:
            ax.text(0.5, 0.5, notes[0], ha="center", transform=ax.transAxes)
        better = "lower" if result.score_direction == "min" else "higher"
        ax.set_ylabel(f"{result.score_kind} ({better} is better)")
        ax.set_xlabel("Recorded iteration")
        integer_ticks(ax)
        ax.set_title(
            "Native score distribution\n"
            "Top-k distinct structures; best representative per structure"
        )
    return figure, notes


__all__ = ["plot_term_presence", "plot_search_score_distribution"]
