
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from torch import Tensor

from kd.core.equation import Form
from kd.core.expr.sympy_bridge import to_latex
from kd.viz._result_data import _NO_SKETCH_SOLUTION, _equation_data
from kd.viz.axes import _get_axes
from kd.viz.equation_display import RENDER_ERRORS, UNRENDERABLE_MARKER
from kd.viz.style import (
    AXIS,
    FIGSIZE_SINGLE,
    INK_2,
    INK_MUTED,
    SERIES,
    style_context,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_NO_DATA_TEXT = "No data"
_BAR_HEIGHT = 0.42

_PAIR_HEIGHT = 0.36
_GT_HEIGHT = 0.18
_PAIR_GAP = 0.04
_SLOT_INCHES = 0.5
_FRAME_INCHES = 1.4







_WIDE_RANGE_RATIO = 100.0
_VALUE_LABEL_FONTSIZE = 9.5


def coefficient_bar_figsize(result: ExperimentResult) -> tuple[float, float]:
    terms, coefficients, selected = _equation_data(result)
    n_terms = 0 if terms is None else len(terms if selected is None else selected)
    height = _FRAME_INCHES + _SLOT_INCHES * max(n_terms, 1)
    return FIGSIZE_SINGLE[0], min(height, 2 * FIGSIZE_SINGLE[1])


def plot_coefficient_bar(
    result: ExperimentResult,
    ax: Axes | None = None,
    *,
    ground_truth: Tensor | None = None,
    style: dict[str, Any] | None = None,
) -> list[str]:
    ax = _get_axes(ax, style, figsize=coefficient_bar_figsize(result))
    warnings: list[str] = []

    terms, coefficients, selected = _equation_data(result)


    if terms is None or coefficients is None:
        warnings.append("No coefficient data available")
        if result.config.get("sketch") is not None:
            warnings.append(_NO_SKETCH_SOLUTION)
        with style_context(style):
            ax.set_title("Coefficients")
            ax.text(
                0.5,
                0.5,
                _NO_DATA_TEXT,
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        return warnings


    coeff_np = np.array(coefficients, dtype=np.float64)


    if selected is not None:
        display_terms = [terms[i] for i in selected]
        display_coeffs = coeff_np[selected]
    else:
        display_terms = list(terms)
        display_coeffs = coeff_np

    if len(display_terms) == 0:
        warnings.append("No terms to display")
        with style_context(style):
            ax.set_title("Coefficients")
            ax.text(
                0.5,
                0.5,
                _NO_DATA_TEXT,
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        return warnings





    finite_mask = np.isfinite(display_coeffs)
    n_nonfinite = int(np.sum(~finite_mask))
    if n_nonfinite > 0:
        warnings.append(
            f"Non-finite coefficient(s) for {n_nonfinite}/{len(display_coeffs)} "
            "displayed terms; shown as N/A"
        )
        display_coeffs = np.where(finite_mask, display_coeffs, np.nan)


    labels = _make_labels(display_terms)
    y = np.arange(len(labels), dtype=np.float64)
    gt_display = (
        None
        if ground_truth is None
        else _aligned_ground_truth(
            ground_truth, result, terms, selected, display_coeffs, warnings
        )
    )
    paired = gt_display is not None
    height = _PAIR_HEIGHT if paired else _BAR_HEIGHT
    bar_y = y - (_GT_HEIGHT + _PAIR_GAP) / 2 if paired else y

    with style_context(style):
        discovered_bars = ax.barh(
            bar_y,
            display_coeffs,
            height,
            label="Published"
            if result.config.get("sketch") is not None
            else "Discovered",
            color=SERIES[0],
        )
        _mark_na_positions(ax, bar_y[~finite_mask])
        _mark_normalization_baseline(ax, result, selected, display_coeffs, bar_y)
        if gt_display is not None:
            gt_y = bar_y + (height + _GT_HEIGHT) / 2 + _PAIR_GAP
            _draw_ground_truth(ax, gt_display, gt_y, warnings)
        _finish_axes(ax, y, labels, legend=paired)




        if _spans_orders_of_magnitude(display_coeffs):
            linthresh = 0.5 * _min_nonzero_abs(display_coeffs)
            ax.set_xscale("symlog", linthresh=linthresh)
            ax.set_xlabel("Coefficient (symlog)")
            _annotate_bar_values(ax, discovered_bars, display_coeffs)

    return warnings


def _mark_normalization_baseline(
    ax: Axes,
    result: ExperimentResult,
    selected: list[int] | None,
    display_coeffs: np.ndarray,
    bar_y: np.ndarray,
) -> None:
    equation = result.equation
    if not (
        equation is not None
        and equation.form is Form.HOMOGENEOUS
        and (selected is None or (len(selected) > 0 and selected[0] == 0))
        and len(display_coeffs) > 0
        and display_coeffs[0] == 1.0
    ):
        return
    ax.annotate(
        "normalization baseline (fixed = 1)",
        xy=(1.0, float(bar_y[0])),
        xytext=(8, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=_VALUE_LABEL_FONTSIZE,
        color=INK_MUTED,
    )


def _aligned_ground_truth(
    ground_truth: Tensor,
    result: ExperimentResult,
    terms: list[str],
    selected: list[int] | None,
    display_coeffs: np.ndarray,
    warnings: list[str],
) -> np.ndarray | None:
    gt_np = np.array(ground_truth.detach().cpu().numpy(), dtype=np.float64)
    equation = result.equation
    aligned: np.ndarray
    if equation is not None and len(gt_np) == len(equation.terms):
        active = equation.active_indices
        if active is None:
            return gt_np
        aligned = gt_np[np.asarray(sorted(active), dtype=int)]
        return aligned
    if selected is not None and len(gt_np) == len(terms):
        aligned = gt_np[np.asarray(selected)]
        return aligned
    if len(gt_np) == len(display_coeffs):
        return gt_np
    catalog_size = len(equation.terms) if equation is not None else len(terms)
    warnings.append(
        f"Ground truth length {len(gt_np)} matches neither the full "
        f"term library ({catalog_size}) nor the displayed terms "
        f"({len(display_coeffs)}); skipping ground-truth overlay"
    )
    return None


def _draw_ground_truth(
    ax: Axes, gt_display: np.ndarray, gt_y: np.ndarray, warnings: list[str]
) -> None:
    gt_finite = np.isfinite(gt_display)
    n_gt_nonfinite = int(np.sum(~gt_finite))
    if n_gt_nonfinite > 0:
        warnings.append(
            f"Non-finite ground-truth coefficient(s) for "
            f"{n_gt_nonfinite}/{len(gt_display)} displayed terms; "
            "shown as N/A"
        )
    ax.barh(
        gt_y,
        np.where(gt_finite, gt_display, np.nan),
        _GT_HEIGHT,
        label="Ground truth",
        color=SERIES[1],
    )
    _mark_na_positions(ax, gt_y[~gt_finite])


def _finish_axes(ax: Axes, y: np.ndarray, labels: list[str], *, legend: bool) -> None:
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(float(y[-1]) + 0.6, float(y[0]) - 0.6)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.axvline(0.0, color=AXIS, linewidth=0.8, zorder=0)
    ax.margins(x=0.12)
    ax.set_xlabel("Coefficient")
    ax.set_title("Coefficients")
    if legend:
        ax.legend(loc="best")


def _mark_na_positions(ax: Axes, positions: np.ndarray) -> None:
    for y_pos in positions:
        ax.text(
            0.02,
            float(y_pos),
            "N/A",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=8,
            color=INK_MUTED,
        )


def _min_nonzero_abs(coeffs: np.ndarray) -> float:
    mags = np.abs(coeffs[np.isfinite(coeffs)])
    nonzero = mags[mags > 0.0]
    return float(nonzero.min()) if nonzero.size else 0.0


def _spans_orders_of_magnitude(coeffs: np.ndarray) -> bool:
    mags = np.abs(coeffs[np.isfinite(coeffs)])
    nonzero = mags[mags > 0.0]
    if nonzero.size < 2:
        return False
    return bool(nonzero.max() / nonzero.min() > _WIDE_RANGE_RATIO)


def _annotate_bar_values(ax: Axes, bars: object, coeffs: np.ndarray) -> None:
    from matplotlib.container import BarContainer

    if not isinstance(bars, BarContainer):
        return
    for bar, value in zip(bars, coeffs, strict=True):
        if not np.isfinite(value):
            continue
        positive = value >= 0.0
        ax.annotate(
            f"{value:.3g}",
            xy=(value, bar.get_y() + bar.get_height() / 2.0),
            xytext=(4 if positive else -4, 0),
            textcoords="offset points",
            ha="left" if positive else "right",
            va="center",
            fontsize=_VALUE_LABEL_FONTSIZE,
            color=INK_2,
        )


def _make_labels(terms: list[str]) -> list[str]:
    labels: list[str] = []
    for term in terms:
        try:
            labels.append(f"${to_latex(term)}$")
        except RENDER_ERRORS:
            logger.exception("Term not renderable as display math; raw IR: %s", term)
            labels.append(UNRENDERABLE_MARKER)
    return labels
