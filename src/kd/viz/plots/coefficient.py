
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
from kd.viz.style import style_context

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_NO_DATA_TEXT = "No data"
_BAR_ALPHA = 0.7
_GT_BAR_ALPHA = 0.4







_WIDE_RANGE_RATIO = 100.0
_VALUE_LABEL_FONTSIZE = 7


def plot_coefficient_bar(
    result: ExperimentResult,
    ax: Axes | None = None,
    *,
    ground_truth: Tensor | None = None,
    style: dict[str, Any] | None = None,
) -> list[str]:
    ax = _get_axes(ax, style)
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


    x = np.arange(len(labels))
    if ground_truth is not None:
        bar_width = 0.35
    else:


        bar_width = max(0.35, min(0.55, 2.0 / len(labels)))

    with style_context(style):

        discovered_bars = ax.bar(
            x,
            display_coeffs,
            bar_width,
            label="Published"
            if result.config.get("sketch") is not None
            else "Discovered",
            color="steelblue",
            alpha=_BAR_ALPHA,
        )
        _mark_na_positions(ax, x[~finite_mask])






        equation = result.equation
        if (
            equation is not None
            and equation.form is Form.HOMOGENEOUS
            and (selected is None or (len(selected) > 0 and selected[0] == 0))
            and len(display_coeffs) > 0
            and display_coeffs[0] == 1.0
        ):
            ax.annotate(
                "normalization\nbaseline (fixed = 1)",
                xy=(float(x[0]), 1.0),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=_VALUE_LABEL_FONTSIZE,
                color="dimgray",
            )







        if ground_truth is not None:
            gt_np = np.array(ground_truth.detach().cpu().numpy(), dtype=np.float64)
            gt_display: np.ndarray | None
            if equation is not None and len(gt_np) == len(equation.terms):
                active = equation.active_indices
                gt_display = (
                    gt_np
                    if active is None
                    else gt_np[np.asarray(sorted(active), dtype=int)]
                )
            elif selected is not None and len(gt_np) == len(terms):
                gt_display = gt_np[np.asarray(selected)]
            elif len(gt_np) == len(display_coeffs):
                gt_display = gt_np
            else:
                gt_display = None
                catalog_size = (
                    len(equation.terms) if equation is not None else len(terms)
                )
                warnings.append(
                    f"Ground truth length {len(gt_np)} matches neither the full "
                    f"term library ({catalog_size}) nor the displayed terms "
                    f"({len(display_coeffs)}); skipping ground-truth overlay"
                )

            if gt_display is not None:


                gt_finite = np.isfinite(gt_display)
                n_gt_nonfinite = int(np.sum(~gt_finite))
                if n_gt_nonfinite > 0:
                    warnings.append(
                        f"Non-finite ground-truth coefficient(s) for "
                        f"{n_gt_nonfinite}/{len(gt_display)} displayed terms; "
                        "shown as N/A"
                    )
                    gt_display = np.where(gt_finite, gt_display, np.nan)
                ax.bar(
                    x + bar_width,
                    gt_display,
                    bar_width,
                    label="Ground Truth",
                    color="firebrick",
                    alpha=_GT_BAR_ALPHA,
                )
                _mark_na_positions(ax, (x + bar_width)[~gt_finite])

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Coefficient")
        ax.set_title("Coefficients")
        ax.legend()




        if _spans_orders_of_magnitude(display_coeffs):
            linthresh = 0.5 * _min_nonzero_abs(display_coeffs)
            ax.set_yscale("symlog", linthresh=linthresh)
            ax.set_ylabel("Coefficient (symlog)")
            _annotate_bar_values(ax, discovered_bars, display_coeffs)

    return warnings


def _mark_na_positions(ax: Axes, positions: np.ndarray) -> None:
    for x_pos in positions:
        ax.text(
            float(x_pos),
            0.02,
            "N/A",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="gray",
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
        above = value >= 0.0
        ax.annotate(
            f"{value:.3g}",
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 3 if above else -3),
            textcoords="offset points",
            ha="center",
            va="bottom" if above else "top",
            fontsize=_VALUE_LABEL_FONTSIZE,
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
