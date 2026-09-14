
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import sympy

from kd.core.equation import Homogeneous
from kd.core.expr import from_sympy, structure_term_key
from kd.core.expr.sympy_bridge import to_latex, to_sympy
from kd.viz._result_data import _equation_data, _equation_lhs, _sketch_fit_note
from kd.viz.equation_display import RENDER_ERRORS, latex_display
from kd.viz.plots.equation import plot_equation
from kd.viz.style import style_context

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from sympy import Expr

    from kd.data.schema import PDEDataset
    from kd.search.result import ExperimentResult


def _coefficients(expression: Expr) -> dict[str, float]:
    coefficients: dict[str, float] = {}
    for term in sympy.Add.make_args(sympy.expand(expression)):
        coefficient, basis = term.as_coeff_Mul()
        value = float(coefficient)
        if not math.isfinite(value):
            raise ValueError("non-finite coefficient")
        term_ir = from_sympy(basis)
        key = structure_term_key("one" if basis == 1 else term_ir)
        coefficients[key] = coefficients.get(key, 0.0) + value
    return coefficients


def _published_coefficients(result: ExperimentResult) -> dict[str, float]:
    terms, values, selected = _equation_data(result)
    if terms is None or values is None:
        raise ValueError("no coefficient data")
    if len(terms) != len(values):
        raise ValueError("term/coefficient length mismatch")
    indices = range(len(terms)) if selected is None else selected
    coefficients: dict[str, float] = {}
    for index in indices:
        for key, value in _coefficients(values[index] * to_sympy(terms[index])).items():
            coefficients[key] = coefficients.get(key, 0.0) + value
    return coefficients


def _truth_coefficients(
    truth: str, result: ExperimentResult, published: dict[str, float]
) -> tuple[dict[str, float], str | None]:
    parts = truth.split("=")
    if len(parts) != 2:
        raise ValueError("expected one equation with '='")
    left, right = (to_sympy(part.strip().replace("^", "**")) for part in parts)
    residual = sympy.expand(left - right)
    if isinstance(result.equation, Homogeneous):
        coefficients = _coefficients(residual)
        if not published:
            raise ValueError("no published normalization term")
        pivot = next(iter(published))
        if pivot not in coefficients or coefficients[pivot] == 0.0:
            raise ValueError("ground truth lacks the published normalization term")
        scale = published[pivot] / coefficients[pivot]
        return (
            {key: value * scale for key, value in coefficients.items()},
            "Homogeneous comparison uses the published normalization.",
        )
    target = to_sympy(_equation_lhs(result))
    factor = residual.coeff(target)
    remainder = residual - factor * target
    if not factor.is_number or factor == 0 or remainder.has(target):
        raise ValueError(
            "ground truth does not have the published target as a linear LHS"
        )
    return _coefficients(-remainder / factor), None


def _comparison_rows(
    result: ExperimentResult, truth: str
) -> tuple[list[list[str]], str | None]:
    published = _published_coefficients(result)
    expected, note = _truth_coefficients(truth, result, published)
    keys = list(dict.fromkeys([*published, *expected]))
    rows = [
        [
            f"${to_latex(key)}$",
            f"{published.get(key, 0.0):.6g}",
            f"{expected.get(key, 0.0):.6g}",
        ]
        for key in keys
    ]
    return rows, note


def _draw_comparison(ax: Axes, rows: list[list[str]], note: str | None) -> None:
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Term", "Published", "Ground truth"],
        cellLoc="center",
        loc="center",
        colWidths=[0.5, 0.25, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#d8dee9")
        if row == 0:
            cell.set_facecolor("#e8eef6")
            cell.set_text_props(weight="bold")
    if note is not None:
        ax.text(0.5, 0.02, note, transform=ax.transAxes, ha="center", fontsize=9)


def plot_equation_card(
    result: ExperimentResult,
    dataset: PDEDataset,
    *,
    style: dict[str, Any] | None = None,
) -> tuple[Figure, list[str]]:
    notes: list[str] = []
    rows: list[list[str]] = []
    normalization = None
    if (
        dataset.ground_truth is not None
        and not latex_display(result, label="card").degraded
    ):
        try:
            rows, normalization = _comparison_rows(result, dataset.ground_truth)
        except RENDER_ERRORS as exc:
            notes.append(f"Ground-truth comparison unavailable: {exc}")
    with style_context(style):
        return _draw_card(result, rows, normalization, notes, style)


def _draw_card(
    result: ExperimentResult,
    rows: list[list[str]],
    normalization: str | None,
    notes: list[str],
    style: dict[str, Any] | None,
) -> tuple[Figure, list[str]]:
    fig, axes = plt.subplots(
        2 if rows else 1,
        1,
        figsize=(10, 3.2 + 0.4 * len(rows)),
        squeeze=False,
        gridspec_kw={"height_ratios": [1, 1.5]} if rows else None,
        layout="constrained",
    )
    ax = axes[0, 0]
    notes.extend(plot_equation(result, ax, style=style))
    metric = result.final_eval.nmse
    badge = f"Fit NMSE: {metric:.4g}" if math.isfinite(metric) else "Fit NMSE: N/A"
    ax.text(0.99, 1.0, badge, transform=ax.transAxes, ha="right", va="top", fontsize=10)
    fit_note = _sketch_fit_note(result)
    if fit_note is not None:
        notes.append(fit_note)
    if notes:
        ax.text(
            0.5,
            0.0,
            "\n".join(notes),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            wrap=True,
        )
    if rows:
        _draw_comparison(axes[1, 0], rows, normalization)
    return fig, notes
