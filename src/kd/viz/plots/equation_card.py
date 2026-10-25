
from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import sympy

from kd.core.equation import Homogeneous, Regression
from kd.core.expr import (
    TermVocabulary,
    from_sympy,
    jet_polynomial,
    monomial_label,
    structure_term_key,
)
from kd.core.expr.executor import DIFF_OPERATOR_PATTERN, UNITY_TOKEN
from kd.core.expr.naming import parse_compound_derivative
from kd.core.expr.sympy_bridge import to_latex, to_sympy
from kd.core.law_judge import judge_law, law_from_equation
from kd.viz._result_data import _equation_data, _equation_lhs, _sketch_fit_note
from kd.viz.equation_display import RENDER_ERRORS, latex_display
from kd.viz.plots.equation import plot_equation
from kd.viz.style import FIGSIZE_SINGLE, INK_2, style_context, style_table

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from sympy import Expr

    from kd.data.schema import PDEDataset
    from kd.search.result import ExperimentResult


TermRow = tuple[str, float, float]

_NORMALIZATION_NOTE = "Homogeneous comparison uses the published normalization."
_EXPANSION_NOTE = "Rows list both laws expanded into monomials."


_IDENTIFIER = re.compile(r"\b([A-Za-z_]\w*)\b(\s*\()?")


def _published_terms(result: ExperimentResult) -> tuple[list[str], list[float]]:
    terms, values, selected = _equation_data(result)
    if terms is None or values is None:
        raise ValueError("no coefficient data")
    if len(terms) != len(values):
        raise ValueError("term/coefficient length mismatch")
    indices = range(len(terms)) if selected is None else selected
    return [terms[index] for index in indices], [values[index] for index in indices]


def _law_vocabulary(texts: list[str]) -> TermVocabulary:
    coordinates: set[str] = set()
    fields: set[str] = set()
    bare: set[str] = set()
    for text in texts:
        for name, call in _IDENTIFIER.findall(text):
            if call:
                head = DIFF_OPERATOR_PATTERN.fullmatch(name)
                if head is not None:
                    coordinates.add(head.group(2))
                continue
            derivative = parse_compound_derivative(name)
            if derivative is None:
                bare.add(name)
                continue
            fields.add(derivative[0])
            coordinates.update(axis for axis, _ in derivative[1])
    fields |= bare - coordinates - {UNITY_TOKEN}
    return TermVocabulary(fields=frozenset(fields), coordinates=frozenset(coordinates))


def _truth_terms(
    truth: str, result: ExperimentResult, vocabulary: TermVocabulary
) -> tuple[list[str], list[float]]:
    parsed = law_from_equation(
        truth, axes=sorted(vocabulary.coordinates), fields=sorted(vocabulary.fields)
    )
    terms, coefficients = list(parsed.terms), list(parsed.coefficients)
    if isinstance(result.equation, Homogeneous):
        if parsed.lhs is None:
            return terms, coefficients
        return [parsed.lhs, *terms], [1.0, *(-value for value in coefficients)]
    target = _equation_lhs(result)
    if parsed.lhs is None and target in terms:
        factor = coefficients.pop(terms.index(target))
        terms.remove(target)
        return terms, [-value / factor for value in coefficients]
    if parsed.lhs != target:
        raise ValueError("ground truth does not have the published target as its LHS")
    return terms, coefficients


def _normalized(
    published: dict[str, float],
    truth: dict[str, float],
    terms: list[str],
    vocabulary: TermVocabulary,
) -> dict[str, float]:
    if not terms:
        raise ValueError("no published normalization term")
    pivot = min(map(monomial_label, jet_polynomial(terms[0], vocabulary)))
    if pivot not in published:
        raise ValueError("the published normalization term sums to zero")
    if pivot not in truth:
        raise ValueError("ground truth lacks the published normalization term")
    scale = published[pivot] / truth[pivot]
    return {label: value * scale for label, value in truth.items()}


def _expanded(
    terms: list[str], coefficients: list[float], vocabulary: TermVocabulary
) -> bool:
    for term, coefficient in zip(terms, coefficients, strict=True):
        if coefficient == 0.0:
            continue
        values = list(jet_polynomial(term, vocabulary).values())
        if len(values) != 1 or abs(values[0]) != 1.0:
            return True
    return False


def _jet_comparison(
    result: ExperimentResult, truth: str
) -> tuple[list[TermRow], str | None]:
    terms, coefficients = _published_terms(result)
    vocabulary = _law_vocabulary([*terms, truth, _equation_lhs(result)])
    truth_terms, truth_coefficients = _truth_terms(truth, result, vocabulary)


    verdict = judge_law(
        terms,
        coefficients,
        truth_terms,
        truth_coefficients,
        axes=sorted(vocabulary.coordinates),
        fields=sorted(vocabulary.fields),
        small_term_rtol=0.0,
        coeff_rtol=0.0,
    )
    published, expected = verdict.law, verdict.truth
    notes: list[str] = []
    if isinstance(result.equation, Homogeneous):
        expected = _normalized(published, expected, terms, vocabulary)
        notes.append(_NORMALIZATION_NOTE)
    if _expanded(terms, coefficients, vocabulary):
        notes.append(_EXPANSION_NOTE)
    labels = list(dict.fromkeys([*published, *expected]))
    rows = [(key, published.get(key, 0.0), expected.get(key, 0.0)) for key in labels]
    return rows, " ".join(notes) or None


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


def _column_comparison(result: ExperimentResult, truth: str) -> list[TermRow]:
    published: dict[str, float] = {}
    for term, value in zip(*_published_terms(result), strict=True):
        for key, part in _coefficients(value * to_sympy(term)).items():
            published[key] = published.get(key, 0.0) + part
    parts = truth.split("=")
    if len(parts) != 2:
        raise ValueError("expected one equation with '='")
    left, right = (to_sympy(part.strip()) for part in parts)
    residual = sympy.expand(left - right)
    target = to_sympy(_equation_lhs(result))
    factor = residual.coeff(target)
    remainder = residual - factor * target
    if not factor.is_number or factor == 0 or remainder.has(target):
        raise ValueError(
            "ground truth does not have the published target as a linear LHS"
        )
    expected = _coefficients(-remainder / factor)
    keys = list(dict.fromkeys([*published, *expected]))
    return [(key, published.get(key, 0.0), expected.get(key, 0.0)) for key in keys]


def _term_comparison(
    result: ExperimentResult, truth: str
) -> tuple[list[TermRow], str | None]:
    if isinstance(result.equation, Regression):
        return _column_comparison(result, truth), None
    return _jet_comparison(result, truth)


def _truth_comparison(
    result: ExperimentResult, truth: str | None
) -> tuple[list[TermRow], str | None, list[str]]:
    if truth is None or latex_display(result, label="card").degraded:
        return [], None, []
    try:
        rows, table_note = _term_comparison(result, truth)
    except (*RENDER_ERRORS, SyntaxError) as exc:
        return [], None, [f"Ground-truth comparison unavailable: {exc}"]
    return rows, table_note, []


def _comparison_rows(rows: list[TermRow]) -> list[list[str]]:
    return [
        [f"${to_latex(key)}$", f"{published:.6g}", f"{expected:.6g}"]
        for key, published, expected in rows
    ]


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
    style_table(table)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
    if note is not None:
        ax.text(
            0.5,
            0.02,
            note,
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
            color=INK_2,
        )


def plot_equation_card(
    result: ExperimentResult,
    dataset: PDEDataset,
    *,
    style: dict[str, Any] | None = None,
) -> tuple[Figure, list[str]]:
    terms, table_note, notes = _truth_comparison(result, dataset.ground_truth)
    with style_context(style):
        return _draw_card(result, _comparison_rows(terms), table_note, notes, style)


def _draw_card(
    result: ExperimentResult,
    rows: list[list[str]],
    table_note: str | None,
    notes: list[str],
    style: dict[str, Any] | None,
) -> tuple[Figure, list[str]]:
    fig, axes = plt.subplots(
        2 if rows else 1,
        1,
        figsize=(FIGSIZE_SINGLE[0], 3.2 + 0.4 * len(rows)),
        squeeze=False,
        gridspec_kw={"height_ratios": [1, 1.5]} if rows else None,
        layout="constrained",
    )
    ax = axes[0, 0]
    notes.extend(plot_equation(result, ax, style=style))
    metric = result.final_eval.nmse
    badge = f"Fit NMSE: {metric:.4g}" if math.isfinite(metric) else "Fit NMSE: N/A"
    ax.text(
        0.99,
        1.0,
        badge,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=INK_2,
    )
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
            color=INK_2,
            wrap=True,
        )
    if rows:
        _draw_comparison(axes[1, 0], rows, table_note)
    return fig, notes
