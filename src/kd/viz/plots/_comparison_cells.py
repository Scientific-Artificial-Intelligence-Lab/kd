
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from kd.core.expr.sympy_bridge import format_pde, to_sympy, to_unicode
from kd.viz.equation_display import (
    RENDER_ERRORS as _RENDER_ERRORS,
)
from kd.viz.equation_display import (
    UNRENDERABLE_MARKER as _UNRENDERABLE_MARKER,
)

if TYPE_CHECKING:
    from kd.core.equation import TermDiff
    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_TRUNCATE_SUFFIX = "..."
_DEFAULT_MAX_LEN = 30


_EXPRESSION_MAX_LEN = 48




_STRUCTURE_ONLY_SUFFIX = " (structure only)"


def _truncate(text: str, *, max_len: int = _DEFAULT_MAX_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - len(_TRUNCATE_SUFFIX)] + _TRUNCATE_SUFFIX


def _single_line(pretty: str, flat: Callable[[], str]) -> str:
    return pretty if "\n" not in pretty else flat()


def _render_term_diff_cell(delta: TermDiff, label: str, warnings: list[str]) -> str:
    markers: list[str] = []
    if delta.form_changed:
        markers.append("form!")
    if delta.lhs_changed:
        markers.append("lhs!")
    parts = (
        markers
        + [f"+{term}" for term in sorted(delta.added)]
        + [f"-{term}" for term in sorted(delta.removed)]
    )
    if not parts:
        return "="
    full = " ".join(parts)
    if len(full) <= _DEFAULT_MAX_LEN:
        return full
    warnings.append(f"{label} term diff abbreviated to counts; full diff: {full}")
    return " ".join(
        markers + [f"+{len(delta.added)}", f"-{len(delta.removed)}", "terms"]
    )


def _fitted_equation_text(result: ExperimentResult) -> str | None:
    final_eval = result.final_eval
    if final_eval.terms is None or final_eval.coefficients is None:
        return None
    try:
        formatted = format_pde(
            final_eval.terms,
            final_eval.coefficients,
            lhs=result.lhs_label,
            selected_indices=final_eval.selected_indices,
        )
    except _RENDER_ERRORS:
        logger.exception("Failed to format summary-table equation as full PDE")
        return None


    return _single_line(formatted.unicode, lambda: f"{formatted.lhs} = {formatted.rhs}")


def _structure_text(
    result: ExperimentResult, label: str, warnings: list[str]
) -> str | None:
    code = result.best_expression
    try:
        return _single_line(to_unicode(code), lambda: str(to_sympy(code)))
    except _RENDER_ERRORS:




        logger.exception(
            "Summary-table expression for run %s is not renderable as display "
            "math; raw IR: %s",
            label,
            code,
        )
        warnings.append(
            f"{label} expression is not renderable as display math; "
            f"the raw expression was written to the log"
        )
        return None


def _fit_to_budget(text: str, label: str, warnings: list[str], max_len: int) -> str:
    if len(text) > max_len:
        warnings.append(f"{label} expression truncated; full text: {text}")
    return _truncate(text, max_len=max_len)


def _expression_cell(result: ExperimentResult, label: str, warnings: list[str]) -> str:
    fitted = _fitted_equation_text(result)
    if fitted is not None:
        return _fit_to_budget(fitted, label, warnings, _EXPRESSION_MAX_LEN)

    structure = _structure_text(result, label, warnings)
    if structure is None:
        return _UNRENDERABLE_MARKER
    warnings.append(
        f"{label} has no fitted terms/coefficients to render; the Expression "
        f"cell shows the bare structure (no LHS, coefficients not applied), "
        f"not the fitted equation"
    )
    budget = _EXPRESSION_MAX_LEN - len(_STRUCTURE_ONLY_SUFFIX)
    return _fit_to_budget(structure, label, warnings, budget) + _STRUCTURE_ONLY_SUFFIX
