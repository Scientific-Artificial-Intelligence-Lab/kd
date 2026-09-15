
from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING

from kd.core.equation import Evolution, LhsSpec, render_lhs_label
from kd.core.expr.registry import PROTECTED_OPERATORS
from kd.viz._result_data import _NO_SKETCH_SOLUTION, _equation_data

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kd.core.integrator import IntegrationResult
    from kd.data.schema import PDEDataset
    from kd.search.result import ExperimentResult

__all__ = ["build_integration_result"]








_PROTECTED_SEMANTICS_NOTE_TEMPLATE = (
    "Protected-operator note: the integrated RHS contains {operators}, which "
    "the platform evaluates with protected semantics (safe_*/clamped "
    "wrappers) — the same semantics under which the equation was scored "
    "during search. Trajectories that would diverge under bare operators "
    "may remain bounded."
)









_INTEGRATION_PRUNE_RTOL = 1e-12


def _prune_near_zero_terms(
    terms: Sequence[str],
    coefficients: Sequence[float],
    active: list[int],
) -> tuple[list[int], list[str]]:
    finite_magnitudes = [
        abs(coefficients[i]) for i in active if math.isfinite(coefficients[i])
    ]
    if not finite_magnitudes:
        return active, []
    threshold = _INTEGRATION_PRUNE_RTOL * max(finite_magnitudes)
    dropped = [
        i
        for i in active
        if math.isfinite(coefficients[i]) and abs(coefficients[i]) < threshold
    ]
    if not dropped:
        return active, []
    keep = [i for i in active if i not in set(dropped)]
    detail = ", ".join(f"'{terms[i]}' (coeff {coefficients[i]:.3g})" for i in dropped)
    note = (
        f"Near-zero term(s) excluded from time integration: {detail} "
        f"(|coeff| < {_INTEGRATION_PRUNE_RTOL:g} * max|coeff|); "
        "reported equation and metrics keep the full term list."
    )
    return keep, [note]


def _assemble_integration_rhs(
    terms: Sequence[str],
    coefficients: Sequence[float],
    keep: Sequence[int],
) -> str:
    non_finite = [i for i in keep if not math.isfinite(coefficients[i])]
    if non_finite:
        detail = ", ".join(
            f"term '{terms[i]}' has coefficient {coefficients[i]!r}" for i in non_finite
        )
        raise ValueError(
            f"Cannot assemble integration RHS: non-finite coefficient(s) — "
            f"{detail}. Coefficients must be finite; a NaN/Inf here points "
            "at a degenerate upstream fit/refit, not at the RHS terms."
        )
    survivors = [i for i in keep if coefficients[i] != 0.0]
    if not survivors:
        return "0"
    return " + ".join(f"({coefficients[i]!r})*({terms[i]})" for i in survivors)


def _protected_semantics_note(rhs: str) -> str | None:
    try:
        tree = ast.parse(rhs, mode="eval")
    except (SyntaxError, ValueError):
        return None
    found = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in PROTECTED_OPERATORS
    }
    if not found:
        return None
    return _PROTECTED_SEMANTICS_NOTE_TEMPLATE.format(operators=", ".join(sorted(found)))


def _integration_lhs_warning(
    result: ExperimentResult,
    dataset: PDEDataset,
) -> str | None:
    equation = result.equation
    expected = LhsSpec(dataset.lhs_field, dataset.lhs_axis, 1)
    if equation is not None:
        if not isinstance(equation, Evolution):
            return (
                "Time integration requires an EVOLUTION equation, "
                f"got {equation.form.value}"
            )
        if equation.lhs_spec == expected:
            return None
        actual_label = render_lhs_label(equation.lhs_spec)
    else:
        actual_label = result.lhs_label
        if actual_label == render_lhs_label(expected):
            return None
    return (
        "Time integration supports only first-order evolution with the "
        f"dataset's field and axis ({render_lhs_label(expected)}); "
        f"the result LHS is {actual_label}."
    )


def _boundary_notes(dataset: PDEDataset) -> list[str]:
    assert dataset.axes is not None
    notes = []
    for name in dataset.spatial_axes:
        if dataset.axes[name].is_periodic:
            notes.append(f"Integration boundary on '{name}': periodic.")
        else:
            notes.append(
                f"Integration boundary on '{name}': both endpoints are fixed "
                "at their initial values. Time-dependent observed boundary "
                "values are not used; Neumann and Robin boundaries are not supported."
            )
    return notes


def build_integration_result(
    result: ExperimentResult,
    dataset: PDEDataset,
) -> tuple[IntegrationResult, list[str]]:
    from kd.core.integrator import IntegrationResult, integrate_pde

    if result.equation is None and result.config.get("sketch") is not None:
        return IntegrationResult(success=False, warning=_NO_SKETCH_SOLUTION), []
    terms, coeffs, selected = _equation_data(result)
    if terms is None or coeffs is None:
        return (
            IntegrationResult(
                success=False,
                warning="Missing terms or coefficients in final_eval",
            ),
            [],
        )
    coeff_values = [float(c) for c in coeffs]
    active = list(selected) if selected is not None else list(range(len(terms)))
    keep, notes = _prune_near_zero_terms(terms, coeff_values, active)
    rhs = _assemble_integration_rhs(terms, coeff_values, keep)
    protected_note = _protected_semantics_note(rhs)
    if protected_note is not None:
        notes = [*notes, protected_note]
    try:
        lhs_warning = _integration_lhs_warning(result, dataset)
        if lhs_warning is not None:
            return IntegrationResult(success=False, warning=lhs_warning), notes
        integrated = integrate_pde(rhs, dataset)
        if integrated.predicted_field is not None:
            notes.extend(_boundary_notes(dataset))
        return integrated, notes
    except Exception as exc:
        return (
            IntegrationResult(
                success=False,
                warning=f"Integration failed: {exc}",
            ),
            notes,
        )
