
from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from kd.core.equation import LhsSpec, Sketch, render_lhs_label
from kd.core.equation._sketch_terms import lower_term
from kd.core.equation.signature import law_term_entry
from kd.core.equation.sketch import PinnedTerm
from kd.core.equation.sketch_builder import dataset_vocabulary
from kd.core.equation.sketch_input import SketchSpec
from kd.core.evaluator import EvaluationResult, Evaluator
from kd.core.expr.jet import Monomial, jet_monomials
from kd.core.linear_solve import r2_score
from kd.core.metrics import aic, nmse
from kd.core.platform.builder import lower_target, resolve_lhs_defaults

if TYPE_CHECKING:
    from torch import Tensor

    from kd.core.expr.term_features import TermVocabulary
    from kd.data.schema import PDEDataset
    from kd.search.protocol import PlatformComponents


@dataclass(frozen=True)
class FixedConstraint:

    pins: tuple[tuple[str, float], ...]
    law: dict[str, float]
    monomials: tuple[frozenset[Monomial], ...]
    vocabulary: TermVocabulary


@dataclass(frozen=True)
class FixedPlacement:

    appended: tuple[str, ...]
    values: dict[str, float]
    positions: frozenset[int]
    rejected: dict[str, str]


def resolve_constraint(
    dataset: PDEDataset,
    *,
    fixed: Mapping[str, float] | None,
    sketch: Sketch | SketchSpec | None,
    lhs_order: int,
) -> FixedConstraint | None:
    if fixed is None and sketch is None:
        return None
    if fixed is not None and sketch is not None:
        raise ValueError(
            "pass fixed or sketch, not both: a sketch's pinned terms are its "
            "fixed coefficients"
        )
    resolved = resolve_lhs_defaults(dataset)
    lhs = LhsSpec(resolved.lhs_field, resolved.lhs_axis, lhs_order)
    if fixed is not None:
        pins = _fixed_pins(fixed)
    else:
        pins = _sketch_pins(cast("Sketch | SketchSpec", sketch), resolved, lhs)
    if not pins:
        return None
    vocabulary = dataset_vocabulary(resolved)
    law, monomials = _checked_pins(pins, lhs, vocabulary)
    return FixedConstraint(tuple(pins), law, monomials, vocabulary)


def _fixed_pins(fixed: Mapping[str, float]) -> list[tuple[str, float]]:
    pins = []
    for term, value in fixed.items():
        try:
            pin = PinnedTerm(lower_term(term), value)
        except ValueError as exc:
            raise ValueError(f"fixed[{term!r}]: {exc}") from exc
        pins.append((term, pin.value))
    return pins


def _sketch_pins(
    sketch: Sketch | SketchSpec, resolved: PDEDataset, lhs: LhsSpec
) -> list[tuple[str, float]]:
    if sketch.lhs_spec is not None and sketch.lhs_spec != lhs:
        raise ValueError(
            f"the sketch's LHS {render_lhs_label(sketch.lhs_spec)} is not this "
            f"refit's LHS {render_lhs_label(lhs)}"
        )
    if isinstance(sketch, SketchSpec):
        view = dataclasses.replace(resolved, lhs_order=lhs.order)
        sketch = sketch.resolve(view)
    return [(pin.term_ir, pin.value) for pin in sketch.pinned]


def _checked_pins(
    pins: Sequence[tuple[str, float]], lhs: LhsSpec, vocabulary: TermVocabulary
) -> tuple[dict[str, float], tuple[frozenset[Monomial], ...]]:
    label = render_lhs_label(lhs)
    lhs_monomials = jet_monomials(label, vocabulary)
    law: dict[str, float] = {}
    owners: dict[str, str] = {}
    monomials: list[frozenset[Monomial]] = []
    for term, value in pins:
        key, folded = law_term_entry(lower_term(term), value)
        if key in law:
            raise ValueError(
                f"fixed terms {owners[key]!r} and {term!r} name one law term {key!r}"
            )
        expansion = _expansion(term, key, vocabulary)
        if expansion & lhs_monomials:
            raise ValueError(
                f"fixed term {term!r} is the LHS {label} up to spelling and "
                "sign; fixing it would state lhs = lhs"
            )
        for (other, _), seen in zip(pins, monomials, strict=False):
            shared = sorted(str(monomial) for monomial in expansion & seen)
            if shared:
                raise ValueError(
                    f"fixed terms {other!r} and {term!r} are alias spellings of "
                    f"one column (shared monomials {shared}); fix it once"
                )
        law[key], owners[key] = folded, term
        monomials.append(expansion)
    return law, tuple(monomials)


def _expansion(term: str, key: str, vocabulary: TermVocabulary) -> frozenset[Monomial]:
    try:
        return jet_monomials(key, vocabulary)
    except ValueError as exc:
        raise ValueError(
            f"fixed term {term!r} has no jet expansion to check it against the "
            f"LHS and the other terms: {exc}"
        ) from exc


def place_fixed(constraint: FixedConstraint, terms: Sequence[str]) -> FixedPlacement:
    identities, rejected = _identities(terms, constraint.vocabulary)
    keys = list(constraint.law)
    matched: dict[int, float] = {}
    appended: list[tuple[str, float]] = []
    for (term, value), key in zip(constraint.pins, keys, strict=True):
        positions = [i for i, identity in identities.items() if identity[0] == key]
        if len(positions) > 1:
            spellings = [terms[i] for i in positions]
            for i in positions:
                rejected[terms[i]] = (
                    f"fixed: the fixed term {term!r} matches the input terms "
                    f"{spellings}; one fixed value cannot be split between them"
                )
        elif positions:
            matched[positions[0]] = constraint.law[key] * identities[positions[0]][1]
        else:
            appended.append((term, value))
    for i, (key, _, expansion) in identities.items():
        if key in constraint.law:
            continue
        for (term, _), seen in zip(constraint.pins, constraint.monomials, strict=True):
            shared = sorted(str(monomial) for monomial in expansion & seen)
            if shared:
                rejected[terms[i]] = (
                    f"fixed: alias spelling of the fixed term {term!r} (shared "
                    f"monomials {shared}); a fixed column is never refitted"
                )
    values = {terms[i]: matched[i] for i in sorted(matched)}
    values.update(appended)
    return FixedPlacement(
        appended=tuple(term for term, _ in appended),
        values=values,
        positions=frozenset(matched),
        rejected=rejected,
    )


def _identities(
    terms: Sequence[str], vocabulary: TermVocabulary
) -> tuple[dict[int, tuple[str, float, frozenset[Monomial]]], dict[str, str]]:
    identities: dict[int, tuple[str, float, frozenset[Monomial]]] = {}
    rejected: dict[str, str] = {}
    for i, term in enumerate(terms):
        try:
            key, sign = law_term_entry(lower_term(term), 1.0)
            identities[i] = (key, sign, jet_monomials(key, vocabulary))
        except ValueError as exc:
            rejected[term] = (
                "fixed: no law identity or jet expansion to compare with the "
                f"fixed terms: {exc}"
            )
    return identities, rejected


def lowered_components(
    components: PlatformComponents, fixed: Mapping[str, float]
) -> PlatformComponents:
    base = cast("Evaluator", components.evaluator)
    pins = [(lower_term(term), value) for term, value in fixed.items()]
    target = lower_target(base.lhs_target, pins, base.executor, base.context)
    evaluator = Evaluator(
        base.executor, base.solver, base.context, target, report_condition_number=True
    )
    return dataclasses.replace(components, evaluator=evaluator)


def merge_fixed(
    fit: EvaluationResult,
    terms: Sequence[str],
    fixed: Mapping[str, float],
    *,
    base: Evaluator,
    rows: Tensor | None,
) -> EvaluationResult:
    assert fit.coefficients is not None
    free = iter(fit.coefficients.tolist())
    values = [fixed[term] if term in fixed else next(free) for term in terms]
    measured = measure_fixed(base, terms, values, rows=rows, complexity=fit.complexity)
    return dataclasses.replace(measured, condition_number=fit.condition_number)


def measure_fixed(
    base: Evaluator,
    terms: Sequence[str],
    values: Sequence[float],
    *,
    rows: Tensor | None,
    complexity: int,
) -> EvaluationResult:
    theta, _ = base.build_theta_matrix([lower_term(t) for t in terms], rows=rows)
    coefficients = torch.tensor(values, dtype=theta.dtype, device=theta.device)
    target = selected_target(base, rows)
    prediction = theta @ coefficients
    residuals = prediction - target
    mse = float((residuals**2).mean().item())
    return EvaluationResult(
        mse=mse,
        nmse=nmse(mse, float(target.var(correction=0).item())),
        r2=r2_score(prediction, target),
        score=aic(mse, complexity, target.numel()),
        complexity=complexity,
        coefficients=coefficients,
        residuals=residuals.detach(),
        terms=list(terms),
        condition_number=None,
    )


def selected_target(evaluator: Evaluator, rows: Tensor | None) -> Tensor:
    target = evaluator.lhs_target
    if rows is None:
        return target
    indices = rows.flatten().nonzero().flatten() if rows.dtype == torch.bool else rows
    return target.index_select(0, indices.to(target.device))


__all__ = [
    "FixedConstraint",
    "FixedPlacement",
    "lowered_components",
    "measure_fixed",
    "merge_fixed",
    "place_fixed",
    "resolve_constraint",
    "selected_target",
]
