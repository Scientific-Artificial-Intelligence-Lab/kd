
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from kd.core.expr.jet import Monomial, jet_polynomial, monomial_label
from kd.core.expr.naming import parse_compound_derivative
from kd.core.expr.term_features import TermVocabulary
from kd.core.recovery import term_set_jaccard
from kd.core.small_terms import small_term_indices

__all__ = ["ParsedLaw", "TermVerdict", "judge_law", "law_from_equation"]

_Polynomial = dict[Monomial, float]


@dataclass(frozen=True)
class ParsedLaw:

    lhs: str | None
    terms: tuple[str, ...]
    coefficients: tuple[float, ...]


@dataclass(frozen=True)
class TermVerdict:

    structure_ok: bool
    coefficients_ok: bool
    law: dict[str, float]
    truth: dict[str, float]
    trimmed: dict[str, float]
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    relative_errors: dict[str, float]
    max_relative_error: float | None
    jaccard: float
    small_term_rtol: float
    coeff_rtol: float
    scale_free: bool
    gauge: str | None
    coefficients_reason: str | None

    @property
    def overall(self) -> bool:
        return self.structure_ok and self.coefficients_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "structure_ok": self.structure_ok,
            "coefficients_ok": self.coefficients_ok,
            "law": dict(self.law),
            "truth": dict(self.truth),
            "trimmed": dict(self.trimmed),
            "missing": list(self.missing),
            "extra": list(self.extra),
            "relative_errors": dict(self.relative_errors),
            "max_relative_error": self.max_relative_error,
            "jaccard": self.jaccard,
            "small_term_rtol": self.small_term_rtol,
            "coeff_rtol": self.coeff_rtol,
            "scale_free": self.scale_free,
            "gauge": self.gauge,
            "coefficients_reason": self.coefficients_reason,
        }


def _vocabulary(axes: Sequence[str], fields: Sequence[str]) -> TermVocabulary:
    return TermVocabulary(fields=frozenset(fields), coordinates=frozenset(axes))


def _labelled(polynomial: _Polynomial) -> dict[str, float]:
    return dict(
        sorted(
            (monomial_label(monomial), value) for monomial, value in polynomial.items()
        )
    )


def _derivative_name(
    left: str, axes: Sequence[str], fields: Sequence[str]
) -> str | None:
    ode = {f"d{field}_d{axis}": f"{field}_{axis}" for field in fields for axis in axes}
    name = ode.get(left, left)
    derivative = parse_compound_derivative(
        name, known_fields=set(fields), known_axes=set(axes)
    )
    return None if derivative is None else name


def law_from_equation(
    equation: str, *, axes: Sequence[str], fields: Sequence[str] = ("u",)
) -> ParsedLaw:
    if equation.count("=") != 1:
        raise ValueError(f"A truth equation needs exactly one '=', got {equation!r}")
    left, right = (side.strip() for side in equation.split("="))
    if right == "0":
        lhs, expression = None, left
    else:
        lhs, expression = _derivative_name(left, axes, fields), right
        if lhs is None:
            raise ValueError(
                f"The left side {left!r} of {equation!r} is not a derivative of "
                f"a field in {sorted(fields)}"
            )
    polynomial = jet_polynomial(
        expression.replace("^", "**"), _vocabulary(axes, fields)
    )
    labelled = _labelled(polynomial)
    return ParsedLaw(
        lhs=lhs, terms=tuple(labelled), coefficients=tuple(labelled.values())
    )


def _side(
    name: str,
    terms: Sequence[str],
    coefficients: Sequence[float],
    vocabulary: TermVocabulary,
) -> _Polynomial:
    if len(terms) != len(coefficients):
        raise ValueError(
            f"{name} has {len(coefficients)} coefficients for {len(terms)} terms"
        )
    total: _Polynomial = {}
    for term, coefficient in zip(terms, coefficients, strict=True):
        if not math.isfinite(coefficient):
            raise ValueError(
                f"{name} coefficient {coefficient!r} of {term!r} is not finite"
            )
        for monomial, value in jet_polynomial(term, vocabulary).items():
            total[monomial] = total.get(monomial, 0.0) + coefficient * value



    return {monomial: value for monomial, value in total.items() if value != 0.0}


def _trim(
    candidate: _Polynomial, small_term_rtol: float
) -> tuple[_Polynomial, _Polynomial]:
    monomials = list(candidate)
    values = [candidate[monomial] for monomial in monomials]
    kept = set(small_term_indices(values, range(len(values)), small_term_rtol))
    support = {m: candidate[m] for i, m in enumerate(monomials) if i in kept}
    small = {m: candidate[m] for i, m in enumerate(monomials) if i not in kept}
    return support, small


def _gauged(
    law: _Polynomial, truth: _Polynomial
) -> tuple[_Polynomial, _Polynomial, Monomial | None]:
    shared = law.keys() & truth.keys()
    if not shared:
        return law, truth, None
    gauge = min(shared, key=monomial_label)
    law_scale, truth_scale = law[gauge], truth[gauge]
    return (
        {monomial: value / law_scale for monomial, value in law.items()},
        {monomial: value / truth_scale for monomial, value in truth.items()},
        gauge,
    )


def judge_law(
    terms: Sequence[str],
    coefficients: Sequence[float],
    truth_terms: Sequence[str],
    truth_coefficients: Sequence[float],
    *,
    axes: Sequence[str],
    fields: Sequence[str] = ("u",),
    small_term_rtol: float,
    coeff_rtol: float,
    scale_free: bool = False,
) -> TermVerdict:
    vocabulary = _vocabulary(axes, fields)
    candidate = _side("candidate", terms, coefficients, vocabulary)
    truth = _side("truth", truth_terms, truth_coefficients, vocabulary)
    law, trimmed = _trim(candidate, small_term_rtol)
    gauge: Monomial | None = None
    if scale_free:
        law, truth, gauge = _gauged(law, truth)
    coefficients_reason = "no_shared_monomial" if scale_free and gauge is None else None
    relative_errors = _labelled(
        {m: abs(law[m] - truth[m]) / abs(truth[m]) for m in law.keys() & truth.keys()}
    )
    missing = tuple(sorted(monomial_label(m) for m in truth.keys() - law.keys()))
    extra = tuple(sorted(monomial_label(m) for m in law.keys() - truth.keys()))
    law_labels, truth_labels = _labelled(law), _labelled(truth)
    return TermVerdict(
        structure_ok=not missing and not extra,
        coefficients_ok=coefficients_reason is None
        and not missing
        and all(error <= coeff_rtol for error in relative_errors.values()),
        law=law_labels,
        truth=truth_labels,
        trimmed=_labelled(trimmed),
        missing=missing,
        extra=extra,
        relative_errors=relative_errors,
        max_relative_error=max(relative_errors.values(), default=None),
        jaccard=term_set_jaccard(list(law_labels), list(truth_labels)),
        small_term_rtol=small_term_rtol,
        coeff_rtol=coeff_rtol,
        scale_free=scale_free,
        gauge=None if gauge is None else monomial_label(gauge),
        coefficients_reason=coefficients_reason,
    )
