
from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from kd.core.equation.canonical import (
    canonicalize_expression,
    skeletonize_constants,
)
from kd.core.equation.types import (
    Equation,
    Evolution,
    Form,
    Homogeneous,
    LhsSpec,
    Regression,
    Term,
    fold_terms,
)


@dataclass(frozen=True)
class StructureFingerprint:

    form: Form
    lhs_spec: LhsSpec | None
    terms: frozenset[str]


@dataclass(frozen=True)
class TermDiff:

    added: frozenset[str]
    removed: frozenset[str]
    common: frozenset[str]
    lhs_changed: bool
    form_changed: bool


def structure(eq: Equation) -> StructureFingerprint:
    match eq:
        case Evolution():
            initial: tuple[int, frozenset[str]] = (0, frozenset())
            _index, terms = fold_terms(
                eq.terms,
                initial,
                _collect_canonical_term,
            )
            return StructureFingerprint(
                form=eq.form,
                lhs_spec=eq.lhs_spec,
                terms=terms,
            )
        case Homogeneous():
            initial = (0, frozenset())
            _index, terms = fold_terms(
                eq.terms,
                initial,
                _collect_canonical_term,
            )
            return StructureFingerprint(
                form=eq.form,
                lhs_spec=None,
                terms=terms,
            )
        case Regression():
            initial = (0, frozenset())
            _index, terms = fold_terms(
                eq.terms,
                initial,
                _collect_skeletonized_term,
            )
            return StructureFingerprint(
                form=eq.form,
                lhs_spec=eq.lhs_spec,
                terms=terms,
            )
    assert_never(eq)


def term_diff(old: Equation, new: Equation) -> TermDiff:
    old_fingerprint = structure(old)
    new_fingerprint = structure(new)
    common = old_fingerprint.terms & new_fingerprint.terms
    return TermDiff(
        added=new_fingerprint.terms - old_fingerprint.terms,
        removed=old_fingerprint.terms - new_fingerprint.terms,
        common=common,
        lhs_changed=old_fingerprint.lhs_spec != new_fingerprint.lhs_spec,
        form_changed=old_fingerprint.form != new_fingerprint.form,
    )


def _collect_canonical_term(
    state: tuple[int, frozenset[str]], term: Term
) -> tuple[int, frozenset[str]]:
    index, terms = state
    term_ir, _coefficient = term
    try:
        canonical_term = canonicalize_expression(term_ir)
    except ValueError as err:
        raise ValueError(
            f"Cannot canonicalize term at index {index}: {term_ir}"
        ) from err
    return index + 1, terms | {canonical_term}


def _collect_skeletonized_term(
    state: tuple[int, frozenset[str]], term: Term
) -> tuple[int, frozenset[str]]:
    index, terms = state
    term_ir, _coefficient = term
    try:
        canonical_term = canonicalize_expression(skeletonize_constants(term_ir))
    except ValueError as err:
        raise ValueError(
            f"Cannot canonicalize term at index {index}: {term_ir}"
        ) from err
    return index + 1, terms | {canonical_term}


__all__ = ["StructureFingerprint", "TermDiff", "structure", "term_diff"]
