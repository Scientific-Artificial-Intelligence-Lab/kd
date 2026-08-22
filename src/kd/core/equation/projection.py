
from __future__ import annotations

from dataclasses import replace
from typing import assert_never

from kd.core.equation.types import (
    Equation,
    EquationAttrs,
    Evolution,
    Homogeneous,
    Regression,
)

_PROVENANCE_NOTE = "active_law"


def _validated_ascending(indices: tuple[int, ...], n_terms: int) -> tuple[int, ...]:
    if len(indices) == 0:
        raise ValueError("active_indices is empty: an empty active set is not a law")
    seen: set[int] = set()
    for index in indices:
        if not 0 <= index < n_terms:
            raise ValueError(
                f"active_indices entry {index} out of range for {n_terms} terms"
            )
        if index in seen:
            raise ValueError(f"duplicate active_indices entry {index}")
        seen.add(index)
    return tuple(sorted(indices))


def _projected_attrs(attrs: EquationAttrs) -> EquationAttrs:
    provenance = (
        _PROVENANCE_NOTE
        if attrs.provenance is None
        else f"{attrs.provenance} | {_PROVENANCE_NOTE}"
    )
    return replace(attrs, provenance=provenance)


def active_law(eq: Equation) -> Equation:
    if eq.active_indices is None:
        return eq
    indices = _validated_ascending(eq.active_indices, len(eq.terms))
    terms = tuple(eq.terms[index] for index in indices)
    match eq:
        case Evolution():
            return replace(
                eq, terms=terms, active_indices=None, attrs=_projected_attrs(eq.attrs)
            )
        case Homogeneous():
            if indices[0] != 0:
                raise ValueError(
                    "homogeneous active_indices must include the pivot (index 0)"
                )
            return replace(
                eq, terms=terms, active_indices=None, attrs=_projected_attrs(eq.attrs)
            )
        case Regression():
            return replace(
                eq, terms=terms, active_indices=None, attrs=_projected_attrs(eq.attrs)
            )
    assert_never(eq)
