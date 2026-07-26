
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias, TypeVar


class Form(Enum):

    EVOLUTION = "EVOLUTION"
    HOMOGENEOUS = "HOMOGENEOUS"
    PARAMETRIC = "PARAMETRIC"
    WEAK = "WEAK"


@dataclass(frozen=True)
class Scalar:

    value: float


@dataclass(frozen=True)
class Field:

    expr: str

    def __post_init__(self) -> None:
        raise NotImplementedError("Field coefficients are reserved for step 4")


@dataclass(frozen=True)
class Hole:

    def __post_init__(self) -> None:
        raise NotImplementedError("Hole coefficients are reserved for step 6")


@dataclass(frozen=True)
class Posterior:

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        raise NotImplementedError("Posterior coefficients are reserved for step 6")


Coefficient: TypeAlias = Scalar | Field | Hole | Posterior
Term: TypeAlias = tuple[str, Coefficient]
AttrValue: TypeAlias = str | int | float | bool | None
_FoldState = TypeVar("_FoldState")


@dataclass(frozen=True)
class LhsSpec:

    field: str
    axis: str
    order: int


@dataclass(frozen=True)
class EquationAttrs:

    units: dict[str, str] | None = None
    provenance: str | None = None
    health: dict[str, AttrValue] | None = None


@dataclass(frozen=True)
class Evolution:

    lhs_spec: LhsSpec
    terms: tuple[Term, ...]
    attrs: EquationAttrs
    active_indices: tuple[int, ...] | None = None

    @property
    def form(self) -> Form:
        return Form.EVOLUTION

    def __post_init__(self) -> None:
        assert isinstance(self.lhs_spec, LhsSpec)
        assert isinstance(self.terms, tuple)
        assert isinstance(self.attrs, EquationAttrs)
        assert self.active_indices is None or isinstance(self.active_indices, tuple)
        for term in self.terms:
            assert isinstance(term, tuple)
            assert len(term) == 2
            term_ir, coefficient = term
            assert isinstance(term_ir, str)
            assert isinstance(coefficient, (Scalar, Field, Hole, Posterior))


@dataclass(frozen=True)
class Homogeneous:

    terms: tuple[Term, ...]
    attrs: EquationAttrs
    active_indices: tuple[int, ...] | None = None

    @property
    def form(self) -> Form:
        return Form.HOMOGENEOUS

    def __post_init__(self) -> None:
        assert isinstance(self.terms, tuple)
        assert isinstance(self.attrs, EquationAttrs)
        assert self.active_indices is None or isinstance(self.active_indices, tuple)
        for term in self.terms:
            assert isinstance(term, tuple)
            assert len(term) == 2
            term_ir, coefficient = term
            assert isinstance(term_ir, str)
            assert isinstance(coefficient, (Scalar, Field, Hole, Posterior))








Equation: TypeAlias = Evolution | Homogeneous


def fold_terms(
    terms: Iterable[Term],
    initial: _FoldState,
    step: Callable[[_FoldState, Term], _FoldState],
) -> _FoldState:
    state = initial
    for term in terms:
        state = step(state, term)
    return state
