
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from kd.core.equation._sketch_terms import lower_term
from kd.core.equation.signature import law_term_key
from kd.core.small_terms import small_term_indices


@dataclass(frozen=True)
class TermVoteRow:

    law_key: str
    term: str
    count: int
    support_indices: tuple[int, ...]


@dataclass(frozen=True)
class TermVoteResult:

    selected: tuple[str, ...]
    rows: tuple[TermVoteRow, ...]
    min_count: int
    n_supports: int


def vote_terms(
    supports: Sequence[Sequence[str]],
    *,
    min_count: int,
    coefficients: Sequence[Sequence[float]] | None = None,
    small_term_rtol: float = 0.0,
) -> TermVoteResult:
    _check_selection(len(supports), min_count, coefficients, small_term_rtol)
    first_spelling: dict[str, str] = {}
    contributors: dict[str, list[int]] = {}
    for index, support in enumerate(supports):
        support_coefficients = None if coefficients is None else coefficients[index]
        for key, term in _counted_terms(
            index, support, support_coefficients, small_term_rtol
        ):
            first_spelling.setdefault(key, term)
            contributors.setdefault(key, []).append(index)
    rows = tuple(
        TermVoteRow(
            law_key=key,
            term=first_spelling[key],
            count=len(indices),
            support_indices=tuple(indices),
        )
        for key, indices in contributors.items()
    )
    return TermVoteResult(
        selected=tuple(row.term for row in rows if row.count >= min_count),
        rows=rows,
        min_count=min_count,
        n_supports=len(supports),
    )


def _check_selection(
    n_supports: int,
    min_count: int,
    coefficients: Sequence[Sequence[float]] | None,
    small_term_rtol: float,
) -> None:
    if not 1 <= min_count <= n_supports:
        raise ValueError(
            f"min_count must be between 1 and the number of supports "
            f"({n_supports}); got {min_count}"
        )
    if not 0.0 <= small_term_rtol < 1.0:
        raise ValueError(f"small_term_rtol must be in [0, 1); got {small_term_rtol}")
    if coefficients is None:
        if small_term_rtol > 0.0:
            raise ValueError("small_term_rtol > 0 requires coefficients")
        return
    if len(coefficients) != n_supports:
        raise ValueError(
            f"coefficients has {len(coefficients)} entries for {n_supports} supports"
        )


def _counted_terms(
    index: int,
    support: Sequence[str],
    coefficients: Sequence[float] | None,
    small_term_rtol: float,
) -> list[tuple[str, str]]:
    keys = [_term_key(index, term) for term in support]
    if coefficients is None:
        kept: Sequence[int] = range(len(support))
    else:
        kept = _kept_indices(index, support, coefficients, small_term_rtol)
    counted: dict[str, str] = {}
    for position in kept:
        counted.setdefault(keys[position], support[position])
    return list(counted.items())


def _term_key(index: int, term: str) -> str:
    try:
        return law_term_key(lower_term(term))
    except ValueError as exc:
        raise ValueError(
            f"supports[{index}] term {term!r} is not valid IR: {exc}"
        ) from exc


def _kept_indices(
    index: int,
    support: Sequence[str],
    coefficients: Sequence[float],
    small_term_rtol: float,
) -> list[int]:
    if len(coefficients) != len(support):
        raise ValueError(
            f"supports[{index}] has {len(support)} terms and "
            f"{len(coefficients)} coefficients"
        )
    values = [float(value) for value in coefficients]
    for term, value in zip(support, values, strict=True):
        if not math.isfinite(value):
            raise ValueError(
                f"supports[{index}] term {term!r} has a non-finite coefficient {value}"
            )
    nonzero = [position for position, value in enumerate(values) if value != 0.0]
    return small_term_indices(values, nonzero, small_term_rtol)


__all__ = ["TermVoteResult", "TermVoteRow", "vote_terms"]
