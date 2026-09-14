
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from kd.core.expr.term_key import structure_term_key
from kd.search.series_keys import SEARCH_TRAJECTORY_KEYS

if TYPE_CHECKING:
    from torch import Tensor

    from kd.search.recorder import VizRecorder


@dataclass(frozen=True)
class SearchTrajectoryCandidate:

    expression: str
    score: float
    terms: tuple[str, ...]
    lhs: str


def active_fit_terms(
    terms: Sequence[str],
    coefficients: Tensor,
    selected_indices: Sequence[int] | None,
) -> tuple[str, ...]:
    values = coefficients.detach().reshape(-1).tolist()
    if len(terms) != len(values):
        raise ValueError("search trajectory terms and coefficients must align")
    indices = range(len(terms)) if selected_indices is None else selected_indices
    return tuple(terms[index] for index in indices if values[index] != 0.0)


def select_trajectory_candidates(
    scores: Sequence[float | None],
    project: Callable[[int], tuple[str, tuple[str, ...], str]],
    *,
    top_k: int,
    direction: Literal["min", "max"],
) -> list[SearchTrajectoryCandidate]:
    if top_k == 0:
        return []
    ranked = [
        (float(score), i)
        for i, score in enumerate(scores)
        if score is not None and math.isfinite(score)
    ]
    ranked.sort(key=lambda item: item[0], reverse=direction == "max")
    selected: list[SearchTrajectoryCandidate] = []
    seen: set[tuple[str, frozenset[str]]] = set()
    for score, index in ranked:
        expression, terms, lhs = project(index)
        key = (lhs, frozenset(structure_term_key(term) for term in terms))
        if key in seen:
            continue
        seen.add(key)
        selected.append(SearchTrajectoryCandidate(expression, score, terms, lhs))
        if len(selected) == top_k:
            break
    return selected


def log_search_trajectory(
    recorder: VizRecorder, candidates: Sequence[SearchTrajectoryCandidate]
) -> None:
    if not recorder.enabled or recorder.trajectory_top_k == 0:
        return
    columns = (
        [candidate.expression for candidate in candidates],
        [candidate.score for candidate in candidates],
        [list(candidate.terms) for candidate in candidates],
        [candidate.lhs for candidate in candidates],
    )
    for key, column in zip(SEARCH_TRAJECTORY_KEYS, columns, strict=True):
        recorder.log(key, column)


def read_search_trajectory(
    recorder: VizRecorder,
) -> list[list[SearchTrajectoryCandidate]]:
    present = set(SEARCH_TRAJECTORY_KEYS) & recorder.keys()
    if not present:
        return []
    if present != set(SEARCH_TRAJECTORY_KEYS):
        raise ValueError("search trajectory requires all four recorder columns")
    columns = [recorder.get(key) for key in SEARCH_TRAJECTORY_KEYS]
    if len({len(column) for column in columns}) != 1:
        raise ValueError("search trajectory columns must have equal iteration counts")
    return [_read_iteration(row) for row in zip(*columns, strict=True)]


def _read_iteration(columns: tuple[Any, ...]) -> list[SearchTrajectoryCandidate]:
    if not all(isinstance(column, list) for column in columns):
        raise ValueError("search trajectory iteration columns must be lists")
    if len({len(column) for column in columns}) != 1:
        raise ValueError("search trajectory iteration columns must have equal lengths")
    return [_read_candidate(*values) for values in zip(*columns, strict=True)]


def _read_candidate(
    expression: Any, score: Any, terms: Any, lhs: Any
) -> SearchTrajectoryCandidate:
    if not isinstance(expression, str) or not isinstance(lhs, str):
        raise ValueError("search trajectory expression and lhs must be strings")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(score)
    ):
        raise ValueError("search trajectory scores must be finite numbers")
    if not isinstance(terms, list) or not all(isinstance(term, str) for term in terms):
        raise ValueError("search trajectory terms must be lists of IR strings")
    return SearchTrajectoryCandidate(expression, float(score), tuple(terms), lhs)
