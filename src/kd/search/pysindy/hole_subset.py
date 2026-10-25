
from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Final

import numpy as np

from kd.core.metrics import aic_no_n




HOLE_SUBSET_MAX: Final = 3
_TIE_RTOL: Final = 1e-12


@dataclass(frozen=True, kw_only=True)
class HoleSubsetChoice:

    indices: tuple[int, ...]
    coefficients: np.ndarray
    score: float
    candidates: int


def _check_term_bounds(min_terms: int, max_terms: int, *, has_anchors: bool) -> None:
    if max_terms < min_terms:
        raise ValueError(f"max_terms={max_terms} is below min_terms={min_terms}")
    if max_terms > HOLE_SUBSET_MAX:
        raise ValueError(
            f"max_terms={max_terms} exceeds HOLE_SUBSET_MAX={HOLE_SUBSET_MAX}"
        )
    if max_terms < 1 and not has_anchors:
        raise ValueError(
            "max_terms must be at least 1 without anchors: the zero law would "
            "be the only candidate"
        )


def _candidate_columns(
    free: list[int], anchors: tuple[int, ...], min_terms: int, max_terms: int
) -> Iterator[tuple[int, ...]]:
    for size in range(min_terms, max_terms + 1):
        for subset in combinations(free, size):
            yield tuple(sorted(anchors + subset))


def _check_columns(
    theta: np.ndarray, anchors: tuple[int, ...], free: list[int], min_terms: int
) -> None:
    if len(free) < min_terms:
        raise ValueError(
            f"hole subset selection has {len(free)} candidate columns outside "
            f"the anchors for holes that need at least {min_terms} terms"
        )
    fitted = (*anchors, *free)
    zero = [index for index in fitted if not np.linalg.norm(theta[:, index]) > 0]
    if zero:
        raise ValueError(
            f"hole subset selection needs non-zero columns; zero columns at {zero}"
        )


def _fit_columns(
    theta: np.ndarray, y: np.ndarray, columns: tuple[int, ...]
) -> tuple[float, np.ndarray]:
    coefficients = np.zeros(theta.shape[1], dtype=np.float64)
    if not columns:
        return float(np.mean(y**2)), coefficients
    block = theta[:, list(columns)]
    scales = np.linalg.norm(block, axis=0)
    unit = np.linalg.lstsq(block / scales, y, rcond=None)[0]
    coefficients[list(columns)] = unit / scales
    residual = y - block @ coefficients[list(columns)]
    return float(np.mean(residual**2)), coefficients


def select_hole_subset(
    theta: np.ndarray,
    y: np.ndarray,
    *,
    anchor_indices: Sequence[int],
    min_terms: int,
    max_terms: int,
    candidate_indices: Sequence[int] | None = None,
    ratio: float = 1.0,
) -> HoleSubsetChoice:
    anchors = tuple(sorted(anchor_indices))
    _check_term_bounds(min_terms, max_terms, has_anchors=bool(anchors))
    if not np.isfinite(y).all():
        raise ValueError("hole subset selection needs a finite target y")
    if candidate_indices is None:
        candidate_indices = range(theta.shape[1])
    free = sorted(index for index in candidate_indices if index not in anchors)
    _check_columns(theta, anchors, free, min_terms)
    best: tuple[tuple[int, ...], np.ndarray, float] | None = None
    candidates = 0
    for columns in _candidate_columns(free, anchors, min_terms, max_terms):
        candidates += 1
        mse, coefficients = _fit_columns(theta, y, columns)
        score = aic_no_n(mse, k=len(columns), ratio=ratio)
        if best is None or (
            score < best[2] and not math.isclose(score, best[2], rel_tol=_TIE_RTOL)
        ):
            best = (columns, coefficients, score)
    if best is None or not math.isfinite(best[2]):
        raise ValueError(
            f"hole subset selection found no finite aic_no_n score among "
            f"{candidates} candidate subsets"
        )
    columns, coefficients, score = best
    return HoleSubsetChoice(
        indices=columns, coefficients=coefficients, score=score, candidates=candidates
    )


__all__ = ["HOLE_SUBSET_MAX", "HoleSubsetChoice", "select_hole_subset"]
