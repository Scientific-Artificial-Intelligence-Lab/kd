
from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Protocol

import numpy as np

from kd.search.discover.evaluation.reward import DEFAULT_ALPHA, compute_reward

if TYPE_CHECKING:
    from kd.core.evaluator import (
        EvaluationResult,
    )



_NEG_PREFIX = "neg("
_NEG_SUFFIX = ")"


class _SparseEvaluator(Protocol):

    def evaluate_expression(self, expr: str) -> EvaluationResult: ...

    def evaluate_terms(
        self, terms: list[str], *, skip_invalid: bool = False,
    ) -> EvaluationResult: ...


@dataclasses.dataclass(frozen=True, slots=True)
class SparseRefitResult:

    sparse_terms: list[str]
    sparse_coefficients: list[float]
    sparse_nmse: float
    sparse_reward: float
    n_dropped: int
    n_kept: int
    eps_relative: float
    abs_threshold: float
    alpha: float


def _strip_neg(term: str) -> tuple[str, int]:
    if term.startswith(_NEG_PREFIX) and term.endswith(_NEG_SUFFIX):
        return term[len(_NEG_PREFIX):-len(_NEG_SUFFIX)], -1
    return term, 1


def consolidate_signed_duplicates(
    terms: list[str], coefficients: list[float],
) -> tuple[list[str], list[float]]:
    if len(terms) != len(coefficients):
        raise ValueError(
            f"terms ({len(terms)}) and coefficients ({len(coefficients)}) "
            f"must have equal length",
        )


    canonical_index: dict[str, int] = {}
    unique_terms: list[str] = []
    net_coefs: list[float] = []



    for term, coef in zip(terms, coefficients, strict=True):
        canonical, sign = _strip_neg(term)
        if canonical in canonical_index:
            net_coefs[canonical_index[canonical]] += sign * coef
        else:
            canonical_index[canonical] = len(unique_terms)
            unique_terms.append(canonical)
            net_coefs.append(sign * coef)
    return unique_terms, net_coefs


def select_kept_indices(
    coefficients: list[float] | np.ndarray, eps_relative: float,
) -> tuple[list[int], list[int], float]:
    if not math.isfinite(eps_relative):
        raise ValueError(
            f"eps_relative must be finite; got {eps_relative}",
        )
    if eps_relative < 0.0 or eps_relative >= 1.0:
        raise ValueError(
            f"eps_relative must be in [0, 1); got {eps_relative}",
        )

    coefs_abs = np.abs(np.asarray(coefficients, dtype=float))
    if eps_relative == 0.0 or coefs_abs.size == 0:
        return list(range(coefs_abs.size)), [], 0.0

    max_abs = float(coefs_abs.max())
    if max_abs == 0.0:

        return list(range(coefs_abs.size)), [], 0.0

    abs_threshold = eps_relative * max_abs
    kept = [i for i, c in enumerate(coefs_abs) if c >= abs_threshold]
    dropped = [i for i, c in enumerate(coefs_abs) if c < abs_threshold]
    return kept, dropped, abs_threshold


def _build_invalid_result(eps_relative: float, alpha: float) -> SparseRefitResult:
    return SparseRefitResult(
        sparse_terms=[],
        sparse_coefficients=[],
        sparse_nmse=float("inf"),
        sparse_reward=0.0,
        n_dropped=0,
        n_kept=0,
        eps_relative=eps_relative,
        abs_threshold=0.0,
        alpha=alpha,
    )


def _build_passthrough_result(
    terms: list[str],
    coefficients: list[float],
    nmse: float,
    reward: float,
    eps_relative: float,
    abs_threshold: float,
    alpha: float,
) -> SparseRefitResult:
    return SparseRefitResult(
        sparse_terms=list(terms),
        sparse_coefficients=list(coefficients),
        sparse_nmse=nmse,
        sparse_reward=reward,
        n_dropped=0,
        n_kept=len(terms),
        eps_relative=eps_relative,
        abs_threshold=abs_threshold,
        alpha=alpha,
    )


def refit_candidate(
    expression: str,
    evaluator: _SparseEvaluator,
    eps_relative: float,
    alpha: float = DEFAULT_ALPHA,
) -> SparseRefitResult:
    initial = evaluator.evaluate_expression(expression)
    if not initial.is_valid:
        return _build_invalid_result(eps_relative, alpha)







    assert initial.terms is not None
    assert initial.coefficients is not None
    initial_terms = list(initial.terms)




    initial_coefs = [float(c) for c in initial.coefficients]
    initial_reward = compute_reward(initial, alpha)



    consolidated_terms, consolidated_coefs = consolidate_signed_duplicates(
        initial_terms, initial_coefs,
    )

    if eps_relative == 0.0:

        return _build_passthrough_result(
            initial_terms, initial_coefs, float(initial.nmse),
            initial_reward, eps_relative, 0.0, alpha,
        )

    kept, dropped, abs_threshold = select_kept_indices(
        consolidated_coefs, eps_relative,
    )



    consolidation_reduced = len(consolidated_terms) < len(initial_terms)
    nothing_dropped = len(dropped) == 0
    if nothing_dropped and not consolidation_reduced:
        return _build_passthrough_result(
            initial_terms, initial_coefs, float(initial.nmse),
            initial_reward, eps_relative, abs_threshold, alpha,
        )

    if not kept:

        return _build_passthrough_result(
            initial_terms, initial_coefs, float(initial.nmse),
            initial_reward, eps_relative, abs_threshold, alpha,
        )

    kept_terms = [consolidated_terms[i] for i in kept]
    refit = evaluator.evaluate_terms(kept_terms)
    if not refit.is_valid:

        return _build_passthrough_result(
            initial_terms, initial_coefs, float(initial.nmse),
            initial_reward, eps_relative, abs_threshold, alpha,
        )

    assert refit.terms is not None
    assert refit.coefficients is not None
    sparse_reward = compute_reward(refit, alpha)
    return SparseRefitResult(
        sparse_terms=list(refit.terms),

        sparse_coefficients=[float(c) for c in refit.coefficients],
        sparse_nmse=float(refit.nmse),
        sparse_reward=sparse_reward,



        n_dropped=len(initial_terms) - len(kept),
        n_kept=len(kept),
        eps_relative=eps_relative,
        abs_threshold=abs_threshold,
        alpha=alpha,
    )
