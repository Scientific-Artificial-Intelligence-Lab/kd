
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from kd.core.evaluator import EvaluationResult
from kd.search.discover.ir.conversion import ir_to_tree
from kd.search.discover.tokens.library import Library

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

logger = logging.getLogger(__name__)


def extract_active_terms(
    result: EvaluationResult,
) -> tuple[list[str], list[float]]:
    if result.terms is None or result.coefficients is None:
        return [], []
    terms = result.terms
    coefficients = result.coefficients.tolist()
    if result.selected_indices is not None:
        terms = [terms[i] for i in result.selected_indices]
        coefficients = [coefficients[i] for i in result.selected_indices]
    return terms, coefficients


@dataclass(frozen=True, slots=True)
class CandidateSnapshot:

    expression: str
    reward: float
    nmse: float
    n_nodes: int
    terms: list[str]


class CycleCandidateTracker:

    def __init__(self, library: Library, capacity: int) -> None:
        self._library = library
        self._capacity = capacity
        self._candidates: list[CandidateSnapshot] = []
        self._seen_expressions: set[str] = set()

    def reset(self) -> None:
        self._candidates = []
        self._seen_expressions = set()

    @property
    def snapshots(self) -> list[CandidateSnapshot]:
        return list(self._candidates)

    def record(
        self,
        unique_irs: Sequence[str],
        unique_rewards: FloatArray,
        unique_eval_valid_mask: BoolArray,
        unique_results: list[EvaluationResult] | None,
    ) -> None:
        if self._capacity == 0 or unique_results is None:
            return
        for idx in np.flatnonzero(unique_eval_valid_mask):
            expression = unique_irs[int(idx)]
            if expression in self._seen_expressions:
                continue
            snapshot = self._make_snapshot(
                expression,
                float(unique_rewards[int(idx)]),
                unique_results[int(idx)],
            )
            if snapshot is None:
                continue
            self._seen_expressions.add(expression)
            self._candidates.append(snapshot)
        self._candidates.sort(key=lambda candidate: candidate.reward, reverse=True)
        del self._candidates[self._capacity:]

    def _make_snapshot(
        self,
        expression: str,
        reward: float,
        result: EvaluationResult,
    ) -> CandidateSnapshot | None:
        if result.terms is None:
            return None
        try:
            n_nodes = ir_to_tree(expression, self._library).n_nodes()
        except ValueError:
            logger.warning(
                "Skipping cycle candidate with unparsable expression '%s'",
                expression,
            )
            return None
        return CandidateSnapshot(
            expression=expression,
            reward=reward,
            nmse=result.nmse,
            n_nodes=n_nodes,
            terms=list(result.terms),
        )


__all__ = [
    "CandidateSnapshot",
    "CycleCandidateTracker",
    "extract_active_terms",
]
