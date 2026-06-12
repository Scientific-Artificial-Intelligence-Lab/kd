
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from kd.search.discover.engine import CandidateSnapshot

CV_DENOM_EPS = 1e-12
DEFAULT_N_BOOTSTRAP = 100
DEFAULT_N_INNER_BOOTSTRAP = 10
NMSE_DUPLICATE_TOL = 1e-5


@runtime_checkable
class StabilityEvaluator(Protocol):

    @property
    def lhs_target(self) -> Tensor: ...

    def build_theta_matrix(
        self, terms: list[str], *, skip_invalid: bool = ...
    ) -> tuple[Tensor, list[str]]: ...


@dataclass(frozen=True, slots=True)
class StabilityCandidateStats:

    candidate: CandidateSnapshot
    mse: np.ndarray
    cv: np.ndarray
    score: np.ndarray


@dataclass(frozen=True, slots=True)
class StabilitySelectionResult:

    selected: CandidateSnapshot
    candidates: list[StabilityCandidateStats]
    vote_counts: list[int]


def stability_select(
    candidates: Sequence[CandidateSnapshot],
    evaluator: StabilityEvaluator,
    *,
    top_k: int,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    n_inner_bootstrap: int = DEFAULT_N_INNER_BOOTSTRAP,
    rng: np.random.Generator | None = None,
) -> StabilitySelectionResult:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    unique_candidates = _drop_duplicate_candidates(candidates)[:top_k]
    if not unique_candidates:
        raise ValueError("No candidates available for stability selection.")

    generator = rng if rng is not None else np.random.default_rng()
    lhs = _to_numpy_vector(evaluator.lhs_target)
    stats = [
        _collect_candidate_stats(
            candidate,
            evaluator,
            lhs,
            n_bootstrap,
            n_inner_bootstrap,
            generator,
        )
        for candidate in unique_candidates
    ]
    score_matrix = np.stack([candidate.score for candidate in stats], axis=0)
    ranking = np.argsort(score_matrix, axis=0)[0]
    vote_counts = np.bincount(ranking, minlength=len(stats))
    selected_idx = int(np.argmax(vote_counts))
    return StabilitySelectionResult(
        selected=stats[selected_idx].candidate,
        candidates=stats,
        vote_counts=vote_counts.tolist(),
    )


def _drop_duplicate_candidates(
    candidates: Sequence[CandidateSnapshot],
) -> list[CandidateSnapshot]:
    unique: list[CandidateSnapshot] = []
    for candidate in candidates:
        duplicate_idx = _find_duplicate_index(unique, candidate)
        if duplicate_idx is None:
            unique.append(candidate)
            continue
        if candidate.n_nodes < unique[duplicate_idx].n_nodes:
            unique[duplicate_idx] = candidate
    return unique


def _find_duplicate_index(
    unique: Sequence[CandidateSnapshot],
    candidate: CandidateSnapshot,
) -> int | None:
    for idx, existing in enumerate(unique):
        if abs(existing.nmse - candidate.nmse) < NMSE_DUPLICATE_TOL:
            return idx
    return None


def _collect_candidate_stats(
    candidate: CandidateSnapshot,
    evaluator: StabilityEvaluator,
    lhs: np.ndarray,
    n_bootstrap: int,
    n_inner_bootstrap: int,
    rng: np.random.Generator,
) -> StabilityCandidateStats:
    theta_tensor, _ = evaluator.build_theta_matrix(candidate.terms)
    theta = _to_numpy_matrix(theta_tensor)
    mse_values, cv_values = _bootstrap_candidate(
        theta,
        lhs,
        n_bootstrap,
        n_inner_bootstrap,
        rng,
    )
    score = mse_values * cv_values
    return StabilityCandidateStats(
        candidate=candidate,
        mse=mse_values,
        cv=cv_values,
        score=score,
    )


def _bootstrap_candidate(
    theta: np.ndarray,
    lhs: np.ndarray,
    n_bootstrap: int,
    n_inner_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    sample_size = _bootstrap_sample_size(theta.shape[0])
    mse_values = np.empty(n_bootstrap, dtype=np.float64)
    cv_values = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        outer_indices = rng.choice(theta.shape[0], sample_size, replace=True)
        theta_outer = theta[outer_indices]
        lhs_outer = lhs[outer_indices]
        coefficients = _solve_lstsq(theta_outer, lhs_outer)
        prediction = theta_outer @ coefficients
        mse_values[idx] = float(np.mean((lhs_outer - prediction) ** 2))
        cv_values[idx] = _bootstrap_cv(
            theta_outer,
            lhs_outer,
            n_inner_bootstrap,
            rng,
        )
    return mse_values, cv_values


def _bootstrap_cv(
    theta_outer: np.ndarray,
    lhs_outer: np.ndarray,
    n_inner_bootstrap: int,
    rng: np.random.Generator,
) -> float:
    sample_size = theta_outer.shape[0]
    coefficients = np.empty(
        (n_inner_bootstrap, theta_outer.shape[1]),
        dtype=np.float64,
    )
    for idx in range(n_inner_bootstrap):
        inner_indices = rng.choice(theta_outer.shape[0], sample_size, replace=True)
        coefficients[idx] = _solve_lstsq(
            theta_outer[inner_indices],
            lhs_outer[inner_indices],
        )
    means = np.mean(coefficients, axis=0)
    stds = np.std(coefficients, axis=0)
    denom = np.maximum(np.abs(means), CV_DENOM_EPS)
    return float(np.mean(np.abs(stds / denom)))


def _solve_lstsq(
    theta: npt.NDArray[np.float64],
    lhs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:





    rcond = max(theta.shape) * float(np.finfo(np.float64).eps)
    solution = torch.linalg.lstsq(
        torch.from_numpy(theta),
        torch.from_numpy(lhs).unsqueeze(1),
        rcond=rcond,
        driver="gelsd",
    ).solution
    return np.asarray(solution.squeeze(-1).numpy(), dtype=np.float64)


def _bootstrap_sample_size(n_rows: int) -> int:
    return max(n_rows // 2, 1)


def _to_numpy_vector(tensor: Tensor) -> npt.NDArray[np.float64]:
    array = cast(
        npt.NDArray[np.float64],
        tensor.detach().cpu().numpy().astype(np.float64, copy=False),
    )
    return array.reshape(-1)


def _to_numpy_matrix(tensor: Tensor) -> npt.NDArray[np.float64]:
    return cast(
        npt.NDArray[np.float64],
        tensor.detach().cpu().numpy().astype(np.float64, copy=False),
    )


__all__ = [
    "CV_DENOM_EPS",
    "StabilityCandidateStats",
    "StabilitySelectionResult",
    "stability_select",
]
