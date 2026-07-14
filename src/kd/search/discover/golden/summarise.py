
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from kd.core.equation.canonical import canonicalize_expression


@dataclass(frozen=True, slots=True)
class GoldenRunResult:

    expression_canonical: str
    expression_raw: str
    term_set_sorted: list[str]
    coefs_by_term: dict[str, float]
    reward: float
    mse: float
    nmse: float
    n_iterations_to_best: int
    wall_time_seconds: float


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_engine_and_summarise(
    *,
    engine: Any,
    evaluator: Any,
    n_iterations: int,
) -> GoldenRunResult:
    start = time.time()
    iter_to_best: int | None = None
    best_seen = float("-inf")
    for i in range(n_iterations):
        engine.run_iteration(evaluator)
        if engine.best_reward > best_seen:
            best_seen = engine.best_reward
            iter_to_best = i + 1
    elapsed = time.time() - start
    return _summarise(
        engine=engine,
        evaluator=evaluator,
        elapsed=elapsed,
        n_iter_to_best=iter_to_best or 0,
    )


def _summarise(
    *,
    engine: Any,
    evaluator: Any,
    elapsed: float,
    n_iter_to_best: int,
) -> GoldenRunResult:
    raw_expr = str(engine.best_expression)
    if not raw_expr:
        return GoldenRunResult(
            expression_canonical="",
            expression_raw="",
            term_set_sorted=[],
            coefs_by_term={},
            reward=float(engine.best_reward),
            mse=float("nan"),
            nmse=float("nan"),
            n_iterations_to_best=n_iter_to_best,
            wall_time_seconds=elapsed,
        )

    eval_result = evaluator.evaluate_expression(raw_expr)
    terms = list(eval_result.terms or [])
    coefs = (
        [float(c) for c in eval_result.coefficients]
        if eval_result.coefficients is not None
        else []
    )
    coefs_by_term = _build_coefs_by_term(terms, coefs)
    canonical = canonicalize_expression(raw_expr)
    return GoldenRunResult(
        expression_canonical=canonical,
        expression_raw=raw_expr,
        term_set_sorted=sorted(terms),
        coefs_by_term=coefs_by_term,
        reward=float(engine.best_reward),
        mse=float(eval_result.mse),
        nmse=float(eval_result.nmse),
        n_iterations_to_best=n_iter_to_best,
        wall_time_seconds=elapsed,
    )


def _build_coefs_by_term(
    terms: list[str], coefs: list[float],
) -> dict[str, float]:
    if len(terms) != len(coefs):
        raise ValueError(
            f"terms / coefficients length mismatch: "
            f"len(terms)={len(terms)} len(coefs)={len(coefs)}",
        )
    return dict(zip(terms, coefs, strict=True))


__all__ = ["GoldenRunResult", "run_engine_and_summarise", "seed_all"]
