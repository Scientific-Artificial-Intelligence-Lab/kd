
from __future__ import annotations

import numpy as np
import pytest
import torch

from kd.core.evaluator import EvaluationResult
from kd.search.discover.builder import build_engine
from kd.search.discover.config import DiscoverConfig
from kd.search.discover.engine import DiscoverEngine


_OUT_OF_BOUNDS_COEF = 1.0e5
_IN_BOUNDS_COEF = 0.5
_GATE_SEED = 0


def _make_engine(*, magnitude_filter: bool) -> DiscoverEngine:
    config = DiscoverConfig(
        batch_size=64,
        max_length=12,
        min_length=2,
        magnitude_filter=magnitude_filter,
        stability_selection=1,
        stability_queue_capacity=10,
    )
    return build_engine(config)


def _out_of_bounds_result(expr: str) -> EvaluationResult:
    return EvaluationResult(
        mse=0.01,
        nmse=0.01,
        r2=0.99,
        aic=1.0,
        complexity=2,
        coefficients=torch.tensor([_OUT_OF_BOUNDS_COEF, _IN_BOUNDS_COEF]),
        is_valid=True,
        error_message="",
        selected_indices=None,
        residuals=None,
        terms=["u", "diff_x_u"],
        expression=expr,
    )


def _seed() -> None:
    torch.manual_seed(_GATE_SEED)
    np.random.seed(_GATE_SEED)


@pytest.mark.integration
def test_gate_on_drops_out_of_bounds_fit_from_training_and_pool() -> None:
    _seed()
    engine = _make_engine(magnitude_filter=True)
    ir_strings = engine.propose()
    assert ir_strings, "propose() must yield at least one candidate to drive the gate"

    engine.receive_results([_out_of_bounds_result(ir) for ir in ir_strings])
    engine.update()

    metrics = engine.last_metrics



    assert metrics["n_eval_valid"] == 0.0

    assert engine.cycle_top_candidates == []

    assert engine.best_reward == 0.0
    assert engine.best_expression == ""


@pytest.mark.integration
def test_gate_off_lets_same_fit_into_training_and_pool() -> None:
    _seed()
    engine = _make_engine(magnitude_filter=False)
    ir_strings = engine.propose()
    assert ir_strings

    engine.receive_results([_out_of_bounds_result(ir) for ir in ir_strings])
    engine.update()

    metrics = engine.last_metrics
    assert metrics["n_eval_valid"] == float(len(ir_strings))
    assert engine.cycle_top_candidates
    assert engine.best_reward > 0.0


@pytest.mark.integration
def test_gate_on_rebase_keeps_best_reward_and_validity_single_sourced() -> None:

    class _ChampionDriftEvaluator:

        def __init__(self, champion_expr: str) -> None:
            self._champion_expr = champion_expr

        def evaluate_expression(self, expr: str) -> EvaluationResult:
            coef = (
                torch.tensor([_OUT_OF_BOUNDS_COEF, _IN_BOUNDS_COEF])
                if expr == self._champion_expr
                else torch.tensor([_IN_BOUNDS_COEF, 0.3])
            )
            return EvaluationResult(
                mse=0.01,
                nmse=0.01,
                r2=0.99,
                aic=1.0,
                complexity=2,
                coefficients=coef,
                is_valid=True,
                error_message="",
                selected_indices=None,
                residuals=None,
                terms=["u", "diff_x_u"],
                expression=expr,
            )

    _seed()
    engine = _make_engine(magnitude_filter=True)
    champion_expr = "add(u, diff_x_u)"


    engine._best_expression = champion_expr
    engine._best_reward = 0.9

    engine.run_cycle(_ChampionDriftEvaluator(champion_expr), n_iterations=0)

    assert engine.best_reward == 0.0
    assert engine.best_result is not None
    assert engine.best_result.is_valid is False
    assert "large_coe" in engine.best_result.error_message
