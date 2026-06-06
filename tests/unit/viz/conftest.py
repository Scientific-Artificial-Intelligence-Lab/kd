
from __future__ import annotations

import matplotlib
import pytest
import torch

matplotlib.use("Agg")

from kd.core.evaluator import EvaluationResult
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult


@pytest.fixture()
def mock_evaluation_result() -> EvaluationResult:
    n_samples = 50
    residuals = torch.randn(n_samples) * 0.1
    return EvaluationResult(
        mse=0.01,
        nmse=0.005,
        r2=0.95,
        aic=-100.0,
        complexity=3,
        coefficients=torch.tensor([1.0, -0.5, 0.3]),
        is_valid=True,
        error_message="",
        selected_indices=[0, 1, 2],
        residuals=residuals,
        terms=["u", "u_x", "u_xx"],
        expression="add(u, add(u_x, u_xx))",
    )


@pytest.fixture()
def mock_recorder() -> VizRecorder:
    recorder = VizRecorder()

    scores = [10.0, 5.0, 3.0, 2.0, 1.5, 1.2, 1.0, 0.8, 0.5, 0.3]
    exprs = [f"expr_{i}" for i in range(10)]
    for score, expr in zip(scores, exprs):
        recorder.log("_best_score", score)
        recorder.log("_best_expr", expr)
        recorder.log("_n_candidates", 20)
    return recorder


@pytest.fixture()
def mock_experiment_result(
    mock_evaluation_result: EvaluationResult,
    mock_recorder: VizRecorder,
) -> ExperimentResult:
    n_samples = 50
    actual = torch.sin(torch.linspace(0, 6.28, n_samples))
    predicted = actual + torch.randn(n_samples) * 0.1
    return ExperimentResult(
        best_expression="add(u, add(u_x, u_xx))",
        best_score=0.3,
        iterations=10,
        early_stopped=False,
        final_eval=mock_evaluation_result,
        actual=actual,
        predicted=predicted,
        dataset_name="test_dataset",
        algorithm_name="SGA",
        config={"max_iter": 10, "population_size": 20},
        recorder=mock_recorder,
    )
