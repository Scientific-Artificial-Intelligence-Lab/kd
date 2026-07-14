
from __future__ import annotations

from typing import Any

import pytest
import torch

from kd.core.evaluator import EvaluationResult
from kd.search.protocol import PlatformComponents
from kd.search.runner import ExperimentRunner
from tests.unit.search._runner_mocks import RecordingAlgorithm


class _LhsNameAlgorithm(RecordingAlgorithm):
    def __init__(self, lhs_name: str | None) -> None:
        super().__init__(score_sequence=[0.0], expression_sequence=["div(u, u)"])
        self._lhs_name = lhs_name

    def build_final_result(self) -> EvaluationResult:
        return EvaluationResult(
            mse=0.0,
            nmse=0.0,
            r2=1.0,
            score=0.0,
            complexity=1,
            coefficients=torch.tensor([1.0]),
            is_valid=True,
            residuals=torch.zeros(3),
            terms=["div(u, u)"],
            expression="div(u, u)",
            lhs_name=self._lhs_name,
        )

    def build_result_target(self) -> torch.Tensor:
        return torch.ones(3)

    @property
    def config(self) -> dict[str, Any]:
        return {"algorithm": "lhs-name-test"}


class TestRunnerLhsName:
    @pytest.mark.unit
    def test_plugin_reported_lhs_name_overrides_dataset_default(
        self, mock_components: PlatformComponents
    ) -> None:
        result = ExperimentRunner(
            _LhsNameAlgorithm("u_tt"),
            max_iterations=1,
            batch_size=1,
        ).run(mock_components)

        assert result.lhs_label == "u_tt"

    @pytest.mark.unit
    def test_missing_lhs_name_keeps_existing_dataset_fallback(
        self, mock_components: PlatformComponents
    ) -> None:
        result = ExperimentRunner(
            _LhsNameAlgorithm(None),
            max_iterations=1,
            batch_size=1,
        ).run(mock_components)

        assert result.lhs_label == "u_t"

