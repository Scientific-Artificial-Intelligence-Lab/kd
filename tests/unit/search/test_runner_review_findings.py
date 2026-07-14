
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor

from kd.core.evaluator import EvaluationResult
from kd.search.protocol import PlatformComponents
from kd.search.runner import ExperimentRunner






class _BadProviderAlgorithm:

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return [f"e{i}" for i in range(n)]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return [EvaluationResult(mse=1.0, nmse=1.0, r2=0.0) for _ in candidates]

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    def build_final_result(self) -> EvaluationResult:


        return EvaluationResult(mse=1.0, nmse=1.0, r2=0.0)

    def build_result_target(self) -> Tensor:


        return None

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def best_expression(self) -> str:
        return "e0"

    @property
    def config(self) -> dict[str, Any]:
        return {"algorithm": "BadProvider"}

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._state)

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._state = dict(value)


class _ShapeMismatchProviderAlgorithm:

    def __init__(
        self,
        target_shape: tuple[int, ...],
        residual_shape: tuple[int, ...],
    ) -> None:
        self._target_shape = target_shape
        self._residual_shape = residual_shape
        self._state: dict[str, Any] = {}

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return [f"e{i}" for i in range(n)]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:

        residuals = torch.zeros(self._residual_shape)
        return [
            EvaluationResult(
                mse=1.0,
                nmse=1.0,
                r2=0.0,
                residuals=residuals,
            )
            for _ in candidates
        ]

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    def build_result_target(self) -> Tensor:
        return torch.zeros(self._target_shape)

    def build_final_result(self) -> EvaluationResult:


        return EvaluationResult(
            mse=1.0,
            nmse=1.0,
            r2=0.0,
            score=0.0,
            complexity=1,
            coefficients=torch.tensor([1.0]),
            is_valid=True,
            error_message="",
            selected_indices=[0],
            residuals=torch.zeros(self._residual_shape),
            terms=["u"],
            expression="e0",
        )

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def best_expression(self) -> str:
        return "e0"

    @property
    def config(self) -> dict[str, Any]:
        return {"algorithm": "ShapeMismatch"}

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._state)

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._state = dict(value)







@pytest.mark.unit
class TestM2BuildResultTargetContract:

    def test_bad_provider_returns_none_raises_typeerror(self) -> None:
        algorithm = _BadProviderAlgorithm()


        assert callable(algorithm.build_result_target)

        components = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=MagicMock(),
            registry=MagicMock(),
        )

        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)
        with pytest.raises(TypeError) as exc_info:
            runner.run(components)


        msg = str(exc_info.value)
        assert "build_result_target" in msg, (
            f"Expected TypeError to mention the protocol method, got: "
            f"{exc_info.value}"
        )
        assert "NoneType" in msg, (
            f"Expected TypeError to name the offending type, got: "
            f"{exc_info.value}"
        )

    def test_bad_provider_does_not_ship_synthetic_target(self) -> None:
        algorithm = _BadProviderAlgorithm()
        components = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=MagicMock(),
            registry=MagicMock(),
        )
        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)

        with pytest.raises(TypeError, match="build_result_target"):
            runner.run(components)







@pytest.mark.unit
class TestM3PredictedShapeGuard:

    def test_residual_shape_mismatch_raises_valueerror(self) -> None:
        algorithm = _ShapeMismatchProviderAlgorithm(
            target_shape=(10,),
            residual_shape=(8,),
        )
        components = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=MagicMock(),
            registry=MagicMock(),
        )
        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)

        with pytest.raises(ValueError) as exc_info:
            runner.run(components)

        msg = str(exc_info.value)


        assert "10" in msg, (
            f"Error message should mention the actual shape (10), got: {msg}"
        )
        assert "8" in msg, (
            f"Error message should mention the residual shape (8), got: {msg}"
        )

    def test_residual_shape_mismatch_2d_raises_valueerror(self) -> None:
        algorithm = _ShapeMismatchProviderAlgorithm(
            target_shape=(4, 5),
            residual_shape=(5, 4),
        )
        components = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=MagicMock(),
            registry=MagicMock(),
        )
        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)

        with pytest.raises(ValueError) as exc_info:
            runner.run(components)
        msg = str(exc_info.value)

        assert "(4, 5)" in msg or "[4, 5]" in msg or "4, 5" in msg, (
            f"Error must mention actual shape, got: {msg}"
        )
        assert "(5, 4)" in msg or "[5, 4]" in msg or "5, 4" in msg, (
            f"Error must mention residual shape, got: {msg}"
        )













class _BadResidualsAlgorithm:

    def __init__(self, residuals: Any, target_shape: tuple[int, ...]) -> None:
        self._residuals = residuals
        self._target_shape = target_shape
        self._state: dict[str, Any] = {}

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return [f"e{i}" for i in range(n)]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return [EvaluationResult(mse=1.0, nmse=1.0, r2=0.0) for _ in candidates]

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    def build_result_target(self) -> Tensor:
        return torch.zeros(self._target_shape)

    def build_final_result(self) -> EvaluationResult:
        return EvaluationResult(
            mse=1.0,
            nmse=1.0,
            r2=0.0,
            score=0.0,
            complexity=1,
            coefficients=torch.tensor([1.0]),
            is_valid=True,
            error_message="",
            selected_indices=[0],

            residuals=self._residuals,
            terms=["u"],
            expression="e0",
        )

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def best_expression(self) -> str:
        return "e0"

    @property
    def config(self) -> dict[str, Any]:
        return {"algorithm": "BadResiduals"}

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._state)

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._state = dict(value)


@pytest.mark.unit
class TestPredictedResidualsTypeGuard:

    def test_list_residuals_raises_typeerror(self) -> None:
        import numpy as np

        algorithm = _BadResidualsAlgorithm(
            residuals=[0.0, 0.0, 0.0],
            target_shape=(3,),
        )
        components = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=MagicMock(),
            registry=MagicMock(),
        )
        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)

        with pytest.raises(TypeError) as exc_info:
            runner.run(components)
        msg = str(exc_info.value).lower()
        assert "residuals" in msg, (
            f"Error must name the offending field, got: {exc_info.value}"
        )


        algorithm2 = _BadResidualsAlgorithm(
            residuals=np.zeros(3),
            target_shape=(3,),
        )
        runner2 = ExperimentRunner(algorithm=algorithm2, max_iterations=1)
        with pytest.raises(TypeError):
            runner2.run(components)
