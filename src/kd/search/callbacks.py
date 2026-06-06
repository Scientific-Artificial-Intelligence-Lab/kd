
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from kd.core.evaluator import EvaluationResult
    from kd.search.protocol import SearchAlgorithm
    from kd.search.recorder import VizRecorder

logger = logging.getLogger(__name__)

__all__ = [
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "LoggingCallback",
    "RunnerCallback",
    "VizDataCollector",
]





_MIN_EVERY_N = 1







@runtime_checkable
class RunnerCallback(Protocol):

    def on_experiment_start(self, algorithm: SearchAlgorithm) -> None:
        ...

    def on_iteration_start(self, iteration: int, algorithm: SearchAlgorithm) -> None:
        ...

    def on_iteration_end(
        self,
        iteration: int,
        algorithm: SearchAlgorithm,
        candidates: list[str],
        results: list[EvaluationResult],
    ) -> None:
        ...

    def on_experiment_end(self, algorithm: SearchAlgorithm) -> None:
        ...

    @property
    def should_stop(self) -> bool:
        ...







class LoggingCallback:

    def __init__(self, every_n: int = 1) -> None:
        if every_n < _MIN_EVERY_N:
            raise ValueError(f"every_n must be >= 1, got {every_n}")
        self._every_n = every_n

    @property
    def should_stop(self) -> bool:
        return False

    def on_experiment_start(self, algorithm: Any) -> None:
        logger.info("Experiment started")

    def on_iteration_start(self, iteration: int, algorithm: Any) -> None:
        pass

    def on_iteration_end(
        self,
        iteration: int,
        algorithm: Any,
        candidates: list[str],
        results: list[Any],
    ) -> None:
        if iteration % self._every_n == 0:
            logger.info(
                "Iteration %d: best_score=%.6g, best_expression='%s'",
                iteration,
                algorithm.best_score,
                algorithm.best_expression,
            )

    def on_experiment_end(self, algorithm: Any) -> None:
        logger.info(
            "Experiment ended: best_score=%.6g, best_expression='%s'",
            algorithm.best_score,
            algorithm.best_expression,
        )


class VizDataCollector:

    def __init__(self, recorder: VizRecorder) -> None:
        self._recorder = recorder

    @property
    def recorder(self) -> VizRecorder:
        return self._recorder

    @property
    def should_stop(self) -> bool:
        return False

    def on_experiment_start(self, algorithm: Any) -> None:
        pass

    def on_iteration_start(self, iteration: int, algorithm: Any) -> None:
        pass

    def on_iteration_end(
        self,
        iteration: int,
        algorithm: Any,
        candidates: list[str],
        results: list[Any],
    ) -> None:
        self._recorder.log("_best_score", algorithm.best_score)
        self._recorder.log("_best_expr", algorithm.best_expression)
        self._recorder.log("_n_candidates", len(candidates))

    def on_experiment_end(self, algorithm: Any) -> None:
        pass







def _initial_best(mode: Literal["min", "max"]) -> float:
    if mode == "min":
        return float("inf")
    return float("-inf")


class EarlyStoppingCallback:

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-6,
        mode: Literal["min", "max"] = "min",
    ) -> None:
        if patience < 0:
            raise ValueError(f"patience must be >= 0, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self._patience = patience
        self._min_delta = min_delta
        self._mode = mode
        self._best: float = _initial_best(mode)
        self._counter: int = 0
        self._should_stop: bool = False

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @property
    def mode(self) -> Literal["min", "max"]:
        return self._mode

    def on_experiment_start(self, algorithm: Any) -> None:
        self._best = _initial_best(self._mode)
        self._counter = 0
        self._should_stop = False

    def on_iteration_start(self, iteration: int, algorithm: Any) -> None:
        pass

    def on_experiment_end(self, algorithm: Any) -> None:
        pass

    def on_iteration_end(
        self,
        iteration: int,
        algorithm: Any,
        candidates: list[str],
        results: list[Any],
    ) -> None:
        score = algorithm.best_score


        if math.isnan(score):
            self._counter += 1
        elif self._is_improvement(score):
            self._best = score
            self._counter = 0
        else:
            self._counter += 1



        if (
            self._counter > 0
            and self._counter >= self._patience
            and not self._should_stop
        ):
            self._should_stop = True
            logger.info(
                "Early stopping at iteration %d (patience=%d)",
                iteration,
                self._patience,
            )

    def _is_improvement(self, score: float) -> bool:
        if self._mode == "min":
            return score < self._best - self._min_delta

        return score > self._best + self._min_delta






_CHECKPOINT_VERSION = 1
_CHECKPOINT_PATTERN = "checkpoint_{iteration:06d}.pt"
_CHECKPOINT_FINAL = "checkpoint_final.pt"


class CheckpointCallback:

    def __init__(self, directory: Path, every_n: int = 10) -> None:
        if every_n < _MIN_EVERY_N:
            raise ValueError(f"every_n must be >= 1, got {every_n}")
        self._directory = Path(directory)
        self._every_n = every_n
        self._last_iteration: int = -1

    @property
    def should_stop(self) -> bool:
        return False

    def on_experiment_start(self, algorithm: Any) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        self._last_iteration = -1

    def on_iteration_start(self, iteration: int, algorithm: Any) -> None:
        pass

    def on_iteration_end(
        self,
        iteration: int,
        algorithm: Any,
        candidates: list[str],
        results: list[Any],
    ) -> None:
        self._last_iteration = iteration
        if iteration % self._every_n == 0:
            path = self._directory / _CHECKPOINT_PATTERN.format(iteration=iteration)
            torch.save(
                {
                    "version": _CHECKPOINT_VERSION,
                    "iteration": iteration,
                    "algorithm_state": algorithm.state,
                    "best_score": algorithm.best_score,
                    "best_expression": algorithm.best_expression,
                },
                path,
            )
            logger.debug("Saved checkpoint to %s", path)

    def on_experiment_end(self, algorithm: Any) -> None:
        path = self._directory / _CHECKPOINT_FINAL
        iteration = max(self._last_iteration, 0)
        torch.save(
            {
                "version": _CHECKPOINT_VERSION,
                "iteration": iteration,
                "algorithm_state": algorithm.state,
                "best_score": algorithm.best_score,
                "best_expression": algorithm.best_expression,
            },
            path,
        )
        logger.debug("Saved final checkpoint to %s", path)
