
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from kd.core.evaluator import EvaluationResult
from kd.search.protocol import PlatformComponents


def _default_final_eval() -> EvaluationResult:
    return EvaluationResult(mse=0.1, nmse=0.1, r2=0.9)







class RecordingAlgorithm:

    def __init__(
        self,
        score_sequence: list[float] | None = None,
        expression_sequence: list[str] | None = None,
    ) -> None:
        self.call_log: list[str] = []
        self.propose_args: list[int] = []
        self.update_args: list[list[EvaluationResult]] = []
        self._iteration: int = 0
        self._score_seq = (
            score_sequence if score_sequence is not None else [1.0, 0.5, 0.1]
        )
        self._expr_seq = (
            expression_sequence
            if expression_sequence is not None
            else ["e0", "e1", "e2"]
        )
        self._state: dict[str, Any] = {}
        self.final_eval_result: EvaluationResult = _default_final_eval()
        self.result_target: Tensor = torch.zeros(0)

    def prepare(self, components: PlatformComponents) -> None:
        self.call_log.append("prepare")

    def propose(self, n: int) -> list[str]:
        self.call_log.append("propose")
        self.propose_args.append(n)
        return [f"candidate_{i}" for i in range(n)]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        self.call_log.append("evaluate")
        return [
            EvaluationResult(mse=0.1 * (i + 1), nmse=0.1 * (i + 1), r2=0.9)
            for i in range(len(candidates))
        ]

    def update(self, results: list[EvaluationResult]) -> None:
        self.call_log.append("update")
        self.update_args.append(results)
        self._iteration += 1

    def build_final_result(self) -> EvaluationResult:
        return self.final_eval_result

    def build_result_target(self) -> Tensor:
        return self.result_target

    @property
    def best_score(self) -> float:
        idx = min(self._iteration, len(self._score_seq) - 1)
        return self._score_seq[idx]

    @property
    def best_expression(self) -> str:
        idx = min(self._iteration, len(self._expr_seq) - 1)
        return self._expr_seq[idx]

    @property
    def config(self) -> dict[str, Any]:
        return {"algorithm": "RecordingAlgorithm", "mock": True}

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._state = value


class ExplodingAlgorithm:

    def __init__(self, explode_at: int = 1) -> None:
        self._explode_at = explode_at
        self._iteration = 0
        self._state: dict[str, Any] = {}
        self.final_eval_result: EvaluationResult = _default_final_eval()
        self.result_target: Tensor = torch.zeros(0)

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        if self._iteration == self._explode_at:
            raise RuntimeError("Algorithm exploded!")
        return [f"c_{i}" for i in range(n)]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return [EvaluationResult(mse=1.0, nmse=1.0, r2=0.0) for _ in candidates]

    def update(self, results: list[EvaluationResult]) -> None:
        self._iteration += 1

    def build_final_result(self) -> EvaluationResult:
        return self.final_eval_result

    def build_result_target(self) -> Tensor:
        return self.result_target

    @property
    def best_score(self) -> float:
        return float("inf")

    @property
    def best_expression(self) -> str:
        return ""

    @property
    def config(self) -> dict[str, Any]:
        return {"algorithm": "ExplodingAlgorithm"}

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._state = value


class StatefulAlgorithm:

    def __init__(self) -> None:
        self._generation: int = 0
        self._population: list[str] = ["init_expr"]
        self._best_score_val: float = float("inf")
        self._best_expr_val: str = ""
        self._prepared = False
        self.final_eval_result: EvaluationResult = _default_final_eval()
        self.result_target: Tensor = torch.zeros(0)

    def prepare(self, components: PlatformComponents) -> None:
        self._prepared = True

    def propose(self, n: int) -> list[str]:
        return [f"gen{self._generation}_c{i}" for i in range(n)]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        results = []
        for i, _c in enumerate(candidates):
            score = 1.0 / (self._generation + 1 + i + 1)
            results.append(EvaluationResult(mse=score, nmse=score, r2=1.0 - score))
        return results

    def update(self, results: list[EvaluationResult]) -> None:
        self._generation += 1
        best = min(results, key=lambda r: r.mse)
        if best.mse < self._best_score_val:
            self._best_score_val = best.mse
            self._best_expr_val = f"found_at_gen{self._generation}"

    def build_final_result(self) -> EvaluationResult:
        return self.final_eval_result

    def build_result_target(self) -> Tensor:
        return self.result_target

    @property
    def best_score(self) -> float:
        return self._best_score_val

    @property
    def best_expression(self) -> str:
        return self._best_expr_val

    @property
    def config(self) -> dict[str, Any]:
        return {"algorithm": "StatefulAlgorithm"}

    @property
    def state(self) -> dict[str, Any]:
        return {
            "generation": self._generation,
            "population": list(self._population),
            "best_score": self._best_score_val,
            "best_expression": self._best_expr_val,
        }

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._generation = value["generation"]
        self._population = value["population"]
        self._best_score_val = value["best_score"]
        self._best_expr_val = value["best_expression"]


class StateVerifyingAlgorithm:

    def __init__(self) -> None:
        self._state: dict[str, Any] = {"restored": False}
        self.final_eval_result: EvaluationResult = _default_final_eval()
        self.result_target: Tensor = torch.zeros(0)

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return [f"e{i}" for i in range(n)]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return [EvaluationResult(mse=1.0, nmse=1.0, r2=0.0) for _ in candidates]

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    def build_final_result(self) -> EvaluationResult:
        return self.final_eval_result

    def build_result_target(self) -> Tensor:
        return self.result_target

    @property
    def best_score(self) -> float:
        return 1.0

    @property
    def best_expression(self) -> str:
        return "e0"

    @property
    def config(self) -> dict[str, Any]:
        return {"algorithm": "StateVerifyingAlgorithm"}

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._state)

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._state = dict(value)
        self._state["restored"] = True







class IterativeRecordingAlgorithm(RecordingAlgorithm):

    def __init__(
        self,
        score_sequence: list[float] | None = None,
        expression_sequence: list[str] | None = None,
    ) -> None:
        super().__init__(score_sequence, expression_sequence)
        self.between_iterations_count: int = 0
        self.between_iterations_at: list[int] = []

    def between_iterations(self) -> None:
        self.call_log.append("between_iterations")
        self.between_iterations_count += 1
        self.between_iterations_at.append(self._iteration)


class TerminatingRecordingAlgorithm(RecordingAlgorithm):

    def __init__(
        self,
        done_after: int = 3,
        score_sequence: list[float] | None = None,
        expression_sequence: list[str] | None = None,
    ) -> None:
        super().__init__(score_sequence, expression_sequence)
        self._done_after = done_after

    @property
    def is_done(self) -> bool:
        return self._iteration >= self._done_after


class DeferredTerminatingAlgorithm(RecordingAlgorithm):

    def __init__(
        self,
        done_after: int = 2,
        score_sequence: list[float] | None = None,
        expression_sequence: list[str] | None = None,
    ) -> None:
        super().__init__(score_sequence, expression_sequence)
        self._done_after = done_after

    @property
    def is_done(self) -> bool:
        if self._iteration == 0:
            raise RuntimeError(
                "is_done was read before the first completed iteration; the "
                "Runner must not probe is_done until after update() has run"
            )
        return self._iteration >= self._done_after


class TerminatingIterativeRecordingAlgorithm(IterativeRecordingAlgorithm):

    def __init__(
        self,
        done_after: int = 3,
        score_sequence: list[float] | None = None,
        expression_sequence: list[str] | None = None,
    ) -> None:
        super().__init__(score_sequence, expression_sequence)
        self._done_after = done_after

    @property
    def is_done(self) -> bool:
        return self._iteration >= self._done_after


class RecordingCallback:

    def __init__(self, stop_at_iteration: int | None = None) -> None:
        self.events: list[str] = []
        self.iteration_starts: list[int] = []
        self.iteration_ends: list[int] = []
        self._stop_at = stop_at_iteration
        self._should_stop = False

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    def on_experiment_start(self, algorithm: Any) -> None:
        self.events.append("experiment_start")

    def on_iteration_start(self, iteration: int, algorithm: Any) -> None:
        self.events.append(f"iteration_start:{iteration}")
        self.iteration_starts.append(iteration)

    def on_iteration_end(
        self,
        iteration: int,
        algorithm: Any,
        candidates: list[str],
        results: list[Any],
    ) -> None:
        self.events.append(f"iteration_end:{iteration}")
        self.iteration_ends.append(iteration)
        if self._stop_at is not None and iteration >= self._stop_at:
            self._should_stop = True

    def on_experiment_end(self, algorithm: Any) -> None:
        self.events.append("experiment_end")
