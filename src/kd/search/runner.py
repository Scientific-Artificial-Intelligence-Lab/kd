
from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor

from kd.core.evaluator import EvaluationResult
from kd.search.callbacks import RunnerCallback, VizDataCollector
from kd.search.protocol import (
    IterativeSearchAlgorithm,
    PlatformComponents,
    SearchAlgorithm,
)
from kd.search.recorder import VizRecorder
from kd.search.result import (
    ExperimentResult,
    ResultBuilder,
    ResultTargetProvider,
    RunResult,
)

logger = logging.getLogger(__name__)





_CHECKPOINT_VERSION = 1
_DEFAULT_LHS_LABEL = "u_t"


class ExperimentRunner:

    def __init__(
        self,
        algorithm: SearchAlgorithm,
        max_iterations: int = 100,
        batch_size: int = 20,
        callbacks: list[RunnerCallback] | None = None,
    ) -> None:
        self._algorithm = algorithm
        self._max_iterations = max_iterations
        self._batch_size = batch_size
        self._callbacks: list[RunnerCallback] = (
            callbacks if callbacks is not None else []
        )
        self._current_iteration: int = 0

    def run(self, components: PlatformComponents) -> ExperimentResult:
        self._current_iteration = 0
        recorder = self._ensure_recorder(components)
        callbacks = self._callbacks_for_run(recorder)
        self._algorithm.prepare(components)

        for cb in callbacks:
            cb.on_experiment_start(self._algorithm)

        early_stopped = False
        iterative_alg = (
            self._algorithm
            if isinstance(self._algorithm, IterativeSearchAlgorithm)
            else None
        )
        try:
            for iteration in range(self._max_iterations):
                self._run_iteration(iteration, callbacks)
                self._current_iteration = iteration + 1
                if any(cb.should_stop for cb in callbacks):
                    early_stopped = True
                    break
                if iterative_alg is not None and iteration < self._max_iterations - 1:
                    iterative_alg.between_iterations()
        finally:
            self._finalize_callbacks(callbacks)

        return self._build_experiment_result(components, recorder, early_stopped)

    def _run_iteration(
        self,
        iteration: int,
        callbacks: list[RunnerCallback],
    ) -> None:
        for cb in callbacks:
            cb.on_iteration_start(iteration, self._algorithm)

        candidates = self._algorithm.propose(self._batch_size)
        results = self._algorithm.evaluate(candidates)
        self._validate_evaluation_results(candidates, results)
        self._algorithm.update(results)

        for cb in callbacks:
            cb.on_iteration_end(iteration, self._algorithm, candidates, results)

    @staticmethod
    def _validate_evaluation_results(
        candidates: list[str],
        results: list[EvaluationResult],
    ) -> None:
        if len(results) != len(candidates):
            raise RuntimeError(
                f"Plugin contract violation: evaluate returned "
                f"{len(results)} results for {len(candidates)} candidates "
                f"(propose/evaluate must maintain 1:1 correspondence)."
            )
        for i, r in enumerate(results):
            if not isinstance(r, EvaluationResult):
                raise TypeError(
                    f"Plugin contract violation: evaluate result[{i}] is "
                    f"{type(r).__name__}, expected EvaluationResult."
                )

    def _ensure_recorder(self, components: PlatformComponents) -> VizRecorder:
        recorder = getattr(components, "recorder", None)
        if recorder is None:
            recorder = self._callback_recorder()
        if recorder is None:
            recorder = VizRecorder()







        if getattr(components, "recorder", None) is None:
            components.recorder = recorder
        return recorder

    def _callback_recorder(self) -> VizRecorder | None:
        for cb in self._callbacks:
            if isinstance(cb, VizDataCollector):
                return cb.recorder
        return None

    def _callbacks_for_run(self, recorder: VizRecorder) -> list[RunnerCallback]:
        callbacks = list(self._callbacks)
        for cb in callbacks:
            if isinstance(cb, VizDataCollector) and cb.recorder is recorder:
                return callbacks
        callbacks.append(VizDataCollector(recorder))
        return callbacks

    def _finalize_callbacks(self, callbacks: list[RunnerCallback]) -> None:
        for cb in callbacks:
            try:
                cb.on_experiment_end(self._algorithm)
            except Exception:
                logger.exception(
                    "Callback %r.on_experiment_end raised",
                    type(cb).__name__,
                )

    def _build_experiment_result(
        self,
        components: PlatformComponents,
        recorder: VizRecorder,
        early_stopped: bool,
    ) -> ExperimentResult:
        final_eval = self._final_eval(components)
        actual = self._actual(components, final_eval)
        predicted = self._predicted(actual, final_eval)
        return ExperimentResult(
            best_expression=self._algorithm.best_expression,
            best_score=self._algorithm.best_score,
            iterations=self._current_iteration,
            early_stopped=early_stopped,
            final_eval=final_eval,
            actual=actual,
            predicted=predicted,
            dataset_name=self._dataset_name(components),
            algorithm_name=type(self._algorithm).__name__,
            config=dict(self._algorithm.config),
            recorder=recorder,
            lhs_label=self._lhs_label(components, final_eval),
        )

    def _final_eval(self, components: PlatformComponents) -> EvaluationResult:
        result: object
        if isinstance(self._algorithm, ResultBuilder):
            result = self._algorithm.build_final_result()
        else:
            result = components.evaluator.evaluate_expression(
                self._algorithm.best_expression
            )
        if isinstance(result, EvaluationResult):
            return result
        return self._invalid_final_eval(
            "Final evaluation did not return EvaluationResult"
        )

    def _actual(
        self,
        components: PlatformComponents,
        final_eval: EvaluationResult,
    ) -> Tensor:
        if isinstance(self._algorithm, ResultTargetProvider):
            actual = self._algorithm.build_result_target()
            if not isinstance(actual, Tensor):
                raise TypeError(
                    f"{type(self._algorithm).__name__}.build_result_target() "
                    f"must return Tensor, got {type(actual).__name__}"
                )
            return actual.detach()
        actual = components.evaluator.lhs_target
        if not isinstance(actual, Tensor):
            raise TypeError(
                f"components.evaluator.lhs_target must be a Tensor, "
                f"got {type(actual).__name__}"
            )
        return actual.detach()

    def _predicted(self, actual: Tensor, final_eval: EvaluationResult) -> Tensor:
        if final_eval.residuals is None:
            return torch.zeros_like(actual)



        if not isinstance(final_eval.residuals, Tensor):
            raise TypeError(
                f"final_eval.residuals must be a Tensor, "
                f"got {type(final_eval.residuals).__name__}"
            )
        if final_eval.residuals.shape != actual.shape:
            raise ValueError(
                f"residuals.shape={tuple(final_eval.residuals.shape)} does not "
                f"match actual.shape={tuple(actual.shape)}"
            )
        return actual + final_eval.residuals

    def _dataset_name(self, components: PlatformComponents) -> str:
        name = components.dataset.name
        if isinstance(name, str):
            return name
        return str(name)

    def _lhs_label(
        self,
        components: PlatformComponents,
        final_eval: EvaluationResult,
    ) -> str:
        if isinstance(final_eval.lhs_name, str) and final_eval.lhs_name:
            return final_eval.lhs_name
        dataset = components.dataset
        lhs_field = getattr(dataset, "lhs_field", None)
        lhs_axis = getattr(dataset, "lhs_axis", None)
        has_field = isinstance(lhs_field, str) and bool(lhs_field)
        has_axis = isinstance(lhs_axis, str) and bool(lhs_axis)
        if has_field and has_axis:
            return f"{lhs_field}_{lhs_axis}"
        return _DEFAULT_LHS_LABEL

    def _invalid_final_eval(self, error_message: str) -> EvaluationResult:
        return EvaluationResult(
            mse=float("inf"),
            nmse=float("inf"),
            r2=-float("inf"),
            aic=float("inf"),
            complexity=0,
            coefficients=None,
            is_valid=False,
            error_message=error_message,
            selected_indices=None,
            residuals=None,
            terms=None,
            expression=self._algorithm.best_expression,
        )

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": _CHECKPOINT_VERSION,
                "iteration": self._current_iteration,
                "algorithm_state": self._algorithm.state,
                "best_score": self._algorithm.best_score,
                "best_expression": self._algorithm.best_expression,
            },
            path,
        )
        logger.debug("Saved checkpoint to %s", path)

    def load_checkpoint(self, path: Path) -> None:
        data = torch.load(Path(path), weights_only=False)
        self._algorithm.state = data["algorithm_state"]
        self._current_iteration = data["iteration"]
        logger.debug(
            "Loaded checkpoint from %s (iteration=%d)", path, self._current_iteration
        )


__all__ = [
    "ExperimentRunner",
    "RunResult",
]
