
from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from kd.core.evaluator import EvaluationResult
from kd.search.callbacks import VizDataCollector
from kd.search.protocol import PlatformComponents
from kd.search.recorder import VizRecorder
from kd.search.runner import ExperimentRunner, RunResult
from tests.unit.search._runner_mocks import (
    ExplodingAlgorithm,
    IterativeRecordingAlgorithm,
    RecordingAlgorithm,
    RecordingCallback,
)






class TestRunnerSmoke:

    @pytest.mark.smoke
    def test_runner_importable(self) -> None:
        from kd.search import ExperimentRunner as Runner
        from kd.search import RunResult as Result

        assert Runner is not None
        assert Result is not None

    @pytest.mark.smoke
    def test_run_result_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(RunResult)

    @pytest.mark.smoke
    def test_run_result_fields(self) -> None:
        field_names = [f.name for f in dataclasses.fields(RunResult)]
        assert field_names == [
            "best_expression",
            "best_score",
            "iterations",
            "early_stopped",
        ]

    @pytest.mark.smoke
    def test_runner_construction(self, recording_algorithm: RecordingAlgorithm) -> None:
        runner = ExperimentRunner(algorithm=recording_algorithm)
        assert runner is not None

    @pytest.mark.smoke
    def test_runner_has_run_method(
        self, recording_algorithm: RecordingAlgorithm
    ) -> None:
        runner = ExperimentRunner(algorithm=recording_algorithm)
        assert callable(getattr(runner, "run", None))

    @pytest.mark.smoke
    def test_runner_has_save_checkpoint(
        self, recording_algorithm: RecordingAlgorithm
    ) -> None:
        runner = ExperimentRunner(algorithm=recording_algorithm)
        assert callable(getattr(runner, "save_checkpoint", None))

    @pytest.mark.smoke
    def test_runner_has_load_checkpoint(
        self, recording_algorithm: RecordingAlgorithm
    ) -> None:
        runner = ExperimentRunner(algorithm=recording_algorithm)
        assert callable(getattr(runner, "load_checkpoint", None))

    @pytest.mark.smoke
    def test_runner_no_evaluator_in_init(self) -> None:
        import inspect

        sig = inspect.signature(ExperimentRunner.__init__)
        params = list(sig.parameters.keys())
        assert "evaluator" not in params, (
            f"Runner must NOT accept evaluator. Params: {params}"
        )







class TestRunnerCoreLoop:

    @pytest.mark.unit
    def test_prepare_called_first(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        runner.run(mock_components)
        assert recording_algorithm.call_log[0] == "prepare"

    @pytest.mark.unit
    def test_loop_order_propose_evaluate_update(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=2)
        runner.run(mock_components)


        loop_calls = [c for c in recording_algorithm.call_log if c != "prepare"]

        for i in range(2):
            offset = i * 3
            assert loop_calls[offset] == "propose"
            assert loop_calls[offset + 1] == "evaluate"
            assert loop_calls[offset + 2] == "update"

    @pytest.mark.unit
    def test_batch_size_forwarded_to_propose(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        batch_size = 7
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=3,
            batch_size=batch_size,
        )
        runner.run(mock_components)


        assert all(n == batch_size for n in recording_algorithm.propose_args)
        assert len(recording_algorithm.propose_args) == 3

    @pytest.mark.unit
    def test_evaluate_results_passed_to_update(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=1,
            batch_size=3,
        )
        runner.run(mock_components)


        assert len(recording_algorithm.update_args) == 1
        results = recording_algorithm.update_args[0]
        assert len(results) == 3
        assert all(isinstance(r, EvaluationResult) for r in results)

    @pytest.mark.unit
    def test_iterations_count_equals_max(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        max_iter = 5
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=max_iter,
        )
        result = runner.run(mock_components)
        assert result.iterations == max_iter

    @pytest.mark.unit
    def test_run_returns_result(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)
        assert hasattr(result, "best_expression")
        assert hasattr(result, "best_score")
        assert hasattr(result, "iterations")
        assert hasattr(result, "early_stopped")

    @pytest.mark.unit
    def test_run_result_best_from_algorithm(
        self,
        mock_components: PlatformComponents,
    ) -> None:

        algo = RecordingAlgorithm(
            score_sequence=[10.0, 5.0, 1.5],
            expression_sequence=["initial", "improving", "final_best"],
        )
        runner = ExperimentRunner(algorithm=algo, max_iterations=3)
        result = runner.run(mock_components)


        assert result.best_score == 1.5
        assert result.best_expression == "final_best"

    @pytest.mark.unit
    def test_not_early_stopped_when_full_run(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=3)
        result = runner.run(mock_components)
        assert result.early_stopped is False







class TestRunnerEarlyStopping:

    @pytest.mark.unit
    def test_early_stop_sets_flag(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        cb = RecordingCallback(stop_at_iteration=1)
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=10,
            callbacks=[cb],
        )
        result = runner.run(mock_components)
        assert result.early_stopped is True

    @pytest.mark.unit
    def test_early_stop_iteration_count(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        cb = RecordingCallback(stop_at_iteration=2)
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=100,
            callbacks=[cb],
        )
        result = runner.run(mock_components)


        assert result.iterations == 3
        assert result.early_stopped is True

    @pytest.mark.unit
    def test_early_stop_any_callback_triggers(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        cb_patient = RecordingCallback(stop_at_iteration=None)
        cb_eager = RecordingCallback(stop_at_iteration=0)
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=10,
            callbacks=[cb_patient, cb_eager],
        )
        result = runner.run(mock_components)
        assert result.early_stopped is True

        assert result.iterations == 1

    @pytest.mark.unit
    def test_early_stop_at_first_iteration(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        cb = RecordingCallback(stop_at_iteration=0)
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=50,
            callbacks=[cb],
        )
        result = runner.run(mock_components)
        assert result.iterations == 1
        assert result.early_stopped is True



        assert "iteration_end:0" in cb.events
        assert 0 in cb.iteration_ends







class TestRunnerCallbackLifecycle:

    @pytest.mark.unit
    def test_callback_lifecycle_order(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        cb = RecordingCallback()
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=2,
            callbacks=[cb],
        )
        runner.run(mock_components)

        assert cb.events[0] == "experiment_start"
        assert cb.events[-1] == "experiment_end"


        inner = cb.events[1:-1]
        assert len(inner) == 4
        assert inner == [
            "iteration_start:0",
            "iteration_end:0",
            "iteration_start:1",
            "iteration_end:1",
        ]

    @pytest.mark.unit
    def test_multiple_callbacks_all_invoked(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        cb1 = RecordingCallback()
        cb2 = RecordingCallback()
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=1,
            callbacks=[cb1, cb2],
        )
        runner.run(mock_components)


        assert cb1.events == cb2.events
        assert "experiment_start" in cb1.events
        assert "experiment_end" in cb1.events

    @pytest.mark.unit
    def test_iteration_start_before_propose(
        self,
        mock_components: PlatformComponents,
    ) -> None:

        events: list[str] = []

        class _OrderAlgo(RecordingAlgorithm):
            def propose(self, n: int) -> list[str]:
                events.append("algo:propose")
                return super().propose(n)

            def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
                events.append("algo:evaluate")
                return super().evaluate(candidates)

            def update(self, results: list[EvaluationResult]) -> None:
                events.append("algo:update")
                return super().update(results)

        class _OrderCallback(RecordingCallback):
            def on_iteration_start(self, iteration: int, algorithm: Any) -> None:
                events.append("cb:iteration_start")

            def on_iteration_end(
                self,
                iteration: int,
                algorithm: Any,
                candidates: list[str],
                results: list[Any],
            ) -> None:
                events.append("cb:iteration_end")

        algo = _OrderAlgo()
        cb = _OrderCallback()
        runner = ExperimentRunner(algorithm=algo, max_iterations=1, callbacks=[cb])
        runner.run(mock_components)



        assert events.index("cb:iteration_start") < events.index("algo:propose")
        assert events.index("algo:update") < events.index("cb:iteration_end")

    @pytest.mark.unit
    def test_iteration_end_receives_candidates_and_results(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        received_candidates: list[list[str]] = []
        received_results: list[list[EvaluationResult]] = []

        class _CapturingCallback(RecordingCallback):
            def on_iteration_end(
                self,
                iteration: int,
                algorithm: Any,
                candidates: list[str],
                results: list[Any],
            ) -> None:
                received_candidates.append(list(candidates))
                received_results.append(list(results))

        cb = _CapturingCallback()
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=2,
            batch_size=3,
            callbacks=[cb],
        )
        runner.run(mock_components)

        assert len(received_candidates) == 2
        assert len(received_results) == 2

        for cands in received_candidates:
            assert len(cands) == 3
        for ress in received_results:
            assert len(ress) == 3
            assert all(isinstance(r, EvaluationResult) for r in ress)

    @pytest.mark.unit
    def test_no_callbacks_default(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=2,
            callbacks=None,
        )
        result = runner.run(mock_components)
        assert result.iterations == 2
        assert result.early_stopped is False







class TestRunnerEdgeCases:

    @pytest.mark.unit
    def test_max_iterations_zero(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        cb = RecordingCallback()
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=0,
            callbacks=[cb],
        )
        result = runner.run(mock_components)

        assert result.iterations == 0
        assert result.early_stopped is False
        assert "experiment_start" in cb.events
        assert "experiment_end" in cb.events

        assert not any(
            "iteration" in e
            for e in cb.events
            if e not in ("experiment_start", "experiment_end")
        )

        assert "prepare" in recording_algorithm.call_log

    @pytest.mark.unit
    def test_single_iteration(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)
        assert result.iterations == 1


        assert recording_algorithm.call_log == [
            "prepare",
            "propose",
            "evaluate",
            "update",
        ]

    @pytest.mark.unit
    def test_batch_size_one(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=3,
            batch_size=1,
        )
        runner.run(mock_components)
        assert all(n == 1 for n in recording_algorithm.propose_args)

    @pytest.mark.unit
    def test_empty_callbacks_list(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=2,
            callbacks=[],
        )
        result = runner.run(mock_components)
        assert result.iterations == 2
        assert result.early_stopped is False

    @pytest.mark.unit
    def test_stale_iteration_after_prior_run(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=0)

        runner._current_iteration = 99

        result = runner.run(mock_components)
        assert result.iterations == 0







class TestRunnerExceptionSafety:

    @pytest.mark.unit
    def test_experiment_end_called_on_exception(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        algo = ExplodingAlgorithm(explode_at=1)
        cb = RecordingCallback()
        runner = ExperimentRunner(
            algorithm=algo,
            max_iterations=5,
            callbacks=[cb],
        )

        with pytest.raises(RuntimeError, match="Algorithm exploded"):
            runner.run(mock_components)


        assert "experiment_end" in cb.events

    @pytest.mark.unit
    def test_experiment_end_called_on_first_iteration_exception(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        algo = ExplodingAlgorithm(explode_at=0)
        cb = RecordingCallback()
        runner = ExperimentRunner(
            algorithm=algo,
            max_iterations=5,
            callbacks=[cb],
        )

        with pytest.raises(RuntimeError, match="Algorithm exploded"):
            runner.run(mock_components)

        assert "experiment_start" in cb.events
        assert "experiment_end" in cb.events

    @pytest.mark.unit
    def test_all_callbacks_get_experiment_end_on_exception(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        algo = ExplodingAlgorithm(explode_at=0)
        cb1 = RecordingCallback()
        cb2 = RecordingCallback()
        runner = ExperimentRunner(
            algorithm=algo,
            max_iterations=5,
            callbacks=[cb1, cb2],
        )

        with pytest.raises(RuntimeError):
            runner.run(mock_components)

        assert "experiment_end" in cb1.events
        assert "experiment_end" in cb2.events

    @pytest.mark.unit
    def test_exception_propagates_after_cleanup(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        algo = ExplodingAlgorithm(explode_at=0)
        runner = ExperimentRunner(
            algorithm=algo,
            max_iterations=5,
            callbacks=[RecordingCallback()],
        )

        with pytest.raises(RuntimeError, match="Algorithm exploded"):
            runner.run(mock_components)







class TestRunResult:

    @pytest.mark.unit
    def test_run_result_construction(self) -> None:
        r = RunResult(
            best_expression="mul(u, u_x)",
            best_score=0.001,
            iterations=42,
            early_stopped=True,
        )
        assert r.best_expression == "mul(u, u_x)"
        assert r.best_score == 0.001
        assert r.iterations == 42
        assert r.early_stopped is True

    @pytest.mark.unit
    def test_run_result_equality(self) -> None:
        r1 = RunResult("e", 0.5, 10, False)
        r2 = RunResult("e", 0.5, 10, False)
        assert r1 == r2

    @pytest.mark.unit
    def test_run_result_inequality(self) -> None:
        r1 = RunResult("e", 0.5, 10, False)
        r2 = RunResult("e", 0.5, 10, True)
        assert r1 != r2







class TestRunnerExperimentResult:

    @pytest.mark.smoke
    def test_experiment_result_importable(self) -> None:
        from kd.search.result import ExperimentResult

        assert ExperimentResult is not None

    @pytest.mark.unit
    def test_run_returns_experiment_result(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.result import ExperimentResult

        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)
        assert isinstance(result, ExperimentResult)

    @pytest.mark.unit
    def test_experiment_result_has_final_eval(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)
        assert hasattr(result, "final_eval")
        assert isinstance(result.final_eval, EvaluationResult)

    @pytest.mark.unit
    def test_experiment_result_has_actual_tensor(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:

        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)
        assert hasattr(result, "actual")
        assert isinstance(result.actual, torch.Tensor)

    @pytest.mark.unit
    def test_experiment_result_has_predicted_tensor(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:

        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)
        assert hasattr(result, "predicted")
        assert isinstance(result.predicted, torch.Tensor)

    @pytest.mark.unit
    def test_experiment_result_has_recorder(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:

        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)
        assert hasattr(result, "recorder")
        assert isinstance(result.recorder, VizRecorder)

    @pytest.mark.unit
    def test_experiment_result_has_config_dict(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)
        assert hasattr(result, "config")
        assert isinstance(result.config, dict)

    @pytest.mark.unit
    def test_experiment_result_preserves_best_from_algorithm(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        algo = RecordingAlgorithm(
            score_sequence=[10.0, 5.0, 1.5],
            expression_sequence=["initial", "improving", "final_best"],
        )
        runner = ExperimentRunner(algorithm=algo, max_iterations=3)
        result = runner.run(mock_components)

        assert result.best_score == 1.5
        assert result.best_expression == "final_best"







class TestRunnerVizDataCollectorInjection:

    @pytest.mark.unit
    def test_recorder_has_best_score_after_run(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=3)
        result = runner.run(mock_components)

        scores = result.recorder.get("_best_score")

        assert len(scores) == 3

    @pytest.mark.unit
    def test_recorder_has_best_expr_after_run(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=2)
        result = runner.run(mock_components)

        exprs = result.recorder.get("_best_expr")
        assert len(exprs) == 2

        assert all(isinstance(e, str) for e in exprs)

    @pytest.mark.unit
    def test_recorder_has_n_candidates_after_run(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        batch_size = 7
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=2,
            batch_size=batch_size,
        )
        result = runner.run(mock_components)

        n_cands = result.recorder.get("_n_candidates")
        assert len(n_cands) == 2

        assert all(n == batch_size for n in n_cands)

    @pytest.mark.unit
    def test_run_without_recorder_creates_default(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:


        assert mock_components.recorder is None

        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=2)
        result = runner.run(mock_components)

        assert isinstance(result.recorder, VizRecorder)

        assert len(result.recorder.get("_best_score")) == 2

    @pytest.mark.unit
    def test_run_with_explicit_recorder(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:

        explicit_recorder = VizRecorder()
        mock_components.recorder = explicit_recorder

        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=2)
        result = runner.run(mock_components)


        assert result.recorder is explicit_recorder

        assert len(result.recorder.get("_best_score")) == 2

    @pytest.mark.unit
    def test_user_callbacks_still_invoked(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        cb = RecordingCallback()
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=2,
            callbacks=[cb],
        )
        result = runner.run(mock_components)


        assert "experiment_start" in cb.events
        assert "experiment_end" in cb.events

        assert len(result.recorder.get("_best_score")) == 2







class TestRunnerResultBuilderFallback:

    @pytest.mark.unit
    def test_result_builder_path(
        self,
        mock_components: PlatformComponents,
    ) -> None:

        from kd.search.result import ResultBuilder

        final_eval = EvaluationResult(
            mse=0.001,
            nmse=0.001,
            r2=0.999,
            aic=-50.0,
            residuals=torch.zeros(10),
        )

        class _BuilderAlgorithm(RecordingAlgorithm):

            @property
            def config(self) -> dict:
                return {"algorithm": "builder_test"}

            def build_final_result(self) -> EvaluationResult:
                return final_eval

        algo = _BuilderAlgorithm()
        assert isinstance(algo, ResultBuilder)



        mock_components.evaluator.lhs_target = torch.zeros(10)

        runner = ExperimentRunner(algorithm=algo, max_iterations=1)
        result = runner.run(mock_components)


        assert result.final_eval.mse == final_eval.mse
        assert result.final_eval.aic == final_eval.aic

    @pytest.mark.unit
    def test_evaluator_fallback_path(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:

        from kd.search.result import ResultBuilder


        assert not isinstance(recording_algorithm, ResultBuilder)


        fallback_eval = EvaluationResult(
            mse=0.5,
            nmse=0.5,
            r2=0.5,
            residuals=torch.zeros(10),
        )
        mock_components.evaluator.evaluate_expression.return_value = fallback_eval
        mock_components.evaluator.lhs_target = torch.randn(10)

        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)


        mock_components.evaluator.evaluate_expression.assert_called_once()
        assert result.final_eval.mse == fallback_eval.mse

    @pytest.mark.unit
    def test_predicted_derived_from_actual_and_residuals(
        self,
        mock_components: PlatformComponents,
    ) -> None:

        actual_data = torch.tensor([1.0, 2.0, 3.0])
        residuals = torch.tensor([0.1, -0.2, 0.3])
        expected_predicted = actual_data + residuals

        fallback_eval = EvaluationResult(
            mse=0.1,
            nmse=0.1,
            r2=0.9,
            residuals=residuals,
        )
        mock_components.evaluator.evaluate_expression.return_value = fallback_eval
        mock_components.evaluator.lhs_target = actual_data

        algo = RecordingAlgorithm()
        runner = ExperimentRunner(algorithm=algo, max_iterations=1)
        result = runner.run(mock_components)

        torch.testing.assert_close(
            result.predicted, expected_predicted, rtol=1e-7, atol=1e-10
        )

    @pytest.mark.unit
    def test_dataset_name_from_components(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:

        mock_components.dataset.name = "test_burgers"
        fallback_eval = EvaluationResult(
            mse=0.1,
            nmse=0.1,
            r2=0.9,
            residuals=torch.zeros(10),
        )
        mock_components.evaluator.evaluate_expression.return_value = fallback_eval
        mock_components.evaluator.lhs_target = torch.randn(10)

        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)

        assert result.dataset_name == "test_burgers"

    @pytest.mark.unit
    def test_algorithm_name_from_class(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:

        fallback_eval = EvaluationResult(
            mse=0.1,
            nmse=0.1,
            r2=0.9,
            residuals=torch.zeros(10),
        )
        mock_components.evaluator.evaluate_expression.return_value = fallback_eval
        mock_components.evaluator.lhs_target = torch.randn(10)

        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)

        assert result.algorithm_name == "RecordingAlgorithm"







class TestRunnerBetweenIterations:

    @pytest.mark.unit
    def test_plain_algorithm_no_between_iterations(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=3)
        runner.run(mock_components)


        assert "between_iterations" not in recording_algorithm.call_log

    @pytest.mark.unit
    def test_iterative_algorithm_between_iterations_called(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        algo = IterativeRecordingAlgorithm()
        runner = ExperimentRunner(algorithm=algo, max_iterations=3)
        runner.run(mock_components)

        assert algo.between_iterations_count > 0
        assert "between_iterations" in algo.call_log

    @pytest.mark.unit
    def test_between_iterations_count(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        max_iter = 5
        algo = IterativeRecordingAlgorithm()
        runner = ExperimentRunner(algorithm=algo, max_iterations=max_iter)
        runner.run(mock_components)

        assert algo.between_iterations_count == max_iter - 1

    @pytest.mark.unit
    def test_between_iterations_not_called_on_early_stop(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        algo = IterativeRecordingAlgorithm()

        cb = RecordingCallback(stop_at_iteration=0)
        runner = ExperimentRunner(algorithm=algo, max_iterations=10, callbacks=[cb])
        result = runner.run(mock_components)

        assert result.early_stopped is True
        assert result.iterations == 1


        assert algo.between_iterations_count == 0

    @pytest.mark.unit
    def test_between_iterations_not_called_after_early_stop_mid_run(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        algo = IterativeRecordingAlgorithm()
        cb = RecordingCallback(stop_at_iteration=2)
        runner = ExperimentRunner(algorithm=algo, max_iterations=10, callbacks=[cb])
        result = runner.run(mock_components)

        assert result.early_stopped is True
        assert result.iterations == 3

        assert algo.between_iterations_count == 2

    @pytest.mark.unit
    def test_between_iterations_not_called_single_iteration(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        algo = IterativeRecordingAlgorithm()
        runner = ExperimentRunner(algorithm=algo, max_iterations=1)
        runner.run(mock_components)

        assert algo.between_iterations_count == 0

    @pytest.mark.unit
    def test_between_iterations_not_called_zero_iterations(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        algo = IterativeRecordingAlgorithm()
        runner = ExperimentRunner(algorithm=algo, max_iterations=0)
        runner.run(mock_components)

        assert algo.between_iterations_count == 0

    @pytest.mark.unit
    def test_between_iterations_after_on_iteration_end(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        global_events: list[str] = []

        class _OrderTrackingAlgo(IterativeRecordingAlgorithm):
            def between_iterations(self) -> None:
                global_events.append("between_iterations")
                super().between_iterations()

        class _OrderTrackingCallback(RecordingCallback):
            def on_iteration_end(
                self,
                iteration: int,
                algorithm: Any,
                candidates: list[str],
                results: list[Any],
            ) -> None:
                global_events.append(f"on_iteration_end:{iteration}")
                super().on_iteration_end(iteration, algorithm, candidates, results)

        algo = _OrderTrackingAlgo()
        cb = _OrderTrackingCallback()
        runner = ExperimentRunner(algorithm=algo, max_iterations=2, callbacks=[cb])
        runner.run(mock_components)



        end_idx = global_events.index("on_iteration_end:0")
        between_idx = global_events.index("between_iterations")
        assert end_idx < between_idx, (
            f"on_iteration_end must precede between_iterations, "
            f"got events: {global_events}"
        )

    @pytest.mark.unit
    def test_between_iterations_loop_ordering(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        algo = IterativeRecordingAlgorithm()
        runner = ExperimentRunner(algorithm=algo, max_iterations=3)
        runner.run(mock_components)

        log = algo.call_log
        assert log[0] == "prepare"


        for i in range(2):
            base = 1 + i * 4
            assert log[base] == "propose", f"iter {i}: expected propose at {base}"
            assert log[base + 1] == "evaluate", f"iter {i}: expected evaluate"
            assert log[base + 2] == "update", f"iter {i}: expected update"
            assert log[base + 3] == "between_iterations", (
                f"iter {i}: expected between_iterations"
            )


        last_base = 1 + 2 * 4
        assert log[last_base] == "propose"
        assert log[last_base + 1] == "evaluate"
        assert log[last_base + 2] == "update"
        assert len(log) == last_base + 3







class _MismatchedLengthAlgorithm:

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return [f"c_{i}" for i in range(n)]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:

        return [EvaluationResult(mse=1.0, nmse=1.0, r2=0.0) for _ in candidates[:-1]]

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    @property
    def best_score(self) -> float:
        return 1.0

    @property
    def best_expression(self) -> str:
        return ""

    @property
    def config(self) -> dict[str, Any]:
        return {"algorithm": "_MismatchedLengthAlgorithm"}

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._state = value


class _WrongTypeAlgorithm:

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return [f"c_{i}" for i in range(n)]

    def evaluate(self, candidates: list[str]) -> list[Any]:

        return [{"mse": 1.0} for _ in candidates]

    def update(self, results: list[Any]) -> None:
        pass

    @property
    def best_score(self) -> float:
        return 1.0

    @property
    def best_expression(self) -> str:
        return ""

    @property
    def config(self) -> dict[str, Any]:
        return {"algorithm": "_WrongTypeAlgorithm"}

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._state = value


class TestRunnerEnforcesBatchContract:

    @pytest.mark.unit
    def test_run_raises_on_length_mismatch(
        self, mock_components: PlatformComponents
    ) -> None:
        algo = _MismatchedLengthAlgorithm()
        runner = ExperimentRunner(algorithm=algo, max_iterations=1, batch_size=4)
        with pytest.raises(RuntimeError, match=r"\b3\b.*\b4\b|evaluate"):
            runner.run(mock_components)

    @pytest.mark.unit
    def test_run_raises_on_wrong_result_type(
        self, mock_components: PlatformComponents
    ) -> None:
        algo = _WrongTypeAlgorithm()
        runner = ExperimentRunner(algorithm=algo, max_iterations=1, batch_size=4)
        with pytest.raises(TypeError, match="EvaluationResult|dict"):
            runner.run(mock_components)


class _RecorderCapturingAlgorithm(RecordingAlgorithm):

    def __init__(self) -> None:
        super().__init__()
        self.prepared_recorder: VizRecorder | None = None

    def prepare(self, components: PlatformComponents) -> None:
        super().prepare(components)
        self.prepared_recorder = components.recorder


class TestEnsureRecorderBackfill:

    @staticmethod
    def _components_no_recorder() -> PlatformComponents:



        evaluator = MagicMock()
        evaluator.lhs_target = torch.zeros(4)
        return PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=evaluator,
            context=MagicMock(),
            registry=MagicMock(),
        )

    @pytest.mark.unit
    def test_backfills_callback_recorder_to_components(self) -> None:
        rec = VizRecorder(enabled=True)
        algo = _RecorderCapturingAlgorithm()
        runner = ExperimentRunner(
            algorithm=algo, max_iterations=1, callbacks=[VizDataCollector(rec)]
        )
        components = self._components_no_recorder()
        assert components.recorder is None

        runner.run(components)


        assert components.recorder is rec, (
            "components.recorder must be backfilled with the callback's "
            "recorder so algorithm.prepare() and the callbacks share one."
        )

        assert algo.prepared_recorder is rec, (
            "algorithm.prepare() must see the backfilled recorder (not None); "
            "otherwise plugin-written metrics are lost."
        )

    @pytest.mark.unit
    def test_backfills_fresh_recorder_when_none_anywhere(self) -> None:
        algo = _RecorderCapturingAlgorithm()
        runner = ExperimentRunner(algorithm=algo, max_iterations=1)
        components = self._components_no_recorder()

        runner.run(components)

        assert components.recorder is not None
        assert algo.prepared_recorder is components.recorder, (
            "prepare() must see the same fresh recorder backfilled to components."
        )
