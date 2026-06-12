
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import torch

from kd.core.evaluator import EvaluationResult


from kd.search.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    RunnerCallback,
)






class _MockAlgorithm:

    def __init__(
        self,
        best_score: float = float("inf"),
        best_expression: str = "",
    ) -> None:
        self._best_score = best_score
        self._best_expression = best_expression
        self._state: dict[str, Any] = {}

    @property
    def best_score(self) -> float:
        return self._best_score

    @best_score.setter
    def best_score(self, value: float) -> None:
        self._best_score = value

    @property
    def best_expression(self) -> str:
        return self._best_expression

    @best_expression.setter
    def best_expression(self, value: str) -> None:
        self._best_expression = value

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._state = value


def _make_eval_result(mse: float = 0.1) -> EvaluationResult:
    return EvaluationResult(mse=mse, nmse=mse, r2=1.0 - mse)







@pytest.fixture
def mock_algo() -> _MockAlgorithm:
    return _MockAlgorithm(best_score=1.0, best_expression="u_x")


@pytest.fixture
def sample_candidates() -> list[str]:
    return ["u_x", "mul(u, u_x)", "u_xx"]


@pytest.fixture
def sample_results() -> list[EvaluationResult]:
    return [
        _make_eval_result(0.1),
        _make_eval_result(0.05),
        _make_eval_result(0.2),
    ]







class TestRunnerCallbackProtocol:

    @pytest.mark.smoke
    def test_protocol_is_importable(self) -> None:
        assert RunnerCallback is not None

    @pytest.mark.smoke
    def test_protocol_is_runtime_checkable(self) -> None:

        obj = LoggingCallback()
        result = isinstance(obj, RunnerCallback)
        assert isinstance(result, bool)

    def test_logging_callback_is_runner_callback(self) -> None:
        cb = LoggingCallback()
        assert isinstance(cb, RunnerCallback)

    def test_early_stopping_is_runner_callback(self) -> None:
        cb = EarlyStoppingCallback()
        assert isinstance(cb, RunnerCallback)

    def test_checkpoint_is_runner_callback(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(directory=tmp_path)
        assert isinstance(cb, RunnerCallback)

    def test_protocol_defines_on_experiment_start(self) -> None:
        assert hasattr(RunnerCallback, "on_experiment_start")

    def test_protocol_defines_on_iteration_start(self) -> None:
        assert hasattr(RunnerCallback, "on_iteration_start")

    def test_protocol_defines_on_iteration_end(self) -> None:
        assert hasattr(RunnerCallback, "on_iteration_end")

    def test_protocol_defines_on_experiment_end(self) -> None:
        assert hasattr(RunnerCallback, "on_experiment_end")

    def test_protocol_defines_should_stop(self) -> None:
        assert hasattr(RunnerCallback, "should_stop")







class TestLoggingCallback:

    def test_default_every_n(self) -> None:
        cb = LoggingCallback()

        assert isinstance(cb, RunnerCallback)

    def test_every_n_custom(self) -> None:
        cb = LoggingCallback(every_n=5)
        assert isinstance(cb, RunnerCallback)

    def test_every_n_less_than_one_raises(self) -> None:
        with pytest.raises(ValueError):
            LoggingCallback(every_n=0)

    def test_every_n_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            LoggingCallback(every_n=-1)

    def test_should_stop_always_false(self, mock_algo: _MockAlgorithm) -> None:
        cb = LoggingCallback()
        assert cb.should_stop is False


        cb.on_experiment_start(mock_algo)
        assert cb.should_stop is False

        cb.on_iteration_end(0, mock_algo, ["u_x"], [_make_eval_result()])
        assert cb.should_stop is False

        cb.on_experiment_end(mock_algo)
        assert cb.should_stop is False

    def test_logs_at_iteration_zero(
        self,
        mock_algo: _MockAlgorithm,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        cb = LoggingCallback(every_n=5)
        cb.on_experiment_start(mock_algo)

        with caplog.at_level(logging.DEBUG):
            cb.on_iteration_end(0, mock_algo, ["u_x"], [_make_eval_result()])


        assert len(caplog.records) > 0

    def test_logs_at_correct_intervals(
        self,
        mock_algo: _MockAlgorithm,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        every_n = 3
        cb = LoggingCallback(every_n=every_n)
        cb.on_experiment_start(mock_algo)

        logged_iterations: list[int] = []

        for i in range(10):
            with caplog.at_level(logging.DEBUG):
                caplog.clear()
                cb.on_iteration_end(i, mock_algo, ["u_x"], [_make_eval_result()])

                if len(caplog.records) > 0:
                    logged_iterations.append(i)


        assert logged_iterations == [0, 3, 6, 9]

    def test_does_not_log_at_non_matching_iterations(
        self,
        mock_algo: _MockAlgorithm,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        cb = LoggingCallback(every_n=5)
        cb.on_experiment_start(mock_algo)

        with caplog.at_level(logging.DEBUG):
            caplog.clear()
            cb.on_iteration_end(1, mock_algo, ["u_x"], [_make_eval_result()])

        assert len(caplog.records) == 0

    def test_logs_best_score_and_expression(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        algo = _MockAlgorithm(best_score=0.42, best_expression="u_xx")
        cb = LoggingCallback(every_n=1)
        cb.on_experiment_start(algo)

        with caplog.at_level(logging.DEBUG):
            cb.on_iteration_end(0, algo, ["u_xx"], [_make_eval_result()])

        log_text = " ".join(r.message for r in caplog.records)

        assert "0.42" in log_text
        assert "u_xx" in log_text

    def test_logs_on_experiment_start(
        self,
        mock_algo: _MockAlgorithm,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        cb = LoggingCallback()
        with caplog.at_level(logging.DEBUG):
            cb.on_experiment_start(mock_algo)

        assert len(caplog.records) > 0

    def test_logs_on_experiment_end(
        self,
        mock_algo: _MockAlgorithm,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        cb = LoggingCallback()
        cb.on_experiment_start(mock_algo)

        with caplog.at_level(logging.DEBUG):
            caplog.clear()
            cb.on_experiment_end(mock_algo)

        assert len(caplog.records) > 0

    def test_every_n_one_logs_every_iteration(
        self,
        mock_algo: _MockAlgorithm,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        cb = LoggingCallback(every_n=1)
        cb.on_experiment_start(mock_algo)

        logged_count = 0
        for i in range(5):
            with caplog.at_level(logging.DEBUG):
                caplog.clear()
                cb.on_iteration_end(i, mock_algo, ["u_x"], [_make_eval_result()])
                if len(caplog.records) > 0:
                    logged_count += 1

        assert logged_count == 5







class TestEarlyStoppingCallback:

    def test_default_parameters(self) -> None:
        cb = EarlyStoppingCallback()
        assert isinstance(cb, RunnerCallback)

    def test_negative_patience_raises(self) -> None:
        with pytest.raises(ValueError, match="patience must be >= 0"):
            EarlyStoppingCallback(patience=-1)

    def test_negative_min_delta_raises(self) -> None:
        with pytest.raises(ValueError, match="min_delta must be >= 0"):
            EarlyStoppingCallback(min_delta=-0.1)

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            EarlyStoppingCallback(mode="minimize")

    def test_should_stop_starts_false(self) -> None:
        cb = EarlyStoppingCallback(patience=5)
        assert cb.should_stop is False

    def test_should_stop_false_after_experiment_start(
        self, mock_algo: _MockAlgorithm
    ) -> None:
        cb = EarlyStoppingCallback(patience=5)
        cb.on_experiment_start(mock_algo)
        assert cb.should_stop is False

    def test_stops_after_patience_stale_iterations_min_mode(self) -> None:
        patience = 3
        cb = EarlyStoppingCallback(patience=patience, mode="min")
        algo = _MockAlgorithm(best_score=1.0)
        cb.on_experiment_start(algo)


        cb.on_iteration_end(0, algo, [], [])


        for i in range(1, patience + 1):
            assert cb.should_stop is False, f"Stopped too early at iteration {i}"
            cb.on_iteration_end(i, algo, [], [])

        assert cb.should_stop is True

    def test_stops_after_patience_stale_iterations_max_mode(self) -> None:
        patience = 3
        cb = EarlyStoppingCallback(patience=patience, mode="max")
        algo = _MockAlgorithm(best_score=1.0)
        cb.on_experiment_start(algo)


        cb.on_iteration_end(0, algo, [], [])


        for i in range(1, patience + 1):
            assert cb.should_stop is False
            cb.on_iteration_end(i, algo, [], [])

        assert cb.should_stop is True

    def test_improvement_resets_counter_min_mode(self) -> None:
        patience = 3
        cb = EarlyStoppingCallback(patience=patience, mode="min")
        algo = _MockAlgorithm(best_score=1.0)
        cb.on_experiment_start(algo)


        cb.on_iteration_end(0, algo, [], [])


        cb.on_iteration_end(1, algo, [], [])
        cb.on_iteration_end(2, algo, [], [])
        assert cb.should_stop is False


        algo.best_score = 0.5
        cb.on_iteration_end(3, algo, [], [])
        assert cb.should_stop is False


        cb.on_iteration_end(4, algo, [], [])
        cb.on_iteration_end(5, algo, [], [])
        assert cb.should_stop is False

    def test_improvement_resets_counter_max_mode(self) -> None:
        patience = 3
        cb = EarlyStoppingCallback(patience=patience, mode="max")
        algo = _MockAlgorithm(best_score=1.0)
        cb.on_experiment_start(algo)


        cb.on_iteration_end(0, algo, [], [])


        cb.on_iteration_end(1, algo, [], [])
        cb.on_iteration_end(2, algo, [], [])
        assert cb.should_stop is False


        algo.best_score = 2.0
        cb.on_iteration_end(3, algo, [], [])
        assert cb.should_stop is False


        cb.on_iteration_end(4, algo, [], [])
        cb.on_iteration_end(5, algo, [], [])
        assert cb.should_stop is False

    def test_min_mode_lower_is_improvement(self) -> None:
        cb = EarlyStoppingCallback(patience=2, min_delta=0.01, mode="min")
        algo = _MockAlgorithm(best_score=1.0)
        cb.on_experiment_start(algo)


        cb.on_iteration_end(0, algo, [], [])


        algo.best_score = 0.5
        cb.on_iteration_end(1, algo, [], [])


        cb.on_iteration_end(2, algo, [], [])
        cb.on_iteration_end(3, algo, [], [])


        assert cb.should_stop is True

    def test_max_mode_higher_is_improvement(self) -> None:
        cb = EarlyStoppingCallback(patience=2, min_delta=0.01, mode="max")
        algo = _MockAlgorithm(best_score=1.0)
        cb.on_experiment_start(algo)


        cb.on_iteration_end(0, algo, [], [])


        algo.best_score = 2.0
        cb.on_iteration_end(1, algo, [], [])


        cb.on_iteration_end(2, algo, [], [])
        cb.on_iteration_end(3, algo, [], [])

        assert cb.should_stop is True

    def test_min_delta_threshold(self) -> None:
        min_delta = 0.1
        cb = EarlyStoppingCallback(patience=2, min_delta=min_delta, mode="min")
        algo = _MockAlgorithm(best_score=1.0)
        cb.on_experiment_start(algo)


        cb.on_iteration_end(0, algo, [], [])


        algo.best_score = 1.0 - 0.05
        cb.on_iteration_end(1, algo, [], [])

        algo.best_score = 1.0 - 0.05
        cb.on_iteration_end(2, algo, [], [])


        assert cb.should_stop is True

    def test_patience_zero_stops_after_first_non_improving(self) -> None:
        cb = EarlyStoppingCallback(patience=0, mode="min")
        algo = _MockAlgorithm(best_score=1.0)
        cb.on_experiment_start(algo)


        cb.on_iteration_end(0, algo, [], [])
        assert cb.should_stop is False


        cb.on_iteration_end(1, algo, [], [])
        assert cb.should_stop is True

    def test_reusable_across_experiments(self) -> None:
        patience = 2
        cb = EarlyStoppingCallback(patience=patience, mode="min")
        algo = _MockAlgorithm(best_score=1.0)


        cb.on_experiment_start(algo)
        cb.on_iteration_end(0, algo, [], [])
        cb.on_iteration_end(1, algo, [], [])
        cb.on_iteration_end(2, algo, [], [])
        assert cb.should_stop is True


        algo.best_score = 5.0
        cb.on_experiment_start(algo)
        assert cb.should_stop is False

        cb.on_iteration_end(0, algo, [], [])
        assert cb.should_stop is False

    def test_continuous_improvement_never_stops(self) -> None:
        cb = EarlyStoppingCallback(patience=2, mode="min")
        algo = _MockAlgorithm(best_score=10.0)
        cb.on_experiment_start(algo)

        for i in range(20):
            algo.best_score = 10.0 - i * 0.5
            cb.on_iteration_end(i, algo, [], [])
            assert cb.should_stop is False

    def test_on_iteration_start_does_not_affect_stopping(
        self, mock_algo: _MockAlgorithm
    ) -> None:
        cb = EarlyStoppingCallback(patience=5)
        cb.on_experiment_start(mock_algo)
        cb.on_iteration_start(0, mock_algo)
        assert cb.should_stop is False







class TestCheckpointCallback:

    def test_default_every_n(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(directory=tmp_path)
        assert isinstance(cb, RunnerCallback)

    def test_every_n_custom(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(directory=tmp_path, every_n=5)
        assert isinstance(cb, RunnerCallback)

    def test_every_n_less_than_one_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            CheckpointCallback(directory=tmp_path, every_n=0)

    def test_every_n_negative_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            CheckpointCallback(directory=tmp_path, every_n=-1)

    def test_should_stop_always_false(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(directory=tmp_path)
        assert cb.should_stop is False

        algo = _MockAlgorithm()
        cb.on_experiment_start(algo)
        assert cb.should_stop is False

        cb.on_iteration_end(0, algo, [], [])
        assert cb.should_stop is False

        cb.on_experiment_end(algo)
        assert cb.should_stop is False

    def test_creates_directory_on_experiment_start(self, tmp_path: Path) -> None:
        ckpt_dir = tmp_path / "checkpoints"
        assert not ckpt_dir.exists()

        cb = CheckpointCallback(directory=ckpt_dir)
        cb.on_experiment_start(_MockAlgorithm())

        assert ckpt_dir.exists()
        assert ckpt_dir.is_dir()

    def test_saves_at_correct_iterations(self, tmp_path: Path) -> None:
        every_n = 3
        cb = CheckpointCallback(directory=tmp_path, every_n=every_n)
        algo = _MockAlgorithm()
        algo.state = {"gen": 0}
        cb.on_experiment_start(algo)

        for i in range(10):
            algo.state = {"gen": i}
            cb.on_iteration_end(i, algo, [], [])


        expected = [
            tmp_path / "checkpoint_000000.pt",
            tmp_path / "checkpoint_000003.pt",
            tmp_path / "checkpoint_000006.pt",
            tmp_path / "checkpoint_000009.pt",
        ]
        for path in expected:
            assert path.exists(), f"Expected checkpoint at {path}"

    def test_does_not_save_at_non_matching_iterations(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(directory=tmp_path, every_n=5)
        algo = _MockAlgorithm()
        cb.on_experiment_start(algo)


        cb.on_iteration_end(1, algo, [], [])

        unexpected = tmp_path / "checkpoint_000001.pt"
        assert not unexpected.exists()

    def test_file_naming_pattern(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(directory=tmp_path, every_n=1)
        algo = _MockAlgorithm()
        cb.on_experiment_start(algo)

        cb.on_iteration_end(0, algo, [], [])
        cb.on_iteration_end(42, algo, [], [])

        assert (tmp_path / "checkpoint_000000.pt").exists()
        assert (tmp_path / "checkpoint_000042.pt").exists()

    def test_saved_dict_has_required_keys(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(directory=tmp_path, every_n=1)
        algo = _MockAlgorithm()
        algo.state = {"population": ["u_x", "u_xx"], "gen": 5}
        cb.on_experiment_start(algo)

        cb.on_iteration_end(0, algo, [], [])

        ckpt_path = tmp_path / "checkpoint_000000.pt"
        data = torch.load(ckpt_path, weights_only=False)

        assert "iteration" in data
        assert "algorithm_state" in data
        assert data["iteration"] == 0
        assert data["algorithm_state"] == algo.state

    def test_saves_final_checkpoint_on_experiment_end(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(directory=tmp_path, every_n=10)
        algo = _MockAlgorithm()
        algo.state = {"final": True}
        cb.on_experiment_start(algo)


        cb.on_iteration_end(0, algo, [], [])

        cb.on_experiment_end(algo)

        final_path = tmp_path / "checkpoint_final.pt"
        assert final_path.exists()

        data = torch.load(final_path, weights_only=False)
        assert "algorithm_state" in data
        assert "iteration" in data
        assert data["iteration"] == 0

    def test_directory_already_exists(self, tmp_path: Path) -> None:
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        assert ckpt_dir.exists()

        cb = CheckpointCallback(directory=ckpt_dir)

        cb.on_experiment_start(_MockAlgorithm())
        assert ckpt_dir.exists()

    def test_nested_directory_creation(self, tmp_path: Path) -> None:
        nested_dir = tmp_path / "a" / "b" / "c"
        cb = CheckpointCallback(directory=nested_dir)
        cb.on_experiment_start(_MockAlgorithm())
        assert nested_dir.exists()







class TestCallbackEdgeCases:

    def test_empty_candidates_and_results(self, mock_algo: _MockAlgorithm) -> None:
        log_cb = LoggingCallback(every_n=1)
        es_cb = EarlyStoppingCallback(patience=5)

        log_cb.on_experiment_start(mock_algo)
        es_cb.on_experiment_start(mock_algo)


        log_cb.on_iteration_end(0, mock_algo, [], [])
        es_cb.on_iteration_end(0, mock_algo, [], [])

    def test_multiple_callbacks_coexist(self, tmp_path: Path) -> None:
        log_cb = LoggingCallback(every_n=1)
        es_cb = EarlyStoppingCallback(patience=2, mode="min")

        algo = _MockAlgorithm(best_score=1.0)

        log_cb.on_experiment_start(algo)
        es_cb.on_experiment_start(algo)


        for i in range(5):
            log_cb.on_iteration_end(i, algo, ["u_x"], [_make_eval_result()])
            es_cb.on_iteration_end(i, algo, ["u_x"], [_make_eval_result()])


        assert log_cb.should_stop is False


        assert es_cb.should_stop is True

    def test_on_iteration_start_is_callable(
        self, mock_algo: _MockAlgorithm, tmp_path: Path
    ) -> None:
        callbacks = [
            LoggingCallback(),
            EarlyStoppingCallback(),
            CheckpointCallback(directory=tmp_path),
        ]

        for cb in callbacks:
            cb.on_experiment_start(mock_algo)
            cb.on_iteration_start(0, mock_algo)

    def test_checkpoint_and_early_stopping_together(self, tmp_path: Path) -> None:
        ckpt_cb = CheckpointCallback(directory=tmp_path, every_n=1)
        es_cb = EarlyStoppingCallback(patience=1, mode="min")

        algo = _MockAlgorithm(best_score=1.0)

        ckpt_cb.on_experiment_start(algo)
        es_cb.on_experiment_start(algo)


        ckpt_cb.on_iteration_end(0, algo, [], [])
        es_cb.on_iteration_end(0, algo, [], [])


        ckpt_cb.on_iteration_end(1, algo, [], [])
        es_cb.on_iteration_end(1, algo, [], [])


        assert es_cb.should_stop is True
        assert ckpt_cb.should_stop is False
        assert (tmp_path / "checkpoint_000000.pt").exists()
        assert (tmp_path / "checkpoint_000001.pt").exists()








def _fire_iteration_end(
    callback: Any,
    iteration: int,
    algorithm: _MockAlgorithm,
) -> None:
    callback.on_iteration_end(
        iteration=iteration,
        algorithm=algorithm,
        candidates=[],
        results=[],
    )







class TestEarlyStoppingNumerical:



    @pytest.mark.numerical
    def test_nan_score_triggers_stop_after_patience(self) -> None:
        patience = 3
        cb = EarlyStoppingCallback(patience=patience, mode="min")
        algo = _MockAlgorithm(best_score=float("nan"))

        for i in range(patience):
            _fire_iteration_end(cb, iteration=i, algorithm=algo)

        assert cb.should_stop is True

    @pytest.mark.numerical
    def test_nan_score_does_not_corrupt_best(self) -> None:
        cb = EarlyStoppingCallback(patience=5, mode="min")


        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))


        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=float("nan")))


        _fire_iteration_end(cb, 2, _MockAlgorithm(best_score=0.5))

        assert cb.should_stop is False

    @pytest.mark.numerical
    def test_nan_interspersed_with_valid_scores(self) -> None:
        cb = EarlyStoppingCallback(patience=2, mode="min")

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=float("nan")))
        _fire_iteration_end(cb, 2, _MockAlgorithm(best_score=float("nan")))

        assert cb.should_stop is True

    @pytest.mark.numerical
    def test_nan_in_max_mode_increments_counter(self) -> None:
        patience = 2
        cb = EarlyStoppingCallback(patience=patience, mode="max")

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=float("nan")))
        _fire_iteration_end(cb, 2, _MockAlgorithm(best_score=float("nan")))

        assert cb.should_stop is True



    @pytest.mark.numerical
    def test_min_mode_neg_inf_is_ultimate_improvement(self) -> None:
        cb = EarlyStoppingCallback(patience=2, mode="min")


        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=1.0))

        _fire_iteration_end(cb, 2, _MockAlgorithm(best_score=float("-inf")))

        assert cb.should_stop is False

    @pytest.mark.numerical
    def test_min_mode_pos_inf_is_no_improvement(self) -> None:
        cb = EarlyStoppingCallback(patience=2, mode="min")

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=float("inf")))
        _fire_iteration_end(cb, 2, _MockAlgorithm(best_score=float("inf")))

        assert cb.should_stop is True

    @pytest.mark.numerical
    def test_max_mode_pos_inf_is_ultimate_improvement(self) -> None:
        cb = EarlyStoppingCallback(patience=2, mode="max")

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=1.0))
        _fire_iteration_end(cb, 2, _MockAlgorithm(best_score=float("inf")))

        assert cb.should_stop is False

    @pytest.mark.numerical
    def test_max_mode_neg_inf_is_no_improvement(self) -> None:
        cb = EarlyStoppingCallback(patience=2, mode="max")

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=float("-inf")))
        _fire_iteration_end(cb, 2, _MockAlgorithm(best_score=float("-inf")))

        assert cb.should_stop is True

    @pytest.mark.numerical
    def test_min_mode_inf_score_against_inf_best_no_nan(self) -> None:
        cb = EarlyStoppingCallback(patience=1, mode="min")



        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=float("inf")))

        assert cb.should_stop is True

    @pytest.mark.numerical
    def test_max_mode_neg_inf_score_against_neg_inf_best_no_nan(self) -> None:
        cb = EarlyStoppingCallback(patience=1, mode="max")


        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=float("-inf")))

        assert cb.should_stop is True



    @pytest.mark.unit
    def test_min_delta_rejects_micro_improvement_min_mode(self) -> None:
        cb = EarlyStoppingCallback(patience=1, min_delta=0.1, mode="min")

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))

        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=0.95))

        assert cb.should_stop is True

    @pytest.mark.unit
    def test_min_delta_accepts_sufficient_improvement_min_mode(self) -> None:
        cb = EarlyStoppingCallback(patience=1, min_delta=0.1, mode="min")

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))

        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=0.85))

        assert cb.should_stop is False

    @pytest.mark.unit
    def test_min_delta_rejects_micro_improvement_max_mode(self) -> None:
        cb = EarlyStoppingCallback(patience=1, min_delta=0.1, mode="max")

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))

        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=1.05))

        assert cb.should_stop is True

    @pytest.mark.unit
    def test_min_delta_accepts_sufficient_improvement_max_mode(self) -> None:
        cb = EarlyStoppingCallback(patience=1, min_delta=0.1, mode="max")

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))

        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=1.15))

        assert cb.should_stop is False

    @pytest.mark.unit
    def test_min_delta_exact_boundary_is_not_improvement(self) -> None:
        cb = EarlyStoppingCallback(patience=1, min_delta=0.1, mode="min")

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))

        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=0.9))

        assert cb.should_stop is True







class TestEarlyStoppingStateReset:

    @pytest.mark.unit
    def test_reset_clears_should_stop(self) -> None:
        cb = EarlyStoppingCallback(patience=1, mode="min")
        algo = _MockAlgorithm(best_score=1.0)


        _fire_iteration_end(cb, 0, algo)
        _fire_iteration_end(cb, 1, algo)
        assert cb.should_stop is True


        cb.on_experiment_start(algo)
        assert cb.should_stop is False

    @pytest.mark.unit
    def test_reset_clears_counter(self) -> None:
        patience = 3
        cb = EarlyStoppingCallback(patience=patience, mode="min")
        algo_stale = _MockAlgorithm(best_score=1.0)


        _fire_iteration_end(cb, 0, algo_stale)
        _fire_iteration_end(cb, 1, algo_stale)


        cb.on_experiment_start(algo_stale)


        _fire_iteration_end(cb, 0, algo_stale)

        for i in range(1, patience):
            _fire_iteration_end(cb, i, algo_stale)
        assert cb.should_stop is False

        _fire_iteration_end(cb, patience, algo_stale)
        assert cb.should_stop is True

    @pytest.mark.unit
    def test_reset_restores_sentinel_best_min_mode(self) -> None:
        cb = EarlyStoppingCallback(patience=5, mode="min")


        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=0.5))


        cb.on_experiment_start(_MockAlgorithm())



        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=100.0))

        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=100.0))

        assert cb.should_stop is False

    @pytest.mark.unit
    def test_reset_restores_sentinel_best_max_mode(self) -> None:
        cb = EarlyStoppingCallback(patience=5, mode="max")


        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=100.0))


        cb.on_experiment_start(_MockAlgorithm())


        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=-100.0))
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=-100.0))

        assert cb.should_stop is False

    @pytest.mark.unit
    def test_full_reuse_two_experiments(self) -> None:
        cb = EarlyStoppingCallback(patience=2, mode="min")


        cb.on_experiment_start(_MockAlgorithm())
        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=1.0))
        _fire_iteration_end(cb, 2, _MockAlgorithm(best_score=1.0))
        assert cb.should_stop is True


        cb.on_experiment_start(_MockAlgorithm())
        assert cb.should_stop is False

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=0.5))
        assert cb.should_stop is False
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=0.3))
        assert cb.should_stop is False
        _fire_iteration_end(cb, 2, _MockAlgorithm(best_score=0.1))
        assert cb.should_stop is False







class TestEarlyStoppingPublicMode:

    @pytest.mark.unit
    @pytest.mark.parametrize("mode", ["min", "max"])
    def test_mode_property_returns_constructor_arg(self, mode: str) -> None:
        cb = EarlyStoppingCallback(mode=mode, patience=5)
        assert cb.mode == mode


        assert isinstance(type(cb).mode, property), (
            "EarlyStoppingCallback.mode must be a @property — not a plain "
            "attribute — so the facade guard can rely on a stable, "
            "documented public API surface."
        )

    @pytest.mark.unit
    def test_mode_property_is_read_only(self) -> None:
        cb = EarlyStoppingCallback(mode="min", patience=5)
        with pytest.raises(AttributeError):
            cb.mode = "max"







class TestCheckpointDesign:



    @pytest.mark.unit
    def test_checkpoint_is_not_raw_state(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(directory=tmp_path, every_n=1)
        algo = _MockAlgorithm()
        algo.state = {"weights": [1.0]}
        cb.on_experiment_start(algo)

        _fire_iteration_end(cb, iteration=0, algorithm=algo)

        pt_files = list(tmp_path.glob("checkpoint_0*.pt"))
        assert len(pt_files) == 1

        data = torch.load(pt_files[0], weights_only=False)


        assert set(data.keys()) == {
            "version",
            "iteration",
            "algorithm_state",
            "best_score",
            "best_expression",
            "algorithm",
        }

    @pytest.mark.unit
    def test_checkpoint_iteration_value_matches(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(directory=tmp_path, every_n=5)
        algo = _MockAlgorithm()
        algo.state = {"gen": 7}
        cb.on_experiment_start(algo)

        _fire_iteration_end(cb, iteration=5, algorithm=algo)

        pt_files = list(tmp_path.glob("checkpoint_0*.pt"))
        assert len(pt_files) == 1

        data = torch.load(pt_files[0], weights_only=False)
        assert data["iteration"] == 5

    @pytest.mark.unit
    def test_checkpoint_algorithm_state_matches(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(directory=tmp_path, every_n=1)
        state = {"generation": 42, "population": ["a", "b"]}
        algo = _MockAlgorithm()
        algo.state = state
        cb.on_experiment_start(algo)

        _fire_iteration_end(cb, iteration=0, algorithm=algo)

        pt_files = list(tmp_path.glob("checkpoint_0*.pt"))
        data = torch.load(pt_files[0], weights_only=False)
        assert data["algorithm_state"] == state



    @pytest.mark.unit
    def test_final_checkpoint_exact_filename(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(directory=tmp_path, every_n=10)
        algo = _MockAlgorithm()
        algo.state = {}
        cb.on_experiment_start(algo)

        cb.on_experiment_end(algo)

        files = [f.name for f in tmp_path.iterdir()]
        assert "checkpoint_final.pt" in files

    @pytest.mark.unit
    def test_final_checkpoint_contains_state(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(directory=tmp_path, every_n=10)
        state = {"generation": 100, "converged": True}
        algo = _MockAlgorithm()
        algo.state = state
        cb.on_experiment_start(algo)

        cb.on_experiment_end(algo)

        final_path = tmp_path / "checkpoint_final.pt"
        data = torch.load(final_path, weights_only=False)
        assert isinstance(data, dict)
        assert "algorithm_state" in data
        assert data["algorithm_state"] == state







class TestIterationIndexing:



    @pytest.mark.unit
    def test_logging_not_one_indexed(self, caplog: pytest.LogCaptureFixture) -> None:
        cb = LoggingCallback(every_n=5)
        algo = _MockAlgorithm(best_score=0.1, best_expression="u_x")
        cb.on_experiment_start(algo)

        for i in [1, 6, 11]:
            caplog.clear()
            with caplog.at_level(logging.INFO):
                _fire_iteration_end(cb, iteration=i, algorithm=algo)
            assert len(caplog.records) == 0, (
                f"Should not log at 1-indexed iteration {i}"
            )

    @pytest.mark.unit
    def test_logging_zero_indexed_full_sequence(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cb = LoggingCallback(every_n=5)
        algo = _MockAlgorithm(best_score=0.1, best_expression="u_x")
        cb.on_experiment_start(algo)

        logged_iterations: list[int] = []
        for i in range(15):
            caplog.clear()
            with caplog.at_level(logging.INFO):
                _fire_iteration_end(cb, iteration=i, algorithm=algo)
            if caplog.records:
                logged_iterations.append(i)

        assert logged_iterations == [0, 5, 10]

    @pytest.mark.unit
    def test_checkpoint_zero_indexed_full_sequence(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(directory=tmp_path, every_n=5)
        algo = _MockAlgorithm()
        algo.state = {"gen": 0}
        cb.on_experiment_start(algo)

        for i in range(15):
            _fire_iteration_end(cb, iteration=i, algorithm=algo)


        pt_files = sorted(
            f for f in tmp_path.glob("checkpoint_*.pt") if "final" not in f.name
        )
        assert len(pt_files) == 3

    @pytest.mark.unit
    def test_checkpoint_not_one_indexed(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(directory=tmp_path, every_n=5)
        algo = _MockAlgorithm()
        algo.state = {"gen": 1}
        cb.on_experiment_start(algo)

        _fire_iteration_end(cb, iteration=1, algorithm=algo)

        pt_files = [
            f for f in tmp_path.glob("checkpoint_*.pt") if "final" not in f.name
        ]
        assert len(pt_files) == 0



    @pytest.mark.unit
    def test_experiment_start_log_contains_start(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cb = LoggingCallback(every_n=1)
        algo = _MockAlgorithm()

        with caplog.at_level(logging.INFO):
            cb.on_experiment_start(algo)

        assert len(caplog.records) > 0
        combined = " ".join(r.message.lower() for r in caplog.records)
        assert "start" in combined

    @pytest.mark.unit
    def test_experiment_end_log_contains_best_score(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cb = LoggingCallback(every_n=1)
        algo = _MockAlgorithm(best_score=0.001, best_expression="mul(u, u_x)")

        with caplog.at_level(logging.INFO):
            cb.on_experiment_end(algo)

        assert len(caplog.records) > 0
        combined = " ".join(r.message for r in caplog.records)

        assert "0.001" in combined or "1e-03" in combined

    @pytest.mark.unit
    def test_experiment_end_log_contains_best_expression(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cb = LoggingCallback(every_n=1)
        algo = _MockAlgorithm(best_score=0.001, best_expression="mul(u, u_x)")

        with caplog.at_level(logging.INFO):
            cb.on_experiment_end(algo)

        assert len(caplog.records) > 0
        combined = " ".join(r.message for r in caplog.records)
        assert "mul(u, u_x)" in combined







class TestVizDataCollectorSmoke:

    @pytest.mark.smoke
    def test_importable(self) -> None:
        from kd.search.callbacks import VizDataCollector

        assert VizDataCollector is not None

    @pytest.mark.smoke
    def test_construction_with_recorder(self) -> None:
        from kd.search.callbacks import VizDataCollector
        from kd.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        assert collector is not None

    @pytest.mark.smoke
    def test_satisfies_runner_callback_protocol(self) -> None:
        from kd.search.callbacks import VizDataCollector
        from kd.search.recorder import VizRecorder

        collector = VizDataCollector(VizRecorder())
        assert isinstance(collector, RunnerCallback)


class TestVizDataCollectorBehavior:

    @pytest.mark.unit
    def test_should_stop_always_false(self) -> None:
        from kd.search.callbacks import VizDataCollector
        from kd.search.recorder import VizRecorder

        collector = VizDataCollector(VizRecorder())

        assert collector.should_stop is False

    @pytest.mark.unit
    def test_should_stop_false_after_many_iterations(self) -> None:
        from kd.search.callbacks import VizDataCollector
        from kd.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm(best_score=1.0, best_expression="u_x")

        for i in range(50):
            collector.on_iteration_end(i, algo, ["u_x"], [_make_eval_result()])

        assert collector.should_stop is False

    @pytest.mark.unit
    def test_on_iteration_end_logs_best_score(self) -> None:
        from kd.search.callbacks import VizDataCollector
        from kd.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm(best_score=0.42, best_expression="u_x")

        collector.on_iteration_end(0, algo, ["u_x"], [_make_eval_result()])

        scores = recorder.get("_best_score")
        assert len(scores) == 1
        assert scores[0] == 0.42

    @pytest.mark.unit
    def test_on_iteration_end_logs_best_expr(self) -> None:
        from kd.search.callbacks import VizDataCollector
        from kd.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm(best_score=1.0, best_expression="mul(u, u_x)")

        collector.on_iteration_end(0, algo, ["u_x"], [_make_eval_result()])

        exprs = recorder.get("_best_expr")
        assert len(exprs) == 1
        assert exprs[0] == "mul(u, u_x)"

    @pytest.mark.unit
    def test_on_iteration_end_logs_n_candidates(self) -> None:
        from kd.search.callbacks import VizDataCollector
        from kd.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm(best_score=1.0, best_expression="u_x")

        candidates = ["u_x", "u_xx", "mul(u, u_x)"]
        results = [_make_eval_result() for _ in candidates]
        collector.on_iteration_end(0, algo, candidates, results)

        n_cands = recorder.get("_n_candidates")
        assert len(n_cands) == 1
        assert n_cands[0] == 3

    @pytest.mark.unit
    def test_multiple_iterations_accumulate(self) -> None:
        from kd.search.callbacks import VizDataCollector
        from kd.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        n_iterations = 5

        for i in range(n_iterations):
            score = 1.0 / (i + 1)
            algo = _MockAlgorithm(best_score=score, best_expression=f"expr_{i}")
            collector.on_iteration_end(i, algo, ["c"], [_make_eval_result()])

        scores = recorder.get("_best_score")
        exprs = recorder.get("_best_expr")
        n_cands = recorder.get("_n_candidates")


        assert len(scores) == n_iterations
        assert len(exprs) == n_iterations
        assert len(n_cands) == n_iterations


        for j in range(1, n_iterations):
            assert scores[j] < scores[j - 1]

    @pytest.mark.unit
    def test_on_experiment_start_is_noop(self) -> None:
        from kd.search.callbacks import VizDataCollector
        from kd.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm()

        collector.on_experiment_start(algo)

        assert len(recorder.keys()) == 0

    @pytest.mark.unit
    def test_on_experiment_end_is_noop(self) -> None:
        from kd.search.callbacks import VizDataCollector
        from kd.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm()

        collector.on_experiment_end(algo)

        assert len(recorder.keys()) == 0

    @pytest.mark.unit
    def test_on_iteration_start_is_noop(self) -> None:
        from kd.search.callbacks import VizDataCollector
        from kd.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm()

        collector.on_iteration_start(0, algo)

        assert len(recorder.keys()) == 0


class TestVizDataCollectorEdgeCases:

    @pytest.mark.unit
    def test_disabled_recorder_no_crash(self) -> None:
        from kd.search.callbacks import VizDataCollector
        from kd.search.recorder import VizRecorder

        recorder = VizRecorder(enabled=False)
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm(best_score=1.0, best_expression="u_x")


        collector.on_iteration_end(0, algo, ["u_x"], [_make_eval_result()])


        assert len(recorder.keys()) == 0

    @pytest.mark.unit
    def test_empty_candidates_list(self) -> None:
        from kd.search.callbacks import VizDataCollector
        from kd.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm(best_score=1.0, best_expression="u_x")

        collector.on_iteration_end(0, algo, [], [])

        n_cands = recorder.get("_n_candidates")
        assert len(n_cands) == 1
        assert n_cands[0] == 0

    @pytest.mark.numerical
    def test_inf_best_score_logged(self) -> None:
        from kd.search.callbacks import VizDataCollector
        from kd.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm(best_score=float("inf"), best_expression="")

        collector.on_iteration_end(0, algo, ["c"], [_make_eval_result()])

        scores = recorder.get("_best_score")
        assert len(scores) == 1
        assert scores[0] == float("inf")

    @pytest.mark.numerical
    def test_nan_best_score_logged(self) -> None:
        import math

        from kd.search.callbacks import VizDataCollector
        from kd.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm(best_score=float("nan"), best_expression="bad")

        collector.on_iteration_end(0, algo, ["c"], [_make_eval_result()])

        scores = recorder.get("_best_score")
        assert len(scores) == 1
        assert math.isnan(scores[0])







class TestAtomicTorchSave:

    @pytest.mark.unit
    def test_roundtrips_and_leaves_no_tmp_residue(self, tmp_path: Path) -> None:
        from kd.search.callbacks import atomic_torch_save

        target = tmp_path / "ckpt.pt"
        payload = {"version": 1, "iteration": 3, "algorithm_state": {"x": 1}}
        atomic_torch_save(payload, target)

        assert target.exists()
        assert torch.load(target, weights_only=False) == payload
        assert list(tmp_path.glob("*.tmp")) == [], "staging file must be cleaned up"

    @pytest.mark.unit
    def test_overwrite_replaces_existing_target_atomically(
        self, tmp_path: Path
    ) -> None:
        from kd.search.callbacks import atomic_torch_save

        target = tmp_path / "ckpt.pt"
        atomic_torch_save({"iteration": 1}, target)
        atomic_torch_save({"iteration": 2}, target)

        assert torch.load(target, weights_only=False) == {"iteration": 2}
        assert list(tmp_path.glob("*.tmp")) == []

    @pytest.mark.unit
    def test_callback_writes_are_atomic_no_residue(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(directory=tmp_path, every_n=1)
        algo = _MockAlgorithm(best_score=1.0, best_expression="u_x")
        algo.state = {"gen": 0}
        cb.on_experiment_start(algo)
        cb.on_iteration_end(0, algo, ["u_x"], [_make_eval_result()])
        cb.on_experiment_end(algo)

        assert (tmp_path / "checkpoint_000000.pt").exists()
        assert (tmp_path / "checkpoint_final.pt").exists()
        assert list(tmp_path.glob("*.tmp")) == []
