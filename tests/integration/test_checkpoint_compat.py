
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from kd.search.callbacks import CheckpointCallback
from kd.search.protocol import PlatformComponents
from kd.search.runner import ExperimentRunner
from tests.unit.search._runner_mocks import (
    RecordingAlgorithm,
    StatefulAlgorithm,
)





AUTHORITATIVE_KEYS = frozenset(
    {
        "version",
        "iteration",
        "algorithm_state",
        "best_score",
        "best_expression",
    }
)







@pytest.fixture
def mock_components() -> PlatformComponents:
    return PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=MagicMock(),
        registry=MagicMock(),
    )







def _run_experiment_with_checkpoint(
    algorithm: RecordingAlgorithm | StatefulAlgorithm,
    components: PlatformComponents,
    checkpoint_dir: Path,
    max_iterations: int = 5,
    every_n: int = 2,
    batch_size: int = 3,
) -> ExperimentRunner:
    cb = CheckpointCallback(directory=checkpoint_dir, every_n=every_n)
    runner = ExperimentRunner(
        algorithm=algorithm,
        max_iterations=max_iterations,
        batch_size=batch_size,
        callbacks=[cb],
    )
    runner.run(components)
    return runner


def _load_checkpoint_data(path: Path) -> dict:
    return torch.load(path, weights_only=False)







@pytest.mark.integration
class TestCheckpointSchemaCompleteness:

    def test_iteration_checkpoint_has_all_keys(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        algo = RecordingAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=3,
            every_n=1,
        )


        ckpt_path = tmp_path / "checkpoint_000000.pt"
        assert ckpt_path.exists(), "Expected iteration checkpoint at iteration 0"

        data = _load_checkpoint_data(ckpt_path)
        missing = AUTHORITATIVE_KEYS - set(data.keys())
        assert not missing, (
            f"Iteration checkpoint missing keys: {missing}. "
            f"Got keys: {set(data.keys())}"
        )

    def test_final_checkpoint_has_all_keys(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        algo = RecordingAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=3,
            every_n=10,
        )

        ckpt_path = tmp_path / "checkpoint_final.pt"
        assert ckpt_path.exists(), "Expected final checkpoint"

        data = _load_checkpoint_data(ckpt_path)
        missing = AUTHORITATIVE_KEYS - set(data.keys())
        assert not missing, (
            f"Final checkpoint missing keys: {missing}. Got keys: {set(data.keys())}"
        )

    def test_version_field_value(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        algo = RecordingAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=2,
            every_n=1,
        )

        data = _load_checkpoint_data(tmp_path / "checkpoint_000000.pt")
        assert "version" in data, "Checkpoint must contain 'version' key"
        assert isinstance(data["version"], int), "version must be int"
        assert data["version"] >= 1, "version must be >= 1"







@pytest.mark.integration
class TestCheckpointLoadCompat:

    def test_load_iteration_checkpoint_no_error(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        algo = RecordingAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=3,
            every_n=1,
        )


        fresh_algo = RecordingAlgorithm()
        fresh_runner = ExperimentRunner(algorithm=fresh_algo)
        ckpt_path = tmp_path / "checkpoint_000000.pt"

        fresh_runner.load_checkpoint(ckpt_path)

    def test_load_final_checkpoint_no_error(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        algo = RecordingAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=3,
            every_n=10,
        )

        fresh_algo = RecordingAlgorithm()
        fresh_runner = ExperimentRunner(algorithm=fresh_algo)
        ckpt_path = tmp_path / "checkpoint_final.pt"
        fresh_runner.load_checkpoint(ckpt_path)







@pytest.mark.integration
class TestCheckpointFieldValues:

    def test_iteration_field_matches_save_time(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        algo = RecordingAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=5,
            every_n=2,
        )


        for expected_iter in [0, 2, 4]:
            ckpt = tmp_path / f"checkpoint_{expected_iter:06d}.pt"
            assert ckpt.exists(), f"Expected checkpoint at iteration {expected_iter}"
            data = _load_checkpoint_data(ckpt)
            assert data["iteration"] == expected_iter, (
                f"Expected iteration={expected_iter}, got {data['iteration']}"
            )

    def test_algorithm_state_is_dict(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        algo = StatefulAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=3,
            every_n=1,
        )

        data = _load_checkpoint_data(tmp_path / "checkpoint_000000.pt")
        assert isinstance(data["algorithm_state"], dict)

    def test_best_score_is_numeric(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        algo = StatefulAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=3,
            every_n=1,
        )


        data = _load_checkpoint_data(tmp_path / "checkpoint_000000.pt")
        assert "best_score" in data, "Checkpoint must contain 'best_score'"
        score = data["best_score"]
        assert isinstance(score, (int, float)), (
            f"best_score must be numeric, got {type(score)}"
        )

    def test_best_expression_is_str(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        algo = StatefulAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=3,
            every_n=1,
        )

        data = _load_checkpoint_data(tmp_path / "checkpoint_000000.pt")
        assert "best_expression" in data, "Checkpoint must contain 'best_expression'"
        assert isinstance(data["best_expression"], str)







@pytest.mark.integration
class TestCheckpointRoundTrip:

    def test_stateful_algorithm_round_trip(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        algo = StatefulAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=5,
            every_n=1,
            batch_size=3,
        )


        ckpt_path = tmp_path / "checkpoint_000002.pt"
        assert ckpt_path.exists()
        data = _load_checkpoint_data(ckpt_path)


        fresh_algo = StatefulAlgorithm()
        fresh_runner = ExperimentRunner(algorithm=fresh_algo)
        fresh_runner.load_checkpoint(ckpt_path)


        assert fresh_algo.state["generation"] == data["algorithm_state"]["generation"]
        assert fresh_algo.state["population"] == data["algorithm_state"]["population"]

    def test_checkpoint_preserves_best_score_and_expression(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        algo = StatefulAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=5,
            every_n=2,
            batch_size=3,
        )


        ckpt_path = tmp_path / "checkpoint_000002.pt"
        assert ckpt_path.exists()
        data = _load_checkpoint_data(ckpt_path)




        algo_internal = data["algorithm_state"]
        assert "best_score" in data, "Checkpoint missing 'best_score' top-level field"
        assert "best_expression" in data, (
            "Checkpoint missing 'best_expression' top-level field"
        )


        assert data["best_score"] == algo_internal["best_score"], (
            f"best_score mismatch: checkpoint top-level has {data['best_score']}, "
            f"but algorithm_state has {algo_internal['best_score']}"
        )
        assert data["best_expression"] == algo_internal["best_expression"], (
            f"best_expression mismatch: checkpoint top-level has "
            f"{data['best_expression']!r}, but algorithm_state has "
            f"{algo_internal['best_expression']!r}"
        )

    def test_final_checkpoint_round_trip(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        algo = StatefulAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=4,
            every_n=10,
            batch_size=2,
        )


        original_state = algo.state.copy()

        ckpt_path = tmp_path / "checkpoint_final.pt"
        assert ckpt_path.exists()

        fresh_algo = StatefulAlgorithm()
        fresh_runner = ExperimentRunner(algorithm=fresh_algo)
        fresh_runner.load_checkpoint(ckpt_path)


        assert fresh_algo.state["generation"] == original_state["generation"]
        assert fresh_algo.state["population"] == original_state["population"]







@pytest.mark.integration
class TestCheckpointBoundary:

    def test_first_iteration_checkpoint(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        algo = StatefulAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=1,
            every_n=1,
        )

        ckpt_path = tmp_path / "checkpoint_000000.pt"
        assert ckpt_path.exists()

        data = _load_checkpoint_data(ckpt_path)
        assert data["iteration"] == 0
        assert isinstance(data["algorithm_state"], dict)

    def test_final_checkpoint_iteration_not_negative_when_zero_iterations(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        algo = StatefulAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=0,
            every_n=1,
        )

        ckpt_path = tmp_path / "checkpoint_final.pt"
        assert ckpt_path.exists(), "Expected final checkpoint"

        data = _load_checkpoint_data(ckpt_path)
        assert data["iteration"] >= 0, (
            f"Final checkpoint iteration must be >= 0, got {data['iteration']}. "
            "CheckpointCallback leaked its internal sentinel (-1) because "
            "no iterations ran and on_iteration_end was never called."
        )

    def test_callback_and_runner_save_same_keys(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        algo = StatefulAlgorithm()
        cb = CheckpointCallback(directory=tmp_path / "cb_dir", every_n=1)
        runner = ExperimentRunner(
            algorithm=algo,
            max_iterations=3,
            batch_size=2,
            callbacks=[cb],
        )
        runner.run(mock_components)


        runner_ckpt = tmp_path / "runner_ckpt.pt"
        runner.save_checkpoint(runner_ckpt)


        runner_data = _load_checkpoint_data(runner_ckpt)
        cb_data = _load_checkpoint_data(tmp_path / "cb_dir" / "checkpoint_final.pt")

        runner_keys = set(runner_data.keys())
        cb_keys = set(cb_data.keys())

        assert runner_keys == cb_keys, (
            f"Key mismatch! Runner has {runner_keys - cb_keys} extra, "
            f"Callback has {cb_keys - runner_keys} extra. "
            f"Runner keys: {runner_keys}, Callback keys: {cb_keys}"
        )

    def test_callback_iteration_ckpt_and_runner_save_same_keys(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        algo = RecordingAlgorithm()
        cb = CheckpointCallback(directory=tmp_path / "cb_dir", every_n=1)
        runner = ExperimentRunner(
            algorithm=algo,
            max_iterations=2,
            batch_size=2,
            callbacks=[cb],
        )
        runner.run(mock_components)

        runner_ckpt = tmp_path / "runner_ckpt.pt"
        runner.save_checkpoint(runner_ckpt)

        runner_keys = set(_load_checkpoint_data(runner_ckpt).keys())
        iter_keys = set(
            _load_checkpoint_data(tmp_path / "cb_dir" / "checkpoint_000000.pt").keys()
        )

        assert runner_keys == iter_keys, (
            f"Iteration checkpoint keys {iter_keys} differ from "
            f"runner checkpoint keys {runner_keys}"
        )
