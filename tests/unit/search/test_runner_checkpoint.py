
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from kd.search.callbacks import CHECKPOINT_VERSION, build_checkpoint_payload
from kd.search.protocol import PlatformComponents
from kd.search.runner import ExperimentRunner
from tests.unit.search._runner_mocks import (
    StatefulAlgorithm,
    StateVerifyingAlgorithm,
)






class TestCheckpointSave:

    @pytest.mark.unit
    def test_save_creates_file(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=2)
        runner.run(mock_components)

        ckpt_path = tmp_path / "checkpoint.pt"
        runner.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

    @pytest.mark.unit
    def test_save_creates_parent_dirs(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=1)
        runner.run(mock_components)

        nested = tmp_path / "deep" / "nested" / "dir" / "ckpt.pt"
        runner.save_checkpoint(nested)
        assert nested.exists()

    @pytest.mark.unit
    def test_save_contains_version(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=1)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)
        data = torch.load(ckpt_path, weights_only=False)
        assert "version" in data
        assert data["version"] == 1

    @pytest.mark.unit
    def test_save_contains_required_keys(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=3)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)
        data = torch.load(ckpt_path, weights_only=False)

        required_keys = {
            "version",
            "iteration",
            "algorithm_state",
            "best_score",
            "best_expression",
        }
        assert required_keys.issubset(set(data.keys())), (
            f"Missing keys: {required_keys - set(data.keys())}"
        )

    @pytest.mark.unit
    def test_save_iteration_matches_run(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        max_iter = 5
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=max_iter)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)
        data = torch.load(ckpt_path, weights_only=False)
        assert data["iteration"] == max_iter

    @pytest.mark.unit
    def test_save_algorithm_state_is_dict(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=1)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)
        data = torch.load(ckpt_path, weights_only=False)
        assert isinstance(data["algorithm_state"], dict)

    @pytest.mark.unit
    def test_save_best_score_from_algorithm(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=3)
        runner.run(mock_components)

        expected_score = stateful_algorithm.best_score
        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)
        data = torch.load(ckpt_path, weights_only=False)
        assert data["best_score"] == expected_score

    @pytest.mark.unit
    def test_save_best_expression_from_algorithm(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=3)
        runner.run(mock_components)

        expected_expr = stateful_algorithm.best_expression
        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)
        data = torch.load(ckpt_path, weights_only=False)
        assert data["best_expression"] == expected_expr







class TestCheckpointLoad:

    @pytest.mark.unit
    def test_load_restores_algorithm_state(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=3)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)


        saved_state = torch.load(ckpt_path, weights_only=False)["algorithm_state"]


        fresh_algo = StatefulAlgorithm()
        fresh_runner = ExperimentRunner(algorithm=fresh_algo, max_iterations=10)
        fresh_runner.load_checkpoint(ckpt_path)


        assert fresh_algo.state == saved_state

    @pytest.mark.unit
    def test_load_sets_algorithm_state_setter(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        algo = StateVerifyingAlgorithm()
        runner = ExperimentRunner(algorithm=algo, max_iterations=1)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)


        fresh_algo = StateVerifyingAlgorithm()
        assert fresh_algo.state.get("restored") is False

        fresh_runner = ExperimentRunner(algorithm=fresh_algo, max_iterations=1)
        fresh_runner.load_checkpoint(ckpt_path)


        assert fresh_algo.state.get("restored") is True

    @pytest.mark.unit
    def test_round_trip_preserves_generation(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=4)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)

        gen_before = stateful_algorithm.state["generation"]

        fresh_algo = StatefulAlgorithm()
        fresh_runner = ExperimentRunner(algorithm=fresh_algo, max_iterations=1)
        fresh_runner.load_checkpoint(ckpt_path)

        assert fresh_algo.state["generation"] == gen_before

    @pytest.mark.unit
    def test_round_trip_preserves_population(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=2)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)

        pop_before = stateful_algorithm.state["population"]

        fresh_algo = StatefulAlgorithm()
        fresh_runner = ExperimentRunner(algorithm=fresh_algo, max_iterations=1)
        fresh_runner.load_checkpoint(ckpt_path)

        assert fresh_algo.state["population"] == pop_before

    @pytest.mark.unit
    def test_load_nonexistent_file_raises(
        self,
        stateful_algorithm: StatefulAlgorithm,
        tmp_path: Path,
    ) -> None:
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=1)
        bogus_path = tmp_path / "does_not_exist.pt"

        with pytest.raises((FileNotFoundError, OSError)):
            runner.load_checkpoint(bogus_path)







class TestCheckpointResumption:

    @pytest.mark.unit
    def test_checkpoint_iteration_stored(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=7)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)

        data = torch.load(ckpt_path, weights_only=False)
        assert data["iteration"] == 7

    @pytest.mark.unit
    def test_load_restores_internal_iteration(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=5)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)


        fresh_algo = StatefulAlgorithm()
        fresh_runner = ExperimentRunner(algorithm=fresh_algo, max_iterations=100)
        fresh_runner.load_checkpoint(ckpt_path)


        ckpt2_path = tmp_path / "ckpt2.pt"
        fresh_runner.save_checkpoint(ckpt2_path)

        data2 = torch.load(ckpt2_path, weights_only=False)
        assert data2["iteration"] == 5







class TestCheckpointEdgeCases:

    @pytest.mark.unit
    def test_save_before_run(
        self,
        stateful_algorithm: StatefulAlgorithm,
        tmp_path: Path,
    ) -> None:
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=10)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)

        data = torch.load(ckpt_path, weights_only=False)
        assert data["version"] == 1

        assert data["iteration"] == 0

    @pytest.mark.unit
    def test_save_after_early_stop(
        self,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:

        class _StopCallback:
            def __init__(self) -> None:
                self._should_stop = False

            @property
            def should_stop(self) -> bool:
                return self._should_stop

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
                if iteration >= 2:
                    self._should_stop = True

            def on_experiment_end(self, algorithm: Any) -> None:
                pass

        algo = StatefulAlgorithm()
        cb = _StopCallback()
        runner = ExperimentRunner(
            algorithm=algo,
            max_iterations=100,
            callbacks=[cb],
        )
        result = runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)

        data = torch.load(ckpt_path, weights_only=False)

        assert data["iteration"] == result.iterations

    @pytest.mark.unit
    def test_overwrite_existing_checkpoint(
        self,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        ckpt_path = tmp_path / "ckpt.pt"


        algo1 = StatefulAlgorithm()
        runner1 = ExperimentRunner(algorithm=algo1, max_iterations=2)
        runner1.run(mock_components)
        runner1.save_checkpoint(ckpt_path)
        data1 = torch.load(ckpt_path, weights_only=False)


        algo2 = StatefulAlgorithm()
        runner2 = ExperimentRunner(algorithm=algo2, max_iterations=5)
        runner2.run(mock_components)
        runner2.save_checkpoint(ckpt_path)
        data2 = torch.load(ckpt_path, weights_only=False)


        assert data1["iteration"] == 2
        assert data2["iteration"] == 5
        assert data2["iteration"] != data1["iteration"]

    @pytest.mark.unit
    def test_checkpoint_uses_torch_save(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=1)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)


        data = torch.load(ckpt_path, weights_only=False)
        assert isinstance(data, dict)







class TestCheckpointVersion:

    @pytest.mark.unit
    def test_version_is_integer(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=1)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)
        data = torch.load(ckpt_path, weights_only=False)
        assert isinstance(data["version"], int)

    @pytest.mark.unit
    def test_version_is_one(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=1)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)
        data = torch.load(ckpt_path, weights_only=False)
        assert data["version"] == 1










class _AnonymousAlgorithm(StatefulAlgorithm):

    @property
    def config(self) -> dict[str, Any]:
        return {"mock": True}


def _valid_payload(algorithm: StatefulAlgorithm, iteration: int = 3) -> dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "iteration": iteration,
        "algorithm_state": algorithm.state,
        "best_score": algorithm.best_score,
        "best_expression": algorithm.best_expression,
    }


def _run_stateful(
    mock_components: PlatformComponents, iterations: int = 3
) -> StatefulAlgorithm:
    algo = StatefulAlgorithm()
    ExperimentRunner(algorithm=algo, max_iterations=iterations).run(mock_components)
    return algo


class TestCheckpointPayloadAlgorithmKey:

    @pytest.mark.unit
    def test_i_build_payload_carries_algorithm_name(self) -> None:
        payload = build_checkpoint_payload(2, StatefulAlgorithm())
        assert payload["algorithm"] == "StatefulAlgorithm"
        assert payload["version"] == CHECKPOINT_VERSION == 1

    @pytest.mark.unit
    def test_i_build_payload_algorithm_none_for_anonymous_plugin(self) -> None:
        payload = build_checkpoint_payload(0, _AnonymousAlgorithm())
        assert payload.get("algorithm") is None

    @pytest.mark.unit
    def test_i_save_checkpoint_payload_carries_algorithm_name(
        self,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        algo = _run_stateful(mock_components, iterations=1)
        runner = ExperimentRunner(algorithm=algo, max_iterations=1)
        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)

        data = torch.load(ckpt_path, weights_only=False)
        assert data["algorithm"] == "StatefulAlgorithm"


class TestLoadCheckpointValidation:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "garbage",
        [[1, 2, 3], "not-a-payload", 42],
        ids=["list", "string", "int"],
    )
    def test_i_load_rejects_non_dict_payload(
        self, tmp_path: Path, garbage: Any
    ) -> None:
        path = tmp_path / "garbage.pt"
        torch.save(garbage, path)
        runner = ExperimentRunner(algorithm=StatefulAlgorithm(), max_iterations=1)
        with pytest.raises(ValueError, match="not a kd checkpoint"):
            runner.load_checkpoint(path)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "missing_key",
        ["version", "iteration", "algorithm_state"],
    )
    def test_i_load_rejects_payload_missing_required_key(
        self,
        mock_components: PlatformComponents,
        tmp_path: Path,
        missing_key: str,
    ) -> None:
        donor = _run_stateful(mock_components)
        payload = _valid_payload(donor)
        del payload[missing_key]
        path = tmp_path / "incomplete.pt"
        torch.save(payload, path)

        runner = ExperimentRunner(algorithm=StatefulAlgorithm(), max_iterations=1)
        with pytest.raises(ValueError, match="not a kd checkpoint"):
            runner.load_checkpoint(path)

    @pytest.mark.unit
    def test_i_load_rejects_version_mismatch_naming_both(
        self,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        donor = _run_stateful(mock_components)
        payload = _valid_payload(donor)
        bad_version = CHECKPOINT_VERSION + 41
        payload["version"] = bad_version
        path = tmp_path / "wrong_version.pt"
        torch.save(payload, path)

        runner = ExperimentRunner(algorithm=StatefulAlgorithm(), max_iterations=1)
        with pytest.raises(ValueError) as exc_info:
            runner.load_checkpoint(path)
        msg = str(exc_info.value)
        assert str(bad_version) in msg
        assert str(CHECKPOINT_VERSION) in msg

    @pytest.mark.unit
    def test_i_load_rejects_algorithm_mismatch_naming_both(
        self,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        donor = _run_stateful(mock_components)
        payload = _valid_payload(donor)
        payload["algorithm"] = "sga"
        path = tmp_path / "mismatch.pt"
        torch.save(payload, path)

        fresh = StatefulAlgorithm()
        runner = ExperimentRunner(algorithm=fresh, max_iterations=1)
        with pytest.raises(ValueError) as exc_info:
            runner.load_checkpoint(path)
        msg = str(exc_info.value)
        assert "sga" in msg
        assert "StatefulAlgorithm" in msg
        assert fresh.state["generation"] == 0, (
            "a rejected load must not partially mutate the plugin state"
        )

    @pytest.mark.unit
    def test_i_load_accepts_matching_algorithm(
        self,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        donor = _run_stateful(mock_components)
        payload = _valid_payload(donor)
        payload["algorithm"] = "StatefulAlgorithm"
        path = tmp_path / "match.pt"
        torch.save(payload, path)

        fresh = StatefulAlgorithm()
        runner = ExperimentRunner(algorithm=fresh, max_iterations=1)
        runner.load_checkpoint(path)
        assert fresh.state["generation"] == donor.state["generation"]

    @pytest.mark.unit
    def test_i_load_tolerates_payload_without_algorithm_key(
        self,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        donor = _run_stateful(mock_components)
        payload = _valid_payload(donor)
        assert "algorithm" not in payload
        path = tmp_path / "legacy.pt"
        torch.save(payload, path)

        fresh = StatefulAlgorithm()
        runner = ExperimentRunner(algorithm=fresh, max_iterations=1)
        runner.load_checkpoint(path)
        assert fresh.state["generation"] == donor.state["generation"]

    @pytest.mark.unit
    def test_i_load_tolerates_plugin_without_algorithm_in_config(
        self,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        donor = _run_stateful(mock_components)
        payload = _valid_payload(donor)
        payload["algorithm"] = "sga"
        path = tmp_path / "anon.pt"
        torch.save(payload, path)

        fresh = _AnonymousAlgorithm()
        runner = ExperimentRunner(algorithm=fresh, max_iterations=1)
        runner.load_checkpoint(path)
        assert fresh.state["generation"] == donor.state["generation"]

    @pytest.mark.unit
    def test_i_load_missing_file_raises_filenotfound_naming_path(
        self, tmp_path: Path
    ) -> None:
        runner = ExperimentRunner(algorithm=StatefulAlgorithm(), max_iterations=1)
        bogus = tmp_path / "missing_checkpoint.pt"
        with pytest.raises(FileNotFoundError) as exc_info:
            runner.load_checkpoint(bogus)
        assert "missing_checkpoint.pt" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "bad_state",
        [None, [], ""],
        ids=["none", "empty-list", "empty-str"],
    )
    def test_fix2_load_rejects_nondict_algorithm_state(
        self,
        mock_components: PlatformComponents,
        tmp_path: Path,
        bad_state: Any,
    ) -> None:
        donor = _run_stateful(mock_components)
        payload = _valid_payload(donor)
        payload["algorithm_state"] = bad_state
        path = tmp_path / "falsy_state.pt"
        torch.save(payload, path)

        fresh = StatefulAlgorithm()
        runner = ExperimentRunner(algorithm=fresh, max_iterations=1)
        with pytest.raises(ValueError, match="not a kd checkpoint"):
            runner.load_checkpoint(path)
        assert fresh.state["generation"] == 0, (
            "a rejected falsy-state load must not mutate the plugin"
        )

    @pytest.mark.unit
    def test_fix2_load_accepts_empty_dict_algorithm_state(
        self,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        received: list[dict[str, Any]] = []

        class _StatelessFake:

            @property
            def state(self) -> dict[str, Any]:
                return {}

            @state.setter
            def state(self, value: dict[str, Any]) -> None:
                received.append(value)

            @property
            def config(self) -> dict[str, Any]:
                return {}

        payload = {
            "version": CHECKPOINT_VERSION,
            "iteration": 3,
            "algorithm_state": {},
        }
        path = tmp_path / "empty_state.pt"
        torch.save(payload, path)

        runner = ExperimentRunner(algorithm=_StatelessFake(), max_iterations=1)
        runner.load_checkpoint(path)
        assert received == [{}], (
            "the empty state must REACH the plugin setter (protocol reset "
            "semantics), not be rejected by the loader"
        )

    @pytest.mark.unit
    def test_fix2_load_rejects_truthy_nondict_algorithm_state(
        self,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        donor = _run_stateful(mock_components)
        payload = _valid_payload(donor)
        payload["algorithm_state"] = [1, 2, 3]
        path = tmp_path / "list_state.pt"
        torch.save(payload, path)

        runner = ExperimentRunner(algorithm=StatefulAlgorithm(), max_iterations=1)
        with pytest.raises(ValueError, match="not a kd checkpoint"):
            runner.load_checkpoint(path)

    @pytest.mark.unit
    def test_fix4a_load_rejects_truncated_file_naming_path(
        self,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        donor = _run_stateful(mock_components)
        good = tmp_path / "good.pt"
        torch.save(_valid_payload(donor), good)
        torn = tmp_path / "torn.pt"
        raw = good.read_bytes()
        torn.write_bytes(raw[: len(raw) // 2])

        runner = ExperimentRunner(algorithm=StatefulAlgorithm(), max_iterations=1)
        with pytest.raises(ValueError, match="not a kd checkpoint") as exc_info:
            runner.load_checkpoint(torn)
        assert "torn.pt" in str(exc_info.value)
