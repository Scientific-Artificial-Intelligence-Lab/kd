
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Literal

import pytest
import torch

from kd.core.evaluator import EvaluationResult
from kd.search.protocol import PlatformComponents
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult
from kd.search.runner import ExperimentRunner
from tests.unit.search._runner_mocks import RecordingAlgorithm

_N_SAMPLES = 16


def _make_eval_result() -> EvaluationResult:
    return EvaluationResult(
        mse=0.01,
        nmse=0.02,
        r2=0.98,
        score=-50.0,
        complexity=2,
        coefficients=torch.tensor([1.0, 0.5]),
        is_valid=True,
        error_message="",
        selected_indices=[0, 1],
        residuals=torch.zeros(_N_SAMPLES),
        terms=["u", "u_x"],
        expression="add(u, u_x)",
    )


def _make_result(
    *,
    algorithm_name: str = "SGAPlugin",
    config: dict[str, Any] | None = None,
    score_kind: str | None = None,
    score_direction: str | None = None,
) -> ExperimentResult:
    recorder = VizRecorder()
    recorder.log("_best_score", 1.0)
    recorder.log("_best_score", 0.5)
    meta_kwargs: dict[str, Any] = {}
    if score_kind is not None:
        meta_kwargs["score_kind"] = score_kind
    if score_direction is not None:
        meta_kwargs["score_direction"] = score_direction
    return ExperimentResult(
        best_expression="add(u, u_x)",
        best_score=0.5,
        iterations=2,
        early_stopped=False,
        final_eval=_make_eval_result(),
        actual=torch.linspace(0.0, 1.0, _N_SAMPLES),
        predicted=torch.linspace(0.0, 1.0, _N_SAMPLES),
        dataset_name="meta_test",
        algorithm_name=algorithm_name,
        config=config if config is not None else {"algorithm": "sga"},
        recorder=recorder,
        **meta_kwargs,
    )







class TestScoreMetaFields:

    def test_defaults_follow_legacy_fallback_semantics(self) -> None:
        result = _make_result()
        assert result.score_kind == "Score"
        assert result.score_direction == "min"

    def test_explicit_non_default_values_carried(self) -> None:
        result = _make_result(score_kind="reward", score_direction="max")
        assert result.score_kind == "reward"
        assert result.score_direction == "max"







class TestScoreMetaSerialization:

    def test_to_dict_emits_both_meta_keys(self) -> None:
        d = _make_result(score_kind="reward", score_direction="max").to_dict()
        assert d["score_kind"] == "reward"
        assert d["score_direction"] == "max"

    def test_round_trip_preserves_non_default_meta(self, tmp_path: Path) -> None:
        path = tmp_path / "meta.json"
        result = _make_result(
            algorithm_name="SGAPlugin",
            score_kind="reward",
            score_direction="max",
        )
        result.save(path)

        loaded = ExperimentResult.load(path)

        assert loaded.score_kind == "reward"
        assert loaded.score_direction == "max"

    def test_loaded_meta_types_are_str(self, tmp_path: Path) -> None:
        path = tmp_path / "meta_types.json"
        _make_result(score_kind="NMSE", score_direction="min").save(path)
        loaded = ExperimentResult.load(path)
        assert isinstance(loaded.score_kind, str)
        assert isinstance(loaded.score_direction, str)







def _legacy_load(
    tmp_path: Path,
    *,
    algorithm_name: str,
    config: dict[str, Any],
) -> ExperimentResult:
    path = tmp_path / "legacy.json"
    _make_result(algorithm_name=algorithm_name, config=config).save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("score_kind", None)
    payload.pop("score_direction", None)
    assert "score_kind" not in payload
    path.write_text(json.dumps(payload), encoding="utf-8")
    return ExperimentResult.load(path)


class TestLegacyPayloadInference:

    @pytest.mark.parametrize(
        ("algorithm_name", "algorithm_key", "kind", "direction"),
        [
            pytest.param("SGAPlugin", "sga", "AIC", "min", id="sga"),
            pytest.param("DLGAPlugin", "dlga", "DLGA fitness", "min", id="dlga"),
            pytest.param("DISCOVERPlugin", "discover", "reward", "max", id="discover"),
            pytest.param("PySRPlugin", "pysr", "NMSE", "min", id="pysr"),
        ],
    )
    def test_real_legacy_shape_infers_declared_meta(
        self,
        tmp_path: Path,
        algorithm_name: str,
        algorithm_key: str,
        kind: str,
        direction: str,
    ) -> None:
        loaded = _legacy_load(
            tmp_path,
            algorithm_name=algorithm_name,
            config={"algorithm": algorithm_key},
        )
        assert loaded.score_kind == kind
        assert loaded.score_direction == direction

    def test_unknown_algorithm_falls_back_to_defaults(self, tmp_path: Path) -> None:
        loaded = _legacy_load(
            tmp_path,
            algorithm_name="MysteryPlugin",
            config={"algorithm": "mystery"},
        )
        assert loaded.score_kind == "Score"
        assert loaded.score_direction == "min"

    def test_missing_algorithm_key_falls_back_without_crash(
        self, tmp_path: Path
    ) -> None:
        loaded = _legacy_load(
            tmp_path,
            algorithm_name="RecordingAlgorithm",
            config={"max_iter": 5},
        )
        assert loaded.score_kind == "Score"
        assert loaded.score_direction == "min"

    def test_partial_payload_inferred_atomically(self, tmp_path: Path) -> None:
        path = tmp_path / "partial.json"
        _make_result(algorithm_name="SGAPlugin", config={"algorithm": "sga"}).save(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["score_kind"] = "reward"
        payload.pop("score_direction", None)
        assert "score_direction" not in payload
        path.write_text(json.dumps(payload), encoding="utf-8")

        loaded = ExperimentResult.load(path)

        assert loaded.score_kind == "AIC"
        assert loaded.score_direction == "min"







class _DeclaringRecordingAlgorithm(RecordingAlgorithm):

    score_kind: ClassVar[str] = "custom-metric"
    score_direction: ClassVar[Literal["min", "max"]] = "max"


class _PartiallyDeclaringAlgorithm(RecordingAlgorithm):

    score_kind: ClassVar[str] = "custom-metric"


@pytest.mark.integration
class TestRunnerFillsScoreMeta:

    def test_meta_filled_from_declaring_algorithm(
        self, mock_components: PlatformComponents
    ) -> None:
        runner = ExperimentRunner(
            algorithm=_DeclaringRecordingAlgorithm(), max_iterations=1
        )

        result = runner.run(mock_components)

        assert result.score_kind == "custom-metric"
        assert result.score_direction == "max"

    def test_meta_defaults_for_non_declaring_algorithm(
        self, mock_components: PlatformComponents
    ) -> None:
        algo = RecordingAlgorithm()
        assert not hasattr(algo, "score_kind")
        runner = ExperimentRunner(algorithm=algo, max_iterations=1)

        result = runner.run(mock_components)

        assert result.score_kind == "Score"
        assert result.score_direction == "min"

    def test_partial_declaration_falls_back_on_both_fields(
        self, mock_components: PlatformComponents
    ) -> None:
        algo = _PartiallyDeclaringAlgorithm()
        assert hasattr(algo, "score_kind")
        assert not hasattr(algo, "score_direction")
        runner = ExperimentRunner(algorithm=algo, max_iterations=1)

        result = runner.run(mock_components)

        assert result.score_kind == "Score"
        assert result.score_direction == "min"

    def test_filled_meta_survives_save_load(
        self, mock_components: PlatformComponents, tmp_path: Path
    ) -> None:
        runner = ExperimentRunner(
            algorithm=_DeclaringRecordingAlgorithm(), max_iterations=1
        )
        result = runner.run(mock_components)
        path = tmp_path / "filled.json"
        result.save(path)

        loaded = ExperimentResult.load(path)

        assert loaded.score_kind == "custom-metric"
        assert loaded.score_direction == "max"
