
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from kd.core.evaluator import EvaluationResult
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult, ResultBuilder






@pytest.fixture
def sample_eval_result() -> EvaluationResult:
    return EvaluationResult(
        mse=0.01,
        nmse=0.02,
        r2=0.98,
        aic=-50.0,
        complexity=3,
        coefficients=torch.tensor([1.0, -6.0, 1.0]),
        is_valid=True,
        selected_indices=[0, 1, 2],
        residuals=torch.randn(100),
        terms=["u", "mul(u, u_x)", "u_xx"],
        expression="add(u, add(mul(u, u_x), u_xx))",
    )


@pytest.fixture
def sample_recorder() -> VizRecorder:
    rec = VizRecorder()
    rec.log("loss", 0.5)
    rec.log("loss", 0.3)
    rec.log("loss", 0.1)
    rec.log("iteration", 1)
    rec.log("iteration", 2)
    rec.log("iteration", 3)
    return rec


@pytest.fixture
def sample_experiment_result(
    sample_eval_result: EvaluationResult,
    sample_recorder: VizRecorder,
) -> ExperimentResult:
    return ExperimentResult(
        best_expression="add(u, add(mul(u, u_x), u_xx))",
        best_score=0.02,
        iterations=50,
        early_stopped=False,
        final_eval=sample_eval_result,
        actual=torch.randn(100),
        predicted=torch.randn(100),
        dataset_name="burgers_1d",
        algorithm_name="sga",
        config={"max_iter": 100, "threshold": 0.1},
        recorder=sample_recorder,
    )







@pytest.mark.smoke
class TestExperimentResultSmoke:

    def test_instantiate(self, sample_experiment_result: ExperimentResult) -> None:
        assert isinstance(sample_experiment_result, ExperimentResult)

    def test_result_builder_is_protocol(self) -> None:
        assert hasattr(ResultBuilder, "build_final_result")







class TestExperimentResultFields:

    def test_run_result_fields(
        self, sample_experiment_result: ExperimentResult
    ) -> None:
        r = sample_experiment_result
        assert r.best_expression == "add(u, add(mul(u, u_x), u_xx))"
        assert r.best_score == pytest.approx(0.02)
        assert r.iterations == 50
        assert r.early_stopped is False

    def test_final_eval_accessible(
        self, sample_experiment_result: ExperimentResult
    ) -> None:
        r = sample_experiment_result
        assert r.final_eval.is_valid is True
        assert r.final_eval.r2 == pytest.approx(0.98)
        assert r.final_eval.complexity == 3

    def test_tensor_fields(self, sample_experiment_result: ExperimentResult) -> None:
        r = sample_experiment_result
        assert isinstance(r.actual, Tensor)
        assert isinstance(r.predicted, Tensor)
        assert r.actual.shape == r.predicted.shape

    def test_metadata_fields(self, sample_experiment_result: ExperimentResult) -> None:
        r = sample_experiment_result
        assert r.dataset_name == "burgers_1d"
        assert r.algorithm_name == "sga"
        assert "max_iter" in r.config

    def test_recorder_field(self, sample_experiment_result: ExperimentResult) -> None:
        r = sample_experiment_result
        assert isinstance(r.recorder, VizRecorder)
        assert r.recorder.get("loss") == [0.5, 0.3, 0.1]







class TestExperimentResultSerialization:

    def test_to_dict_contains_required_keys(
        self, sample_experiment_result: ExperimentResult
    ) -> None:
        d = sample_experiment_result.to_dict()
        assert isinstance(d, dict)
        for key in [
            "best_expression",
            "best_score",
            "iterations",
            "early_stopped",
            "dataset_name",
            "algorithm_name",
            "config",
            "final_eval",
            "actual",
            "predicted",
            "recorder",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_is_json_safe(
        self, sample_experiment_result: ExperimentResult
    ) -> None:
        import json

        d = sample_experiment_result.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_save_load_round_trip(
        self, sample_experiment_result: ExperimentResult, tmp_path: Path
    ) -> None:
        fpath = tmp_path / "result.pt"
        sample_experiment_result.save(fpath)
        assert fpath.exists()

        loaded = ExperimentResult.load(fpath)


        assert loaded.best_expression == sample_experiment_result.best_expression
        assert loaded.best_score == pytest.approx(sample_experiment_result.best_score)
        assert loaded.iterations == sample_experiment_result.iterations
        assert loaded.early_stopped == sample_experiment_result.early_stopped
        assert loaded.dataset_name == sample_experiment_result.dataset_name
        assert loaded.algorithm_name == sample_experiment_result.algorithm_name

    def test_save_load_preserves_tensors(
        self, sample_experiment_result: ExperimentResult, tmp_path: Path
    ) -> None:
        fpath = tmp_path / "result.pt"
        sample_experiment_result.save(fpath)
        loaded = ExperimentResult.load(fpath)

        torch.testing.assert_close(
            loaded.actual,
            sample_experiment_result.actual,
            rtol=1e-5,
            atol=1e-8,
        )
        torch.testing.assert_close(
            loaded.predicted,
            sample_experiment_result.predicted,
            rtol=1e-5,
            atol=1e-8,
        )

    def test_save_load_preserves_final_eval(
        self, sample_experiment_result: ExperimentResult, tmp_path: Path
    ) -> None:
        fpath = tmp_path / "result.pt"
        sample_experiment_result.save(fpath)
        loaded = ExperimentResult.load(fpath)

        orig = sample_experiment_result.final_eval
        fe = loaded.final_eval

        assert fe.mse == pytest.approx(orig.mse)
        assert fe.nmse == pytest.approx(orig.nmse)
        assert fe.r2 == pytest.approx(orig.r2)
        assert fe.aic == pytest.approx(orig.aic)
        assert fe.complexity == orig.complexity
        assert fe.is_valid == orig.is_valid
        assert fe.error_message == orig.error_message
        assert fe.expression == orig.expression

        assert fe.selected_indices == orig.selected_indices
        assert fe.terms == orig.terms

        assert fe.coefficients is not None
        torch.testing.assert_close(fe.coefficients, orig.coefficients)
        assert fe.residuals is not None
        torch.testing.assert_close(fe.residuals, orig.residuals)

    def test_save_load_preserves_recorder(
        self, sample_experiment_result: ExperimentResult, tmp_path: Path
    ) -> None:
        fpath = tmp_path / "result.pt"
        sample_experiment_result.save(fpath)
        loaded = ExperimentResult.load(fpath)

        assert isinstance(loaded.recorder, VizRecorder)
        assert loaded.recorder.get("loss") == [0.5, 0.3, 0.1]
        assert loaded.recorder.keys() == {"loss", "iteration"}

    def test_save_creates_parent_dirs(
        self, sample_experiment_result: ExperimentResult, tmp_path: Path
    ) -> None:
        fpath = tmp_path / "nested" / "deep" / "result.pt"
        sample_experiment_result.save(fpath)
        assert fpath.exists()







class TestResultBuilderProtocol:

    def test_conforming_class_passes_isinstance(self) -> None:
        class MyBuilder:
            def build_final_result(self) -> EvaluationResult:
                return EvaluationResult(mse=0.0, nmse=0.0, r2=1.0)

        assert isinstance(MyBuilder(), ResultBuilder)

    def test_non_conforming_class_fails_isinstance(self) -> None:
        class NotABuilder:
            def some_other_method(self) -> None:
                pass

        assert not isinstance(NotABuilder(), ResultBuilder)

    def test_protocol_not_instantiable_directly(self) -> None:
        with pytest.raises(TypeError):
            ResultBuilder()







@pytest.mark.numerical
class TestExperimentResultNegative:

    def test_load_nonexistent_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises((FileNotFoundError, OSError)):
            ExperimentResult.load(tmp_path / "does_not_exist.pt")

    def test_early_stopped_true(self, sample_eval_result: EvaluationResult) -> None:
        r = ExperimentResult(
            best_expression="u",
            best_score=1.0,
            iterations=5,
            early_stopped=True,
            final_eval=sample_eval_result,
            actual=torch.tensor([1.0]),
            predicted=torch.tensor([1.0]),
            dataset_name="test",
            algorithm_name="test",
            config={},
            recorder=VizRecorder(),
        )
        assert r.early_stopped is True

    def test_empty_recorder_survives_round_trip(
        self, sample_eval_result: EvaluationResult, tmp_path: Path
    ) -> None:
        r = ExperimentResult(
            best_expression="u_xx",
            best_score=0.5,
            iterations=10,
            early_stopped=False,
            final_eval=sample_eval_result,
            actual=torch.randn(20),
            predicted=torch.randn(20),
            dataset_name="heat",
            algorithm_name="discover",
            config={"alpha": 0.01},
            recorder=VizRecorder(),
        )
        fpath = tmp_path / "empty_rec.pt"
        r.save(fpath)
        loaded = ExperimentResult.load(fpath)
        assert loaded.recorder.keys() == set()
        assert loaded.recorder.get("anything") == []

    def test_save_with_inf_best_score(
        self, sample_eval_result: EvaluationResult, tmp_path: Path
    ) -> None:
        import math

        r = ExperimentResult(
            best_expression="",
            best_score=float("inf"),
            iterations=0,
            early_stopped=False,
            final_eval=sample_eval_result,
            actual=torch.randn(10),
            predicted=torch.randn(10),
            dataset_name="test",
            algorithm_name="sga",
            config={},
            recorder=VizRecorder(),
        )
        fpath = tmp_path / "inf_score.json"
        r.save(fpath)
        loaded = ExperimentResult.load(fpath)


        assert isinstance(loaded.best_score, float), (
            "loaded.best_score must remain a float to satisfy "
            "RunResult.best_score: float"
        )
        assert math.isnan(loaded.best_score), (
            "non-finite best_score must collapse to NaN on load (was "
            "previously returning None which violates float type)"
        )

    def test_save_with_nan_aic(self, tmp_path: Path) -> None:
        eval_result = EvaluationResult(
            mse=0.01,
            nmse=0.02,
            r2=0.98,
            aic=float("-inf"),
        )
        r = ExperimentResult(
            best_expression="u",
            best_score=0.02,
            iterations=10,
            early_stopped=False,
            final_eval=eval_result,
            actual=torch.randn(10),
            predicted=torch.randn(10),
            dataset_name="test",
            algorithm_name="sga",
            config={},
            recorder=VizRecorder(),
        )
        fpath = tmp_path / "nan_aic.json"
        r.save(fpath)
        loaded = ExperimentResult.load(fpath)
        assert loaded.final_eval.aic is None

    def test_save_with_non_finite_final_eval_metrics(self, tmp_path: Path) -> None:
        import math

        eval_result = EvaluationResult(
            mse=float("inf"),
            nmse=float("inf"),
            r2=float("-inf"),
            aic=None,
        )
        r = ExperimentResult(
            best_expression="",
            best_score=0.02,
            iterations=0,
            early_stopped=False,
            final_eval=eval_result,
            actual=torch.randn(10),
            predicted=torch.randn(10),
            dataset_name="test",
            algorithm_name="sga",
            config={},
            recorder=VizRecorder(),
        )
        fpath = tmp_path / "nonfinite_eval.json"
        r.save(fpath)
        loaded = ExperimentResult.load(fpath)

        fe = loaded.final_eval

        for name, value in (("mse", fe.mse), ("nmse", fe.nmse), ("r2", fe.r2)):
            assert isinstance(value, float), (
                f"final_eval.{name} must stay a float (None breaks "
                f"math.isfinite / f-format downstream), got {value!r}"
            )
            assert math.isnan(value), (
                f"non-finite final_eval.{name} must load as NaN, got {value!r}"
            )

        assert fe.aic is None


        assert math.isfinite(fe.r2) is False
        _ = f"{fe.nmse:.4g}"

    def test_load_preserves_float_type_for_best_score_with_nan(
        self, sample_eval_result: EvaluationResult, tmp_path: Path
    ) -> None:
        import math

        r = ExperimentResult(
            best_expression="",
            best_score=float("nan"),
            iterations=0,
            early_stopped=False,
            final_eval=sample_eval_result,
            actual=torch.randn(10),
            predicted=torch.randn(10),
            dataset_name="test",
            algorithm_name="sga",
            config={},
            recorder=VizRecorder(),
        )
        fpath = tmp_path / "nan_score.json"
        r.save(fpath)
        loaded = ExperimentResult.load(fpath)
        assert isinstance(loaded.best_score, float)
        assert math.isnan(loaded.best_score)
