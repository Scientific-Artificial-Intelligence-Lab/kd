
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from kd.core.evaluator import EvaluationResult
from kd.search.recorder import VizRecorder
from kd.search.records import (
    EVIDENCE_HASH_SCHEME,
    RECORD_HASH_SCHEME,
    RUN_RECORD_SCHEMA_VERSION,
    EvidenceRecord,
    RunCost,
    RunRecord,
    seal_record_hash,
)
from kd.search.result import ExperimentResult, RunManifest, default_final_result
from kd.search.run_spec import RUN_SPEC_HASH_SCHEME, RunSpec






@pytest.fixture
def sample_eval_result() -> EvaluationResult:
    return EvaluationResult(
        mse=0.01,
        nmse=0.02,
        r2=0.98,
        score=-50.0,
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


def _sample_run_record() -> RunRecord:
    evidence = EvidenceRecord(
        instrument="sga",
        dataset_name="burgers_1d",
        dataset_cache_fingerprint="sha256:dataset",
        seed=0,
        is_valid=True,
        expression="mul(u, u_x)",
        score_kind="AIC",
        score_direction="min",
        headline_coefficient_source="native",
    )
    run_spec = RunSpec(
        kd_version="0.1.0",
        config={"algorithm": "sga"},
        dataset_cache_fingerprint="sha256:dataset",
    )
    return seal_record_hash(
        RunRecord(
            schema_version=RUN_RECORD_SCHEMA_VERSION,
            evidence_hash_scheme=EVIDENCE_HASH_SCHEME,
            created_at="2026-07-17T00:00:00+00:00",
            cost=RunCost(
                wallclock_seconds=1.0,
                search_seconds=1.0,
                boundary_results=0,
                boundary_invalid_results=0,
            ),
            evidence=evidence,
            evidence_hash=evidence.content_hash(),
            run_spec=run_spec,
            run_spec_hash=run_spec.run_spec_hash,
            run_spec_hash_scheme=RUN_SPEC_HASH_SCHEME,
            record_hash="",
            record_hash_scheme=RECORD_HASH_SCHEME,
        )
    )


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

    def test_default_final_result_is_callable(self) -> None:
        assert callable(default_final_result)







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


            "manifest",


            "run_record",
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

        assert fe.score == pytest.approx(orig.score)
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

    def test_save_load_preserves_attached_run_record(
        self, sample_experiment_result: ExperimentResult, tmp_path: Path
    ) -> None:
        record = _sample_run_record()
        sample_experiment_result.run_record = record
        fpath = tmp_path / "result.json"

        sample_experiment_result.save(fpath)
        loaded = ExperimentResult.load(fpath)

        assert loaded.run_record == record

    def test_legacy_payload_without_run_record_loads_none(
        self, sample_experiment_result: ExperimentResult, tmp_path: Path
    ) -> None:
        import json

        payload = sample_experiment_result.to_dict()
        payload.pop("run_record")
        fpath = tmp_path / "legacy_result.json"
        with fpath.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False)

        loaded = ExperimentResult.load(fpath)

        assert loaded.run_record is None

    def test_save_creates_parent_dirs(
        self, sample_experiment_result: ExperimentResult, tmp_path: Path
    ) -> None:
        fpath = tmp_path / "nested" / "deep" / "result.pt"
        sample_experiment_result.save(fpath)
        assert fpath.exists()







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

    def test_save_with_nan_score(self, tmp_path: Path) -> None:
        eval_result = EvaluationResult(
            mse=0.01,
            nmse=0.02,
            r2=0.98,
            score=float("-inf"),
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
        fpath = tmp_path / "nan_score.json"
        r.save(fpath)
        loaded = ExperimentResult.load(fpath)
        assert loaded.final_eval.score is None

    def test_save_with_non_finite_final_eval_metrics(self, tmp_path: Path) -> None:
        import math

        eval_result = EvaluationResult(
            mse=float("inf"),
            nmse=float("inf"),
            r2=float("-inf"),
            score=None,
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

        assert fe.score is None


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







@pytest.fixture
def sample_manifest() -> RunManifest:
    return RunManifest(
        dataset_cache_fingerprint="burgers_1d:grid:u:t:x,t_u(64, 32)_a1b2c3d4",
        kd_version="0.1.0",
        seed=42,
        terms=["u", "u_x", "u_xx"],
        artifacts={"weights.pt": {"sha256": "deadbeef", "size": 12345}},
    )


@pytest.fixture
def minimal_manifest() -> RunManifest:
    return RunManifest(
        dataset_cache_fingerprint="heat:grid:u:x:x_u(100,)_00000000",
        kd_version="0.1.0",
        seed=None,
    )







@pytest.mark.smoke
class TestRunManifestSmoke:

    def test_instantiate(self, sample_manifest: RunManifest) -> None:
        assert isinstance(sample_manifest, RunManifest)

    def test_field_values(self, sample_manifest: RunManifest) -> None:
        m = sample_manifest
        assert m.dataset_cache_fingerprint.startswith("burgers_1d:")
        assert m.kd_version == "0.1.0"
        assert m.seed == 42
        assert m.terms == ["u", "u_x", "u_xx"]
        assert m.artifacts == {"weights.pt": {"sha256": "deadbeef", "size": 12345}}

    def test_optional_fields_default_none(self) -> None:
        m = RunManifest(dataset_cache_fingerprint="fp", kd_version="0.1.0", seed=7)
        assert m.terms is None
        assert m.artifacts is None







class TestRunManifestRoundTrip:

    def test_full_manifest_round_trip(self, sample_manifest: RunManifest) -> None:
        restored = RunManifest.from_dict(sample_manifest.to_dict())
        assert restored == sample_manifest

    def test_minimal_manifest_round_trip(self, minimal_manifest: RunManifest) -> None:
        restored = RunManifest.from_dict(minimal_manifest.to_dict())
        assert restored == minimal_manifest

        assert restored.seed is None
        assert restored.terms is None
        assert restored.artifacts is None

    @pytest.mark.parametrize("seed", [0, 1, 42, -1, 2**31])
    def test_round_trip_preserves_seed_int(self, seed: int) -> None:
        m = RunManifest(dataset_cache_fingerprint="fp", kd_version="0.1.0", seed=seed)
        restored = RunManifest.from_dict(m.to_dict())
        assert restored.seed == seed
        assert restored == m

    def test_to_dict_is_plain_dict(self, sample_manifest: RunManifest) -> None:
        d = sample_manifest.to_dict()
        assert isinstance(d, dict)
        for key in [
            "dataset_cache_fingerprint",
            "kd_version",
            "seed",
            "terms",
            "artifacts",
            "resumed",
        ]:
            assert key in d, f"manifest dict missing key: {key}"

    def test_to_dict_carries_field_values(self, sample_manifest: RunManifest) -> None:
        d = sample_manifest.to_dict()
        assert (
            d["dataset_cache_fingerprint"]
            == sample_manifest.dataset_cache_fingerprint
        )
        assert d["kd_version"] == sample_manifest.kd_version
        assert d["seed"] == sample_manifest.seed
        assert d["terms"] == sample_manifest.terms


        assert d["artifacts"] == sample_manifest.artifacts

    def test_from_dict_tolerates_missing_optional_keys(self) -> None:
        m = RunManifest.from_dict(
            {"dataset_cache_fingerprint": "fp", "kd_version": "0.1.0", "seed": 7}
        )
        assert m.dataset_cache_fingerprint == "fp"
        assert m.kd_version == "0.1.0"
        assert m.seed == 7
        assert m.terms is None
        assert m.artifacts is None







class TestRunManifestJsonSafe:

    def test_full_manifest_json_dumps(self, sample_manifest: RunManifest) -> None:
        import json

        encoded = json.dumps(sample_manifest.to_dict(), allow_nan=False)

        decoded = RunManifest.from_dict(json.loads(encoded))
        assert decoded == sample_manifest

    def test_minimal_manifest_json_dumps(self, minimal_manifest: RunManifest) -> None:
        import json

        encoded = json.dumps(minimal_manifest.to_dict(), allow_nan=False)
        assert "null" in encoded
        decoded = RunManifest.from_dict(json.loads(encoded))
        assert decoded == minimal_manifest

    def test_to_dict_values_are_json_native(self, sample_manifest: RunManifest) -> None:
        d = sample_manifest.to_dict()
        json_types = (str, int, float, bool, type(None), list, dict)
        for key, value in d.items():
            assert isinstance(value, json_types), (
                f"manifest[{key!r}] is {type(value).__name__}, not JSON-native"
            )







class TestExperimentResultManifest:

    def test_construct_without_manifest_defaults_none(
        self, sample_experiment_result: ExperimentResult
    ) -> None:
        assert sample_experiment_result.manifest is None

    def test_construct_with_manifest(
        self,
        sample_eval_result: EvaluationResult,
        sample_manifest: RunManifest,
    ) -> None:
        r = ExperimentResult(
            best_expression="u",
            best_score=0.02,
            iterations=10,
            early_stopped=False,
            final_eval=sample_eval_result,
            actual=torch.randn(10),
            predicted=torch.randn(10),
            dataset_name="test",
            algorithm_name="sga",
            config={},
            recorder=VizRecorder(),
            manifest=sample_manifest,
        )
        assert r.manifest is sample_manifest

    def test_to_dict_manifest_none_when_absent(
        self, sample_experiment_result: ExperimentResult
    ) -> None:
        d = sample_experiment_result.to_dict()
        assert d["manifest"] is None

    def test_to_dict_includes_manifest_dict_when_present(
        self,
        sample_eval_result: EvaluationResult,
        sample_manifest: RunManifest,
    ) -> None:
        r = ExperimentResult(
            best_expression="u",
            best_score=0.02,
            iterations=10,
            early_stopped=False,
            final_eval=sample_eval_result,
            actual=torch.randn(10),
            predicted=torch.randn(10),
            dataset_name="test",
            algorithm_name="sga",
            config={},
            recorder=VizRecorder(),
            manifest=sample_manifest,
        )
        d = r.to_dict()
        assert d["manifest"] == sample_manifest.to_dict()

    def test_to_dict_with_manifest_is_json_safe(
        self,
        sample_eval_result: EvaluationResult,
        sample_manifest: RunManifest,
    ) -> None:
        import json

        r = ExperimentResult(
            best_expression="u",
            best_score=0.02,
            iterations=10,
            early_stopped=False,
            final_eval=sample_eval_result,
            actual=torch.randn(10),
            predicted=torch.randn(10),
            dataset_name="test",
            algorithm_name="sga",
            config={},
            recorder=VizRecorder(),
            manifest=sample_manifest,
        )
        encoded = json.dumps(r.to_dict(), allow_nan=False)
        assert isinstance(encoded, str)

        reloaded = json.loads(encoded)
        assert reloaded["manifest"]["dataset_cache_fingerprint"] == (
            sample_manifest.dataset_cache_fingerprint
        )

    def test_save_load_preserves_manifest(
        self,
        sample_eval_result: EvaluationResult,
        sample_manifest: RunManifest,
        tmp_path: Path,
    ) -> None:
        r = ExperimentResult(
            best_expression="u",
            best_score=0.02,
            iterations=10,
            early_stopped=False,
            final_eval=sample_eval_result,
            actual=torch.randn(10),
            predicted=torch.randn(10),
            dataset_name="test",
            algorithm_name="sga",
            config={},
            recorder=VizRecorder(),
            manifest=sample_manifest,
        )
        fpath = tmp_path / "with_manifest.json"
        r.save(fpath)
        loaded = ExperimentResult.load(fpath)
        assert loaded.manifest == sample_manifest

    def test_save_load_preserves_minimal_manifest(
        self,
        sample_eval_result: EvaluationResult,
        minimal_manifest: RunManifest,
        tmp_path: Path,
    ) -> None:
        r = ExperimentResult(
            best_expression="u",
            best_score=0.02,
            iterations=10,
            early_stopped=False,
            final_eval=sample_eval_result,
            actual=torch.randn(10),
            predicted=torch.randn(10),
            dataset_name="test",
            algorithm_name="sga",
            config={},
            recorder=VizRecorder(),
            manifest=minimal_manifest,
        )
        fpath = tmp_path / "minimal_manifest.json"
        r.save(fpath)
        loaded = ExperimentResult.load(fpath)
        assert loaded.manifest == minimal_manifest
        assert loaded.manifest is not None
        assert loaded.manifest.seed is None







class TestExperimentResultManifestBackwardCompat:

    def test_save_without_manifest_loads_as_none(
        self, sample_experiment_result: ExperimentResult, tmp_path: Path
    ) -> None:
        fpath = tmp_path / "no_manifest.json"
        sample_experiment_result.save(fpath)
        loaded = ExperimentResult.load(fpath)
        assert loaded.manifest is None

    def test_load_legacy_dict_without_manifest_key(
        self, sample_experiment_result: ExperimentResult, tmp_path: Path
    ) -> None:
        import json

        fpath = tmp_path / "legacy.json"
        sample_experiment_result.save(fpath)


        with fpath.open("r", encoding="utf-8") as handle:
            data = json.load(handle)



        assert "manifest" in data
        data.pop("manifest", None)
        with fpath.open("w", encoding="utf-8") as handle:
            json.dump(data, handle)


        with fpath.open("r", encoding="utf-8") as handle:
            assert "manifest" not in json.load(handle)

        loaded = ExperimentResult.load(fpath)
        assert loaded.manifest is None
