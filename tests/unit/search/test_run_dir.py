
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

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
from kd.search.result import ExperimentResult, RunManifest
from kd.search.run_dir import (
    RUNDIR_SCHEME,
    RunDirPaths,
    create_run_dir,
    finalize_run_dir,
    new_run_id,
    run_id_of_run_dir,
)
from kd.search.run_spec import RUN_SPEC_HASH_SCHEME, RunSpec

pytestmark = pytest.mark.unit


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
            created_at="2026-08-09T00:00:00+00:00",
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


def _sample_result() -> ExperimentResult:
    recorder = VizRecorder()
    recorder.log("loss", 0.5)
    return ExperimentResult(
        best_expression="mul(u, u_x)",
        best_score=-50.0,
        iterations=3,
        early_stopped=False,
        final_eval=EvaluationResult(
            mse=0.01,
            nmse=0.02,
            r2=0.98,
            score=-50.0,
            complexity=1,
            coefficients=None,
            is_valid=True,
            selected_indices=None,
            residuals=None,
            terms=["mul(u, u_x)"],
            expression="mul(u, u_x)",
        ),
        actual=torch.randn(8),
        predicted=torch.randn(8),
        dataset_name="burgers_1d",
        algorithm_name="SGAPlugin",
        config={"algorithm": "sga"},
        recorder=recorder,
        manifest=RunManifest(
            dataset_cache_fingerprint="sha256:dataset",
            kd_version="0.1.0",
            seed=0,
        ),
        run_record=_sample_run_record(),
    )


class TestLayout:
    def test_create_claims_fresh_dir_and_derives_paths(
        self, tmp_path: Path
    ) -> None:
        paths = create_run_dir(tmp_path / "run-a")
        assert isinstance(paths, RunDirPaths)
        assert paths.root.is_dir()
        assert paths.checkpoints == paths.root / "checkpoints"
        assert paths.events.name == "events.jsonl"
        assert paths.phases.name == "phases.jsonl"

    def test_create_refuses_non_empty_dir(self, tmp_path: Path) -> None:
        (tmp_path / "stray.txt").write_text("x")
        with pytest.raises(ValueError, match="not empty"):
            create_run_dir(tmp_path)

    def test_new_run_id_carries_instrument_and_is_unique(self) -> None:
        first, second = new_run_id("sga"), new_run_id("sga")
        assert first.startswith("sga-")
        assert first != second


class TestFinalize:
    def test_embed_mode_writes_record_recorder_manifest(
        self, tmp_path: Path
    ) -> None:
        paths = create_run_dir(tmp_path / "run-a")
        result = _sample_result()
        payload = finalize_run_dir(
            paths,
            result,
            run_id="sga-x",
            instrument="sga",
            status="completed",
        )
        assert paths.record.is_file()
        assert paths.recorder.is_file()
        manifest = json.loads(paths.manifest.read_text())
        assert manifest == json.loads(json.dumps(payload))
        assert manifest["scheme"] == RUNDIR_SCHEME
        assert manifest["config_hash"] == result.run_record.run_spec_hash

        record_bytes = paths.record.read_bytes()
        assert manifest["artifacts"]["record.json"] == {
            "sha256": hashlib.sha256(record_bytes).hexdigest(),
            "bytes": len(record_bytes),
        }

        assert manifest["artifacts"]["events.jsonl"] is None

        assert RunRecord.load(paths.record).record_hash is not None

    def test_pointer_mode_references_external_record(
        self, tmp_path: Path
    ) -> None:
        paths = create_run_dir(tmp_path / "run-a")
        result = _sample_result()
        payload = finalize_run_dir(
            paths,
            result,
            run_id="sga-x",
            instrument="sga",
            status="completed",
            record_ref="../../records/entry-0000.json",
        )
        assert not paths.record.exists()
        assert payload["record_ref"] == {
            "path": "../../records/entry-0000.json",
            "record_hash": result.run_record.record_hash,
        }

    def test_raised_episode_writes_manifest_only(self, tmp_path: Path) -> None:
        paths = create_run_dir(tmp_path / "run-a")
        payload = finalize_run_dir(
            paths,
            None,
            run_id="sga-x",
            instrument="sga",
            status="raised",
        )
        assert not paths.record.exists()
        assert not paths.recorder.exists()
        assert payload["status"] == "raised"
        assert payload["dataset_cache_fingerprint"] is None

    def test_rejects_unknown_status_and_bad_lineage(self, tmp_path: Path) -> None:
        paths = create_run_dir(tmp_path / "run-a")
        with pytest.raises(ValueError, match="status"):
            finalize_run_dir(
                paths, None, run_id="x", instrument="sga", status="exploded"
            )
        with pytest.raises(ValueError, match="lineage"):
            finalize_run_dir(
                paths,
                None,
                run_id="x",
                instrument="sga",
                status="raised",
                lineage={"resume_from": "a"},
            )


class TestRunIdProbe:
    def test_probe_reads_own_manifest(self, tmp_path: Path) -> None:
        paths = create_run_dir(tmp_path / "run-a")
        finalize_run_dir(
            paths, None, run_id="sga-abc", instrument="sga", status="raised"
        )
        assert run_id_of_run_dir(paths.root) == "sga-abc"

    def test_probe_returns_none_for_foreign_or_absent_manifest(
        self, tmp_path: Path
    ) -> None:
        assert run_id_of_run_dir(tmp_path) is None
        (tmp_path / "manifest.json").write_text('{"scheme": "other"}')
        assert run_id_of_run_dir(tmp_path) is None
        (tmp_path / "manifest.json").write_text("not json")
        assert run_id_of_run_dir(tmp_path) is None
