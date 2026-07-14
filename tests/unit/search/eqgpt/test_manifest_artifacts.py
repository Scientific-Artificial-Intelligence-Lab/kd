
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.search.eqgpt import vocab as vmod
from kd.search.eqgpt.artifacts import (
    build_run_artifacts,
    resolve_weights_fingerprint_path,
)
from kd.search.eqgpt.backend import FakeGPTBackend
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.plugin import EqGPTPlugin
from kd.search.eqgpt.vocab import vocab_asset_path
from kd.search.protocol import PlatformComponents
from kd.search.result import RunManifest
from kd.search.runner import ExperimentRunner
from tests.unit.search._runner_mocks import RecordingAlgorithm


def _tiny_components() -> PlatformComponents:
    x = torch.linspace(0.0, 1.0, 10)
    t = torch.linspace(0.0, 0.5, 5)
    gx, gt = torch.meshgrid(x, t, indexing="ij")
    dataset = PDEDataset(
        name="artifacts_tiny",
        task_type=TaskType.PDE,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=torch.sin(gx) * torch.cos(gt))},
        lhs_field="u",
        lhs_axis="t",
    )
    return PlatformBuilder(dataset, DerivativeReqs()).build()


class _ArtifactAlgorithm(RecordingAlgorithm):

    def __init__(self, artifacts: dict[str, Any]) -> None:
        super().__init__()
        self.artifacts = artifacts







def test_run_manifest_round_trips_artifacts() -> None:
    artifacts = {"vocab": {"sha256": "abc", "size": 1, "path": "v.json"}}
    manifest = RunManifest(
        dataset_fingerprint="fp", kd_version="0.1.0", seed=0, artifacts=artifacts
    )
    assert RunManifest.from_dict(manifest.to_dict()) == manifest
    assert manifest.to_dict()["artifacts"] == artifacts







def test_build_run_artifacts_fingerprints_vocab() -> None:
    artifacts = build_run_artifacts(vocab_path=vocab_asset_path(), variables=("t", "x"))
    assert artifacts["vocab"]["sha256"] == vmod.VOCAB_SHA256
    assert artifacts["vocab"]["size"] == vmod.VOCAB_SIZE_BYTES
    assert "vocab_mask_version" in artifacts


def test_build_run_artifacts_missing_weights_omits_them() -> None:
    artifacts = build_run_artifacts(vocab_path=vocab_asset_path(), weights_path=None)
    assert "vocab" in artifacts
    assert artifacts.get("weights") is None







def test_runner_fills_manifest_artifacts_from_algorithm() -> None:
    artifacts = {"weights": {"sha256": "deadbeef", "size": 42, "path": "w.pt"}}
    runner = ExperimentRunner(algorithm=_ArtifactAlgorithm(artifacts), max_iterations=1)
    result = runner.run(_tiny_components())
    assert result.manifest is not None
    assert result.manifest.artifacts == artifacts


def test_runner_manifest_artifacts_none_when_algorithm_has_none() -> None:
    runner = ExperimentRunner(algorithm=RecordingAlgorithm(), max_iterations=1)
    result = runner.run(_tiny_components())
    assert result.manifest is not None
    assert result.manifest.artifacts is None







def _eqgpt_plugin(*, weights_path: Any = None) -> EqGPTPlugin:
    config = EqGPTConfig(
        sparsity_alpha=0.02, variables=("t", "x"), weights_path=weights_path
    )
    return EqGPTPlugin(config, backend=FakeGPTBackend(57, seed=0))


def test_plugin_artifacts_carry_vocab_fingerprint() -> None:
    artifacts = _eqgpt_plugin().artifacts
    assert artifacts["vocab"]["sha256"] == vmod.VOCAB_SHA256
    assert "vocab_mask_version" in artifacts


def test_plugin_artifacts_fingerprint_weights_when_configured(tmp_path) -> None:
    weights_file = tmp_path / "weights.pt"
    torch.save({"w": torch.zeros(1)}, weights_file)
    weights = _eqgpt_plugin(weights_path=weights_file).artifacts.get("weights")
    assert weights is not None
    assert weights["sha256"]
    assert weights["path"]







class _BackendWithResolved:

    def __init__(self, resolved: Path | None) -> None:
        self.resolved_weights_path = resolved


def test_resolve_weights_fingerprint_prefers_backend_resolved_over_configured() -> None:
    backend = _BackendWithResolved(Path("/resolved/actual.pt"))
    got = resolve_weights_fingerprint_path(backend, Path("/configured/other.pt"))
    assert got == Path("/resolved/actual.pt")


def test_resolve_weights_fingerprint_falls_back_to_configured_without_attr() -> None:
    backend = FakeGPTBackend(57, seed=0)
    assert not hasattr(backend, "resolved_weights_path")
    configured = Path("/configured/w.pt")
    assert resolve_weights_fingerprint_path(backend, configured) == configured


def test_resolve_weights_fingerprint_none_when_both_absent() -> None:
    assert resolve_weights_fingerprint_path(_BackendWithResolved(None), None) is None

    assert resolve_weights_fingerprint_path(FakeGPTBackend(57, seed=0), None) is None


def test_plugin_artifacts_fingerprint_resolved_weights_asset_free(tmp_path) -> None:
    weights_file = tmp_path / "resolved.pt"
    torch.save({"w": torch.zeros(3)}, weights_file)
    raw = weights_file.read_bytes()

    backend = FakeGPTBackend(57, seed=0)
    backend.resolved_weights_path = weights_file
    config = EqGPTConfig(
        sparsity_alpha=0.02, variables=("t", "x"), weights_path=None
    )
    weights = EqGPTPlugin(config, backend=backend).artifacts.get("weights")

    assert weights is not None
    assert weights["sha256"] == hashlib.sha256(raw).hexdigest()
    assert weights["size"] == len(raw)
    assert weights["path"] == str(weights_file)


def test_plugin_artifacts_omit_weights_for_injected_backend_no_disk() -> None:
    plugin = _eqgpt_plugin()
    assert not hasattr(plugin._backend, "resolved_weights_path")
    artifacts = plugin.artifacts
    assert artifacts.get("weights") is None
    assert artifacts["vocab"]["sha256"] == vmod.VOCAB_SHA256







def test_build_run_artifacts_wave_surrogates_and_grids(tmp_path: Path) -> None:
    surr_a = tmp_path / "case_a.pkl"
    surr_b = tmp_path / "case_b.pkl"
    surr_a.write_bytes(b"weights-A")
    surr_b.write_bytes(b"weights-B")
    pkl = tmp_path / "WaveBreaking.pkl"
    pkl.write_bytes(b"scatter-bytes")

    artifacts = build_run_artifacts(
        vocab_path=vocab_asset_path(),
        variables=("t", "x"),
        surrogate_paths={"N_caseA": surr_a, "N_caseB": surr_b},
        wave_pkl_path=pkl,
        grid_params={"reward_points_per_window": 50, "coeff_points_per_window": 100},
    )

    surrogates = artifacts["surrogates"]
    assert set(surrogates) == {"N_caseA", "N_caseB"}

    assert surrogates["N_caseA"]["sha256"] == hashlib.sha256(b"weights-A").hexdigest()
    assert surrogates["N_caseB"]["sha256"] == hashlib.sha256(b"weights-B").hexdigest()
    assert surrogates["N_caseA"]["size"] == len(b"weights-A")

    assert surrogates["N_caseA"]["sha256"] != surrogates["N_caseB"]["sha256"]

    assert artifacts["wave_data"]["sha256"] == hashlib.sha256(b"scatter-bytes").hexdigest()
    assert artifacts["grids"] == {
        "reward_points_per_window": 50,
        "coeff_points_per_window": 100,
    }


def test_build_run_artifacts_single_case_omits_wave_keys() -> None:
    artifacts = build_run_artifacts(vocab_path=vocab_asset_path(), variables=("t", "x"))
    assert "surrogates" not in artifacts
    assert "wave_data" not in artifacts
    assert "grids" not in artifacts
