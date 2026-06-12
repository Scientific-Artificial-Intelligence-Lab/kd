
from __future__ import annotations

import math
import re
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor

import kd
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
    compute_dataset_fingerprint,
)
from kd.search.discover import DiscoverConfig
from kd.search.discover.plugin import DISCOVERPlugin
from kd.search.dlga import DLGAConfig
from kd.search.dlga.plugin import DLGAPlugin
from kd.search.protocol import PlatformComponents
from kd.search.pysr import PySRConfig, PySRPlugin
from kd.search.result import RunManifest
from kd.search.runner import ExperimentRunner
from kd.search.sga import SGAConfig
from kd.search.sga.plugin import SGAPlugin
from tests.unit.search._runner_mocks import RecordingAlgorithm
from tests.unit.search.pysr.conftest import FakePySRBackend, make_backend_factory






class SeededRecordingAlgorithm(RecordingAlgorithm):

    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        self._seed = seed

    @property
    def config(self) -> dict[str, Any]:
        return {"algorithm": "SeededRecordingAlgorithm", "seed": self._seed}


def _make_1d_dataset(name: str = "manifest_1d") -> PDEDataset:
    n_points = 32
    x = torch.linspace(0, 2 * math.pi, n_points, dtype=torch.float64)
    return PDEDataset(
        name=name,
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo(name="x", values=x)},
        axis_order=["x"],
        fields={"u": FieldData(name="u", values=torch.sin(x))},
        lhs_field="u",
        lhs_axis="x",
    )


def _real_components(dataset: PDEDataset, target: Tensor) -> PlatformComponents:
    components = PlatformComponents(
        dataset=dataset,
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=MagicMock(),
        registry=MagicMock(),
    )
    components.evaluator.lhs_target = target
    return components


def _mock_components_with_target() -> PlatformComponents:
    components = PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=MagicMock(),
        registry=MagicMock(),
    )
    components.evaluator.lhs_target = torch.zeros(0)
    return components





_ID_LEAK_PATTERN = re.compile(r"id='\d|0x[0-9a-fA-F]{6,}")







@pytest.mark.integration
class TestRunnerPopulatesManifest:

    def test_run_produces_non_none_manifest(self) -> None:
        dataset = _make_1d_dataset()
        target = dataset.get_field("u")
        runner = ExperimentRunner(algorithm=RecordingAlgorithm(), max_iterations=1)

        result = runner.run(_real_components(dataset, target))

        assert result.manifest is not None
        assert isinstance(result.manifest, RunManifest)

    def test_manifest_fingerprint_and_version_non_empty(self) -> None:
        dataset = _make_1d_dataset()
        target = dataset.get_field("u")
        runner = ExperimentRunner(algorithm=RecordingAlgorithm(), max_iterations=1)

        manifest = runner.run(_real_components(dataset, target)).manifest

        assert manifest is not None
        assert isinstance(manifest.dataset_fingerprint, str)
        assert manifest.dataset_fingerprint != ""
        assert isinstance(manifest.kd_version, str)
        assert manifest.kd_version != ""

    def test_fingerprint_equals_compute_dataset_fingerprint(self) -> None:
        dataset = _make_1d_dataset()
        target = dataset.get_field("u")
        runner = ExperimentRunner(algorithm=RecordingAlgorithm(), max_iterations=1)

        manifest = runner.run(_real_components(dataset, target)).manifest

        assert manifest is not None
        assert manifest.dataset_fingerprint == compute_dataset_fingerprint(dataset)

    def test_kd_version_matches_installed_version(self) -> None:
        import importlib.metadata

        dataset = _make_1d_dataset()
        target = dataset.get_field("u")
        runner = ExperimentRunner(algorithm=RecordingAlgorithm(), max_iterations=1)

        manifest = runner.run(_real_components(dataset, target)).manifest

        assert manifest is not None

        assert manifest.kd_version == kd.__version__
        assert manifest.kd_version == importlib.metadata.version("kd")

    def test_terms_none_for_current_algorithms(self) -> None:
        dataset = _make_1d_dataset()
        target = dataset.get_field("u")
        runner = ExperimentRunner(algorithm=RecordingAlgorithm(), max_iterations=1)

        manifest = runner.run(_real_components(dataset, target)).manifest

        assert manifest is not None
        assert manifest.terms is None







@pytest.mark.integration
class TestManifestSeed:

    def test_seed_none_when_config_has_no_seed(self) -> None:
        dataset = _make_1d_dataset()
        target = dataset.get_field("u")

        assert RecordingAlgorithm().config.get("seed") is None
        runner = ExperimentRunner(algorithm=RecordingAlgorithm(), max_iterations=1)

        manifest = runner.run(_real_components(dataset, target)).manifest

        assert manifest is not None
        assert manifest.seed is None

    def test_seed_propagated_from_config(self) -> None:
        dataset = _make_1d_dataset()
        target = dataset.get_field("u")
        algorithm = SeededRecordingAlgorithm(seed=1234)
        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)

        manifest = runner.run(_real_components(dataset, target)).manifest

        assert manifest is not None
        assert manifest.seed == 1234

    def test_seed_zero_is_not_dropped(self) -> None:
        dataset = _make_1d_dataset()
        target = dataset.get_field("u")
        algorithm = SeededRecordingAlgorithm(seed=0)
        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)

        manifest = runner.run(_real_components(dataset, target)).manifest

        assert manifest is not None
        assert manifest.seed == 0







@pytest.mark.integration
class TestManifestDeterminism:

    def test_two_runs_independent_datasets_same_manifest(self) -> None:
        import json

        ds_a = _make_1d_dataset()
        ds_b = _make_1d_dataset()
        assert ds_a is not ds_b

        m1 = (
            ExperimentRunner(
                algorithm=SeededRecordingAlgorithm(seed=7), max_iterations=1
            )
            .run(_real_components(ds_a, ds_a.get_field("u")))
            .manifest
        )
        m2 = (
            ExperimentRunner(
                algorithm=SeededRecordingAlgorithm(seed=7), max_iterations=1
            )
            .run(_real_components(ds_b, ds_b.get_field("u")))
            .manifest
        )

        assert m1 is not None
        assert m2 is not None


        assert json.dumps(m1.to_dict(), sort_keys=True) == json.dumps(
            m2.to_dict(), sort_keys=True
        )
        assert m1.to_dict() == m2.to_dict()
        assert m1 == m2

    def test_fingerprint_premise_independent_datasets_equal(self) -> None:
        assert compute_dataset_fingerprint(
            _make_1d_dataset()
        ) == compute_dataset_fingerprint(_make_1d_dataset())

    def test_manifest_has_no_object_id_or_address(self) -> None:
        import json

        dataset = _make_1d_dataset()
        target = dataset.get_field("u")
        manifest = (
            ExperimentRunner(
                algorithm=SeededRecordingAlgorithm(seed=7), max_iterations=1
            )
            .run(_real_components(dataset, target))
            .manifest
        )

        assert manifest is not None
        blob = json.dumps(manifest.to_dict(), allow_nan=False)
        assert _ID_LEAK_PATTERN.search(blob) is None, (
            f"manifest leaked an object id / memory address: {blob!r}"
        )

    def test_manifest_has_no_timestamp_key(self) -> None:
        dataset = _make_1d_dataset()
        target = dataset.get_field("u")
        manifest = (
            ExperimentRunner(
                algorithm=SeededRecordingAlgorithm(seed=7), max_iterations=1
            )
            .run(_real_components(dataset, target))
            .manifest
        )

        assert manifest is not None
        keys = {k.lower() for k in manifest.to_dict()}
        forbidden = {"timestamp", "time", "date", "created_at", "datetime"}
        assert keys.isdisjoint(forbidden), (
            f"manifest carries a non-deterministic time field: {keys & forbidden}"
        )







@pytest.mark.integration
class TestManifestRobustnessToMock:

    def test_run_with_mock_dataset_does_not_crash(self) -> None:
        runner = ExperimentRunner(algorithm=RecordingAlgorithm(), max_iterations=1)

        result = runner.run(_mock_components_with_target())

        assert result.manifest is not None
        assert isinstance(result.manifest.dataset_fingerprint, str)

    def test_mock_dataset_fingerprint_is_deterministic(self) -> None:
        runner_a = ExperimentRunner(algorithm=RecordingAlgorithm(), max_iterations=1)
        runner_b = ExperimentRunner(algorithm=RecordingAlgorithm(), max_iterations=1)

        fp_a = runner_a.run(_mock_components_with_target()).manifest
        fp_b = runner_b.run(_mock_components_with_target()).manifest

        assert fp_a is not None
        assert fp_b is not None
        assert fp_a.dataset_fingerprint == fp_b.dataset_fingerprint

    def test_mock_dataset_fingerprint_has_no_object_id(self) -> None:
        runner = ExperimentRunner(algorithm=RecordingAlgorithm(), max_iterations=1)

        manifest = runner.run(_mock_components_with_target()).manifest

        assert manifest is not None
        assert _ID_LEAK_PATTERN.search(manifest.dataset_fingerprint) is None, (
            f"mock fingerprint leaked an object id: {manifest.dataset_fingerprint!r}"
        )

    def test_mock_dataset_manifest_is_json_safe(self) -> None:
        import json

        runner = ExperimentRunner(algorithm=RecordingAlgorithm(), max_iterations=1)

        manifest = runner.run(_mock_components_with_target()).manifest

        assert manifest is not None
        encoded = json.dumps(manifest.to_dict(), allow_nan=False)
        assert isinstance(encoded, str)







@pytest.mark.integration
class TestManifestThreeAlgorithmCompat:

    @pytest.mark.parametrize(
        ("make_plugin", "seed"),
        [
            pytest.param(lambda s: SGAPlugin(SGAConfig(seed=s)), 11, id="sga"),
            pytest.param(lambda s: DLGAPlugin(DLGAConfig(seed=s)), 22, id="dlga"),
            pytest.param(
                lambda s: DISCOVERPlugin(DiscoverConfig(seed=s)), 33, id="discover"
            ),
        ],
    )
    def test_real_plugin_manifest_populated(self, make_plugin: Any, seed: int) -> None:
        dataset = _make_1d_dataset()
        plugin = make_plugin(seed)
        runner = ExperimentRunner(algorithm=plugin, max_iterations=1)

        manifest = runner._build_manifest(
            _real_components(dataset, dataset.get_field("u"))
        )


        assert manifest.seed == seed

        assert manifest.dataset_fingerprint == compute_dataset_fingerprint(dataset)
        assert manifest.kd_version == kd.__version__
        assert manifest.terms is None
        assert RunManifest.from_dict(manifest.to_dict()) == manifest















_TERMS_2D_NX = 32
_TERMS_2D_NT = 16


def _make_pysr_2d_dataset(name: str = "manifest_pysr_2d") -> PDEDataset:
    x = torch.linspace(0.0, 2.0 * math.pi, _TERMS_2D_NX, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, _TERMS_2D_NT, dtype=torch.float64)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(xg) * torch.exp(-tg)
    return PDEDataset(
        name=name,
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


class _TermsExposingAlgorithm(RecordingAlgorithm):

    def __init__(self, terms: list[str] | None) -> None:
        super().__init__()
        self._terms = terms

    @property
    def terms(self) -> list[str] | None:
        return None if self._terms is None else list(self._terms)


@pytest.mark.integration
class TestManifestTermsFromPlugin:

    def test_terms_populated_for_pysr_plugin(self) -> None:
        dataset = _make_pysr_2d_dataset()
        from kd.core.platform.builder import PlatformBuilder
        from kd.core.platform.requirements import DerivativeReqs
        from kd.search.recorder import VizRecorder

        reqs = DerivativeReqs(
            provider_kind="finite_diff",
            max_atomic_order=2,
            lhs_order=1,
            needs_surrogate=False,
        )
        components = PlatformBuilder(dataset, reqs).build()
        components.recorder = VizRecorder()

        backend = FakePySRBackend()
        plugin = PySRPlugin(PySRConfig(), backend_factory=make_backend_factory(backend))
        runner = ExperimentRunner(algorithm=plugin, max_iterations=1)

        result = runner.run(components)

        assert result.manifest is not None

        assert result.manifest.terms is not None
        assert isinstance(result.manifest.terms, list)
        assert result.manifest.terms == plugin.terms

        assert RunManifest.from_dict(result.manifest.to_dict()) == result.manifest

    def test_terms_present_when_algorithm_exposes_them(self) -> None:
        dataset = _make_1d_dataset()
        target = dataset.get_field("u")
        algorithm = _TermsExposingAlgorithm(terms=["u", "u_x", "mul(u, u_x)"])
        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)

        manifest = runner.run(_real_components(dataset, target)).manifest

        assert manifest is not None
        assert manifest.terms == ["u", "u_x", "mul(u, u_x)"]

    def test_terms_none_when_algorithm_terms_is_none(self) -> None:
        dataset = _make_1d_dataset()
        target = dataset.get_field("u")
        algorithm = _TermsExposingAlgorithm(terms=None)
        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)

        manifest = runner.run(_real_components(dataset, target)).manifest

        assert manifest is not None
        assert manifest.terms is None

    def test_terms_none_for_algorithm_without_attribute(self) -> None:
        dataset = _make_1d_dataset()
        target = dataset.get_field("u")
        plain = RecordingAlgorithm()
        assert not hasattr(plain, "terms")
        runner = ExperimentRunner(algorithm=plain, max_iterations=1)

        manifest = runner.run(_real_components(dataset, target)).manifest

        assert manifest is not None
        assert manifest.terms is None
