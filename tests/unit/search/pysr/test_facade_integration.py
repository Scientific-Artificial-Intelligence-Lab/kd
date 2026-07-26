
from __future__ import annotations

import json
import math

import pytest

from kd.search.protocol import PlatformComponents
from kd.search.pysr import PySRConfig, PySRPlugin
from kd.search.result import ExperimentResult, RunManifest
from kd.search.runner import ExperimentRunner
from tests.unit.search.pysr.conftest import (
    FakePySRBackend,
    _sqrt_best_recipe,
    make_backend_factory,
)


_PARETO_KEYS = {
    "pareto_complexity",
    "pareto_loss",
    "pareto_nmse",
    "selected_complexity",
}







@pytest.mark.integration
class TestPySRFacadeEndToEnd:

    def test_run_produces_valid_experiment_result(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = PySRPlugin(PySRConfig(), backend_factory=make_backend_factory(backend))
        runner = ExperimentRunner(algorithm=plugin, max_iterations=1)

        result = runner.run(real_pysr_components)

        assert isinstance(result, ExperimentResult)
        assert result.best_expression
        assert result.final_eval.is_valid
        assert math.isfinite(result.best_score)
        assert result.algorithm_name == "PySRPlugin"

    def test_run_populates_manifest_with_terms(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = PySRPlugin(PySRConfig(), backend_factory=make_backend_factory(backend))
        runner = ExperimentRunner(algorithm=plugin, max_iterations=1)

        result = runner.run(real_pysr_components)

        assert result.manifest is not None
        assert isinstance(result.manifest, RunManifest)
        assert result.manifest.dataset_cache_fingerprint != ""
        assert result.manifest.terms is not None
        assert isinstance(result.manifest.terms, list)
        assert result.manifest.terms == plugin.terms

    def test_run_logs_pareto_whitelist_keys(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = PySRPlugin(PySRConfig(), backend_factory=make_backend_factory(backend))
        runner = ExperimentRunner(algorithm=plugin, max_iterations=1)

        result = runner.run(real_pysr_components)

        logged = set(result.recorder.to_dict().keys())
        assert logged >= _PARETO_KEYS, f"missing Pareto series: {_PARETO_KEYS - logged}"

    def test_pareto_series_are_internally_consistent(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = PySRPlugin(PySRConfig(), backend_factory=make_backend_factory(backend))
        runner = ExperimentRunner(algorithm=plugin, max_iterations=1)

        result = runner.run(real_pysr_components)
        rec = result.recorder.to_dict()

        complexity = rec["pareto_complexity"][-1]
        loss = rec["pareto_loss"][-1]
        nmse = rec["pareto_nmse"][-1]
        assert len(complexity) == len(loss) == len(nmse)
        assert len(loss) >= 1

        assert all(a > b for a, b in zip(loss, loss[1:], strict=False))

    def test_run_fits_backend_exactly_once(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = PySRPlugin(PySRConfig(), backend_factory=make_backend_factory(backend))
        runner = ExperimentRunner(algorithm=plugin, max_iterations=1)

        runner.run(real_pysr_components)

        assert backend.fit_calls == 1

    def test_extra_iterations_do_not_refit(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = PySRPlugin(PySRConfig(), backend_factory=make_backend_factory(backend))
        runner = ExperimentRunner(algorithm=plugin, max_iterations=4)

        runner.run(real_pysr_components)

        assert backend.fit_calls == 1

    def test_result_to_dict_json_round_trips(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = PySRPlugin(PySRConfig(), backend_factory=make_backend_factory(backend))
        runner = ExperimentRunner(algorithm=plugin, max_iterations=1)

        result = runner.run(real_pysr_components)
        encoded = json.dumps(result.to_dict(), allow_nan=False)

        assert isinstance(encoded, str)
        reloaded = json.loads(encoded)

        assert reloaded["manifest"]["terms"] == plugin.terms







@pytest.mark.integration
class TestPySRFacadeFailureInjection:

    def test_unconvertible_best_raises_runtime_error(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend(best_recipe=_sqrt_best_recipe)
        plugin = PySRPlugin(PySRConfig(), backend_factory=make_backend_factory(backend))
        runner = ExperimentRunner(algorithm=plugin, max_iterations=1)

        with pytest.raises(RuntimeError):
            runner.run(real_pysr_components)
