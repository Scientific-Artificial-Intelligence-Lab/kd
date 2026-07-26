
from __future__ import annotations

import json

import pytest

from kd.core.equation import to_dict as equation_to_dict
from kd.search.protocol import PlatformComponents
from kd.search.pysindy import PySINDyConfig, PySINDyPlugin
from kd.search.runner import ExperimentRunner
from tests.unit.search.pysindy.conftest import FakeSINDyBackend, make_backend_factory

_TERMS = ("u", "u_x", "u_xx", "mul(u,u_x)")


def _run(components: PlatformComponents) -> tuple[object, FakeSINDyBackend]:
    backend = FakeSINDyBackend()
    plugin = PySINDyPlugin(
        PySINDyConfig(terms=_TERMS, seed=23),
        backend_factory=make_backend_factory(backend),
    )
    result = ExperimentRunner(
        algorithm=plugin,
        max_iterations=1,
        batch_size=plugin.runner_batch_size,
    ).run(components)
    return result, backend


@pytest.mark.integration
def test_runner_preserves_native_support_across_result_boundaries(
    real_pysindy_components: PlatformComponents,
) -> None:
    result, backend = _run(real_pysindy_components)

    assert backend.fit_calls == 1
    final_eval = result.final_eval
    assert final_eval.terms == list(_TERMS)
    assert final_eval.selected_indices == [2]
    assert final_eval.coefficients is not None
    assert final_eval.coefficients.tolist() == [0.0, 0.0, 1.0, 0.0]

    run_record = result.run_record
    assert run_record is not None
    assert run_record.evidence.support == ["u_xx"]
    assert run_record.evidence.coefficients == [1.0]


@pytest.mark.integration
def test_equation_payload_retains_full_catalog_and_active_indices(
    real_pysindy_components: PlatformComponents,
) -> None:
    result, _backend = _run(real_pysindy_components)
    assert result.equation is not None
    payload = equation_to_dict(result.equation)

    assert [entry[0] for entry in payload["terms"]] == list(_TERMS)
    assert payload["active_indices"] == [2]
    assert [entry[1]["value"] for entry in payload["terms"]] == [
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    assert result.run_record is not None
    assert result.run_record.evidence.catalog_fit == payload


@pytest.mark.integration
def test_manifest_recorder_and_serialization_are_populated(
    real_pysindy_components: PlatformComponents,
) -> None:
    result, _backend = _run(real_pysindy_components)

    assert result.manifest is not None
    assert result.manifest.seed == 23
    assert result.manifest.terms == list(_TERMS)
    assert {"native_nmse", "refit_nmse", "support_size"} <= result.recorder.keys()

    encoded = json.dumps(result.to_dict(), allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["manifest"]["terms"] == list(_TERMS)
    assert decoded["run_record"]["evidence"]["support"] == ["u_xx"]
