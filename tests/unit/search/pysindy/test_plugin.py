
from __future__ import annotations

import json
import math
import pickle
from dataclasses import replace

import numpy as np
import pytest
import torch

from kd.core.platform.requirements import DerivativeReqs
from kd.search.protocol import PlatformComponents, SearchAlgorithm
from kd.search.pysindy.config import PySINDyConfig
from kd.search.pysindy.plugin import PySINDyPlugin
from tests.unit.search.pysindy.conftest import (
    FakeSINDyBackend,
    all_zero_coefficient_recipe,
    make_backend_factory,
    make_invalid_result,
    nonfinite_coefficient_recipe,
    wrong_shape_coefficient_recipe,
)

_TERMS = ("u", "u_x", "u_xx", "mul(u,u_x)")


def _make_plugin(
    backend: FakeSINDyBackend,
    *,
    terms: tuple[str, ...] = _TERMS,
    seed: int = 0,
    threshold: float = 0.1,
) -> PySINDyPlugin:
    config = PySINDyConfig(
        terms=terms,
        seed=seed,
        threshold=threshold,
    )
    return PySINDyPlugin(config, backend_factory=make_backend_factory(backend))


def _prepared_plugin(
    components: PlatformComponents,
    backend: FakeSINDyBackend,
    *,
    terms: tuple[str, ...] = _TERMS,
) -> PySINDyPlugin:
    plugin = _make_plugin(backend, terms=terms)
    plugin.prepare(components)
    return plugin


def test_default_construction_is_lazy_and_protocol_conformant() -> None:
    plugin = PySINDyPlugin()
    assert isinstance(plugin, SearchAlgorithm)
    assert plugin.one_shot is True
    assert plugin.runner_batch_size == 1
    assert plugin.score_kind == "NMSE"
    assert plugin.score_direction == "min"


def test_prepare_requires_an_evaluator(
    real_pysindy_components: PlatformComponents,
) -> None:
    plugin = _make_plugin(FakeSINDyBackend())
    without_evaluator = replace(real_pysindy_components, evaluator=None)
    with pytest.raises(TypeError, match="evaluator"):
        plugin.prepare(without_evaluator)


def test_propose_is_one_shot_and_fits_exactly_once(
    real_pysindy_components: PlatformComponents,
) -> None:
    backend = FakeSINDyBackend()
    plugin = _prepared_plugin(real_pysindy_components, backend)

    first = plugin.propose(8)
    second = plugin.propose(8)

    assert first == ["u_xx"]
    assert second == []

    assert backend.fit_calls == 1


def test_backend_receives_exported_theta_and_lhs_as_float64(
    real_pysindy_components: PlatformComponents,
) -> None:
    evaluator = real_pysindy_components.evaluator
    assert evaluator is not None
    theta, valid_terms = evaluator.build_theta_matrix(list(_TERMS))
    expected_y = evaluator.lhs_target.detach().cpu().numpy().astype(np.float64)
    backend = FakeSINDyBackend()
    plugin = _prepared_plugin(real_pysindy_components, backend)

    plugin.propose(1)

    assert valid_terms == list(_TERMS)
    assert backend.captured_X is not None
    assert backend.captured_y is not None
    assert backend.captured_X.dtype == np.float64
    assert backend.captured_y.dtype == np.float64
    np.testing.assert_array_equal(
        backend.captured_X,
        theta.detach().cpu().numpy().astype(np.float64),
    )
    np.testing.assert_array_equal(backend.captured_y, expected_y.reshape(-1))


def test_native_result_is_full_length_and_catalog_aligned(
    real_pysindy_components: PlatformComponents,
) -> None:
    plugin = _prepared_plugin(real_pysindy_components, FakeSINDyBackend())
    plugin.propose(1)

    result = plugin.build_final_result()

    assert result.is_valid
    assert result.expression == "u_xx"
    assert result.terms == list(_TERMS)
    assert result.selected_indices == [2]
    assert result.coefficients is not None
    torch.testing.assert_close(
        result.coefficients,
        torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=result.coefficients.dtype),
    )
    assert plugin.best_score == result.nmse
    assert plugin.best_expression == result.expression
    assert plugin.terms == list(_TERMS)
    exposed_terms = plugin.terms
    assert exposed_terms is not None
    exposed_terms.clear()
    assert plugin.terms == list(_TERMS)


def test_empty_support_fails_with_actionable_threshold_name(
    real_pysindy_components: PlatformComponents,
) -> None:
    backend = FakeSINDyBackend(all_zero_coefficient_recipe)
    plugin = _make_plugin(backend, threshold=0.75)
    plugin.prepare(real_pysindy_components)
    with pytest.raises(RuntimeError, match="threshold"):
        plugin.propose(1)


@pytest.mark.parametrize(
    ("recipe", "fragment"),
    [
        pytest.param(wrong_shape_coefficient_recipe, "shape", id="wrong-shape"),
        pytest.param(nonfinite_coefficient_recipe, "finite", id="nonfinite"),
    ],
)
def test_backend_output_is_validated_at_plugin_boundary(
    real_pysindy_components: PlatformComponents,
    recipe: object,
    fragment: str,
) -> None:
    backend = FakeSINDyBackend(recipe)
    plugin = _prepared_plugin(real_pysindy_components, backend)
    with pytest.raises(RuntimeError, match=fragment):
        plugin.propose(1)


def test_fresh_prepare_resets_fit_state(
    real_pysindy_components: PlatformComponents,
) -> None:
    backend = FakeSINDyBackend()
    plugin = _prepared_plugin(real_pysindy_components, backend)
    plugin.propose(1)
    plugin.prepare(real_pysindy_components)

    assert plugin.best_expression == ""
    assert math.isinf(plugin.best_score)
    assert plugin.propose(1) == ["u_xx"]
    assert backend.fit_calls == 2


def test_restored_state_preserves_native_coefficients_without_refit(
    real_pysindy_components: PlatformComponents,
) -> None:
    def two_term_recipe(n_features: int) -> np.ndarray:
        values = np.zeros(n_features, dtype=np.float64)
        values[0] = 1.5
        values[2] = -0.25
        return values

    donor = _prepared_plugin(
        real_pysindy_components,
        FakeSINDyBackend(two_term_recipe),
    )
    donor.propose(1)
    expected = donor.build_final_result()
    saved = pickle.loads(pickle.dumps(donor.state))

    restored_backend = FakeSINDyBackend()
    restored = _make_plugin(restored_backend)
    restored.state = saved
    restored.prepare(real_pysindy_components)

    assert restored.propose(1) == []
    assert restored_backend.fit_calls == 0
    actual = restored.build_final_result()
    assert actual.coefficients is not None
    assert expected.coefficients is not None
    torch.testing.assert_close(actual.coefficients, expected.coefficients)
    assert actual.selected_indices == expected.selected_indices
    assert actual.terms == expected.terms
    assert actual.expression == expected.expression
    assert actual.residuals is not None
    assert expected.residuals is not None
    torch.testing.assert_close(actual.residuals, expected.residuals)


def test_build_result_target_is_detached_independent_clone(
    real_pysindy_components: PlatformComponents,
) -> None:
    evaluator = real_pysindy_components.evaluator
    assert evaluator is not None
    original = evaluator.lhs_target.clone()
    plugin = _prepared_plugin(real_pysindy_components, FakeSINDyBackend())

    target = plugin.build_result_target()
    target.zero_()

    assert target.requires_grad is False
    torch.testing.assert_close(evaluator.lhs_target, original)


def test_config_is_json_safe_and_carries_catalog_identity() -> None:
    plugin = _make_plugin(FakeSINDyBackend(), seed=17)
    assert plugin.config["algorithm"] == "pysindy"
    assert plugin.config["seed"] == 17
    assert isinstance(plugin.config["library_fingerprint"], str)
    json.dumps(plugin.config)


@pytest.mark.parametrize(
    ("terms", "max_order"),
    [
        pytest.param(("u", "mul(u,u)"), 1, id="derivative-free-floor"),
        pytest.param(("u", "u_xx"), 2, id="second-order"),
        pytest.param(("u_xxx",), 3, id="third-order"),
    ],
)
def test_derivative_requirements_follow_catalog(
    terms: tuple[str, ...], max_order: int
) -> None:
    reqs = _make_plugin(FakeSINDyBackend(), terms=terms).derivative_requirements
    assert isinstance(reqs, DerivativeReqs)
    assert reqs == DerivativeReqs(
        provider_kind="finite_diff",
        max_atomic_order=max_order,
        lhs_order=1,
        needs_surrogate=False,
    )


def test_update_logs_native_and_refit_metrics_once(
    real_pysindy_components: PlatformComponents,
) -> None:
    recorder = real_pysindy_components.recorder
    assert recorder is not None
    plugin = _prepared_plugin(real_pysindy_components, FakeSINDyBackend())
    plugin.propose(1)
    invalid_refit = make_invalid_result()

    plugin.update([invalid_refit])
    plugin.update([invalid_refit])

    assert recorder.keys() == {"native_nmse", "refit_nmse", "support_size"}
    assert recorder.get("native_nmse") == [plugin.best_score]
    assert recorder.get("refit_nmse") == [None]
    assert recorder.get("support_size") == [1]


def test_restore_rejects_checkpoint_from_a_different_catalog(
    real_pysindy_components: PlatformComponents,
) -> None:
    donor = _prepared_plugin(real_pysindy_components, FakeSINDyBackend())
    donor.propose(1)
    saved = donor.state

    other = _make_plugin(FakeSINDyBackend(), terms=("u", "u_x"))
    with pytest.raises(ValueError, match="library_fingerprint"):
        other.state = saved


def test_restore_rejects_present_but_null_fingerprint(
    real_pysindy_components: PlatformComponents,
) -> None:
    donor = _prepared_plugin(real_pysindy_components, FakeSINDyBackend())
    donor.propose(1)
    saved = dict(donor.state)
    saved["library_fingerprint"] = None

    other = _make_plugin(FakeSINDyBackend())
    with pytest.raises(ValueError, match="library_fingerprint"):
        other.state = saved


def test_legacy_payload_without_fingerprint_restores(
    real_pysindy_components: PlatformComponents,
) -> None:
    donor = _prepared_plugin(real_pysindy_components, FakeSINDyBackend())
    donor.propose(1)
    saved = dict(donor.state)
    saved.pop("library_fingerprint")

    restored_backend = FakeSINDyBackend()
    restored = _make_plugin(restored_backend)
    restored.state = saved
    assert restored.best_expression == donor.best_expression
    restored.prepare(real_pysindy_components)
    assert restored.propose(1) == []
    assert restored_backend.fit_calls == 0


def test_restored_best_score_is_repriced_from_the_rebuild(
    real_pysindy_components: PlatformComponents,
) -> None:
    donor = _prepared_plugin(real_pysindy_components, FakeSINDyBackend())
    donor.propose(1)
    saved = dict(donor.state)
    saved["best_score"] = 123.456

    restored = _make_plugin(FakeSINDyBackend())
    restored.state = saved
    restored.prepare(real_pysindy_components)
    result = restored.build_final_result()

    assert restored.best_score == result.nmse
    assert restored.best_score != 123.456
