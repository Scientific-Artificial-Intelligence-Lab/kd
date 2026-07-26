
from __future__ import annotations

from typing import Any

import pytest
import torch

from kd.api import _PLUGIN_CLASS_BY_ALGORITHM
from kd.search import resume_policy
from kd.search.callbacks import build_checkpoint_payload
from kd.search.run_spec import CONFIG_CANON_SCHEME, canonicalize_config
from tests.unit.search import _resume_conformance_helpers as helpers

pytestmark = pytest.mark.unit

_REG = _PLUGIN_CLASS_BY_ALGORITHM







def _distinct(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, str):
        return value + "_DISTINCT"
    if isinstance(value, (list, tuple)):
        return [*list(value), "_DISTINCT"]
    if value is None:
        return "_DISTINCT_NON_NONE"
    return "_DISTINCT_FALLBACK"


def _stored_and_live_differing_in(
    algorithm: str, field: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    plugin = helpers.make_plugin(_REG[algorithm])
    live = dict(plugin.config)
    stored = canonicalize_config(dict(plugin.config))
    original = stored[field]
    override = _distinct(original)
    assert override != original, f"{field} override is not distinct"
    stored = {**stored, field: override}
    return stored, live


def _check(stored: dict[str, Any], live: dict[str, Any], algorithm: str) -> None:
    resume_policy.check_resume_config(
        stored,
        CONFIG_CANON_SCHEME,
        algorithm=algorithm,
        plugin_cls=_REG[algorithm],
        live_config=live,
    )











_INIT_ONLY_CASES = [
    pytest.param("sga", "aic_ratio", id="sga-aic_ratio"),
    pytest.param("sga", "p_mute", id="sga-p_mute-nonknob"),
    pytest.param("sga", "seed", id="sga-seed"),
    pytest.param("dlga", "epsilon", id="dlga-epsilon"),
    pytest.param("discover", "max_length", id="discover-max_length"),
    pytest.param("discover", "num_layers", id="discover-num_layers-structural"),
    pytest.param("pysr", "population_size", id="pysr-population_size"),
    pytest.param("eqgpt", "sparsity_alpha", id="eqgpt-sparsity_alpha"),
    pytest.param("llm4ed", "reward_limit", id="llm4ed-reward_limit"),
    pytest.param("pysindy", "threshold", id="pysindy-threshold"),
]


@pytest.mark.parametrize(("algorithm", "field"), _INIT_ONLY_CASES)
def test_init_only_change_is_rejected_naming_the_field(
    algorithm: str, field: str
) -> None:


    assert (
        resume_policy.resolve_field_tier(_REG[algorithm], algorithm, field)
        == "init_only"
    )
    stored, live = _stored_and_live_differing_in(algorithm, field)
    with pytest.raises(ValueError, match="init_only") as exc:
        _check(stored, live, algorithm)
    assert field in str(exc.value)
    assert "fresh fit" in str(exc.value)


    assert field in {c["field"] for c in exc.value.changes}








_IDENTITY_CASES = [
    pytest.param("sga", "use_autograd", id="sga-use_autograd"),
    pytest.param("dlga", "library", id="dlga-library"),
    pytest.param("discover", "max_diff_order", id="discover-max_diff_order"),
    pytest.param("pysr", "terms", id="pysr-terms"),
    pytest.param("eqgpt", "start_words", id="eqgpt-start_words"),
    pytest.param("pysindy", "terms", id="pysindy-terms"),
]


@pytest.mark.parametrize(("algorithm", "field"), _IDENTITY_CASES)
def test_identity_breaking_change_points_to_new_lineage(
    algorithm: str, field: str
) -> None:
    assert (
        resume_policy.resolve_field_tier(_REG[algorithm], algorithm, field)
        == "identity_breaking"
    )
    stored, live = _stored_and_live_differing_in(algorithm, field)
    with pytest.raises(ValueError, match="identity_breaking") as exc:
        _check(stored, live, algorithm)
    assert field in str(exc.value)
    assert "new lineage" in str(exc.value)

    assert field in {c["field"] for c in exc.value.changes}






_FINGERPRINT_ALGORITHMS = [
    pytest.param("pysr", id="pysr"),
    pytest.param("pysindy", id="pysindy"),
]


@pytest.mark.parametrize("algorithm", _FINGERPRINT_ALGORITHMS)
def test_fingerprint_cross_catalog_restore_is_rejected(algorithm: str) -> None:
    donor_fp = helpers.plugin_fingerprint(algorithm, ("u", "u_x"))
    subject = helpers.make_fingerprint_plugin(algorithm, ("u", "u_x", "u_xx"))
    assert donor_fp != subject._library.fingerprint
    saved = {"library_fingerprint": donor_fp, "fitted": True}
    with pytest.raises(ValueError, match="library_fingerprint"):
        subject.state = saved


@pytest.mark.parametrize("algorithm", _FINGERPRINT_ALGORITHMS)
def test_fingerprint_present_null_restore_is_rejected(algorithm: str) -> None:
    subject = helpers.make_fingerprint_plugin(algorithm, ("u", "u_x", "u_xx"))
    saved = {"library_fingerprint": None, "fitted": True}
    with pytest.raises(ValueError, match="library_fingerprint"):
        subject.state = saved


@pytest.mark.parametrize("algorithm", _FINGERPRINT_ALGORITHMS)
def test_fingerprint_absent_key_is_legacy_and_not_rejected(algorithm: str) -> None:
    donor = helpers.make_fingerprint_plugin(algorithm, ("u", "u_x", "u_xx"))
    legacy_payload = dict(donor.state)
    legacy_payload["best_expression"] = "u_xx"
    legacy_payload["fitted"] = True
    legacy_payload.pop("library_fingerprint")
    assert "library_fingerprint" not in legacy_payload

    subject = helpers.make_fingerprint_plugin(algorithm, ("u", "u_x", "u_xx"))
    subject.state = legacy_payload


    assert subject.best_expression == "u_xx"










def test_resume_safe_sga_num_shrink_truncates_population() -> None:
    donor = helpers.make_sga_plugin(num=8)
    donor.prepare(helpers.sga_components())
    payload = donor.state
    assert len(payload["population"]) == 8

    subject = helpers.make_sga_plugin(num=4)
    subject.state = payload
    subject.prepare(helpers.sga_components())
    candidates = subject.propose(subject.runner_batch_size)
    subject.update(subject.evaluate(candidates))



    assert len(subject._population) == 4


def test_resume_safe_dlga_pop_size_grows_on_next_generation() -> None:
    donor = helpers.make_dlga_plugin(pop_size=4)
    donor.prepare(helpers.dlga_components(4))
    payload = donor.state
    assert len(payload["population"]) == 4

    subject = helpers.make_dlga_plugin(pop_size=8)
    subject.state = payload
    subject.prepare(helpers.dlga_components(8))
    candidates = subject.propose(subject.runner_batch_size)
    subject.update(subject.evaluate(candidates))
    subject.between_iterations()


    assert len(subject._population) == 8


def test_resume_safe_discover_reward_alpha_reprices_champion() -> None:
    from kd.search.discover.config import DiscoverConfig

    config_a = DiscoverConfig(reward_alpha=0.01)
    config_b = DiscoverConfig(reward_alpha=0.05)
    spec = helpers.DiscoverStubSpec()
    expected_a = helpers.discover_expected_reprice(config_a, spec)
    expected_b = helpers.discover_expected_reprice(config_b, spec)
    assert expected_a != expected_b

    payload = helpers.discover_donor_payload(
        config_a, champion_reward=0.85, champion_expr=helpers.DISCOVER_CHAMPION_IR
    )
    subject = helpers.discover_restore_then_prepare(
        config_b, payload, helpers.DiscoverCountingEvaluator(spec)
    )

    assert subject.best_score == expected_b
    assert subject.best_score != expected_a


def test_resume_safe_discover_learning_rate_reasserts_on_deferred_load() -> None:
    from kd.search.discover.config import DiscoverConfig
    from kd.search.discover.plugin import DISCOVERPlugin

    lr_a, lr_b = 0.001, 0.005
    assert lr_a != lr_b
    payload = helpers.discover_donor_with_trained_optimizer(
        DiscoverConfig(learning_rate=lr_a)
    )
    donor_moments = payload["engine_state"]["optimizer_state"]["state"]
    assert donor_moments, "donor optimizer carried no moments to restore"
    assert any(
        bool(torch.any(entry["exp_avg"] != 0)) for entry in donor_moments.values()
    ), "donor moments are trivially zero -- the survival assertion would be vacuous"

    subject = DISCOVERPlugin(DiscoverConfig(learning_rate=lr_b))
    subject.state = payload
    subject.prepare(helpers.discover_components(helpers.DiscoverCountingEvaluator()))

    strategy = subject._engine._strategy
    controller = subject._engine._generator
    optimizer = strategy._get_optimizer(controller)

    assert all(group["lr"] == lr_b for group in optimizer.param_groups)
    restored_moments = optimizer.state_dict()["state"]
    assert all(
        torch.equal(restored_moments[key]["exp_avg"], donor_moments[key]["exp_avg"])
        for key in donor_moments
    ), "donor Adam moments did not survive the deferred load (lr is independent)"


def test_resume_safe_eqgpt_samples_per_epoch_drives_batch() -> None:
    donor = helpers.make_eqgpt_plugin(samples_per_epoch=2)
    donor.prepare(helpers.eqgpt_components())
    helpers.run_eqgpt_epochs(donor, 2)
    payload = donor.state

    subject = helpers.make_eqgpt_plugin(samples_per_epoch=5)
    subject.state = payload
    subject.prepare(helpers.eqgpt_components())

    assert donor.runner_batch_size == 2

    assert subject.runner_batch_size == 5
    assert len(subject.propose(subject.runner_batch_size)) == 5


def test_resume_safe_eqgpt_finetune_lr_reasserts_on_restore() -> None:
    lr_a, lr_b = 1e-3, 5e-3
    assert lr_a != lr_b
    donor = helpers.make_eqgpt_plugin(finetune_lr=lr_a)
    donor.prepare(helpers.eqgpt_components())
    helpers.run_eqgpt_epochs(donor, 1)
    payload = donor.state
    donor_moments = payload["optimizer_state"]["state"]
    assert any(
        bool(torch.any(entry["exp_avg"] != 0)) for entry in donor_moments.values()
    ), "donor moments are trivially zero"

    subject = helpers.make_eqgpt_plugin(finetune_lr=lr_b)
    subject.state = payload
    subject.prepare(helpers.eqgpt_components())


    optimizer = subject._optimizer
    assert optimizer is not None
    assert all(group["lr"] == lr_b for group in optimizer.param_groups)
    restored_moments = optimizer.state_dict()["state"]
    assert all(
        torch.equal(restored_moments[key]["exp_avg"], donor_moments[key]["exp_avg"])
        for key in donor_moments
    ), "donor Adam moments did not survive restore"


def test_resume_safe_llm4ed_temperature_is_live_read() -> None:
    from kd.search.llm4ed.plugin import Llm4edPlugin
    from tests.unit.search.llm4ed._plugin_helpers import (
        GOOD,
        FakeProvider,
        components_for,
        make_config,
        prepared,
        run,
    )

    temp_a, temp_b = 0.30, 0.90
    assert temp_a != temp_b
    donor, _ = prepared(
        config=make_config(temperature=temp_a, stop_threshold=0.995),
        provider=FakeProvider(GOOD),
    )
    run(donor, rounds=2)
    payload = donor.state

    subject_provider = FakeProvider(GOOD)
    subject = Llm4edPlugin(
        make_config(temperature=temp_b, stop_threshold=0.5), provider=subject_provider
    )
    subject.state = payload
    subject.prepare(components_for())
    subject.propose(4)



    request_temps = [req.params.temperature for req in subject_provider.requests]
    assert request_temps, "propose emitted no provider request"
    assert request_temps[-1] == temp_b
    assert temp_a not in request_temps



    assert subject.is_done is True
    strict = Llm4edPlugin(
        make_config(temperature=temp_b, stop_threshold=0.999),
        provider=FakeProvider(GOOD),
    )
    strict.state = payload
    strict.prepare(components_for())
    assert strict.is_done is False











_SYMMETRY_PLUGINS = [pytest.param(algorithm, id=algorithm) for algorithm in _REG]


def _gated_live_artifacts(plugin: Any, algorithm: str) -> Any:
    if resume_policy.CONFIG_ARTIFACT_KEYS.get(algorithm):
        return getattr(plugin, "artifacts", None)
    return None


@pytest.mark.parametrize("algorithm", _SYMMETRY_PLUGINS)
def test_same_config_resume_is_accepted_for_every_plugin(algorithm: str) -> None:
    donor, fresh = helpers.resume_symmetry_pair(algorithm)
    payload = build_checkpoint_payload(1, donor)


    assert payload["config"] is not None
    assert payload["config_canon_scheme"] == CONFIG_CANON_SCHEME
    assert (
        resume_policy.check_resume_config(
            payload["config"],
            payload["config_canon_scheme"],
            algorithm=algorithm,
            plugin_cls=_REG[algorithm],
            live_config=dict(fresh.config),
            live_artifacts=_gated_live_artifacts(fresh, algorithm),
        )
        is None
    )


def test_symmetry_sentinel_sweeps_the_full_live_registry() -> None:
    swept = {p.values[0] for p in _SYMMETRY_PLUGINS}
    assert swept == set(_REG)







def _tiers_with_members(algorithm: str) -> set[str]:
    plugin_cls = _REG[algorithm]
    tiers = {"init_only"}
    for knob in plugin_cls.descriptor.knobs:
        tiers.add(knob.resume_tier)
    if resume_policy.SCIENCE_AXIS_DENYLISTS[algorithm]:
        tiers.add("identity_breaking")
    return tiers






_RESUME_SAFE_BEHAVIORAL_COVERAGE = {"sga", "dlga", "discover", "eqgpt", "llm4ed"}


def _policy_leg_coverage() -> set[tuple[str, str]]:
    covered: set[tuple[str, str]] = set()
    for algorithm, _field in [(p.values[0], p.values[1]) for p in _INIT_ONLY_CASES]:
        covered.add((algorithm, "init_only"))
    for algorithm, _field in [(p.values[0], p.values[1]) for p in _IDENTITY_CASES]:
        covered.add((algorithm, "identity_breaking"))
    for algorithm in _RESUME_SAFE_BEHAVIORAL_COVERAGE:
        covered.add((algorithm, "resume_safe"))
    return covered


def test_conformance_sweeps_the_full_live_registry() -> None:
    swept = {p.values[0] for p in _INIT_ONLY_CASES}
    assert swept == set(_REG)


def test_every_plugin_tier_with_members_has_a_conformance_case() -> None:
    covered = _policy_leg_coverage()
    required = {
        (algorithm, tier)
        for algorithm in _REG
        for tier in _tiers_with_members(algorithm)
    }
    missing = required - covered
    assert not missing, f"tiers with members but no conformance case: {sorted(missing)}"


def test_resume_safe_behavioral_coverage_equals_live_membership() -> None:
    live_resume_safe_plugins = {
        algorithm
        for algorithm in _REG
        if "resume_safe" in _tiers_with_members(algorithm)
    }
    assert live_resume_safe_plugins == _RESUME_SAFE_BEHAVIORAL_COVERAGE


def test_pysr_and_pysindy_have_no_resume_safe_members() -> None:
    for algorithm in ("pysr", "pysindy"):
        assert "resume_safe" not in _tiers_with_members(algorithm)


def test_llm4ed_has_no_identity_breaking_members() -> None:
    assert resume_policy.SCIENCE_AXIS_DENYLISTS["llm4ed"] == frozenset()
    assert "identity_breaking" not in _tiers_with_members("llm4ed")
