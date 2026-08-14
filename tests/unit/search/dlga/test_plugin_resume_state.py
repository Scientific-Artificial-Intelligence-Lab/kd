
from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from kd.core.evaluator import EvaluationResult
from kd.core.platform.builder import PlatformBuilder
from kd.data import generate_advection_data
from kd.data.schema import PDEDataset
from kd.models.field_model import FieldModel
from kd.search.dlga import DLGAConfig, DLGAPlugin
from kd.search.dlga.genes import Genome, random_population
from kd.search.protocol import PlatformComponents

pytestmark = pytest.mark.unit

_POP_SIZE = 8
_SEED = 0

_LAST_FITNESS_KEY = "last_fitness"
_SURROGATE_KEY = "surrogate_model"


def _config(**overrides: Any) -> DLGAConfig:
    return DLGAConfig(**{"pop_size": _POP_SIZE, "seed": _SEED, **overrides})


def _cheap_model() -> FieldModel:
    return FieldModel(
        coord_names=["x", "t"],
        field_names=["u"],
        hidden_sizes=[8],
        activation="tanh",
    ).to(dtype=torch.float64)


@pytest.fixture(scope="module")
def dataset() -> PDEDataset:
    return generate_advection_data(
        speeds=(1.0,), waves=(2.0,), grid_sizes=(32,), nt=16, seed=0
    )


@pytest.fixture(scope="module")
def injected_model() -> FieldModel:
    return _cheap_model()


@pytest.fixture(scope="module")
def components(
    dataset: PDEDataset, injected_model: FieldModel
) -> PlatformComponents:
    reqs = DLGAPlugin(_config(), surrogate_model=injected_model)
    return PlatformBuilder(dataset, reqs.derivative_requirements).build()


def _seeded_random_population(config: DLGAConfig) -> list[Genome]:
    rng = torch.Generator()
    rng.manual_seed(config.seed)
    return random_population(
        pop_size=config.pop_size,
        rng=rng,
        library_size=len(config.library),
        max_modules=config.max_modules,
        max_module_length=config.max_module_length,
        partial_prob=config.partial_prob,
        genes_prob=config.genes_prob,
    )


def _run_one_generation(
    plugin: DLGAPlugin,
) -> list[EvaluationResult]:
    results = plugin.evaluate(plugin.propose(_POP_SIZE))
    plugin.update(results)
    return results


def _prepared_plugin(
    components: PlatformComponents,
    injected_model: FieldModel,
    **config_overrides: Any,
) -> DLGAPlugin:
    plugin = DLGAPlugin(
        _config(**config_overrides), surrogate_model=injected_model
    )
    plugin.prepare(components)
    return plugin







def test_state_carries_the_archived_generations_fitness(
    components: PlatformComponents, injected_model: FieldModel
) -> None:
    plugin = _prepared_plugin(components, injected_model)
    results = _run_one_generation(plugin)

    state = plugin.state
    assert _LAST_FITNESS_KEY in state, (
        "state carries no archived fitness, so a resume cannot perform the "
        f"evolution step the archiving run owed; keys are {sorted(state)}."
    )
    archived = state[_LAST_FITNESS_KEY]
    assert archived is not None
    assert len(archived) == len(results)
    for fitness, result in zip(archived, results, strict=True):
        if result.is_valid:
            assert fitness == pytest.approx(result.score)
        else:
            assert fitness == float("inf")


def test_state_has_no_archived_fitness_before_the_first_generation(
    components: PlatformComponents, injected_model: FieldModel
) -> None:
    plugin = _prepared_plugin(components, injected_model)
    state = plugin.state
    assert _LAST_FITNESS_KEY in state, sorted(state)
    assert state[_LAST_FITNESS_KEY] is None


def test_state_has_no_archived_fitness_once_the_population_advances(
    components: PlatformComponents, injected_model: FieldModel
) -> None:
    plugin = _prepared_plugin(components, injected_model)
    _run_one_generation(plugin)
    plugin.between_iterations()

    state = plugin.state
    assert _LAST_FITNESS_KEY in state, sorted(state)
    assert state[_LAST_FITNESS_KEY] is None


def test_resume_with_a_larger_population_reaches_the_new_size(
    components: PlatformComponents, injected_model: FieldModel
) -> None:
    source = _prepared_plugin(components, injected_model)
    _run_one_generation(source)
    saved = source.state

    grown = DLGAPlugin(
        _config(pop_size=_POP_SIZE * 3), surrogate_model=injected_model
    )
    grown.state = saved
    grown.prepare(components)

    population = grown.state["population"]
    assert population is not None
    assert len(population) == _POP_SIZE * 3, (
        f"resume kept the archived population size {len(population)} instead of "
        f"the requested {_POP_SIZE * 3}."
    )







def test_state_carries_the_surrogate_module(
    components: PlatformComponents, injected_model: FieldModel
) -> None:
    plugin = _prepared_plugin(components, injected_model)

    state = plugin.state
    assert _SURROGATE_KEY in state, (
        "state carries no surrogate module, so a resume has to retrain NN_1; "
        f"keys are {sorted(state)}."
    )
    stored = state[_SURROGATE_KEY]
    assert isinstance(stored, nn.Module)
    stored_params = dict(stored.state_dict())
    expected_params = dict(injected_model.state_dict())
    assert stored_params.keys() == expected_params.keys()
    for name, tensor in expected_params.items():
        torch.testing.assert_close(stored_params[name], tensor, rtol=0.0, atol=0.0)


def test_restored_surrogate_reaches_the_platform_via_derivative_requirements(
    components: PlatformComponents, injected_model: FieldModel
) -> None:
    source = _prepared_plugin(components, injected_model)
    _run_one_generation(source)
    restored_module = _cheap_model()
    payload = {**source.state, _SURROGATE_KEY: restored_module}

    plugin = DLGAPlugin(_config())
    plugin.state = payload

    assert plugin.derivative_requirements.surrogate_model is restored_module
    assert _SURROGATE_KEY not in plugin.config, (
        "a restored surrogate leaked into the live config; the resume gate "
        "reads config identity and would refuse this resume."
    )
    assert plugin.artifacts is None

    plugin.prepare(components)
    assert plugin.derivative_requirements.surrogate_model is None, (
        "the restored module must be consumed by prepare(): a reused "
        "instance's NEXT build reads derivative_requirements before any "
        "prepare() can reset, so lingering residue would inject the previous "
        "dataset's surrogate into a fresh run."
    )


def test_state_round_trips_before_prepare(
    components: PlatformComponents, injected_model: FieldModel
) -> None:
    source = _prepared_plugin(components, injected_model)
    _run_one_generation(source)
    restored_module = _cheap_model()
    payload = {**source.state, _SURROGATE_KEY: restored_module}

    plugin = DLGAPlugin(_config())
    plugin.state = payload
    echoed = plugin.state

    assert echoed[_LAST_FITNESS_KEY] == payload[_LAST_FITNESS_KEY]
    assert echoed[_SURROGATE_KEY] is restored_module







def test_fresh_prepare_starts_from_the_seeded_random_population(
    components: PlatformComponents, injected_model: FieldModel
) -> None:
    config = _config()
    plugin = DLGAPlugin(config, surrogate_model=injected_model)
    plugin.prepare(components)

    assert plugin.state["population"] == _seeded_random_population(config)


def test_prepare_after_a_reset_ignores_the_abandoned_restore(
    components: PlatformComponents, injected_model: FieldModel
) -> None:
    source = _prepared_plugin(components, injected_model)
    _run_one_generation(source)
    payload = {**source.state, _SURROGATE_KEY: _cheap_model()}

    config = _config()
    plugin = DLGAPlugin(config)
    plugin.state = payload
    plugin.state = {}

    assert plugin.derivative_requirements.surrogate_model is None
    plugin.prepare(components)
    assert plugin.state["population"] == _seeded_random_population(config)


def test_second_prepare_after_a_consumed_restore_starts_fresh(
    components: PlatformComponents, injected_model: FieldModel
) -> None:
    source = _prepared_plugin(components, injected_model)
    _run_one_generation(source)
    payload = {**source.state, _SURROGATE_KEY: _cheap_model()}

    config = _config()
    plugin = DLGAPlugin(config, surrogate_model=injected_model)
    plugin.state = payload
    plugin.prepare(components)
    plugin.prepare(components)

    assert plugin.state["population"] == _seeded_random_population(config)
