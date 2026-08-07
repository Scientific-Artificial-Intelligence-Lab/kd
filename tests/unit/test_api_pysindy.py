
from __future__ import annotations

import pytest

from kd.api import Model, _SUPPORTED_ALGORITHMS
from kd.data.schema import PDEDataset
from kd.search.iteration_events import IterationEvent, IterationEventEmitter
from kd.search.pysindy import PySINDyConfig, PySINDyPlugin
from kd.search.pysr import PySRConfig


def test_pysindy_is_a_supported_algorithm() -> None:
    assert "pysindy" in _SUPPORTED_ALGORITHMS


def test_model_builds_pysindy_plugin_with_one_shot_batch() -> None:
    model = Model(algorithm="pysindy", verbose=False)
    plugin, batch_size = model._build_plugin()
    assert isinstance(plugin, PySINDyPlugin)
    assert batch_size == 1


def test_default_config_threads_seed_but_not_generations() -> None:
    with pytest.warns(UserWarning, match="pysindy"):
        model = Model(algorithm="pysindy", generations=73, seed=19, verbose=False)
    config = model._build_pysindy_config()
    assert config.seed == 19
    assert config.max_iter == PySINDyConfig().max_iter


def test_user_config_is_deep_copied_verbatim() -> None:
    supplied = PySINDyConfig(threshold=0.25, max_iter=31, seed=7)
    with pytest.warns(UserWarning, match="pysindy"):
        model = Model(
            algorithm="pysindy",
            generations=99,
            verbose=False,
            config=supplied,
        )
    resolved = model._build_pysindy_config()
    assert resolved == supplied
    assert resolved is not supplied


def test_mismatched_config_names_required_type() -> None:
    model = Model(
        algorithm="pysindy",
        verbose=False,
        config=PySRConfig(),
    )
    with pytest.raises(TypeError, match="PySINDyConfig"):
        model._build_pysindy_config()


def test_fit_emits_exactly_one_iteration_event(
    simple_2d_dataset: PDEDataset,
) -> None:
    pytest.importorskip("pysindy")
    events: list[IterationEvent] = []
    emitter = IterationEventEmitter(on_event=events.append)
    model = Model(algorithm="pysindy", verbose=False, callbacks=[emitter])
    model.fit(simple_2d_dataset)

    assert len(events) == 1
    assert events[0].iteration == 0
