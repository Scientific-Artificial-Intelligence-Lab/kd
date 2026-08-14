
from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from kd.core.platform.sketch_compile import SKETCH_CONFIG_KEY

from kd.api import _PLUGIN_CLASS_BY_ALGORITHM
from kd.core.equation.sketch import PinnedTerm, sketch_to_dict
from kd.search.checkpoint_payload import build_checkpoint_payload
from kd.search.protocol import DiscoveryTask, FacadeWiringContract
from kd.search.pysindy import PySINDyPlugin
from kd.search.resume_policy import check_resume_config, resolve_field_tier
from kd.search.run_spec import CONFIG_CANON_SCHEME, canonicalize_config
from tests.unit.search._sketch_fakes import (
    PINNED_ADVECTION,
    SketchFakePlugin,
    build_components,
    burgers_sketch,
    run_plugin,
    tiny_burgers_dataset,
)

_BASE_CONFIG: dict[str, Any] = {
    "algorithm": "pysindy",
    "terms": ["u", "u_x"],
    "threshold": 0.1,
    "seed": 0,
}


def stored_snapshot(payload: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(_BASE_CONFIG)
    if payload is not None:
        config[SKETCH_CONFIG_KEY] = payload
    return canonicalize_config(config)


def guard(stored: dict[str, Any], live: dict[str, Any]) -> None:
    check_resume_config(
        stored,
        CONFIG_CANON_SCHEME,
        algorithm="pysindy",
        plugin_cls=PySINDyPlugin,
        live_config=live,
    )


@pytest.mark.parametrize("algorithm", sorted(_PLUGIN_CLASS_BY_ALGORITHM))
def test_reserved_key_is_not_shadowed_by_any_plugin_declaration(
    algorithm: str,
) -> None:
    plugin_cls: type[FacadeWiringContract] = _PLUGIN_CLASS_BY_ALGORITHM[algorithm]
    config_fields = {field.name for field in dataclasses.fields(plugin_cls.config_cls)}
    knob_names = {knob.name for knob in plugin_cls.descriptor.knobs}
    assert SKETCH_CONFIG_KEY not in config_fields
    assert SKETCH_CONFIG_KEY not in knob_names


@pytest.mark.parametrize("algorithm", sorted(_PLUGIN_CLASS_BY_ALGORITHM))
def test_reserved_key_resolves_to_the_identity_breaking_tier(
    algorithm: str,
) -> None:
    tier = resolve_field_tier(
        _PLUGIN_CLASS_BY_ALGORITHM[algorithm], algorithm, SKETCH_CONFIG_KEY
    )
    assert tier == "identity_breaking"


def test_checkpoint_snapshot_carries_the_task_payload() -> None:
    task = DiscoveryTask.from_sketch(burgers_sketch())

    payload = build_checkpoint_payload(3, SketchFakePlugin(), task=task)

    assert payload["config"][SKETCH_CONFIG_KEY] == canonicalize_config(
        dict(task.payload)
    )


def test_checkpoint_snapshot_without_a_task_gains_no_key() -> None:
    payload = build_checkpoint_payload(3, SketchFakePlugin())

    assert SKETCH_CONFIG_KEY not in payload["config"]


def test_resume_accepts_the_same_sketch() -> None:
    payload = sketch_to_dict(burgers_sketch())
    guard(stored_snapshot(payload), {**_BASE_CONFIG, SKETCH_CONFIG_KEY: payload})


@pytest.mark.parametrize(
    ("stored_payload", "live_has_sketch"),
    [
        (sketch_to_dict(burgers_sketch()), True),
        (sketch_to_dict(burgers_sketch()), False),
        (None, True),
    ],
    ids=["changed", "removed", "added"],
)
def test_resume_rejects_a_moved_sketch_naming_the_field(
    stored_payload: dict[str, Any] | None, live_has_sketch: bool
) -> None:
    live = dict(_BASE_CONFIG)
    if live_has_sketch:
        live[SKETCH_CONFIG_KEY] = sketch_to_dict(
            burgers_sketch(pinned=(PinnedTerm(PINNED_ADVECTION, -0.5),))
        )

    with pytest.raises(ValueError, match=SKETCH_CONFIG_KEY) as excinfo:
        guard(stored_snapshot(stored_payload), live)

    assert "identity_breaking" in str(excinfo.value)


def test_run_identity_carries_the_sketch_only_on_a_task_run() -> None:
    dataset = tiny_burgers_dataset()
    task = DiscoveryTask.from_sketch(burgers_sketch())

    plain = run_plugin(build_components(dataset), SketchFakePlugin())
    lowered = run_plugin(
        build_components(dataset, task=task), SketchFakePlugin()
    )

    assert plain.run_record is not None
    assert lowered.run_record is not None
    assert SKETCH_CONFIG_KEY not in plain.run_record.run_spec.config
    assert lowered.run_record.run_spec.config[SKETCH_CONFIG_KEY] == task.payload


    assert lowered.config[SKETCH_CONFIG_KEY] is not task.payload
    assert plain.run_record.run_spec_hash != lowered.run_record.run_spec_hash
