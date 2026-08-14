
from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np
import pytest

from kd.harness import plan as plan_module
from kd.harness.plan import (
    PLAN_HASH_SCHEME,
    PLAN_SCHEMA_VERSION,
    ExperimentPlan,
    PlanEntry,
)
from kd.search.records import StrictDecodeError
from kd.search.run_spec import ConfigCanonicalizationError



RESERVED_MODEL_KWARGS = (
    "algorithm",
    "seed",
    "config",
    "callbacks",
    "provider",
    "surrogate_model",
    "checkpoint_dir",
    "checkpoint_every",
    "checkpoint_keep_last",
    "phases_path",
    "verbose",
    "device",
)


def _make_entry(**overrides: Any) -> PlanEntry:
    kwargs: dict[str, Any] = {
        "instrument": "sga",
        "dataset_ref": "burgers_tiny",
        "seed": 0,
        "model_kwargs": {"generations": 2, "population": 4},
    }
    kwargs.update(overrides)
    return PlanEntry(**kwargs)


def _make_plan(**overrides: Any) -> ExperimentPlan:
    kwargs: dict[str, Any] = {
        "name": "demo-plan",
        "entries": (
            _make_entry(),
            _make_entry(instrument="discover", seed=1),
        ),
    }
    kwargs.update(overrides)
    return ExperimentPlan(**kwargs)







def test_plan_hash_scheme_constant() -> None:
    assert PLAN_HASH_SCHEME == "kd-plan-v1"
    assert PLAN_SCHEMA_VERSION == 1


def test_plan_hash_deterministic_across_independent_constructions() -> None:

    first = _make_plan()
    second = _make_plan()
    assert first is not second
    assert first.plan_hash() == second.plan_hash()
    digest = first.plan_hash()
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_plan_hash_insensitive_to_model_kwargs_key_order() -> None:

    ordered = _make_plan(
        entries=(_make_entry(model_kwargs={"a": 1, "b": 2}),)
    )
    reversed_keys = _make_plan(
        entries=(_make_entry(model_kwargs={"b": 2, "a": 1}),)
    )
    assert ordered.plan_hash() == reversed_keys.plan_hash()


def test_plan_hash_sensitive_to_name() -> None:
    assert _make_plan().plan_hash() != _make_plan(name="other-plan").plan_hash()


def test_plan_hash_sensitive_to_entry_order() -> None:

    a = _make_entry()
    b = _make_entry(instrument="discover", seed=1)
    forward = ExperimentPlan(name="demo-plan", entries=(a, b))
    backward = ExperimentPlan(name="demo-plan", entries=(b, a))
    assert forward.plan_hash() != backward.plan_hash()


def test_plan_hash_sensitive_to_instrument() -> None:
    base = _make_plan(entries=(_make_entry(),))
    changed = _make_plan(entries=(_make_entry(instrument="pysindy"),))
    assert base.plan_hash() != changed.plan_hash()


def test_plan_hash_sensitive_to_dataset_ref() -> None:
    base = _make_plan(entries=(_make_entry(),))
    changed = _make_plan(entries=(_make_entry(dataset_ref="kdv_tiny"),))
    assert base.plan_hash() != changed.plan_hash()


def test_plan_hash_sensitive_to_seed() -> None:
    base = _make_plan(entries=(_make_entry(seed=0),))
    changed = _make_plan(entries=(_make_entry(seed=1),))
    assert base.plan_hash() != changed.plan_hash()


def test_plan_hash_sensitive_to_model_kwargs() -> None:
    base = _make_plan(entries=(_make_entry(model_kwargs={"generations": 2}),))
    changed = _make_plan(entries=(_make_entry(model_kwargs={"generations": 3}),))
    assert base.plan_hash() != changed.plan_hash()


def test_plan_hash_sensitive_to_entry_count() -> None:
    one = _make_plan(entries=(_make_entry(),))
    two = _make_plan(entries=(_make_entry(), _make_entry(seed=1)))
    assert one.plan_hash() != two.plan_hash()







def test_plan_json_round_trip_preserves_equality_and_hash() -> None:
    original = _make_plan()
    payload = json.loads(json.dumps(original.to_dict()))
    rebuilt = ExperimentPlan.from_dict(payload)
    assert rebuilt == original
    assert rebuilt.plan_hash() == original.plan_hash()


def test_to_dict_is_pure_content_payload() -> None:
    payload = _make_plan().to_dict()
    assert set(payload) == {"plan_schema_version", "name", "entries"}
    assert payload["plan_schema_version"] == PLAN_SCHEMA_VERSION

    assert "plan_hash" not in json.dumps(payload)


def test_to_dict_returns_defensive_copy() -> None:
    plan = _make_plan()
    payload = plan.to_dict()
    payload["entries"][0]["model_kwargs"]["generations"] = 999
    assert plan.entries[0].model_kwargs["generations"] == 2


def test_from_dict_rejects_unknown_plan_key() -> None:
    payload = _make_plan().to_dict()
    payload["extra"] = 1
    with pytest.raises(StrictDecodeError, match="Unknown"):
        ExperimentPlan.from_dict(payload)


def test_from_dict_rejects_missing_plan_key() -> None:
    payload = _make_plan().to_dict()
    del payload["name"]
    with pytest.raises(StrictDecodeError, match="name"):
        ExperimentPlan.from_dict(payload)


@pytest.mark.parametrize("version", [0, 2, "1", None, True])
def test_from_dict_rejects_bad_version(version: object) -> None:
    payload = _make_plan().to_dict()
    payload["plan_schema_version"] = version
    with pytest.raises(StrictDecodeError, match="plan_schema_version"):
        ExperimentPlan.from_dict(payload)


def test_from_dict_rejects_missing_version() -> None:
    payload = _make_plan().to_dict()
    del payload["plan_schema_version"]
    with pytest.raises(StrictDecodeError, match="plan_schema_version"):
        ExperimentPlan.from_dict(payload)


def test_from_dict_rejects_unknown_entry_key() -> None:
    payload = _make_plan().to_dict()
    payload["entries"][0]["extra"] = 1
    with pytest.raises(StrictDecodeError, match="Unknown"):
        ExperimentPlan.from_dict(payload)


def test_from_dict_rejects_missing_entry_key() -> None:
    payload = _make_plan().to_dict()
    del payload["entries"][0]["seed"]
    with pytest.raises(StrictDecodeError, match="seed"):
        ExperimentPlan.from_dict(payload)


def test_from_dict_rejects_non_array_entries() -> None:
    payload = _make_plan().to_dict()
    payload["entries"] = "not-a-list"
    with pytest.raises(StrictDecodeError, match="entries"):
        ExperimentPlan.from_dict(payload)


def test_from_dict_rejects_non_object_entry() -> None:
    payload = _make_plan().to_dict()
    payload["entries"][0] = "not-a-dict"
    with pytest.raises(StrictDecodeError, match="entries"):
        ExperimentPlan.from_dict(payload)


def test_from_dict_rejects_non_object_payload() -> None:
    with pytest.raises(StrictDecodeError, match="object"):
        ExperimentPlan.from_dict("not-a-dict")







@pytest.mark.parametrize("field", ["instrument", "dataset_ref"])
@pytest.mark.parametrize("value", ["", None, 3])
def test_entry_rejects_bad_string_field(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _make_entry(**{field: value})


@pytest.mark.parametrize("value", [True, False])
def test_entry_rejects_bool_seed(value: bool) -> None:
    with pytest.raises(ValueError, match="seed"):
        _make_entry(seed=value)


@pytest.mark.parametrize("value", ["0", 0.0, None])
def test_entry_rejects_non_int_seed(value: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        _make_entry(seed=value)


@pytest.mark.parametrize("key", RESERVED_MODEL_KWARGS)
def test_entry_rejects_reserved_model_kwargs_key(key: str) -> None:
    with pytest.raises(ValueError, match="reserved"):
        _make_entry(model_kwargs={key: 1})


def test_entry_rejects_non_dict_model_kwargs() -> None:
    with pytest.raises(ValueError, match="model_kwargs"):
        _make_entry(model_kwargs=[("a", 1)])


def test_entry_rejects_non_json_safe_model_kwargs() -> None:


    with pytest.raises(ConfigCanonicalizationError, match="int64"):
        _make_entry(model_kwargs={"n": np.int64(3)})


def test_entry_rejects_non_finite_float_model_kwargs() -> None:

    with pytest.raises(ConfigCanonicalizationError, match="Non-finite"):
        _make_entry(model_kwargs={"x": float("nan")})


def test_entry_canonicalizes_model_kwargs_at_construction() -> None:
    entry = _make_entry(model_kwargs={"a": (1, 2)})
    assert entry.model_kwargs == {"a": [1, 2]}


def test_plan_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        _make_plan(name="")


def test_plan_rejects_list_entries() -> None:

    with pytest.raises(ValueError, match="tuple"):
        _make_plan(entries=[_make_entry()])


def test_plan_rejects_empty_entries() -> None:
    with pytest.raises(ValueError, match="entries"):
        _make_plan(entries=())


def test_plan_rejects_non_entry_elements() -> None:
    with pytest.raises(ValueError, match="entries"):
        _make_plan(entries=(_make_entry().to_dict(),))







def test_plan_v1_tables_match_dataclass_fields_today() -> None:



    entry_fields = {field.name for field in dataclasses.fields(PlanEntry)}
    assert plan_module._PLAN_V1_ENTRY_FIELDS == entry_fields
    plan_fields = {field.name for field in dataclasses.fields(ExperimentPlan)}
    assert plan_module._PLAN_V1_FIELDS == plan_fields | {"plan_schema_version"}








def test_reserved_model_kwargs_cover_harness_owned_keys() -> None:



    assert frozenset(
        {
            "algorithm",
            "seed",
            "config",
            "callbacks",
            "provider",
            "surrogate_model",
            "checkpoint_dir",
            "checkpoint_every",
            "checkpoint_keep_last",
            "phases_path",
            "verbose",
            "device",
        }
    ) == plan_module._RESERVED_MODEL_KWARGS


def test_entry_rejects_device_in_model_kwargs() -> None:


    with pytest.raises(ValueError, match="reserved"):
        _make_entry(model_kwargs={"device": "cuda"})


def test_plan_v1_entry_fields_stay_frozen_without_device() -> None:


    assert frozenset(
        {"instrument", "dataset_ref", "seed", "model_kwargs"}
    ) == plan_module._PLAN_V1_ENTRY_FIELDS


def test_plan_v1_golden_hash_byte_lock() -> None:





    golden = ExperimentPlan(
        name="kd-plan-v1-golden",
        entries=(
            PlanEntry(
                instrument="sga",
                dataset_ref="burgers_tiny",
                seed=0,
                model_kwargs={"generations": 2, "population": 8},
            ),
            PlanEntry(
                instrument="discover",
                dataset_ref="kdv_tiny",
                seed=1,
                model_kwargs={},
            ),
        ),
    )
    assert golden.plan_hash() == (
        "c54cfc06a278d051776e9bf40129f35a489f0cbfc4caf7d2ced50a1ed3a48799"
    )
