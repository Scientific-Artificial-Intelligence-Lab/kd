
from __future__ import annotations

import json
from typing import Any

import pytest

import kd
from kd.api import (
    _PLUGIN_CLASS_BY_ALGORITHM,
    _SUPPORTED_ALGORITHMS,
    instrument_schemas,
)
from kd.search.descriptor import tool_schema

pytestmark = pytest.mark.unit


def _assert_json_primitives(value: Any) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for item in value:
            _assert_json_primitives(item)
        return
    assert isinstance(value, dict)
    for key, item in value.items():
        assert isinstance(key, str)
        _assert_json_primitives(item)


class TestInstrumentSchemasFacade:

    def test_exported_alongside_model(self) -> None:
        assert kd.instrument_schemas is instrument_schemas

    def test_algorithms_derive_exactly_from_registry_in_order(self) -> None:
        schemas = instrument_schemas()
        algorithms = [schema["algorithm"] for schema in schemas]
        assert algorithms == list(_SUPPORTED_ALGORITHMS)
        assert algorithms == list(_PLUGIN_CLASS_BY_ALGORITHM)
        assert len(algorithms) == len(set(algorithms))

    def test_facade_composes_every_registry_class(self) -> None:
        assert instrument_schemas() == [
            tool_schema(plugin_cls)
            for plugin_cls in _PLUGIN_CLASS_BY_ALGORITHM.values()
        ]

    def test_json_round_trip_contains_only_primitives(self) -> None:
        schemas = instrument_schemas()
        restored = json.loads(json.dumps(schemas))
        assert restored == schemas
        _assert_json_primitives(schemas)

    def test_capability_sets_are_sorted_lists(self) -> None:
        for schema in instrument_schemas():
            for mode in schema["modes"]:
                assert isinstance(mode["forms"], list)
                assert mode["forms"] == sorted(mode["forms"])
                assert isinstance(mode["topologies"], list)
                assert mode["topologies"] == sorted(mode["topologies"])

