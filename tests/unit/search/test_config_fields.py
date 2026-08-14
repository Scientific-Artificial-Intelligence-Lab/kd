
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Literal, get_type_hints

import pytest

from kd.api import _PLUGIN_CLASS_BY_ALGORITHM
from kd.search.config_fields import FACADE_MAPPED, _json_kind, _render_type
from kd.search.descriptor import tool_schema


@pytest.mark.unit
@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (bool, "bool"),
        (int, "int"),
        (float, "float"),
        (str, "str"),
        (int | None, "int"),
        (Literal["tanh", "sin", "relu"], "str"),
        (list[bool], "bool_list"),
        (list[int], "int_list"),
        (frozenset[float], "float_list"),
        (tuple[str, ...], "str_list"),
        (dict[str, Any], "dict"),
        (Path, None),
        (tuple[int, str], None),
        (tuple[int, int], None),
        (dict[str, int], None),
    ],
    ids=[
        "bool",
        "int",
        "float",
        "str",
        "optional_scalar",
        "literal",
        "bool_list",
        "int_list",
        "float_frozenset",
        "variadic_str_tuple",
        "dict_escape_hatch",
        "path",
        "heterogeneous_tuple",
        "fixed_tuple",
        "typed_dict",
    ],
)
def test_json_closure_classifier(annotation: Any, expected: str | None) -> None:
    assert _json_kind(annotation) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (int, "int"),
        (float | None, "float | None"),
        (
            Literal["tanh", "sin", "relu"],
            "Literal['tanh','sin','relu']",
        ),
        (tuple[str, ...], "tuple[str, ...]"),
        (dict[str, Any] | None, "dict[str, Any] | None"),
    ],
)
def test_annotation_renderer(annotation: Any, expected: str) -> None:
    assert _render_type(annotation) == expected


@pytest.mark.unit
def test_facade_mapped_table_is_the_frozen_collision_contract() -> None:
    assert FACADE_MAPPED == {
        "sga": {
            "num": "population",
            "depth": "depth",
            "width": "width",
            "aic_ratio": "aic_ratio",
            "use_autograd": "derivatives",
            "seed": "seed",
        },
        "dlga": {"seed": "seed"},
        "discover": {"seed": "seed"},
        "pysr": {"niterations": "generations", "seed": "seed"},
        "eqgpt": {"seed": "seed"},
        "llm4ed": {"seed": "seed"},
        "pysindy": {"seed": "seed"},
    }


def _rows(algorithm: str) -> dict[str, dict[str, Any]]:
    plugin_cls = _PLUGIN_CLASS_BY_ALGORITHM[algorithm]
    return {row["name"]: row for row in tool_schema(plugin_cls)["fields"]}


@pytest.mark.unit
def test_representative_rows_publish_honest_wire_defaults() -> None:
    assert _rows("pysindy")["terms"]["default"] == [
        "u",
        "u_x",
        "u_xx",
        "mul(u, u_x)",
    ]
    assert _rows("eqgpt")["masked_tokens"]["default"] == []
    assert _rows("discover")["library"]["default"] is None
    assert _rows("eqgpt")["sparsity_alpha"]["required"] is True
    assert _rows("pysr")["niterations"]["default"] == 40


@pytest.mark.unit
def test_pysr_facade_default_remains_fifty_iterations() -> None:
    from kd.api import Model

    assert Model("pysr", verbose=False)._build_pysr_config().niterations == 50


@pytest.mark.unit
def test_sga_has_no_forwardable_json_container_field() -> None:
    plugin_cls = _PLUGIN_CLASS_BY_ALGORITHM["sga"]
    hints = get_type_hints(plugin_cls.config_cls)
    forwardable = {
        field.name
        for field in dataclasses.fields(plugin_cls.config_cls)
        if field.name not in FACADE_MAPPED["sga"]
    }
    container_fields = {
        name
        for name in forwardable
        if (_json_kind(hints[name]) or "").endswith("_list")
    }
    assert container_fields == set()


@pytest.mark.unit
def test_json_kind_describes_the_type_and_settable_from_the_reachability() -> None:
    seed = _rows("sga")["seed"]
    assert seed["settable_from"] == "facade"
    assert seed["json_kind"] == "int"
    assert seed["facade_param"] == "seed"

    weights_path = _rows("eqgpt")["weights_path"]
    assert weights_path["settable_from"] == "python"
    assert weights_path["json_kind"] is None
