
from __future__ import annotations

import inspect
import json
from typing import Any

import pytest

import kd
from kd.api import (
    _FACADE_PARAM_DEFAULTS,
    _GENERATIONS_INTO_CONFIG,
    _PLUGIN_CLASS_BY_ALGORITHM,
    Model,
)

_RESUME_TIERS = frozenset({"resume_safe", "init_only", "identity_breaking"})
_SETTABLE_FROM = frozenset({"json", "python", "facade"})



_REQUIRED_VALUES: dict[str, dict[str, Any]] = {
    "eqgpt": {"sparsity_alpha": 0.02},
}




_PERTURBATION: dict[str, tuple[str, Any]] = {
    "sga": ("lam", 0.25),
    "dlga": ("pop_size", 37),
    "discover": ("entropy_weight", 0.02),
    "pysr": ("maxsize", 11),
    "pysindy": ("threshold", 0.25),
    "eqgpt": ("top_k", 7),
    "llm4ed": ("pool_size", 9),
}

_ALGORITHMS = sorted(_PLUGIN_CLASS_BY_ALGORITHM)


def _schema(algorithm: str) -> dict[str, Any]:
    return next(
        item for item in kd.instrument_schemas() if item["algorithm"] == algorithm
    )


def _build_config(algorithm: str, **kwargs: Any) -> Any:
    model = Model(algorithm=algorithm, verbose=False, **kwargs)
    builder = getattr(model, model._CONFIG_BUILDER_BY_ALGORITHM[algorithm])
    return builder()


def _json_settable_defaults(algorithm: str) -> dict[str, Any]:
    return {
        row["name"]: row["default"]
        for row in _schema(algorithm)["fields"]
        if row["settable_from"] == "json" and not row["required"]
    }







@pytest.mark.unit
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_every_advertised_field_is_accepted_under_its_own_name(
    algorithm: str,
) -> None:
    required = _REQUIRED_VALUES.get(algorithm, {})
    expected = _build_config(algorithm, **required)
    actual = _build_config(
        algorithm, **{**_json_settable_defaults(algorithm), **required}
    )
    assert actual == expected







@pytest.mark.unit
def test_perturbation_table_covers_exactly_the_registry() -> None:
    assert set(_PERTURBATION) == set(_PLUGIN_CLASS_BY_ALGORITHM)


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_a_non_default_value_reaches_the_built_config(algorithm: str) -> None:
    field, value = _PERTURBATION[algorithm]
    config = _build_config(
        algorithm, **{**_REQUIRED_VALUES.get(algorithm, {}), field: value}
    )
    assert getattr(config, field) == value







@pytest.mark.unit
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_an_unknown_field_name_is_rejected(algorithm: str) -> None:
    with pytest.raises(TypeError) as exc_info:
        Model(
            algorithm=algorithm,
            verbose=False,
            definitely_not_a_field_xyz=1,
            **_REQUIRED_VALUES.get(algorithm, {}),
        )
    message = str(exc_info.value)
    assert algorithm in message
    assert "definitely_not_a_field_xyz" in message







@pytest.mark.unit
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_every_row_names_a_real_config_field(algorithm: str) -> None:
    import dataclasses

    config_cls = _PLUGIN_CLASS_BY_ALGORITHM[algorithm].config_cls
    field_names = [field.name for field in dataclasses.fields(config_cls)]
    row_names = [row["name"] for row in _schema(algorithm)["fields"]]
    assert row_names == field_names


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_row_shape_is_json_safe_and_self_consistent(algorithm: str) -> None:
    for row in _schema(algorithm)["fields"]:
        assert row["settable_from"] in _SETTABLE_FROM
        assert row["resume_tier"] in _RESUME_TIERS





        assert (row["json_kind"] is None) == (row["settable_from"] == "python")

        if row["required"]:
            assert row["default"] is None
        assert json.loads(json.dumps(row["default"])) == row["default"]


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_facade_owned_fields_name_the_parameter_that_reaches_them(
    algorithm: str,
) -> None:
    for row in _schema(algorithm)["fields"]:
        if row["settable_from"] == "facade":
            assert row["facade_param"]







@pytest.mark.unit
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_facade_params_cover_the_facade_route_and_the_budget(
    algorithm: str,
) -> None:
    published = {row["name"] for row in _schema(algorithm)["facade_params"]}
    mapped = {
        row["facade_param"]
        for row in _schema(algorithm)["fields"]
        if row["settable_from"] == "facade"
    }
    assert published == mapped | {"generations"}


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_facade_param_rows_are_json_safe_and_self_consistent(
    algorithm: str,
) -> None:
    for row in _schema(algorithm)["facade_params"]:
        assert row["effect"] in {"config_field", "max_iterations", "unused"}
        assert (row["config_field"] is not None) == (row["effect"] == "config_field")
        assert row["kind"] in {"bool", "int", "float", "str"}
        assert json.loads(json.dumps(row)) == row


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_published_facade_defaults_are_what_omitting_them_produces(
    algorithm: str,
) -> None:
    config = _build_config(algorithm, **_REQUIRED_VALUES.get(algorithm, {}))
    fields = {row["name"]: row for row in _schema(algorithm)["fields"]}
    for row in _schema(algorithm)["facade_params"]:
        field = fields.get(row["config_field"] or "")
        if field is None or field["json_kind"] != row["kind"]:
            continue
        assert getattr(config, row["config_field"]) == row["default"]


@pytest.mark.unit
def test_the_budget_effect_matches_what_the_facade_itself_does() -> None:
    for algorithm in _ALGORITHMS:
        row = next(
            row
            for row in _schema(algorithm)["facade_params"]
            if row["name"] == "generations"
        )
        if algorithm in _GENERATIONS_INTO_CONFIG:
            assert row["effect"] == "config_field"
        elif _PLUGIN_CLASS_BY_ALGORITHM[algorithm].one_shot:
            assert row["effect"] == "unused"
        else:
            assert row["effect"] == "max_iterations"


@pytest.mark.unit
def test_every_facade_parameter_has_a_declared_default() -> None:
    signature = inspect.signature(Model.__init__).parameters
    for name in _FACADE_PARAM_DEFAULTS:
        assert name in signature
    for algorithm in _ALGORITHMS:
        for row in _schema(algorithm)["facade_params"]:
            assert row["name"] in _FACADE_PARAM_DEFAULTS


@pytest.mark.unit
def test_seed_is_facade_owned_on_every_algorithm() -> None:
    for algorithm in _ALGORITHMS:
        rows = {row["name"]: row for row in _schema(algorithm)["fields"]}
        assert rows["seed"]["settable_from"] == "facade"
        assert rows["seed"]["facade_param"] == "seed"
