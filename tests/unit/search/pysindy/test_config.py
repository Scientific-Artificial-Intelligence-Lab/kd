
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict, fields, is_dataclass

import pytest

from kd.search.pysindy.config import PySINDyConfig


def test_defaults_are_the_documented_stlsq_surface() -> None:
    config = PySINDyConfig()

    assert config.terms == ("u", "u_x", "u_xx", "mul(u, u_x)")
    assert config.threshold == 0.1
    assert config.max_iter == 20
    assert config.normalize_columns is False
    assert config.unbias is True
    assert config.seed == 0
    assert config.extra_optimizer_kwargs is None


def test_config_is_frozen_and_terms_are_a_tuple() -> None:
    config = PySINDyConfig()
    assert is_dataclass(config)
    assert isinstance(config.terms, tuple)
    with pytest.raises(FrozenInstanceError):
        config.seed = 4


def test_stlsq_only_config_has_no_optimizer_selector() -> None:
    assert "optimizer" not in {field.name for field in fields(PySINDyConfig)}


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        pytest.param({"terms": ()}, "terms", id="empty-terms"),
        pytest.param({"threshold": -0.01}, "threshold", id="negative-threshold"),
        pytest.param({"threshold": float("nan")}, "threshold", id="nan-threshold"),
        pytest.param({"threshold": float("inf")}, "threshold", id="inf-threshold"),
        pytest.param({"max_iter": 0}, "max_iter", id="zero-max-iter"),
        pytest.param({"max_iter": -1}, "max_iter", id="negative-max-iter"),
    ],
)
def test_invalid_values_fail_loud(
    kwargs: dict[str, object], fragment: str
) -> None:
    with pytest.raises(ValueError, match=fragment):
        PySINDyConfig(**kwargs)


def test_zero_threshold_is_allowed() -> None:
    assert PySINDyConfig(threshold=0.0).threshold == 0.0


def test_asdict_is_json_safe_with_default_extra_kwargs() -> None:
    payload = asdict(PySINDyConfig())
    assert json.loads(json.dumps(payload))["extra_optimizer_kwargs"] is None


def test_extra_optimizer_kwargs_are_preserved() -> None:
    extra = {"alpha": 0.02, "verbose": True}
    assert PySINDyConfig(extra_optimizer_kwargs=extra).extra_optimizer_kwargs == extra


def test_extra_optimizer_kwargs_may_not_shadow_typed_fields() -> None:
    with pytest.raises(ValueError, match="extra_optimizer_kwargs"):
        PySINDyConfig(threshold=0.1, extra_optimizer_kwargs={"threshold": 0.5})
    with pytest.raises(ValueError, match="unbias"):
        PySINDyConfig(extra_optimizer_kwargs={"alpha": 0.02, "unbias": False})
