
from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from typing import Any

import pytest


from kd.search.pysr.config import PySRConfig

pytestmark = pytest.mark.unit







class TestPySRConfigDefaults:

    @pytest.mark.smoke
    def test_constructs_with_no_arguments(self) -> None:
        cfg = PySRConfig()
        assert cfg is not None

    def test_is_a_dataclass(self) -> None:
        assert is_dataclass(PySRConfig)

    def test_default_terms(self) -> None:
        cfg = PySRConfig()
        assert cfg.terms == ("u", "u_x", "u_xx", "mul(u, u_x)")

    def test_default_scalar_knobs(self) -> None:
        cfg = PySRConfig()
        assert cfg.seed == 0
        assert cfg.niterations == 40
        assert cfg.population_size == 33
        assert cfg.populations == 15
        assert cfg.maxsize == 20

    def test_default_operators(self) -> None:
        cfg = PySRConfig()
        assert cfg.binary_operators == ("+", "-", "*", "/")
        assert cfg.unary_operators == ("sin", "cos", "exp", "log")

    def test_default_extra_kwargs_is_none(self) -> None:
        cfg = PySRConfig()
        assert cfg.extra_pysr_kwargs is None







class TestPySRConfigFieldTypes:

    def test_terms_is_tuple(self) -> None:
        assert isinstance(PySRConfig().terms, tuple)

    def test_binary_operators_is_tuple(self) -> None:
        assert isinstance(PySRConfig().binary_operators, tuple)

    def test_unary_operators_is_tuple(self) -> None:
        assert isinstance(PySRConfig().unary_operators, tuple)

    def test_no_mutable_default_shared_between_instances(self) -> None:
        a = PySRConfig()
        b = PySRConfig()
        assert a.extra_pysr_kwargs is None
        assert b.extra_pysr_kwargs is None

        assert a.terms == b.terms







class TestPySRConfigCustom:

    def test_custom_terms(self) -> None:
        cfg = PySRConfig(terms=("u", "u_xx"))
        assert cfg.terms == ("u", "u_xx")

    def test_custom_scalar_knobs(self) -> None:
        cfg = PySRConfig(
            seed=7,
            niterations=100,
            population_size=50,
            populations=20,
            maxsize=30,
        )
        assert cfg.seed == 7
        assert cfg.niterations == 100
        assert cfg.population_size == 50
        assert cfg.populations == 20
        assert cfg.maxsize == 30

    def test_custom_operators(self) -> None:
        cfg = PySRConfig(
            binary_operators=("+", "*"),
            unary_operators=("sin",),
        )
        assert cfg.binary_operators == ("+", "*")
        assert cfg.unary_operators == ("sin",)

    def test_extra_pysr_kwargs_passthrough(self) -> None:
        extra: dict[str, Any] = {"parsimony": 0.001, "model_selection": "best"}
        cfg = PySRConfig(extra_pysr_kwargs=extra)
        assert cfg.extra_pysr_kwargs == extra







class TestPySRConfigImmutability:

    def test_field_assignment_is_blocked(self) -> None:
        cfg = PySRConfig()


        with pytest.raises((AttributeError,)):
            cfg.seed = 99

    def test_hashable_when_extra_is_none(self) -> None:
        cfg = PySRConfig()

        assert isinstance(hash(cfg), int)







class TestPySRConfigSerialization:

    def test_asdict_returns_dict(self) -> None:
        cfg = PySRConfig()
        data = asdict(cfg)
        assert isinstance(data, dict)
        field_names = {f.name for f in fields(PySRConfig)}
        assert set(data) == field_names

    def test_asdict_is_json_safe_with_default_config(self) -> None:
        cfg = PySRConfig()

        encoded = json.dumps(asdict(cfg))
        assert isinstance(encoded, str)

    def test_asdict_json_roundtrip_preserves_scalars(self) -> None:
        cfg = PySRConfig(seed=3, niterations=11, maxsize=7)
        decoded = json.loads(json.dumps(asdict(cfg)))
        assert decoded["seed"] == 3
        assert decoded["niterations"] == 11
        assert decoded["maxsize"] == 7

    def test_asdict_json_safe_with_json_safe_extra(self) -> None:
        cfg = PySRConfig(extra_pysr_kwargs={"parsimony": 0.01})
        encoded = json.dumps(asdict(cfg))
        assert "parsimony" in encoded







class TestPySRConfigValidation:

    def test_empty_terms_raises(self) -> None:
        with pytest.raises(ValueError):
            PySRConfig(terms=())

    @pytest.mark.parametrize("bad", [0, -1, -40])
    def test_non_positive_niterations_raises(self, bad: int) -> None:
        with pytest.raises(ValueError):
            PySRConfig(niterations=bad)

    @pytest.mark.parametrize("bad", [0, -1, -20])
    def test_non_positive_maxsize_raises(self, bad: int) -> None:
        with pytest.raises(ValueError):
            PySRConfig(maxsize=bad)

    @pytest.mark.parametrize("bad", [0, -1, -33])
    def test_non_positive_population_size_raises(self, bad: int) -> None:
        with pytest.raises(ValueError):
            PySRConfig(population_size=bad)

    def test_valid_minimal_config_does_not_raise(self) -> None:
        cfg = PySRConfig(
            terms=("u",),
            niterations=1,
            maxsize=1,
            population_size=1,
        )
        assert cfg.terms == ("u",)
        assert cfg.niterations == 1
        assert cfg.maxsize == 1
        assert cfg.population_size == 1
