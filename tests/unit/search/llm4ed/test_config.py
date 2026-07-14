
from __future__ import annotations

import dataclasses
import json

import pytest

from kd.search.llm4ed.config import Llm4edConfig, config_to_json_safe_dict


class TestEdlDefaults:

    def test_temperature_is_edl_08_not_generic_10(self) -> None:


        assert Llm4edConfig().temperature == pytest.approx(0.8)

    def test_max_tokens_default_1024(self) -> None:
        assert Llm4edConfig().max_tokens == 1024

    def test_stop_threshold_default_0995(self) -> None:

        assert Llm4edConfig().stop_threshold == pytest.approx(0.995)

    def test_reward_limit_default_05(self) -> None:

        assert Llm4edConfig().reward_limit == pytest.approx(0.5)

    def test_pool_size_default_5(self) -> None:

        assert Llm4edConfig().pool_size == 5

    def test_init_num_default_20(self) -> None:

        assert Llm4edConfig().init_num == 20

    def test_samples_per_epoch_default_8(self) -> None:


        assert Llm4edConfig().samples_per_epoch == 8

    def test_frozen(self) -> None:
        cfg = Llm4edConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.temperature = 0.1

    def test_overrides_take(self) -> None:
        cfg = Llm4edConfig(temperature=0.3, pool_size=8, reward_limit=0.7)
        assert cfg.temperature == pytest.approx(0.3)
        assert cfg.pool_size == 8
        assert cfg.reward_limit == pytest.approx(0.7)


class TestGuardrails:

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"temperature": -0.1},
            {"temperature": float("nan")},
            {"temperature": float("inf")},
            {"temperature": "hot"},
        ],
    )
    def test_bad_temperature_rejected(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            Llm4edConfig(**kwargs)

    @pytest.mark.parametrize("bad", [0, -1, True, 3.5])
    def test_bad_max_tokens_rejected(self, bad: object) -> None:

        with pytest.raises(ValueError):
            Llm4edConfig(max_tokens=bad)



    @pytest.mark.parametrize("bad", [0.0, -0.5, 1.5, 2.0, True, "x"])
    def test_bad_stop_threshold_rejected(self, bad: object) -> None:

        with pytest.raises(ValueError):
            Llm4edConfig(stop_threshold=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.5, True, "x"])
    def test_bad_reward_limit_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError):
            Llm4edConfig(reward_limit=bad)

    @pytest.mark.parametrize("field", ["pool_size", "init_num", "samples_per_epoch"])
    @pytest.mark.parametrize("bad", [0, -1, True])
    def test_bad_positive_int_fields_rejected(
        self, field: str, bad: object
    ) -> None:
        with pytest.raises(ValueError):
            Llm4edConfig(**{field: bad})

    @pytest.mark.parametrize(
        "field", ["max_llm_calls_per_propose", "max_llm_calls_per_run"]
    )
    @pytest.mark.parametrize("bad", [0, -3, True, "x"])
    def test_bad_call_bounds_rejected(self, field: str, bad: object) -> None:

        with pytest.raises(ValueError):
            Llm4edConfig(**{field: bad})

    @pytest.mark.parametrize("bad", [-1, True])
    def test_bad_seed_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError):
            Llm4edConfig(seed=bad)

    def test_empty_model_rejected(self) -> None:
        with pytest.raises(ValueError):
            Llm4edConfig(model="")

    def test_valid_edl_defaults_construct(self) -> None:

        Llm4edConfig()


class TestNoLoopField:

    def test_no_loop_counter_fields(self) -> None:
        names = {f.name for f in dataclasses.fields(Llm4edConfig)}
        forbidden = {
            "generations",
            "epochs",
            "max_iterations",
            "n_iterations",
            "optimize_epochs",
            "iterations",
            "max_epochs",
        }
        assert names.isdisjoint(forbidden), (
            f"loop-counter field(s) leaked in: {names & forbidden}"
        )

    def test_no_provider_retry_field(self) -> None:

        names = {f.name for f in dataclasses.fields(Llm4edConfig)}
        assert "max_retries_per_call" not in names


class TestConfigProperty:

    def test_algorithm_prefix(self) -> None:
        out = Llm4edConfig().config
        assert out["algorithm"] == "llm4ed"

    def test_json_round_trips(self) -> None:
        out = Llm4edConfig().config
        restored = json.loads(json.dumps(out))
        assert restored["algorithm"] == "llm4ed"
        assert restored["temperature"] == pytest.approx(0.8)
        assert restored["max_tokens"] == 1024

    def test_carries_all_dataclass_fields(self) -> None:
        out = Llm4edConfig().config
        field_names = {f.name for f in dataclasses.fields(Llm4edConfig)}
        assert field_names <= set(out), (
            f"config property dropped fields: {field_names - set(out)}"
        )

    def test_none_base_url_is_json_null(self) -> None:
        out = Llm4edConfig(base_url=None).config
        assert out["base_url"] is None


class TestJsonSafeSerializer:

    def test_returns_json_safe_dict(self) -> None:
        raw = config_to_json_safe_dict(Llm4edConfig())


        json.loads(json.dumps(raw))
        assert "algorithm" not in raw
        assert raw["reward_limit"] == pytest.approx(0.5)
