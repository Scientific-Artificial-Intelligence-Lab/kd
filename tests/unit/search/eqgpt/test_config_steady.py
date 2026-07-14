
from __future__ import annotations

import json

import pytest

from kd.data.schema import DataTopology
from kd.search.eqgpt.config import EqGPTConfig, config_to_json_safe_dict
from kd.search.eqgpt.plugin import EqGPTPlugin

pytestmark = pytest.mark.unit



_PRESET_NAME = "steady_preset"


_STEADY_SPARSITY_ALPHA = 1.0


def _steady_config(**overrides: object) -> EqGPTConfig:





    steady_preset = getattr(EqGPTConfig, _PRESET_NAME)
    return steady_preset("smile", **overrides)







class TestSteadyPreset:
    def test_builds_valid_config(self) -> None:
        cfg = _steady_config()
        assert isinstance(cfg, EqGPTConfig)

    def test_free_pivot_start_words(self) -> None:


        cfg = _steady_config()
        assert cfg.start_words == ("S",)

    def test_pins_sparsity_alpha_one(self) -> None:
        cfg = _steady_config()
        assert cfg.sparsity_alpha == pytest.approx(_STEADY_SPARSITY_ALPHA)

    def test_variables_derived_at_prepare(self) -> None:

        cfg = _steady_config()
        assert cfg.variables is None

    def test_is_not_wave_multicase(self) -> None:
        cfg = _steady_config()
        assert cfg.is_wave_multicase is False

    def test_routes_to_steady_lhs_order_zero(self) -> None:
        cfg = _steady_config()
        reqs = EqGPTPlugin(cfg, backend=None).derivative_requirements
        assert reqs.lhs_order == 0
        assert DataTopology.SCATTERED in reqs.supported_topologies







class TestSteadyOverrides:
    def test_overrides_win(self) -> None:
        cfg = _steady_config(seed=7, samples_per_epoch=100)
        assert cfg.seed == 7
        assert cfg.samples_per_epoch == 100

        assert cfg.sparsity_alpha == pytest.approx(_STEADY_SPARSITY_ALPHA)
        assert cfg.start_words == ("S",)

    def test_sparsity_alpha_is_overridable(self) -> None:
        cfg = _steady_config(sparsity_alpha=0.5)
        assert cfg.sparsity_alpha == pytest.approx(0.5)

    def test_validation_still_applies(self) -> None:

        steady_preset = getattr(EqGPTConfig, _PRESET_NAME)
        with pytest.raises(ValueError):
            steady_preset("smile", samples_per_epoch=0)







def test_steady_and_wave_are_mutually_exclusive() -> None:
    steady_preset = getattr(EqGPTConfig, _PRESET_NAME)
    with pytest.raises(ValueError, match=r"steady|wave|case_filter|mutual"):
        steady_preset("smile", case_filter="N")







class TestSteadyJsonSafe:
    def test_round_trips_through_json(self) -> None:
        cfg = _steady_config(seed=3)
        raw = config_to_json_safe_dict(cfg)
        encoded = json.dumps(raw)
        decoded = json.loads(encoded)
        assert decoded["seed"] == 3
        assert decoded["sparsity_alpha"] == pytest.approx(_STEADY_SPARSITY_ALPHA)

        assert list(decoded["start_words"]) == ["S"]
