
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kd.search.eqgpt.config import EqGPTConfig, config_to_json_safe_dict

pytestmark = pytest.mark.unit







class TestWavePreset:
    def test_pins_case_filter_and_sparsity(self) -> None:
        cfg = EqGPTConfig.wave_preset()
        assert cfg.case_filter == "N"
        assert cfg.sparsity_alpha == pytest.approx(0.02)

    def test_activates_multicase_mode(self) -> None:
        cfg = EqGPTConfig.wave_preset()
        assert cfg.is_wave_multicase is True

    def test_dual_grids_default_50_and_100(self) -> None:
        cfg = EqGPTConfig.wave_preset()

        assert cfg.reward_points_per_window == 50
        assert cfg.coeff_points_per_window == 100

    def test_start_words_pin_evolution_lhs(self) -> None:

        cfg = EqGPTConfig.wave_preset()
        assert cfg.start_words == ("S", "ut", "+")

    def test_overrides_win(self) -> None:
        cfg = EqGPTConfig.wave_preset(
            seed=7,
            sparsity_alpha=0.005,
            reward_points_per_window=25,
            coeff_points_per_window=40,
            primary_case="N_G2Tp12A090_broad",
        )
        assert cfg.seed == 7
        assert cfg.sparsity_alpha == pytest.approx(0.005)
        assert cfg.reward_points_per_window == 25
        assert cfg.coeff_points_per_window == 40
        assert cfg.primary_case == "N_G2Tp12A090_broad"

        assert cfg.case_filter == "N"

    def test_case_filter_override_wins(self) -> None:
        cfg = EqGPTConfig.wave_preset(case_filter="L")
        assert cfg.case_filter == "L"
        assert cfg.is_wave_multicase is True







class TestIsWaveMulticase:
    def test_single_case_config_is_not_wave(self) -> None:
        cfg = EqGPTConfig(sparsity_alpha=0.02)
        assert cfg.case_filter is None
        assert cfg.is_wave_multicase is False

    def test_case_filter_set_activates_mode(self) -> None:
        cfg = EqGPTConfig(sparsity_alpha=0.02, case_filter="N")
        assert cfg.is_wave_multicase is True







class TestWaveValidation:
    @pytest.mark.parametrize(
        "field", ["reward_points_per_window", "coeff_points_per_window"]
    )
    @pytest.mark.parametrize("bad", [0, -5])
    def test_non_positive_grid_rejected(self, field: str, bad: int) -> None:
        with pytest.raises(ValueError, match=field):
            EqGPTConfig(sparsity_alpha=0.02, **{field: bad})

    def test_grid_must_be_int_not_bool(self) -> None:

        with pytest.raises(ValueError, match="reward_points_per_window"):
            EqGPTConfig(sparsity_alpha=0.02, reward_points_per_window=True)

    def test_empty_case_filter_rejected(self) -> None:
        with pytest.raises(ValueError, match="case_filter"):
            EqGPTConfig(sparsity_alpha=0.02, case_filter="")

    def test_empty_primary_case_rejected(self) -> None:
        with pytest.raises(ValueError, match="primary_case"):
            EqGPTConfig(sparsity_alpha=0.02, case_filter="N", primary_case="")

    def test_primary_case_requires_case_filter(self) -> None:

        with pytest.raises(ValueError, match="primary_case requires case_filter"):
            EqGPTConfig(sparsity_alpha=0.02, primary_case="N_G2Tp12A080_broad")

    def test_primary_case_with_case_filter_ok(self) -> None:
        cfg = EqGPTConfig(
            sparsity_alpha=0.02,
            case_filter="N",
            primary_case="N_G2Tp12A080_broad",
        )
        assert cfg.primary_case == "N_G2Tp12A080_broad"







class TestJsonSafeDict:
    def test_stringifies_wave_paths(self) -> None:
        cfg = EqGPTConfig.wave_preset(
            wave_pkl_path=Path("/data/WaveBreaking.pkl"),
            v1_asset_dir=Path("/assets/EqGPT_wave_breaking"),
        )
        raw = config_to_json_safe_dict(cfg)
        assert raw["wave_pkl_path"] == "/data/WaveBreaking.pkl"
        assert raw["v1_asset_dir"] == "/assets/EqGPT_wave_breaking"
        assert isinstance(raw["wave_pkl_path"], str)
        assert isinstance(raw["v1_asset_dir"], str)

    def test_none_wave_paths_stay_none(self) -> None:
        cfg = EqGPTConfig.wave_preset()
        raw = config_to_json_safe_dict(cfg)
        assert raw["wave_pkl_path"] is None
        assert raw["v1_asset_dir"] is None

    def test_round_trips_through_json(self) -> None:
        cfg = EqGPTConfig.wave_preset(
            wave_pkl_path=Path("/data/WaveBreaking.pkl"),
            v1_asset_dir=Path("/assets/EqGPT_wave_breaking"),
            primary_case="N_G2Tp12A080_broad",
        )
        raw = config_to_json_safe_dict(cfg)

        encoded = json.dumps(raw)
        decoded = json.loads(encoded)
        assert decoded["case_filter"] == "N"
        assert decoded["reward_points_per_window"] == 50
        assert decoded["coeff_points_per_window"] == 100
        assert decoded["primary_case"] == "N_G2Tp12A080_broad"
