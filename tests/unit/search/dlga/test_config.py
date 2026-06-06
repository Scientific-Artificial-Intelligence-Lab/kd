
from __future__ import annotations

import dataclasses

import pytest

from kd.search.dlga import DLGAConfig, DLGAPlugin


class TestDLGAConfig:
    @pytest.mark.smoke
    def test_defaults_match_stage1_decision(self) -> None:
        config = DLGAConfig()

        assert dataclasses.is_dataclass(config)
        assert config.mode == "constant"
        assert config.library == ["u", "u_x", "u_xx", "u_xxx"]
        assert config.solver == "svd_null_space"
        assert config.lhs_auto_select is True




        assert config.epsilon == pytest.approx(1e-3)
        assert config.pop_size == 400
        assert config.seed == 0
        assert config.auto_upgrade_threshold == pytest.approx(1e-3)


        assert config.surrogate_hidden_sizes == [50, 50, 50, 50, 50]
        assert config.surrogate_activation == "sin"
        assert config.surrogate_max_epochs == 50000

    @pytest.mark.unit
    def test_default_library_is_not_shared(self) -> None:
        first = DLGAConfig()
        second = DLGAConfig()

        first.library.append("x")

        assert second.library == ["u", "u_x", "u_xx", "u_xxx"]

    @pytest.mark.unit
    def test_constant_mode_is_implemented(self) -> None:
        plugin = DLGAPlugin(DLGAConfig(mode="constant"))

        assert plugin.config["mode"] == "constant"

    @pytest.mark.unit
    @pytest.mark.parametrize("mode", ["adaptive", "auto"])
    def test_unimplemented_modes_raise(self, mode: str) -> None:
        with pytest.raises(NotImplementedError, match="mode='constant'"):
            DLGAPlugin(DLGAConfig(mode=mode))

    @pytest.mark.unit
    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            DLGAConfig(mode="paper")

    @pytest.mark.unit
    def test_auto_upgrade_threshold_is_documented_as_reserved(self) -> None:
        fields = {f.name: f for f in dataclasses.fields(DLGAConfig)}
        doc = fields["auto_upgrade_threshold"].metadata.get("doc", "")
        assert "reserved" in doc.lower(), (
            "auto_upgrade_threshold must document that it is reserved for Stage II"
        )
        assert "auto" in doc.lower() or "stage ii" in doc.lower(), (
            "reserved doc should name the mode='auto' / Stage II feature it gates"
        )

    @pytest.mark.unit
    def test_auto_upgrade_threshold_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="auto_upgrade_threshold"):
            DLGAConfig(auto_upgrade_threshold=-1.0)


class TestDLGAPresets:

    @pytest.mark.unit
    def test_kdv_preset_locks_tight_epsilon(self) -> None:
        config = DLGAConfig.kdv_preset()
        assert config.epsilon == pytest.approx(1e-6)

        assert config.pop_size == 400
        assert config.surrogate_max_epochs == 50000

    @pytest.mark.unit
    def test_burgers_preset_epsilon(self) -> None:
        assert DLGAConfig.burgers_preset().epsilon == pytest.approx(1e-3)

    @pytest.mark.unit
    def test_wave_preset_epsilon(self) -> None:
        assert DLGAConfig.wave_preset().epsilon == pytest.approx(1e-3)

    @pytest.mark.unit
    def test_chafee_preset_epsilon(self) -> None:
        assert DLGAConfig.chafee_preset().epsilon == pytest.approx(1e-5)

    @pytest.mark.unit
    def test_preset_accepts_overrides(self) -> None:
        config = DLGAConfig.kdv_preset(surrogate_max_epochs=2000, pop_size=50)
        assert config.epsilon == pytest.approx(1e-6)
        assert config.surrogate_max_epochs == 2000
        assert config.pop_size == 50

    @pytest.mark.unit
    def test_preset_returns_dlgaconfig_instance(self) -> None:
        assert isinstance(DLGAConfig.kdv_preset(), DLGAConfig)

    @pytest.mark.unit
    def test_preset_is_usable_under_facade(self) -> None:
        import warnings

        import kd

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m = kd.Model(algorithm="dlga", config=DLGAConfig.kdv_preset())
        assert m.algorithm == "dlga"
