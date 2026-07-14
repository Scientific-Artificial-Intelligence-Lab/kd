
from __future__ import annotations

import dataclasses

import pytest

from kd.search.eqgpt.config import EqGPTConfig


BURGERS_SPARSITY_ALPHA = 0.02







_PRESET_NAME = "burgers_preset"


def test_burgers_preset_matches_hand_built_default() -> None:
    burgers_preset = getattr(EqGPTConfig, _PRESET_NAME)
    preset = burgers_preset()

    assert isinstance(preset, EqGPTConfig)
    assert preset.sparsity_alpha == pytest.approx(BURGERS_SPARSITY_ALPHA)

    assert preset.seed == 0
    assert preset.samples_per_epoch == 400
    assert preset.top_k == 10

    expected = EqGPTConfig(sparsity_alpha=BURGERS_SPARSITY_ALPHA)
    for f in dataclasses.fields(EqGPTConfig):
        assert getattr(preset, f.name) == getattr(expected, f.name), f.name
    assert preset == expected


def test_burgers_preset_applies_overrides() -> None:
    burgers_preset = getattr(EqGPTConfig, _PRESET_NAME)
    preset = burgers_preset(seed=7, samples_per_epoch=100)

    assert preset.seed == 7
    assert preset.samples_per_epoch == 100

    assert preset.sparsity_alpha == pytest.approx(BURGERS_SPARSITY_ALPHA)
    assert preset.top_k == 10


def test_burgers_preset_sparsity_alpha_is_overridable() -> None:
    burgers_preset = getattr(EqGPTConfig, _PRESET_NAME)
    preset = burgers_preset(sparsity_alpha=0.001)

    assert preset.sparsity_alpha == pytest.approx(0.001)


def test_burgers_preset_validation_still_applies() -> None:
    burgers_preset = getattr(EqGPTConfig, _PRESET_NAME)
    with pytest.raises(ValueError):
        burgers_preset(samples_per_epoch=0)


def test_burgers_preset_returns_frozen_dataclass() -> None:
    burgers_preset = getattr(EqGPTConfig, _PRESET_NAME)
    preset = burgers_preset()

    with pytest.raises(dataclasses.FrozenInstanceError):
        preset.sparsity_alpha = 0.5
