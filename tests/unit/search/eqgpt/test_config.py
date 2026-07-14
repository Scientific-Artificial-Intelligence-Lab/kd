
from __future__ import annotations

import pytest

from kd.search.eqgpt.config import EqGPTConfig


def test_config_defaults() -> None:
    cfg = EqGPTConfig(sparsity_alpha=0.02)
    assert cfg.samples_per_epoch == 400
    assert cfg.top_k == 10
    assert cfg.finetune_lr == pytest.approx(1e-5)
    assert cfg.finetune_steps == 5
    assert cfg.exploration_rate == pytest.approx(0.2)
    assert cfg.max_length == 49
    assert cfg.start_words == ("S", "ut", "+")
    assert cfg.variables is None


def test_config_requires_sparsity_alpha() -> None:
    with pytest.raises(TypeError):
        EqGPTConfig()


def test_config_has_no_loop_field() -> None:
    cfg = EqGPTConfig(sparsity_alpha=0.02)
    for forbidden in (
        "optimize_epochs",
        "generations",
        "n_iterations",
        "max_iterations",
    ):
        assert not hasattr(cfg, forbidden), forbidden


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sparsity_alpha": 0.02, "samples_per_epoch": 0},
        {"sparsity_alpha": 0.02, "top_k": 0},
        {"sparsity_alpha": 0.02, "finetune_steps": -1},
        {"sparsity_alpha": 0.02, "max_length": 0},
        {"sparsity_alpha": 0.02, "seed": -1},
        {"sparsity_alpha": 0.02, "finetune_lr": 0.0},
        {"sparsity_alpha": 0.02, "exploration_rate": 1.5},
        {"sparsity_alpha": 0.02, "variables": ()},
        {"sparsity_alpha": 0.02, "max_length": 3},
        {"sparsity_alpha": 0.02, "seed": True},
        {"sparsity_alpha": 0.02, "finetune_lr": True},
    ],
)
def test_config_rejects_invalid(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        EqGPTConfig(**kwargs)
