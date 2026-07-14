
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from kd.data.schema import PDEDataset
from kd.search.eqgpt._steady import SteadyEvaluator, _best_finite_snapshot
from kd.search.eqgpt.config import EqGPTConfig

pytestmark = pytest.mark.unit


class TestBestFiniteSnapshot:

    def test_skips_poisoned_lower_value_snapshot(self) -> None:



        finite = {"w": torch.tensor([1.0, 2.0])}
        poisoned = {"w": torch.tensor([float("nan"), 1.0])}
        assert _best_finite_snapshot([(0.5, finite), (0.1, poisoned)]) is finite

    def test_all_poisoned_returns_none(self) -> None:
        poisoned = {"w": torch.tensor([float("inf")])}
        assert _best_finite_snapshot([(0.1, poisoned)]) is None

    def test_empty_returns_none(self) -> None:
        assert _best_finite_snapshot([]) is None

    def test_picks_min_among_finite(self) -> None:
        higher = {"w": torch.tensor([1.0])}
        lower = {"w": torch.tensor([2.0])}
        assert _best_finite_snapshot([(0.9, higher), (0.2, lower)]) is lower


def _steady_config(**overrides: object) -> EqGPTConfig:
    base: dict[str, object] = {
        "sparsity_alpha": 1.0,
        "steady": True,
        "steady_activation": "sin",
        "start_words": ("S",),
        "steady_train_iters": 5,
    }
    base.update(overrides)
    return EqGPTConfig(**base)


def _scatter(coords: dict[str, torch.Tensor]) -> PDEDataset:
    n = next(iter(coords.values())).numel()
    fields = {"u": torch.rand(n, dtype=torch.float64)}
    return PDEDataset.from_scatter(coords=coords, fields=fields, lhs="", name="t")


class TestSteadyAxisValidation:

    def test_rejects_extra_axis(self) -> None:
        torch.manual_seed(0)
        ds = _scatter(
            {
                "x": torch.rand(8, dtype=torch.float64),
                "y": torch.rand(8, dtype=torch.float64),
                "z": torch.rand(8, dtype=torch.float64),
            }
        )

        with pytest.raises(ValueError, match=r"EXACTLY"):
            SteadyEvaluator.from_components(ds, MagicMock(), _steady_config())

    def test_rejects_missing_axis(self) -> None:
        torch.manual_seed(0)
        ds = _scatter({"x": torch.rand(8, dtype=torch.float64)})
        with pytest.raises(ValueError, match=r"EXACTLY"):
            SteadyEvaluator.from_components(ds, MagicMock(), _steady_config())
