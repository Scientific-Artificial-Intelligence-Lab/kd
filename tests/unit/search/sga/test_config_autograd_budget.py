
from __future__ import annotations

import pytest

from kd.search.sga.config import SGAConfig






class TestAutogradBudgetDefaults:

    @pytest.mark.unit
    def test_patience_default_is_none(self) -> None:
        config = SGAConfig()
        assert config.autograd_train_patience is None, (
            "autograd_train_patience must default to None (v1 fixed-step "
            f"budget); got {config.autograd_train_patience!r}"
        )

    @pytest.mark.unit
    def test_val_ratio_default_is_zero(self) -> None:
        config = SGAConfig()
        assert config.autograd_train_val_ratio == 0.0, (
            "autograd_train_val_ratio must default to 0.0 (full-data train); "
            f"got {config.autograd_train_val_ratio!r}"
        )

    @pytest.mark.unit
    def test_default_combo_is_constructible(self) -> None:
        config = SGAConfig()
        assert config.autograd_train_patience is None
        assert config.autograd_train_val_ratio == 0.0

    @pytest.mark.unit
    def test_explicit_valid_early_stop_combo_constructs(self) -> None:
        config = SGAConfig(
            autograd_train_patience=50,
            autograd_train_val_ratio=0.2,
        )
        assert config.autograd_train_patience == 50
        assert config.autograd_train_val_ratio == pytest.approx(0.2)







class TestAutogradBudgetValidation:

    @pytest.mark.unit
    def test_patience_orphan_without_val_ratio_raises(self) -> None:
        with pytest.raises(ValueError):
            SGAConfig(
                autograd_train_patience=100,
                autograd_train_val_ratio=0.0,
            )

    @pytest.mark.unit
    def test_patience_orphan_uses_default_val_ratio(self) -> None:
        with pytest.raises(ValueError):
            SGAConfig(autograd_train_patience=100)

    @pytest.mark.unit
    @pytest.mark.parametrize("bad_patience", [0, -1, -100])
    def test_patience_below_one_raises(self, bad_patience: int) -> None:
        with pytest.raises(ValueError):
            SGAConfig(
                autograd_train_patience=bad_patience,
                autograd_train_val_ratio=0.2,
            )

    @pytest.mark.unit
    @pytest.mark.parametrize("bad_ratio", [-0.1, -1.0, 1.0, 1.5, 2.0, float("nan")])
    def test_val_ratio_out_of_range_raises(self, bad_ratio: float) -> None:
        with pytest.raises(ValueError):
            SGAConfig(autograd_train_val_ratio=bad_ratio)

    @pytest.mark.unit
    def test_val_ratio_just_below_one_is_valid(self) -> None:
        config = SGAConfig(
            autograd_train_patience=5,
            autograd_train_val_ratio=0.99,
        )
        assert config.autograd_train_val_ratio == pytest.approx(0.99)

    @pytest.mark.unit
    def test_patience_none_with_nonzero_val_ratio_is_valid(self) -> None:
        config = SGAConfig(
            autograd_train_patience=None,
            autograd_train_val_ratio=0.3,
        )
        assert config.autograd_train_patience is None
        assert config.autograd_train_val_ratio == pytest.approx(0.3)
