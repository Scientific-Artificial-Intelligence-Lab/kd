
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from kd.search.discover.config import PINNConfig
from kd.search.discover.pinn.model import _early_stop_warmup

if TYPE_CHECKING:
    from kd.search.discover.pinn.model import PINNModel





class TestEarlyStopWarmup:

    @pytest.mark.unit
    def test_warmup_auto_pretrain(self) -> None:
        config = PINNConfig(early_stop_patience=100, pretrain_epoch=5000)
        assert _early_stop_warmup(config, max_epochs=config.pretrain_epoch) == 1000

    @pytest.mark.unit
    def test_warmup_auto_pinn(self) -> None:
        config = PINNConfig(early_stop_patience=100, pinn_epoch=500)
        assert _early_stop_warmup(config, max_epochs=config.pinn_epoch) == 500

    @pytest.mark.unit
    def test_warmup_default_pinn_is_dormant(self) -> None:
        config = PINNConfig()
        warmup = _early_stop_warmup(config, max_epochs=config.pinn_epoch)
        assert warmup == config.pinn_epoch == 1000

    @pytest.mark.unit
    def test_warmup_auto_small_patience(self) -> None:
        config = PINNConfig(early_stop_patience=3, pinn_epoch=1000)
        assert _early_stop_warmup(config, max_epochs=config.pinn_epoch) == 50

    @pytest.mark.unit
    def test_warmup_explicit(self) -> None:
        config = PINNConfig(
            early_stop_warmup=200,
            early_stop_patience=100,
            pretrain_epoch=5000,
        )
        assert _early_stop_warmup(config, max_epochs=config.pretrain_epoch) == 200

    @pytest.mark.unit
    def test_warmup_explicit_capped(self) -> None:
        config = PINNConfig(early_stop_warmup=9999, pinn_epoch=500)
        assert _early_stop_warmup(config, max_epochs=config.pinn_epoch) == 500

    @pytest.mark.unit
    def test_warmup_zero_disables(self) -> None:
        config = PINNConfig(early_stop_warmup=0, pinn_epoch=1000)
        assert _early_stop_warmup(config, max_epochs=config.pinn_epoch) == 0

    @pytest.mark.unit
    def test_warmup_default_pretrain_backward_compatible(self) -> None:
        config = PINNConfig()
        assert _early_stop_warmup(config, max_epochs=config.pretrain_epoch) == 5000






class TestPINNConfigWarmup:

    @pytest.mark.unit
    def test_config_default_none(self) -> None:
        config = PINNConfig()
        assert config.early_stop_warmup is None

    @pytest.mark.unit
    def test_config_explicit_value(self) -> None:
        config = PINNConfig(early_stop_warmup=300)
        assert config.early_stop_warmup == 300

    @pytest.mark.unit
    def test_config_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="early_stop_warmup"):
            PINNConfig(early_stop_warmup=-1)






class TestPINNTrainingWarmup:

    @pytest.mark.unit
    def test_pinn_es_blocked_during_warmup(self) -> None:
        import torch

        from kd.search.discover.pinn.model import _PINNBestState, _update_pinn_best


        config = PINNConfig(
            early_stop_patience=10,
            early_stop_warmup=100,
            pinn_epoch=500,
        )


        model = torch.nn.Linear(1, 1)
        best = _PINNBestState(
            state_dict=model.state_dict(),
            data_loss=0.1,
            physics_loss=0.1,
            total_loss=0.2,
        )



        data_loss = torch.tensor(0.5)
        physics_loss = torch.tensor(0.5)
        total_loss = torch.tensor(1.0)
        epochs_without_improvement = config.early_stop_patience

        new_best, new_ewi, result = _update_pinn_best(
            cast("PINNModel", model), best, data_loss, physics_loss,
            total_loss,
            epoch=50,
            epochs_without_improvement=epochs_without_improvement,
            config=config,
        )


        assert result is None, "ES should not fire during warmup"
        assert new_ewi == 0, "Counter should reset during warmup"

    @pytest.mark.unit
    def test_pinn_es_fires_after_warmup(self) -> None:
        import torch

        from kd.search.discover.pinn.model import _PINNBestState, _update_pinn_best

        config = PINNConfig(
            early_stop_patience=10,
            early_stop_warmup=100,
            pinn_epoch=500,
        )

        model = torch.nn.Linear(1, 1)
        best = _PINNBestState(
            state_dict=model.state_dict(),
            data_loss=0.1,
            physics_loss=0.1,
            total_loss=0.2,
        )

        data_loss = torch.tensor(0.5)
        physics_loss = torch.tensor(0.5)
        total_loss = torch.tensor(1.0)
        epochs_without_improvement = config.early_stop_patience

        new_best, new_ewi, result = _update_pinn_best(
            cast("PINNModel", model), best, data_loss, physics_loss,
            total_loss,
            epoch=150,
            epochs_without_improvement=epochs_without_improvement,
            config=config,
        )


        assert result is not None, "ES should fire after warmup"

    @pytest.mark.unit
    def test_pinn_es_warmup_zero_fires_immediately(self) -> None:
        import torch

        from kd.search.discover.pinn.model import _PINNBestState, _update_pinn_best

        config = PINNConfig(
            early_stop_patience=10,
            early_stop_warmup=0,
            pinn_epoch=500,
        )

        model = torch.nn.Linear(1, 1)
        best = _PINNBestState(
            state_dict=model.state_dict(),
            data_loss=0.1,
            physics_loss=0.1,
            total_loss=0.2,
        )

        data_loss = torch.tensor(0.5)
        physics_loss = torch.tensor(0.5)
        total_loss = torch.tensor(1.0)

        _, _, result = _update_pinn_best(
            cast("PINNModel", model), best, data_loss, physics_loss,
            total_loss,
            epoch=11,
            epochs_without_improvement=config.early_stop_patience,
            config=config,
        )

        assert result is not None, "warmup=0 should allow immediate ES"
