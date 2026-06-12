
from __future__ import annotations

import math
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as functional
from torch import Tensor

from kd.search.discover.config import DiscoverConfig, PINNConfig
from kd.search.discover.pinn.cycle import PINNCycleRunner
from kd.search.discover.pinn.executor import make_pinn_dataset
from kd.search.discover.pinn.model import PINNModel, PretrainResult
from kd.search.discover.tokens.library import LibraryConfig

_COORD_NAMES = ["x", "t"]
_FIELD_NAMES = ["u"]
_TRAIN_N = 64
_VAL_N = 16
_MODEL_SEED = 0
_DATA_SEED = 1
_VAL_DATA_SEED = 2




_EMPTY_VAL_CONFIG = PINNConfig(number_layer=2, n_hidden=16, pretrain_epoch=80, lr=0.01)


_EARLY_STOP_PRONE_CONFIG = PINNConfig(
    number_layer=2,
    n_hidden=16,
    pretrain_epoch=60,
    lr=0.01,
    early_stop_patience=5,
    early_stop_warmup=10,
)
_ZERO_EPOCH_CONFIG = PINNConfig(number_layer=2, n_hidden=16, pretrain_epoch=0, lr=0.01)
_NAN_LOCK_CONFIG = PINNConfig(number_layer=2, n_hidden=16, pretrain_epoch=20, lr=0.01)


def _make_sine_data(n: int, seed: int) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(n, generator=gen) * 2 - 1
    t = torch.rand(n, generator=gen) * 2 - 1
    u = torch.sin(torch.pi * x) * torch.cos(torch.pi * t)
    return {"x": x, "t": t}, {"u": u}


def _empty_like(data: dict[str, Tensor]) -> dict[str, Tensor]:
    return {k: v[:0] for k, v in data.items()}


def _param_snapshot(model: PINNModel) -> dict[str, Tensor]:
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def _params_moved(before: dict[str, Tensor], model: PINNModel) -> bool:
    return any(
        not torch.allclose(before[name], param.detach())
        for name, param in model.named_parameters()
    )


def _train_mse(
    model: PINNModel,
    coords: dict[str, Tensor],
    targets: dict[str, Tensor],
) -> float:
    model.eval()
    with torch.no_grad():
        predictions = model(**coords)
        return functional.mse_loss(predictions["u"], targets["u"]).item()


def _fresh_model(config: PINNConfig) -> PINNModel:
    torch.manual_seed(_MODEL_SEED)
    return PINNModel(_COORD_NAMES, _FIELD_NAMES, config)





@pytest.mark.unit
def test_empty_val_keeps_trained_weights() -> None:
    model = _fresh_model(_EMPTY_VAL_CONFIG)
    coords, targets = _make_sine_data(_TRAIN_N, seed=_DATA_SEED)
    before = _param_snapshot(model)

    model.pretrain(
        coords,
        targets,
        _empty_like(coords),
        _empty_like(targets),
        _EMPTY_VAL_CONFIG,
    )

    assert _params_moved(before, model), (
        "pretrain with an empty val set returned epoch-0 weights: all "
        f"{_EMPTY_VAL_CONFIG.pretrain_epoch} training epochs were "
        "silently discarded by the best-state rollback"
    )


@pytest.mark.unit
def test_empty_val_training_improves_train_mse_over_initial() -> None:
    model = _fresh_model(_EMPTY_VAL_CONFIG)
    twin = _fresh_model(_ZERO_EPOCH_CONFIG)
    coords, targets = _make_sine_data(_TRAIN_N, seed=_DATA_SEED)

    model.pretrain(
        coords,
        targets,
        _empty_like(coords),
        _empty_like(targets),
        _EMPTY_VAL_CONFIG,
    )
    twin.pretrain(
        coords,
        targets,
        _empty_like(coords),
        _empty_like(targets),
        _ZERO_EPOCH_CONFIG,
    )

    trained_mse = _train_mse(model, coords, targets)
    epoch0_mse = _train_mse(twin, coords, targets)
    assert math.isfinite(trained_mse)
    assert trained_mse < epoch0_mse, (
        f"train MSE after {_EMPTY_VAL_CONFIG.pretrain_epoch} epochs "
        f"({trained_mse}) is not below the epoch-0 baseline "
        f"({epoch0_mse}) — training results were thrown away"
    )


@pytest.mark.unit
@pytest.mark.numerical
def test_empty_val_runs_all_epochs_without_phantom_early_stop() -> None:
    model = _fresh_model(_EARLY_STOP_PRONE_CONFIG)
    coords, targets = _make_sine_data(_TRAIN_N, seed=_DATA_SEED)

    result = model.pretrain(
        coords,
        targets,
        _empty_like(coords),
        _empty_like(targets),
        _EARLY_STOP_PRONE_CONFIG,
    )

    assert result.stopped_early is False, (
        "early stop fired off a NaN validation plateau (no val data)"
    )
    assert result.epochs_run == _EARLY_STOP_PRONE_CONFIG.pretrain_epoch


@pytest.mark.unit
@pytest.mark.numerical
def test_empty_val_reports_nan_val_loss() -> None:
    model = _fresh_model(_NAN_LOCK_CONFIG)
    coords, targets = _make_sine_data(_TRAIN_N, seed=_DATA_SEED)

    result = model.pretrain(
        coords,
        targets,
        _empty_like(coords),
        _empty_like(targets),
        _NAN_LOCK_CONFIG,
    )

    assert isinstance(result, PretrainResult)
    assert math.isnan(result.val_loss), (
        f"val_loss must remain NaN for an empty val set, got {result.val_loss!r}"
    )
    assert math.isfinite(result.train_loss)





@pytest.mark.unit
def test_nonempty_val_control_moves_weights_and_restores_best() -> None:
    model = _fresh_model(_EMPTY_VAL_CONFIG)
    coords, targets = _make_sine_data(_TRAIN_N, seed=_DATA_SEED)
    val_coords, val_targets = _make_sine_data(_VAL_N, seed=_VAL_DATA_SEED)
    before = _param_snapshot(model)

    result = model.pretrain(coords, targets, val_coords, val_targets, _EMPTY_VAL_CONFIG)

    assert _params_moved(before, model), (
        "control config failed to move weights — the empty-val tests "
        "in this module would be vacuous; fix the config, not the SUT"
    )
    assert math.isfinite(result.val_loss)
    model.eval()
    with torch.no_grad():
        current_val = functional.mse_loss(
            model(**val_coords)["u"], val_targets["u"]
        ).item()
    torch.testing.assert_close(current_val, result.val_loss, rtol=1e-5, atol=1e-8)





class _StubComponent:
    pass


def _build_min_runner(
    n_obs: int, pinn_config: PINNConfig
) -> tuple[PINNCycleRunner, PINNModel, dict[str, Tensor]]:
    config = DiscoverConfig(
        n_iterations=1,
        batch_size=2,
        max_length=10,
        library=LibraryConfig(
            operators=["add"], state_vars=["u"], coord_vars=["x", "t"]
        ),
        pinn=pinn_config,
    )
    model = _fresh_model(pinn_config)
    gen = torch.Generator().manual_seed(3)
    obs_coords = {
        "x": torch.rand(n_obs, generator=gen),
        "t": torch.rand(n_obs, generator=gen),
    }
    obs_targets = {"u": torch.rand(n_obs, generator=gen)}
    runner = PINNCycleRunner(
        engine=cast(Any, _StubComponent()),
        pinn_model=model,
        pinn_executor=cast(Any, _StubComponent()),
        initial_evaluator=cast(Any, _StubComponent()),
        observation_coords=obs_coords,
        observation_targets=obs_targets,
        colloc_coords={"x": torch.rand(8), "t": torch.rand(8)},
        dataset_metadata=make_pinn_dataset(
            ["x", "t"], ["u"], lhs_field="u", lhs_axis="t"
        ),
        config=config,
    )
    return runner, model, obs_coords


@pytest.mark.unit
def test_cycle_pretrain_int_truncation_keeps_trained_weights() -> None:
    assert int(4 * _EARLY_STOP_PRONE_CONFIG.pretrain_val_ratio) == 0
    runner, model, _ = _build_min_runner(n_obs=4, pinn_config=_EARLY_STOP_PRONE_CONFIG)
    before = _param_snapshot(model)

    result = runner._pretrain()

    assert _params_moved(before, model), (
        "cycle _pretrain with n_obs=4 (val split truncates to 0 rows) "
        "rolled the model back to its epoch-0 weights"
    )
    assert result.stopped_early is False


@pytest.mark.unit
def test_cycle_pretrain_int_truncation_passes_empty_val(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kd.search.discover.pinn.model as model_module

    captured: dict[str, int] = {}

    def _capturing_pretrain(
        self_model: Any,
        coords: dict[str, Tensor],
        targets: dict[str, Tensor],
        val_coords: dict[str, Tensor],
        val_targets: dict[str, Tensor],
        config: PINNConfig,
    ) -> PretrainResult:
        captured["train_rows"] = int(coords["x"].shape[0])
        captured["val_rows"] = int(val_coords["x"].shape[0])
        captured["val_target_rows"] = int(val_targets["u"].shape[0])
        return PretrainResult(
            train_loss=0.0,
            val_loss=float("nan"),
            epochs_run=1,
            stopped_early=False,
        )

    monkeypatch.setattr(model_module.PINNModel, "pretrain", _capturing_pretrain)
    runner, _, _ = _build_min_runner(n_obs=4, pinn_config=_EARLY_STOP_PRONE_CONFIG)
    runner._pretrain()

    assert captured["val_rows"] == 0
    assert captured["val_target_rows"] == 0
    assert captured["train_rows"] == 4
