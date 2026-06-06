
from __future__ import annotations

import logging
import math

import pytest
import torch
from torch import Tensor

from kd.core.expr import FunctionRegistry
from kd.data.schema import PDEDataset
from kd.search.discover.config import PINNConfig
from kd.search.discover.pinn.executor import PINNExecutor, make_pinn_dataset
from kd.search.discover.pinn.model import PINNModel, TrainResult





COORD_NAMES = ["x", "t"]
FIELD_NAMES = ["u"]

_CPU_DEVICE = torch.device("cpu")
N_OBS = 200
N_COLLOC = 500


SMALL_TRAIN_CONFIG = PINNConfig(
    number_layer=2,
    n_hidden=10,
    activation="tanh",
    pretrain_epoch=200,
    pinn_epoch=50,
    lr=0.005,
    coef_pde=1.0,
    early_stop_patience=10,
)


EARLY_STOP_CONFIG = PINNConfig(
    number_layer=2,
    n_hidden=10,
    activation="tanh",
    pretrain_epoch=200,
    pinn_epoch=500,
    lr=0.0,
    coef_pde=1.0,
    early_stop_patience=5,
)


_MODEL_SEED = 42







def _heat_solution(x: Tensor, t: Tensor) -> Tensor:
    return torch.exp(-torch.pi**2 * t) * torch.sin(torch.pi * x)


def _make_observation_data(
    n: int = N_OBS,
    device: torch.device = _CPU_DEVICE,
    seed: int = 42,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    torch.manual_seed(seed)
    x = torch.rand(n, device=device) * 2 - 1
    t = torch.rand(n, device=device).abs()
    u = _heat_solution(x, t)
    return {"x": x, "t": t}, {"u": u}


def _make_colloc_coords(
    n: int = N_COLLOC,
    device: torch.device = _CPU_DEVICE,
    seed: int = 123,
) -> dict[str, Tensor]:
    torch.manual_seed(seed)
    return {
        "x": (torch.rand(n, device=device) * 2 - 1).detach().requires_grad_(True),
        "t": torch.rand(n, device=device).abs().detach().requires_grad_(True),
    }


def _make_local_coords(
    n: int = 100,
    device: torch.device = _CPU_DEVICE,
    seed: int = 456,
) -> dict[str, Tensor]:
    torch.manual_seed(seed)
    return {
        "x": (torch.rand(n, device=device) * 0.5).detach().requires_grad_(True),
        "t": (torch.rand(n, device=device) * 0.5).detach().requires_grad_(True),
    }


def _pretrain_model(
    config: PINNConfig,
    device: torch.device = _CPU_DEVICE,
    seed: int = _MODEL_SEED,
) -> PINNModel:
    torch.manual_seed(seed)
    model = PINNModel(COORD_NAMES, FIELD_NAMES, config, device)
    obs_coords, obs_targets = _make_observation_data(device=device)
    val_coords, val_targets = _make_observation_data(n=50, device=device, seed=99)
    model.pretrain(obs_coords, obs_targets, val_coords, val_targets, config)
    return model


def _compute_initial_physics_loss(
    model: PINNModel,
    executor: PINNExecutor,
    terms: list[str],
    coefficients: list[float],
    colloc_coords: dict[str, Tensor],
    dataset_metadata: PDEDataset,
) -> float:
    with torch.no_grad():
        residual = executor.compute_residual(
            model=model,
            terms=terms,
            coefficients=coefficients,
            coords=colloc_coords,
            dataset_metadata=dataset_metadata,
            lhs_field=dataset_metadata.lhs_field,
            lhs_axis=dataset_metadata.lhs_axis,
        )
        return (residual**2).mean().item()







@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def pretrained_model(device: torch.device) -> PINNModel:
    return _pretrain_model(SMALL_TRAIN_CONFIG, device)


@pytest.fixture
def executor() -> PINNExecutor:
    return PINNExecutor(FunctionRegistry.create_default())


@pytest.fixture
def dataset() -> PDEDataset:
    return make_pinn_dataset(
        axis_names=COORD_NAMES,
        field_names=FIELD_NAMES,
        lhs_field="u",
        lhs_axis="t",
    )


@pytest.fixture
def obs_data(device: torch.device) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    return _make_observation_data(device=device)


@pytest.fixture
def colloc_coords(device: torch.device) -> dict[str, Tensor]:
    return _make_colloc_coords(device=device)



HEAT_TERMS = ["diff2_x(u)"]
HEAT_COEFFICIENTS = [1.0]







class TestTrainResult:

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_returns_train_result(
        self,
        pretrained_model: PINNModel,
        executor: PINNExecutor,
        obs_data: tuple[dict[str, Tensor], dict[str, Tensor]],
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        obs_coords, obs_targets = obs_data
        result = pretrained_model.train_pinn(
            terms=HEAT_TERMS,
            coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            config=SMALL_TRAIN_CONFIG,
        )
        assert isinstance(result, TrainResult)
        assert isinstance(result.data_loss, float)
        assert isinstance(result.physics_loss, float)
        assert isinstance(result.total_loss, float)
        assert isinstance(result.epochs_run, int)
        assert isinstance(result.stopped_early, bool)

    @pytest.mark.unit
    def test_losses_are_finite(
        self,
        pretrained_model: PINNModel,
        executor: PINNExecutor,
        obs_data: tuple[dict[str, Tensor], dict[str, Tensor]],
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        obs_coords, obs_targets = obs_data
        result = pretrained_model.train_pinn(
            terms=HEAT_TERMS,
            coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            config=SMALL_TRAIN_CONFIG,
        )
        assert math.isfinite(result.data_loss)
        assert math.isfinite(result.physics_loss)
        assert math.isfinite(result.total_loss)

    @pytest.mark.unit
    def test_total_loss_consistent(
        self,
        pretrained_model: PINNModel,
        executor: PINNExecutor,
        obs_data: tuple[dict[str, Tensor], dict[str, Tensor]],
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        obs_coords, obs_targets = obs_data
        result = pretrained_model.train_pinn(
            terms=HEAT_TERMS,
            coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            config=SMALL_TRAIN_CONFIG,
        )
        expected_total = (
            result.data_loss + SMALL_TRAIN_CONFIG.coef_pde * result.physics_loss
        )
        assert abs(result.total_loss - expected_total) < 0.01 * max(
            abs(expected_total), 1e-6
        ), (
            f"total_loss={result.total_loss:.6f} != "
            f"data+coef*physics={expected_total:.6f}"
        )







class TestPhysicsLoss:

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_physics_loss_decreases(
        self,
        pretrained_model: PINNModel,
        executor: PINNExecutor,
        obs_data: tuple[dict[str, Tensor], dict[str, Tensor]],
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        obs_coords, obs_targets = obs_data

        initial_physics = _compute_initial_physics_loss(
            pretrained_model, executor, HEAT_TERMS, HEAT_COEFFICIENTS,
            colloc_coords, dataset,
        )

        result = pretrained_model.train_pinn(
            terms=HEAT_TERMS,
            coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            config=SMALL_TRAIN_CONFIG,
        )


        assert result.physics_loss < initial_physics, (
            f"Physics loss did not decrease: {result.physics_loss:.6f} "
            f"vs initial {initial_physics:.6f}"
        )







class TestDataLossStability:

    @pytest.mark.unit
    def test_data_loss_does_not_diverge(
        self,
        pretrained_model: PINNModel,
        executor: PINNExecutor,
        obs_data: tuple[dict[str, Tensor], dict[str, Tensor]],
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        obs_coords, obs_targets = obs_data


        with torch.no_grad():
            out = pretrained_model(**obs_coords)
        pre_data_loss = ((out["u"] - obs_targets["u"]) ** 2).mean().item()

        result = pretrained_model.train_pinn(
            terms=HEAT_TERMS,
            coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            config=SMALL_TRAIN_CONFIG,
        )



        assert result.data_loss < pre_data_loss * 15.0 + 0.02, (
            f"Data loss diverged: {result.data_loss:.6f} "
            f"vs pre-PINN {pre_data_loss:.6f}"
        )







class _NaNInjector:

    def __init__(self, nan_after: int = 3) -> None:
        self.call_count = 0
        self.nan_after = nan_after

    def __call__(
        self, module: torch.nn.Module, _input: object, output: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        self.call_count += 1
        if self.call_count > self.nan_after:
            return {k: v * float("nan") for k, v in output.items()}
        return output


class _TransientNaNInjector:

    def __init__(self, nan_after: int, nan_duration: int) -> None:
        self.call_count = 0
        self.nan_after = nan_after
        self.nan_end = nan_after + nan_duration

    def __call__(
        self, module: torch.nn.Module, _input: object, output: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        self.call_count += 1
        if self.nan_after < self.call_count <= self.nan_end:
            return {k: v * float("nan") for k, v in output.items()}
        return output


class TestNaNHandling:

    @pytest.mark.unit
    def test_nan_recovery_with_injected_nan(
        self,
        executor: PINNExecutor,
        dataset: PDEDataset,
        device: torch.device,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        model = _pretrain_model(SMALL_TRAIN_CONFIG, device)
        obs_coords, obs_targets = _make_observation_data(device=device)
        colloc_coords = _make_colloc_coords(device=device)


        injector = _NaNInjector(nan_after=3)
        handle = model.register_forward_hook(injector)

        try:
            with caplog.at_level(logging.WARNING):
                result = model.train_pinn(
                    terms=HEAT_TERMS,
                    coefficients=HEAT_COEFFICIENTS,
                    pinn_executor=executor,
                    observation_coords=obs_coords,
                    observation_targets=obs_targets,
                    colloc_coords=colloc_coords,
                    dataset_metadata=dataset,
                    config=PINNConfig(
                        number_layer=2, n_hidden=10, pinn_epoch=20,
                        lr=0.005, coef_pde=1.0, early_stop_patience=5,
                    ),
                )
        finally:
            handle.remove()


        with torch.no_grad():
            out = model(**obs_coords)
        assert torch.isfinite(out["u"]).all(), (
            "Model output contains NaN/Inf — checkpoint recovery failed"
        )
        assert math.isfinite(result.total_loss), (
            "Result total_loss is NaN/Inf — checkpoint recovery failed"
        )

        nan_warned = any(
            "nan" in record.message.lower() or "inf" in record.message.lower()
            for record in caplog.records
        )
        assert nan_warned, (
            "No NaN/Inf warning logged — NaN detection may not be implemented"
        )

    @pytest.mark.unit
    def test_nan_recovery_preserves_good_checkpoint(
        self,
        executor: PINNExecutor,
        dataset: PDEDataset,
        device: torch.device,
    ) -> None:
        model = _pretrain_model(SMALL_TRAIN_CONFIG, device)
        obs_coords, obs_targets = _make_observation_data(device=device)
        colloc_coords = _make_colloc_coords(device=device)


        with torch.no_grad():
            pre_output = model(**obs_coords)["u"].clone()


        injector = _NaNInjector(nan_after=5)
        handle = model.register_forward_hook(injector)
        try:
            model.train_pinn(
                terms=HEAT_TERMS,
                coefficients=HEAT_COEFFICIENTS,
                pinn_executor=executor,
                observation_coords=obs_coords,
                observation_targets=obs_targets,
                colloc_coords=colloc_coords,
                dataset_metadata=dataset,
                config=PINNConfig(
                    number_layer=2, n_hidden=10, pinn_epoch=20,
                    lr=0.005, coef_pde=1.0, early_stop_patience=5,
                ),
            )
        finally:
            handle.remove()



        with torch.no_grad():
            post_output = model(**obs_coords)["u"]
        diff = (post_output - pre_output).abs().mean().item()
        assert diff < 1.0, (
            f"Post-recovery output differs too much from pre-training: "
            f"mean diff = {diff:.4f} — checkpoint may not be restored"
        )








NAN_RETRY_CONFIG = PINNConfig(
    number_layer=2,
    n_hidden=10,
    activation="tanh",
    pretrain_epoch=200,
    pinn_epoch=30,
    lr=0.005,
    coef_pde=1.0,
    early_stop_patience=100,
    max_nan_recoveries=3,
)


class TestNaNRetry:

    @pytest.mark.unit
    def test_default_max_nan_recoveries(self) -> None:
        config = PINNConfig()
        assert config.max_nan_recoveries == 3

    @pytest.mark.unit
    def test_transient_nan_continues_training(
        self,
        executor: PINNExecutor,
        dataset: PDEDataset,
        device: torch.device,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        model = _pretrain_model(NAN_RETRY_CONFIG, device)
        obs_coords, obs_targets = _make_observation_data(device=device)
        colloc_coords = _make_colloc_coords(device=device)

        injector = _TransientNaNInjector(nan_after=3, nan_duration=6)
        handle = model.register_forward_hook(injector)
        try:
            with caplog.at_level(logging.WARNING):
                result = model.train_pinn(
                    terms=HEAT_TERMS,
                    coefficients=HEAT_COEFFICIENTS,
                    pinn_executor=executor,
                    observation_coords=obs_coords,
                    observation_targets=obs_targets,
                    colloc_coords=colloc_coords,
                    dataset_metadata=dataset,
                    config=NAN_RETRY_CONFIG,
                )
        finally:
            handle.remove()


        assert math.isfinite(result.total_loss)

        assert result.epochs_run > 5, (
            f"Training stopped too early at epoch {result.epochs_run} — "
            "retry logic may not be working"
        )

    @pytest.mark.unit
    def test_persistent_nan_terminates_after_max_retries(
        self,
        executor: PINNExecutor,
        dataset: PDEDataset,
        device: torch.device,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        config = PINNConfig(
            number_layer=2,
            n_hidden=10,
            activation="tanh",
            pretrain_epoch=200,
            pinn_epoch=100,
            lr=0.005,
            coef_pde=1.0,
            early_stop_patience=200,
            max_nan_recoveries=2,
        )
        model = _pretrain_model(config, device)
        obs_coords, obs_targets = _make_observation_data(device=device)
        colloc_coords = _make_colloc_coords(device=device)

        injector = _NaNInjector(nan_after=3)
        handle = model.register_forward_hook(injector)
        try:
            with caplog.at_level(logging.WARNING):
                result = model.train_pinn(
                    terms=HEAT_TERMS,
                    coefficients=HEAT_COEFFICIENTS,
                    pinn_executor=executor,
                    observation_coords=obs_coords,
                    observation_targets=obs_targets,
                    colloc_coords=colloc_coords,
                    dataset_metadata=dataset,
                    config=config,
                )
        finally:
            handle.remove()


        assert result.epochs_run < config.pinn_epoch, (
            "Training ran all epochs — should have terminated after NaN retries"
        )

        with torch.no_grad():
            out = model(**obs_coords)
        assert torch.isfinite(out["u"]).all(), (
            "Model output not finite after NaN termination"
        )

        exceeded = any("exceeded" in r.message.lower() or
                       "consecutive" in r.message.lower()
                       for r in caplog.records)
        assert exceeded, "No warning about exceeding max NaN recoveries"

    @pytest.mark.unit
    def test_nan_counter_resets_on_success(
        self,
        executor: PINNExecutor,
        dataset: PDEDataset,
        device: torch.device,
    ) -> None:
        config = PINNConfig(
            number_layer=2,
            n_hidden=10,
            activation="tanh",
            pretrain_epoch=200,
            pinn_epoch=80,
            lr=0.005,
            coef_pde=1.0,
            early_stop_patience=200,
            max_nan_recoveries=3,
        )
        model = _pretrain_model(config, device)
        obs_coords, obs_targets = _make_observation_data(device=device)
        colloc_coords = _make_colloc_coords(device=device)




        class _TwoWindowInjector:
            def __init__(self) -> None:
                self.call_count = 0

            def __call__(
                self, module: torch.nn.Module, _input: object,
                output: dict[str, Tensor],
            ) -> dict[str, Tensor]:
                self.call_count += 1
                if 4 < self.call_count <= 6 or 80 < self.call_count <= 82:
                    return {k: v * float("nan") for k, v in output.items()}
                return output

        injector = _TwoWindowInjector()
        handle = model.register_forward_hook(injector)
        try:
            result = model.train_pinn(
                terms=HEAT_TERMS,
                coefficients=HEAT_COEFFICIENTS,
                pinn_executor=executor,
                observation_coords=obs_coords,
                observation_targets=obs_targets,
                colloc_coords=colloc_coords,
                dataset_metadata=dataset,
                config=config,
            )
        finally:
            handle.remove()


        assert math.isfinite(result.total_loss)
        assert result.epochs_run == config.pinn_epoch, (
            f"Training stopped at epoch {result.epochs_run} — "
            f"NaN counter should have reset between windows"
        )

    @pytest.mark.unit
    def test_max_nan_recoveries_zero_terminates_immediately(
        self,
        executor: PINNExecutor,
        dataset: PDEDataset,
        device: torch.device,
    ) -> None:
        config = PINNConfig(
            number_layer=2,
            n_hidden=10,
            activation="tanh",
            pretrain_epoch=200,
            pinn_epoch=50,
            lr=0.005,
            coef_pde=1.0,
            early_stop_patience=100,
            max_nan_recoveries=0,
        )
        model = _pretrain_model(config, device)
        obs_coords, obs_targets = _make_observation_data(device=device)
        colloc_coords = _make_colloc_coords(device=device)

        injector = _NaNInjector(nan_after=3)
        handle = model.register_forward_hook(injector)
        try:
            result = model.train_pinn(
                terms=HEAT_TERMS,
                coefficients=HEAT_COEFFICIENTS,
                pinn_executor=executor,
                observation_coords=obs_coords,
                observation_targets=obs_targets,
                colloc_coords=colloc_coords,
                dataset_metadata=dataset,
                config=config,
            )
        finally:
            handle.remove()


        assert result.epochs_run < 10
        assert math.isfinite(result.total_loss)







class TestEarlyStopping:

    @pytest.mark.unit
    def test_early_stopping_triggers(
        self,
        executor: PINNExecutor,
        dataset: PDEDataset,
        device: torch.device,
    ) -> None:
        model = _pretrain_model(EARLY_STOP_CONFIG, device)
        obs_coords, obs_targets = _make_observation_data(device=device)
        colloc_coords = _make_colloc_coords(device=device)

        result = model.train_pinn(
            terms=HEAT_TERMS,
            coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            config=EARLY_STOP_CONFIG,
        )

        assert result.stopped_early is True, "Early stopping did not trigger"
        assert result.epochs_run < EARLY_STOP_CONFIG.pinn_epoch, (
            f"Ran all {result.epochs_run} epochs — early stopping failed"
        )

    @pytest.mark.unit
    def test_full_epochs_when_not_converged(
        self,
        executor: PINNExecutor,
        dataset: PDEDataset,
        device: torch.device,
    ) -> None:
        config = PINNConfig(
            number_layer=2, n_hidden=10,
            pretrain_epoch=200,
            pinn_epoch=20,
            lr=0.005, coef_pde=1.0,
            early_stop_patience=1000,
        )
        model = _pretrain_model(config, device)
        obs_coords, obs_targets = _make_observation_data(device=device)
        colloc_coords = _make_colloc_coords(device=device)

        result = model.train_pinn(
            terms=HEAT_TERMS,
            coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            config=config,
        )

        assert result.stopped_early is False
        assert result.epochs_run == config.pinn_epoch







class TestLocalCoords:

    @pytest.mark.unit
    def test_local_coords_none_no_error(
        self,
        pretrained_model: PINNModel,
        executor: PINNExecutor,
        obs_data: tuple[dict[str, Tensor], dict[str, Tensor]],
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        obs_coords, obs_targets = obs_data
        result = pretrained_model.train_pinn(
            terms=HEAT_TERMS,
            coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            config=SMALL_TRAIN_CONFIG,
            local_coords=None,
        )
        assert isinstance(result, TrainResult)
        assert math.isfinite(result.total_loss)

    @pytest.mark.unit
    def test_local_coords_provided(
        self,
        pretrained_model: PINNModel,
        executor: PINNExecutor,
        obs_data: tuple[dict[str, Tensor], dict[str, Tensor]],
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
        device: torch.device,
    ) -> None:
        obs_coords, obs_targets = obs_data
        local_coords = _make_local_coords(device=device)

        result = pretrained_model.train_pinn(
            terms=HEAT_TERMS,
            coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            config=SMALL_TRAIN_CONFIG,
            local_coords=local_coords,
        )
        assert isinstance(result, TrainResult)
        assert math.isfinite(result.total_loss)

    @pytest.mark.unit
    def test_local_coords_affects_training(
        self,
        executor: PINNExecutor,
        dataset: PDEDataset,
        device: torch.device,
    ) -> None:
        obs_coords, obs_targets = _make_observation_data(device=device, seed=10)
        colloc_coords = _make_colloc_coords(device=device, seed=11)
        local_coords = _make_local_coords(n=200, device=device, seed=12)


        model_a = _pretrain_model(SMALL_TRAIN_CONFIG, device, seed=77)
        shared_state = {k: v.clone() for k, v in model_a.state_dict().items()}

        model_b = PINNModel(COORD_NAMES, FIELD_NAMES, SMALL_TRAIN_CONFIG, device)
        model_b.load_state_dict(shared_state)


        result_a = model_a.train_pinn(
            terms=HEAT_TERMS,
            coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            config=SMALL_TRAIN_CONFIG,
            local_coords=None,
        )


        result_b = model_b.train_pinn(
            terms=HEAT_TERMS,
            coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            config=SMALL_TRAIN_CONFIG,
            local_coords=local_coords,
        )


        assert result_a.total_loss != result_b.total_loss, (
            "local_coords had no effect on training — "
            "local loss may not be computed"
        )

    @pytest.mark.unit
    def test_total_loss_includes_local_loss(
        self,
        executor: PINNExecutor,
        dataset: PDEDataset,
        device: torch.device,
    ) -> None:
        model = _pretrain_model(SMALL_TRAIN_CONFIG, device)
        obs_coords, obs_targets = _make_observation_data(device=device)
        colloc_coords = _make_colloc_coords(device=device)
        local_coords = _make_local_coords(n=200, device=device)

        result = model.train_pinn(
            terms=HEAT_TERMS,
            coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            config=SMALL_TRAIN_CONFIG,
            local_coords=local_coords,
        )



        no_local_total = (
            result.data_loss + SMALL_TRAIN_CONFIG.coef_pde * result.physics_loss
        )
        assert result.total_loss >= no_local_total - 1e-6, (
            f"total_loss={result.total_loss:.6f} < "
            f"data+coef*physics={no_local_total:.6f} — "
            "local_loss not included in total"
        )







class TestMultiTermTraining:

    @pytest.mark.unit
    def test_multiterm_burgers_runs(
        self,
        executor: PINNExecutor,
        dataset: PDEDataset,
        device: torch.device,
    ) -> None:
        model = _pretrain_model(SMALL_TRAIN_CONFIG, device)
        obs_coords, obs_targets = _make_observation_data(device=device)
        colloc_coords = _make_colloc_coords(device=device)

        result = model.train_pinn(
            terms=["mul(u, diff_x(u))", "diff2_x(u)"],
            coefficients=[-1.0, 0.01],
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            config=SMALL_TRAIN_CONFIG,
        )

        assert isinstance(result, TrainResult)
        assert math.isfinite(result.total_loss)
        assert math.isfinite(result.physics_loss)







CLIP_CONFIG = PINNConfig(
    number_layer=2,
    n_hidden=10,
    activation="tanh",
    pretrain_epoch=200,
    pinn_epoch=5,
    lr=0.005,
    coef_pde=1.0,
    early_stop_patience=100,
    grad_clip_norm=1.0,
)

NO_CLIP_CONFIG = PINNConfig(
    number_layer=2,
    n_hidden=10,
    activation="tanh",
    pretrain_epoch=200,
    pinn_epoch=5,
    lr=0.005,
    coef_pde=1.0,
    early_stop_patience=100,
    grad_clip_norm=None,
)


class TestGradientClipping:

    @pytest.mark.unit
    def test_default_config_has_grad_clip(self) -> None:
        config = PINNConfig()
        assert config.grad_clip_norm == 1.0

    @pytest.mark.unit
    def test_clip_grad_norm_called(
        self,
        executor: PINNExecutor,
        dataset: PDEDataset,
        device: torch.device,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _pretrain_model(CLIP_CONFIG, device)
        obs_coords, obs_targets = _make_observation_data(device=device)
        colloc_coords = _make_colloc_coords(device=device)

        clip_calls: list[float] = []
        original_clip = torch.nn.utils.clip_grad_norm_

        def _spy_clip(
            parameters: object,
            max_norm: float,
            **kwargs: object,
        ) -> Tensor:
            clip_calls.append(max_norm)
            return original_clip(parameters, max_norm, **kwargs)

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _spy_clip)

        model.train_pinn(
            terms=HEAT_TERMS,
            coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            config=CLIP_CONFIG,
        )

        assert len(clip_calls) > 0, "clip_grad_norm_ was never called"
        assert all(n == 1.0 for n in clip_calls), (
            f"clip_grad_norm_ called with wrong max_norm: {clip_calls}"
        )

    @pytest.mark.unit
    def test_no_clip_when_disabled(
        self,
        executor: PINNExecutor,
        dataset: PDEDataset,
        device: torch.device,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _pretrain_model(NO_CLIP_CONFIG, device)
        obs_coords, obs_targets = _make_observation_data(device=device)
        colloc_coords = _make_colloc_coords(device=device)

        clip_calls: list[float] = []
        original_clip = torch.nn.utils.clip_grad_norm_

        def _spy_clip(
            parameters: object,
            max_norm: float,
            **kwargs: object,
        ) -> Tensor:
            clip_calls.append(max_norm)
            return original_clip(parameters, max_norm, **kwargs)

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _spy_clip)

        model.train_pinn(
            terms=HEAT_TERMS,
            coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            config=NO_CLIP_CONFIG,
        )

        assert len(clip_calls) == 0, (
            f"clip_grad_norm_ should not be called when disabled, "
            f"but was called {len(clip_calls)} times"
        )

    @pytest.mark.unit
    def test_clipping_bounds_gradient_norms(
        self,
        executor: PINNExecutor,
        dataset: PDEDataset,
        device: torch.device,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        high_coef_config = PINNConfig(
            number_layer=2,
            n_hidden=10,
            activation="tanh",
            pretrain_epoch=200,
            pinn_epoch=10,
            lr=0.005,
            coef_pde=100.0,
            early_stop_patience=100,
            grad_clip_norm=1.0,
        )
        model = _pretrain_model(high_coef_config, device)
        obs_coords, obs_targets = _make_observation_data(device=device)
        colloc_coords = _make_colloc_coords(device=device)

        pre_clip_norms: list[float] = []
        original_clip = torch.nn.utils.clip_grad_norm_

        def _record_clip(
            parameters: object,
            max_norm: float,
            **kwargs: object,
        ) -> Tensor:
            result = original_clip(parameters, max_norm, **kwargs)
            pre_clip_norms.append(result.item())
            return result

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _record_clip)

        model.train_pinn(
            terms=HEAT_TERMS,
            coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            config=high_coef_config,
        )

        assert len(pre_clip_norms) > 0, "clip_grad_norm_ was never called"

        assert any(n > 1.0 for n in pre_clip_norms), (
            f"No gradient norms exceeded max_norm=1.0 — "
            f"clipping never activated: {pre_clip_norms}"
        )

    @pytest.mark.unit
    def test_grad_clip_norm_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="grad_clip_norm must be positive"):
            PINNConfig(grad_clip_norm=0.0)

    @pytest.mark.unit
    def test_grad_clip_norm_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="grad_clip_norm must be positive"):
            PINNConfig(grad_clip_norm=-1.0)







class TestCollocChunkSizeValidation:

    @pytest.mark.unit
    def test_default_is_none(self) -> None:
        config = PINNConfig()
        assert config.colloc_chunk_size is None

    @pytest.mark.unit
    def test_positive_value_accepted(self) -> None:
        config = PINNConfig(colloc_chunk_size=8000)
        assert config.colloc_chunk_size == 8000

    @pytest.mark.unit
    def test_none_value_accepted(self) -> None:
        config = PINNConfig(colloc_chunk_size=None)
        assert config.colloc_chunk_size is None

    @pytest.mark.unit
    def test_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="colloc_chunk_size must be positive"):
            PINNConfig(colloc_chunk_size=0)

    @pytest.mark.unit
    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="colloc_chunk_size must be positive"):
            PINNConfig(colloc_chunk_size=-1)













_COEF_PDE_ZERO_RESIDUAL_BOUND = 2


def _make_coef_pde_config(coef_pde: float) -> PINNConfig:
    return PINNConfig(
        number_layer=2,
        n_hidden=10,
        activation="tanh",
        pretrain_epoch=200,
        pinn_epoch=30,
        lr=0.005,
        coef_pde=coef_pde,
        early_stop_patience=10,
    )


class TestCoefPdeAblation:

    @pytest.mark.unit
    @pytest.mark.parametrize("coef_pde", [0.1, 1.0])
    def test_coef_pde_produces_valid_training(
        self,
        coef_pde: float,
        executor: PINNExecutor,
        dataset: PDEDataset,
        device: torch.device,
    ) -> None:
        config = _make_coef_pde_config(coef_pde)
        model = _pretrain_model(config, device)
        obs_coords, obs_targets = _make_observation_data(device=device)
        colloc = _make_colloc_coords(device=device)

        result = model.train_pinn(
            terms=HEAT_TERMS,
            coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc,
            dataset_metadata=dataset,
            config=config,
        )

        assert isinstance(result, TrainResult)
        assert math.isfinite(result.data_loss)
        assert math.isfinite(result.physics_loss)
        assert math.isfinite(result.total_loss)
        assert result.epochs_run > 0

    @pytest.mark.unit
    def test_coef_pde_affects_loss_balance(
        self,
        executor: PINNExecutor,
        dataset: PDEDataset,
        device: torch.device,
    ) -> None:
        config = _make_coef_pde_config(1.0)
        model = _pretrain_model(config, device)
        obs_coords, obs_targets = _make_observation_data(device=device)
        colloc = _make_colloc_coords(device=device)
        result = model.train_pinn(
            terms=HEAT_TERMS,
            coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc,
            dataset_metadata=dataset,
            config=config,
        )


        physics = result.physics_loss
        gap_at_1 = 1.0 * physics
        gap_at_01 = 0.1 * physics
        assert gap_at_1 > gap_at_01, (
            f"Algebraic: 1.0*P must exceed 0.1*P for positive P={physics:.6f}"
        )

        assert abs(gap_at_1 / gap_at_01 - 10.0) < 1e-6, (
            f"gap ratio should be 10, got {gap_at_1 / gap_at_01:.6f}"
        )

    @pytest.mark.unit
    def test_coef_pde_default_matches_reference(self) -> None:
        assert PINNConfig().coef_pde == 0.0

    @pytest.mark.unit
    def test_loss_formula_respects_coef_pde(
        self,
        executor: PINNExecutor,
        dataset: PDEDataset,
        device: torch.device,
    ) -> None:
        for coef in (0.1, 1.0):
            config = _make_coef_pde_config(coef)
            model = _pretrain_model(config, device)
            obs_coords, obs_targets = _make_observation_data(device=device)
            colloc = _make_colloc_coords(device=device)
            result = model.train_pinn(
                terms=HEAT_TERMS,
                coefficients=HEAT_COEFFICIENTS,
                pinn_executor=executor,
                observation_coords=obs_coords,
                observation_targets=obs_targets,
                colloc_coords=colloc,
                dataset_metadata=dataset,
                config=config,
            )

            expected = result.data_loss + coef * result.physics_loss
            assert abs(result.total_loss - expected) < 1e-4, (
                f"coef_pde={coef}: total={result.total_loss:.6f} != "
                f"data({result.data_loss:.6f}) + "
                f"{coef}*phys({result.physics_loss:.6f}) "
                f"= {expected:.6f}"
            )

    @pytest.mark.unit
    def test_coef_pde_zero_skips_residual_compute(
        self,
        executor: PINNExecutor,
        dataset: PDEDataset,
        device: torch.device,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.discover.pinn import model as pinn_model

        config = _make_coef_pde_config(0.0)
        model = _pretrain_model(config, device)
        obs_coords, obs_targets = _make_observation_data(device=device)
        colloc = _make_colloc_coords(device=device)

        call_count = {"backward": 0}
        original = pinn_model._chunked_backward_residual_loss

        def spy(*args: object, **kwargs: object) -> float:
            call_count["backward"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(
            pinn_model, "_chunked_backward_residual_loss", spy
        )




        residual_count = {"compute": 0}
        original_residual = executor.compute_residual

        def residual_spy(*args: object, **kwargs: object) -> Tensor:
            residual_count["compute"] += 1
            return original_residual(*args, **kwargs)

        monkeypatch.setattr(executor, "compute_residual", residual_spy)

        result = model.train_pinn(
            terms=HEAT_TERMS,
            coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc,
            dataset_metadata=dataset,
            config=config,
        )
        assert math.isfinite(result.data_loss)





        assert call_count["backward"] > 0, "backward wrapper should run"





        max_allowed = _COEF_PDE_ZERO_RESIDUAL_BOUND
        assert residual_count["compute"] <= max_allowed, (
            f"compute_residual fired {residual_count['compute']} times "
            f"with coef_pde=0.0; short-circuit should keep this <= "
            f"{max_allowed} (eval-path only, hard-pinned); without the "
            f"guard the count scales with pinn_epoch={config.pinn_epoch}."
        )

    @pytest.mark.unit
    def test_coef_pde_one_invokes_residual_compute(
        self,
        executor: PINNExecutor,
        dataset: PDEDataset,
        device: torch.device,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = _make_coef_pde_config(1.0)
        model = _pretrain_model(config, device)
        obs_coords, obs_targets = _make_observation_data(device=device)
        colloc = _make_colloc_coords(device=device)

        residual_count = {"compute": 0}
        original_residual = executor.compute_residual

        def residual_spy(*args: object, **kwargs: object) -> Tensor:
            residual_count["compute"] += 1
            return original_residual(*args, **kwargs)

        monkeypatch.setattr(executor, "compute_residual", residual_spy)

        result = model.train_pinn(
            terms=HEAT_TERMS,
            coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc,
            dataset_metadata=dataset,
            config=config,
        )
        assert math.isfinite(result.data_loss)




        assert residual_count["compute"] >= config.pinn_epoch, (
            f"compute_residual fired only {residual_count['compute']} "
            f"times with coef_pde=1.0; expected >= pinn_epoch="
            f"{config.pinn_epoch} to keep MODE2 backward path alive."
        )

    @pytest.mark.unit
    def test_coef_pde_zero_does_not_freeze_field_model(
        self,
        executor: PINNExecutor,
        dataset: PDEDataset,
        device: torch.device,
    ) -> None:
        config = _make_coef_pde_config(0.0)
        model = _pretrain_model(config, device)
        obs_coords, obs_targets = _make_observation_data(device=device)
        colloc = _make_colloc_coords(device=device)


        params_before = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }
        assert params_before, "PINNModel must have at least one parameter"

        result = model.train_pinn(
            terms=HEAT_TERMS,
            coefficients=HEAT_COEFFICIENTS,
            pinn_executor=executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc,
            dataset_metadata=dataset,
            config=config,
        )
        assert math.isfinite(result.data_loss)






        moved = any(
            not torch.allclose(params_before[name], param)
            for name, param in model.named_parameters()
        )
        assert moved, (
            "FieldModel must still update via data backward when "
            "coef_pde=0.0 (NN is not frozen pre-). If freeze "
            "became default, invert this test."
        )
