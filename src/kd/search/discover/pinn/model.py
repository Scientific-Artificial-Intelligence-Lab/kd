
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

from kd.models import FieldModel
from kd.search.discover.config import PINNConfig
from kd.search.discover.pinn._memory_log import _log_memory

if TYPE_CHECKING:
    from kd.data.schema import PDEDataset
    from kd.search.discover.pinn.executor import PINNExecutor

logger = logging.getLogger(__name__)

_STATE_DICT_KEY = "state_dict"
_EARLY_STOP_MIN_DELTA = 1e-7
_EARLY_STOP_MIN_WARMUP_EPOCHS = 50
_EARLY_STOP_WARMUP_MULTIPLIER = 10
_PINN_EARLY_STOP_MIN_DELTA = 1e-7



















_LARGE_BATCH_AUTO_CHUNK = 16_384
_LARGE_BATCH_AUTO_THRESHOLD = 65_536


_CPU_DEVICE = torch.device("cpu")











_EMPTY_CACHE_EVERY_N_CHUNKS = 16



_EMPTY_CACHE_PRESSURE_THRESHOLD = 0.75


@dataclass(frozen=True, slots=True)
class PretrainResult:

    train_loss: float
    val_loss: float
    epochs_run: int
    stopped_early: bool


@dataclass(frozen=True, slots=True)
class TrainResult:

    data_loss: float
    physics_loss: float
    total_loss: float
    epochs_run: int
    stopped_early: bool


class PINNModel(nn.Module):

    def __init__(
        self,
        coord_names: list[str],
        field_names: list[str],
        config: PINNConfig,
        device: torch.device = _CPU_DEVICE,
    ) -> None:
        super().__init__()
        self.coord_names = list(coord_names)
        self.field_names = list(field_names)
        self.config = config
        self.field_model = FieldModel(
            coord_names=self.coord_names,
            field_names=self.field_names,
            hidden_sizes=_hidden_sizes(config),
            activation=config.activation,
        )
        self.to(device)

    def forward(self, **coords: Tensor) -> dict[str, Tensor]:
        return cast(dict[str, Tensor], self.field_model(**coords))

    def pretrain(
        self,
        coords: dict[str, Tensor],
        targets: dict[str, Tensor],
        val_coords: dict[str, Tensor],
        val_targets: dict[str, Tensor],
        config: PINNConfig,
    ) -> PretrainResult:
        _log_memory("pretrain_start", logger)
        dtype = _infer_dtype(coords, targets)
        device = _module_device(self)
        self.to(device=device, dtype=dtype)
        train_coords = _prepare_data_dict(coords, device, dtype)
        train_targets = _prepare_data_dict(targets, device, dtype)
        eval_coords = _prepare_data_dict(val_coords, device, dtype)
        eval_targets = _prepare_data_dict(val_targets, device, dtype)
        self.field_model.set_normalization(
            _compute_stats(train_coords, self.coord_names),
            _compute_stats(train_targets, self.field_names),
        )
        if config.pretrain_epoch == 0:
            result_zero = _evaluate_result(
                self,
                train_coords,
                train_targets,
                eval_coords,
                eval_targets,
                0,
                False,
            )
            _log_memory("pretrain_end", logger)
            return result_zero

        if not _has_validation_rows(eval_coords, eval_targets):
            return _pretrain_no_val(self, train_coords, train_targets, config)

        optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        best = _BestState.capture(
            self,
            train_coords,
            train_targets,
            eval_coords,
            eval_targets,
        )
        epochs_run, stopped_early = _run_pretrain_loop(
            self,
            optimizer,
            train_coords,
            train_targets,
            eval_coords,
            eval_targets,
            config,
            best,
        )
        self.load_state_dict(best.state_dict)
        _log_memory("pretrain_end", logger)
        return PretrainResult(
            train_loss=best.train_loss,
            val_loss=best.val_loss,
            epochs_run=epochs_run,
            stopped_early=stopped_early,
        )

    def train_pinn(
        self,
        terms: list[str],
        coefficients: list[float],
        pinn_executor: PINNExecutor,
        observation_coords: dict[str, Tensor],
        observation_targets: dict[str, Tensor],
        colloc_coords: dict[str, Tensor],
        dataset_metadata: PDEDataset,
        config: PINNConfig,
        local_coords: dict[str, Tensor] | None = None,
    ) -> TrainResult:
        _log_memory("train_pinn_start", logger)
        dtype = _infer_dtype(observation_coords, observation_targets)
        device = _module_device(self)
        self.to(device=device, dtype=dtype)
        obs_coords = _prepare_data_dict(observation_coords, device, dtype)
        obs_targets = _prepare_data_dict(observation_targets, device, dtype)
        colloc_inputs = _prepare_grad_data_dict(colloc_coords, device, dtype)
        local_inputs = _prepare_optional_grad_data(local_coords, device, dtype)
        optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        best = _capture_pinn_best(
            self,
            pinn_executor,
            terms,
            coefficients,
            obs_coords,
            obs_targets,
            colloc_inputs,
            dataset_metadata,
            config,
            local_inputs,
        )
        if config.pinn_epoch == 0:
            result_zero = _make_train_result(best, 0, False)
            _log_memory("train_pinn_end", logger)
            return result_zero
        result = _run_pinn_training(
            self,
            optimizer,
            pinn_executor,
            terms,
            coefficients,
            obs_coords,
            obs_targets,
            colloc_inputs,
            dataset_metadata,
            config,
            best,
            local_inputs,
        )
        _log_memory("train_pinn_end", logger)
        return result

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save({_STATE_DICT_KEY: self.state_dict()}, Path(path))

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(
            Path(path), map_location=_module_device(self), weights_only=True
        )
        state_dict = checkpoint.get(_STATE_DICT_KEY, checkpoint)
        self.load_state_dict(state_dict)


@dataclass
class _BestState:

    state_dict: dict[str, Tensor]
    train_loss: float
    val_loss: float
    epochs_without_improvement: int = 0

    @classmethod
    def capture(
        cls,
        model: PINNModel,
        train_coords: dict[str, Tensor],
        train_targets: dict[str, Tensor],
        val_coords: dict[str, Tensor],
        val_targets: dict[str, Tensor],
    ) -> _BestState:
        train_loss, val_loss = _evaluate_losses(
            model, train_coords, train_targets, val_coords, val_targets
        )
        return cls(
            state_dict=_clone_state_dict(model.state_dict()),
            train_loss=train_loss,
            val_loss=val_loss,
        )


@dataclass(slots=True)
class _PINNBestState:

    state_dict: dict[str, Tensor]
    data_loss: float
    physics_loss: float
    total_loss: float


def _hidden_sizes(config: PINNConfig) -> list[int]:
    return [config.n_hidden] * config.number_layer


def _module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def _infer_dtype(
    coords: dict[str, Tensor],
    targets: dict[str, Tensor],
) -> torch.dtype:
    for tensors in (coords, targets):
        for value in tensors.values():
            return value.dtype
    return torch.float32


def _prepare_data_dict(
    data: dict[str, Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    return {
        name: value.detach().to(device=device, dtype=dtype)
        for name, value in data.items()
    }


def _compute_stats(
    data: dict[str, Tensor],
    names: list[str],
) -> dict[str, tuple[Tensor, Tensor]]:
    stats: dict[str, tuple[Tensor, Tensor]] = {}
    for name in names:
        values = data[name]
        stats[name] = (values.mean(), values.std(correction=0))
    return stats


_PRETRAIN_LOG_INTERVAL = 10_000
_PINN_LOG_INTERVAL = 100


def _run_pretrain_loop(
    model: PINNModel,
    optimizer: torch.optim.Optimizer,
    train_coords: dict[str, Tensor],
    train_targets: dict[str, Tensor],
    val_coords: dict[str, Tensor],
    val_targets: dict[str, Tensor],
    config: PINNConfig,
    best: _BestState,
) -> tuple[int, bool]:
    total = config.pretrain_epoch
    for epoch in range(1, total + 1):
        _train_epoch(model, optimizer, train_coords, train_targets)
        train_loss, val_loss = _evaluate_losses(
            model, train_coords, train_targets, val_coords, val_targets
        )
        if epoch % _PRETRAIN_LOG_INTERVAL == 0 or epoch == 1:
            logger.info(
                "pretrain %d/%d train=%.6f val=%.6f best_val=%.6f",
                epoch,
                total,
                train_loss,
                val_loss,
                best.val_loss,
            )
        if _is_improved(val_loss, best.val_loss):
            best.state_dict = _clone_state_dict(model.state_dict())
            best.train_loss = train_loss
            best.val_loss = val_loss
            best.epochs_without_improvement = 0
            continue
        best.epochs_without_improvement += 1
        if best.epochs_without_improvement >= config.early_stop_patience:
            if epoch < _early_stop_warmup(config, config.pretrain_epoch):
                best.epochs_without_improvement = 0
                continue
            logger.info(
                "pretrain early stop at epoch %d best_val=%.6f",
                epoch,
                best.val_loss,
            )
            return epoch, True
    return total, False


def _has_validation_rows(
    val_coords: dict[str, Tensor],
    val_targets: dict[str, Tensor],
) -> bool:
    tensors = [*val_coords.values(), *val_targets.values()]
    if not tensors:
        return False
    return all(int(tensor.shape[0]) > 0 for tensor in tensors)


def _pretrain_no_val(
    model: PINNModel,
    train_coords: dict[str, Tensor],
    train_targets: dict[str, Tensor],
    config: PINNConfig,
) -> PretrainResult:
    logger.warning(
        "Pretrain received an empty validation set: validation-based "
        "early stopping and best-weight restoration are DISABLED; "
        "training will run all %d epochs and keep the final-epoch "
        "weights (val_loss reported as NaN).",
        config.pretrain_epoch,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    final_train_loss = _run_pretrain_loop_no_val(
        model, optimizer, train_coords, train_targets, config
    )
    _log_memory("pretrain_end", logger)
    return PretrainResult(
        train_loss=final_train_loss,
        val_loss=math.nan,
        epochs_run=config.pretrain_epoch,
        stopped_early=False,
    )


def _run_pretrain_loop_no_val(
    model: PINNModel,
    optimizer: torch.optim.Optimizer,
    train_coords: dict[str, Tensor],
    train_targets: dict[str, Tensor],
    config: PINNConfig,
) -> float:
    total = config.pretrain_epoch
    for epoch in range(1, total + 1):
        _train_epoch(model, optimizer, train_coords, train_targets)
        if epoch % _PRETRAIN_LOG_INTERVAL == 0 or epoch == 1:
            logger.info(
                "pretrain (no-val) %d/%d train=%.6f",
                epoch,
                total,
                _evaluate_train_loss(model, train_coords, train_targets),
            )
    return _evaluate_train_loss(model, train_coords, train_targets)


@torch.no_grad()
def _evaluate_train_loss(
    model: PINNModel,
    train_coords: dict[str, Tensor],
    train_targets: dict[str, Tensor],
) -> float:
    model.eval()
    return _mse_loss(model(**train_coords), train_targets).item()


def _train_epoch(
    model: PINNModel,
    optimizer: torch.optim.Optimizer,
    coords: dict[str, Tensor],
    targets: dict[str, Tensor],
) -> None:
    model.train()
    optimizer.zero_grad()
    loss = _mse_loss(model(**coords), targets)
    loss.backward()
    optimizer.step()


def _prepare_grad_data_dict(
    data: dict[str, Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    return {
        name: value.detach().clone().to(device=device, dtype=dtype).requires_grad_(True)
        for name, value in data.items()
    }


def _prepare_optional_grad_data(
    data: dict[str, Tensor] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor] | None:
    if data is None:
        return None
    return _prepare_grad_data_dict(data, device, dtype)


def _run_pinn_training(
    model: PINNModel,
    optimizer: torch.optim.Optimizer,
    pinn_executor: PINNExecutor,
    terms: list[str],
    coefficients: list[float],
    observation_coords: dict[str, Tensor],
    observation_targets: dict[str, Tensor],
    colloc_coords: dict[str, Tensor],
    dataset_metadata: PDEDataset,
    config: PINNConfig,
    best: _PINNBestState,
    local_coords: dict[str, Tensor] | None,
) -> TrainResult:
    _log_memory("_run_pinn_training_start", logger)
    epochs_without_improvement = 0
    consecutive_nan = 0
    total = config.pinn_epoch
    for epoch in range(1, total + 1):
        if epoch % _PINN_LOG_INTERVAL == 0 or epoch == 1:
            logger.info(
                "pinn_train %d/%d data=%.6f phys=%.6f total=%.6f",
                epoch,
                total,
                best.data_loss,
                best.physics_loss,
                best.total_loss,
            )
        best, epochs_without_improvement, result, nan_detected = _run_pinn_epoch(
            model,
            optimizer,
            pinn_executor,
            terms,
            coefficients,
            observation_coords,
            observation_targets,
            colloc_coords,
            dataset_metadata,
            config,
            best,
            local_coords,
            epoch,
            epochs_without_improvement,
        )
        if nan_detected:
            consecutive_nan += 1
            if consecutive_nan > config.max_nan_recoveries:
                logger.warning(
                    "Exceeded %d consecutive NaN recoveries at epoch %d; "
                    "terminating PINN training",
                    config.max_nan_recoveries,
                    epoch,
                )
                _log_memory("_run_pinn_training_end", logger)
                return _make_train_result(best, epoch, False)
            continue
        consecutive_nan = 0
        if result is not None:
            _log_memory("_run_pinn_training_end", logger)
            return result
    model.load_state_dict(best.state_dict)
    _log_memory("_run_pinn_training_end", logger)
    return _make_train_result(best, config.pinn_epoch, False)


def _run_pinn_epoch(
    model: PINNModel,
    optimizer: torch.optim.Optimizer,
    pinn_executor: PINNExecutor,
    terms: list[str],
    coefficients: list[float],
    observation_coords: dict[str, Tensor],
    observation_targets: dict[str, Tensor],
    colloc_coords: dict[str, Tensor],
    dataset_metadata: PDEDataset,
    config: PINNConfig,
    best: _PINNBestState,
    local_coords: dict[str, Tensor] | None,
    epoch: int,
    epochs_without_improvement: int,
) -> tuple[_PINNBestState, int, TrainResult | None, bool]:
    log_this_epoch = epoch % 100 == 0 or epoch == 1
    if log_this_epoch:
        _log_memory(f"epoch_{epoch}_start", logger)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _clear_coordinate_grads(colloc_coords, local_coords)

    losses = _compute_and_backward_pinn_losses(
        model,
        pinn_executor,
        terms,
        coefficients,
        observation_coords,
        observation_targets,
        colloc_coords,
        dataset_metadata,
        config,
        local_coords,
    )
    total_value = losses["total"]




    if not math.isfinite(total_value) or not _all_grads_finite(model):
        _recover_best_checkpoint(model, optimizer, best, epoch, config.lr)
        if log_this_epoch:
            _log_memory(f"epoch_{epoch}_end", logger)
        return best, epochs_without_improvement, None, True
    best, epochs_without_improvement, result = _update_pinn_best_from_values(
        model,
        best,
        data_value=losses["data"],
        physics_value=losses["physics"],
        total_value=total_value,
        epoch=epoch,
        epochs_without_improvement=epochs_without_improvement,
        config=config,
    )
    if result is not None:
        if log_this_epoch:
            _log_memory(f"epoch_{epoch}_end", logger)
        return best, epochs_without_improvement, result, False
    if config.grad_clip_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
    optimizer.step()
    if log_this_epoch:
        _log_memory(f"epoch_{epoch}_end", logger)
    return best, epochs_without_improvement, None, False


def _compute_and_backward_pinn_losses(
    model: PINNModel,
    pinn_executor: PINNExecutor,
    terms: list[str],
    coefficients: list[float],
    observation_coords: dict[str, Tensor],
    observation_targets: dict[str, Tensor],
    colloc_coords: dict[str, Tensor],
    dataset_metadata: PDEDataset,
    config: PINNConfig,
    local_coords: dict[str, Tensor] | None,
) -> dict[str, float]:
    data_value = _backward_data_loss(model, observation_coords, observation_targets)
    physics_value = _chunked_backward_residual_loss(
        model,
        pinn_executor,
        terms,
        coefficients,
        colloc_coords,
        dataset_metadata,
        weight=config.coef_pde,
        chunk_size=config.colloc_chunk_size,
    )
    local_value = 0.0
    if local_coords is not None:
        local_value = _chunked_backward_residual_loss(
            model,
            pinn_executor,
            terms,
            coefficients,
            local_coords,
            dataset_metadata,
            weight=config.coef_pde,
            chunk_size=config.colloc_chunk_size,
        )
    total_value = data_value + config.coef_pde * (physics_value + local_value)
    return {
        "data": data_value,
        "physics": physics_value,
        "local": local_value,
        "total": total_value,
    }


def _backward_data_loss(
    model: PINNModel,
    observation_coords: dict[str, Tensor],
    observation_targets: dict[str, Tensor],
) -> float:
    predictions = model(**observation_coords)
    loss = _mse_loss(predictions, observation_targets)
    loss.backward()
    value = loss.detach().item()
    del loss, predictions
    return value


def _all_grads_finite(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True


def _compute_pinn_losses(
    model: PINNModel,
    pinn_executor: PINNExecutor,
    terms: list[str],
    coefficients: list[float],
    observation_coords: dict[str, Tensor],
    observation_targets: dict[str, Tensor],
    colloc_coords: dict[str, Tensor],
    dataset_metadata: PDEDataset,
    config: PINNConfig,
    local_coords: dict[str, Tensor] | None,
) -> tuple[Tensor, Tensor, Tensor]:
    predictions = model(**observation_coords)
    data_loss = _mse_loss(predictions, observation_targets)
    physics_loss = _compute_residual_loss(
        model,
        pinn_executor,
        terms,
        coefficients,
        colloc_coords,
        dataset_metadata,
    )
    local_loss = _compute_local_loss(
        model,
        pinn_executor,
        terms,
        coefficients,
        local_coords,
        dataset_metadata,
        data_loss,
    )
    coef_pde = data_loss.new_tensor(config.coef_pde)
    total_loss = data_loss + coef_pde * (physics_loss + local_loss)
    return data_loss, physics_loss, total_loss


def _compute_residual_loss(
    model: PINNModel,
    pinn_executor: PINNExecutor,
    terms: list[str],
    coefficients: list[float],
    coords: dict[str, Tensor],
    dataset_metadata: PDEDataset,
) -> Tensor:
    residual = pinn_executor.compute_residual(
        model=model,
        terms=terms,
        coefficients=coefficients,
        coords=coords,
        dataset_metadata=dataset_metadata,
        lhs_field=dataset_metadata.lhs_field,
        lhs_axis=dataset_metadata.lhs_axis,
    )
    return residual.square().mean()


def _compute_local_loss(
    model: PINNModel,
    pinn_executor: PINNExecutor,
    terms: list[str],
    coefficients: list[float],
    local_coords: dict[str, Tensor] | None,
    dataset_metadata: PDEDataset,
    reference: Tensor,
) -> Tensor:
    if local_coords is None:
        return reference.new_zeros(())
    return _compute_residual_loss(
        model,
        pinn_executor,
        terms,
        coefficients,
        local_coords,
        dataset_metadata,
    )


def _clear_coordinate_grads(*coord_sets: dict[str, Tensor] | None) -> None:
    for coords in coord_sets:
        if coords is None:
            continue
        for tensor in coords.values():
            tensor.grad = None


def _resolve_effective_chunk_size(
    chunk_size: int | None,
    n_total: int,
) -> int | None:
    if chunk_size is not None:
        return chunk_size
    if n_total <= _LARGE_BATCH_AUTO_THRESHOLD:
        return None
    logger.info(
        "colloc_chunk_size=None but n_total=%d > %d; auto-chunking at %d to "
        "avoid OOM on deep expressions (E29).",
        n_total,
        _LARGE_BATCH_AUTO_THRESHOLD,
        _LARGE_BATCH_AUTO_CHUNK,
    )
    return _LARGE_BATCH_AUTO_CHUNK


def _maybe_empty_cache(chunk_idx: int) -> None:



    if chunk_idx <= 0 or chunk_idx % _EMPTY_CACHE_EVERY_N_CHUNKS != 0:
        return
    if not torch.cuda.is_available():
        return
    try:
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device)
        total = torch.cuda.get_device_properties(device).total_memory
        if total > 0 and (allocated / total) > _EMPTY_CACHE_PRESSURE_THRESHOLD:
            torch.cuda.empty_cache()
    except RuntimeError:


        return


def _chunked_backward_residual_loss(
    model: PINNModel,
    pinn_executor: PINNExecutor,
    terms: list[str],
    coefficients: list[float],
    coords: dict[str, Tensor],
    dataset_metadata: PDEDataset,
    *,
    weight: float,
    chunk_size: int | None,
) -> float:
    if weight == 0.0:
        return 0.0
    n_total = next(iter(coords.values())).shape[0]
    effective_chunk_size = _resolve_effective_chunk_size(chunk_size, n_total)
    if effective_chunk_size is None or effective_chunk_size >= n_total:
        residual = pinn_executor.compute_residual(
            model=model,
            terms=terms,
            coefficients=coefficients,
            coords=coords,
            dataset_metadata=dataset_metadata,
            lhs_field=dataset_metadata.lhs_field,
            lhs_axis=dataset_metadata.lhs_axis,
        )
        loss_sum_sq = residual.pow(2).sum()
        loss_mean = loss_sum_sq / n_total
        (weight * loss_mean).backward()
        value = loss_mean.detach().item()
        del residual, loss_sum_sq, loss_mean
        return value

    total_sum_sq = 0.0
    for chunk_idx, start in enumerate(range(0, n_total, effective_chunk_size)):
        end = min(start + effective_chunk_size, n_total)
        chunk_coords = {
            k: v[start:end].detach().requires_grad_(True) for k, v in coords.items()
        }
        try:
            chunk_residual = pinn_executor.compute_residual(
                model=model,
                terms=terms,
                coefficients=coefficients,
                coords=chunk_coords,
                dataset_metadata=dataset_metadata,
                lhs_field=dataset_metadata.lhs_field,
                lhs_axis=dataset_metadata.lhs_axis,
            )
            chunk_sum_sq = chunk_residual.pow(2).sum()
            chunk_loss = chunk_sum_sq / n_total
            (weight * chunk_loss).backward()
            total_sum_sq += chunk_sum_sq.detach().item()
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.error(
                "Chunk OOM at start=%d, chunk_size=%d, n_total=%d. "
                "Reduce colloc_chunk_size.",
                start,
                effective_chunk_size,
                n_total,
            )
            raise
        del chunk_residual, chunk_sum_sq, chunk_loss, chunk_coords
        _maybe_empty_cache(chunk_idx)
    return total_sum_sq / n_total


def _chunked_eval_residual_loss(
    model: PINNModel,
    pinn_executor: PINNExecutor,
    terms: list[str],
    coefficients: list[float],
    coords: dict[str, Tensor],
    dataset_metadata: PDEDataset,
    *,
    weight: float,
    chunk_size: int | None,
) -> float:
    if weight == 0.0:
        return 0.0
    n_total = next(iter(coords.values())).shape[0]
    effective_chunk_size = _resolve_effective_chunk_size(chunk_size, n_total)
    if effective_chunk_size is None or effective_chunk_size >= n_total:
        with torch.enable_grad():
            residual = pinn_executor.compute_residual(
                model=model,
                terms=terms,
                coefficients=coefficients,
                coords=coords,
                dataset_metadata=dataset_metadata,
                lhs_field=dataset_metadata.lhs_field,
                lhs_axis=dataset_metadata.lhs_axis,
            )
            value = residual.pow(2).mean().detach().item()
            del residual
            return value

    total_sum_sq = 0.0
    for chunk_idx, start in enumerate(range(0, n_total, effective_chunk_size)):
        end = min(start + effective_chunk_size, n_total)
        chunk_coords = {
            k: v[start:end].detach().requires_grad_(True) for k, v in coords.items()
        }
        try:
            with torch.enable_grad():
                chunk_residual = pinn_executor.compute_residual(
                    model=model,
                    terms=terms,
                    coefficients=coefficients,
                    coords=chunk_coords,
                    dataset_metadata=dataset_metadata,
                    lhs_field=dataset_metadata.lhs_field,
                    lhs_axis=dataset_metadata.lhs_axis,
                )
                chunk_sum_sq_value = chunk_residual.pow(2).sum().detach().item()
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.error(
                "Eval chunk OOM at start=%d, chunk_size=%d, n_total=%d. "
                "Reduce colloc_chunk_size.",
                start,
                effective_chunk_size,
                n_total,
            )
            raise
        total_sum_sq += chunk_sum_sq_value
        del chunk_residual, chunk_coords
        _maybe_empty_cache(chunk_idx)
    return total_sum_sq / n_total


@torch.no_grad()
def _capture_pinn_best(
    model: PINNModel,
    pinn_executor: PINNExecutor,
    terms: list[str],
    coefficients: list[float],
    observation_coords: dict[str, Tensor],
    observation_targets: dict[str, Tensor],
    colloc_coords: dict[str, Tensor],
    dataset_metadata: PDEDataset,
    config: PINNConfig,
    local_coords: dict[str, Tensor] | None,
) -> _PINNBestState:
    _log_memory("capture_best_start", logger)
    d_val = _mse_loss(model(**observation_coords), observation_targets).detach().item()
    p_val = _chunked_eval_residual_loss(
        model,
        pinn_executor,
        terms,
        coefficients,
        colloc_coords,
        dataset_metadata,
        weight=config.coef_pde,
        chunk_size=config.colloc_chunk_size,
    )
    l_val = 0.0
    if local_coords is not None:
        l_val = _chunked_eval_residual_loss(
            model,
            pinn_executor,
            terms,
            coefficients,
            local_coords,
            dataset_metadata,
            weight=config.coef_pde,
            chunk_size=config.colloc_chunk_size,
        )
    t_val = d_val + config.coef_pde * (p_val + l_val)
    if not math.isfinite(t_val):
        logger.warning("Initial PINN losses are non-finite; using inf as baseline")
        d_val = float("inf")
        p_val = float("inf")
        t_val = float("inf")
    result = _PINNBestState(
        state_dict=_clone_state_dict(model.state_dict()),
        data_loss=d_val,
        physics_loss=p_val,
        total_loss=t_val,
    )
    _log_memory("capture_best_end", logger)
    return result


def _update_pinn_best(
    model: PINNModel,
    best: _PINNBestState,
    data_loss: Tensor,
    physics_loss: Tensor,
    total_loss: Tensor,
    epoch: int,
    epochs_without_improvement: int,
    config: PINNConfig,
) -> tuple[_PINNBestState, int, TrainResult | None]:
    total_value = total_loss.detach().item()
    if _is_pinn_improved(total_value, best.total_loss):
        return (
            _PINNBestState(
                state_dict=_clone_state_dict(model.state_dict()),
                data_loss=data_loss.detach().item(),
                physics_loss=physics_loss.detach().item(),
                total_loss=total_value,
            ),
            0,
            None,
        )
    epochs_without_improvement += 1
    if epochs_without_improvement < config.early_stop_patience:
        return best, epochs_without_improvement, None
    warmup = _early_stop_warmup(config, config.pinn_epoch)
    if epoch < warmup:
        return best, 0, None
    model.load_state_dict(best.state_dict)
    logger.info(
        "pinn_train early stop at epoch %d total=%.6f",
        epoch,
        best.total_loss,
    )
    return best, epochs_without_improvement, _make_train_result(best, epoch, True)


def _update_pinn_best_from_values(
    model: PINNModel,
    best: _PINNBestState,
    *,
    data_value: float,
    physics_value: float,
    total_value: float,
    epoch: int,
    epochs_without_improvement: int,
    config: PINNConfig,
) -> tuple[_PINNBestState, int, TrainResult | None]:
    if _is_pinn_improved(total_value, best.total_loss):
        return (
            _PINNBestState(
                state_dict=_clone_state_dict(model.state_dict()),
                data_loss=data_value,
                physics_loss=physics_value,
                total_loss=total_value,
            ),
            0,
            None,
        )
    epochs_without_improvement += 1
    if epochs_without_improvement < config.early_stop_patience:
        return best, epochs_without_improvement, None
    warmup = _early_stop_warmup(config, config.pinn_epoch)
    if epoch < warmup:
        return best, 0, None
    model.load_state_dict(best.state_dict)
    logger.info(
        "pinn_train early stop at epoch %d total=%.6f",
        epoch,
        best.total_loss,
    )
    return best, epochs_without_improvement, _make_train_result(best, epoch, True)


def _recover_best_checkpoint(
    model: PINNModel,
    optimizer: torch.optim.Optimizer,
    best: _PINNBestState,
    epoch: int,
    lr: float,
) -> None:
    logger.warning(
        "Detected NaN/Inf during PINN training at epoch %d; restoring best checkpoint",
        epoch,
    )
    model.load_state_dict(best.state_dict)

    optimizer.param_groups.clear()
    fresh = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.state = fresh.state
    optimizer.param_groups[:] = fresh.param_groups
    optimizer.zero_grad(set_to_none=True)


@torch.no_grad()
def _evaluate_result(
    model: PINNModel,
    train_coords: dict[str, Tensor],
    train_targets: dict[str, Tensor],
    val_coords: dict[str, Tensor],
    val_targets: dict[str, Tensor],
    epochs_run: int,
    stopped_early: bool,
) -> PretrainResult:
    train_loss, val_loss = _evaluate_losses(
        model, train_coords, train_targets, val_coords, val_targets
    )
    return PretrainResult(
        train_loss=train_loss,
        val_loss=val_loss,
        epochs_run=epochs_run,
        stopped_early=stopped_early,
    )


@torch.no_grad()
def _evaluate_losses(
    model: PINNModel,
    train_coords: dict[str, Tensor],
    train_targets: dict[str, Tensor],
    val_coords: dict[str, Tensor],
    val_targets: dict[str, Tensor],
) -> tuple[float, float]:
    model.eval()
    train_loss = _mse_loss(model(**train_coords), train_targets).item()
    val_loss = _mse_loss(model(**val_coords), val_targets).item()
    return train_loss, val_loss


def _mse_loss(predictions: dict[str, Tensor], targets: dict[str, Tensor]) -> Tensor:
    losses = [functional.mse_loss(predictions[name], targets[name]) for name in targets]
    return torch.stack(losses).mean()


def _clone_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in state_dict.items()}


def _make_train_result(
    best: _PINNBestState,
    epochs_run: int,
    stopped_early: bool,
) -> TrainResult:
    return TrainResult(
        data_loss=best.data_loss,
        physics_loss=best.physics_loss,
        total_loss=best.total_loss,
        epochs_run=epochs_run,
        stopped_early=stopped_early,
    )


def _is_finite_loss(loss: Tensor) -> bool:
    return bool(torch.isfinite(loss.detach()).item())


def _is_pinn_improved(current: float, best: float) -> bool:
    return current < best - _PINN_EARLY_STOP_MIN_DELTA


def _is_improved(current: float, best: float) -> bool:
    return current < best - _EARLY_STOP_MIN_DELTA


def _early_stop_warmup(config: PINNConfig, max_epochs: int) -> int:
    if config.early_stop_warmup is not None:
        return min(max_epochs, config.early_stop_warmup)
    return min(
        max_epochs,
        max(
            _EARLY_STOP_MIN_WARMUP_EPOCHS,
            config.early_stop_patience * _EARLY_STOP_WARMUP_MULTIPLIER,
        ),
    )
