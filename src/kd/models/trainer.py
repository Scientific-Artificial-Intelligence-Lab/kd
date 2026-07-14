
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from kd.models.field_model import FieldModel

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:

    final_loss: float
    epochs_run: int
    early_stopped: bool
    val_loss: float | None
    best_val_loss: float | None = None
    best_epoch: int | None = None
    best_restored: bool = False
    loss_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] | None = None


class FieldModelTrainer:

    def __init__(
        self,
        model: FieldModel,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        self._model = model
        self._lr = lr
        self._weight_decay = weight_decay

    def fit(
        self,
        coords: dict[str, Tensor],
        targets: dict[str, Tensor],
        max_epochs: int = 10000,
        patience: int | None = 100,
        val_ratio: float = 0.2,
        seed: int = 0,
        restore_best: bool = False,
        device: str | torch.device | None = None,
    ) -> TrainingResult:
        _validate_val_ratio(val_ratio)
        entry_device = next(self._model.parameters()).device
        target_device = _normalize_device(device)
        _validate_device_available(target_device)


        _data_dtype = _infer_dtype(coords, targets)
        _model_dtype = next(self._model.parameters()).dtype
        if _data_dtype != _model_dtype:
            logger.debug(
                "Casting model from %s to %s to match input data",
                _model_dtype,
                _data_dtype,
            )
            self._model = self._model.to(dtype=_data_dtype)


        cuda_devices: list[int] = []
        if torch.cuda.is_available():
            cuda_devices = list(range(torch.cuda.device_count()))

        try:
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(seed)
                result = self._fit_inner(
                    coords,
                    targets,
                    max_epochs,
                    patience,
                    val_ratio,
                    seed,
                    restore_best,
                    target_device,
                )
        finally:
            self._model = self._model.to(entry_device)
        return result



    def _fit_inner(
        self,
        coords: dict[str, Tensor],
        targets: dict[str, Tensor],
        max_epochs: int,
        patience: int | None,
        val_ratio: float,
        seed: int,
        restore_best: bool,
        target_device: torch.device | None,
    ) -> TrainingResult:

        _reinit_parameters(self._model)








        coords = {k: v.detach() for k, v in coords.items()}
        targets = {k: v.detach() for k, v in targets.items()}
        if target_device is not None:
            self._model = self._model.to(target_device)
            coords = _move_tensors(coords, target_device)
            targets = _move_tensors(targets, target_device)


        self._set_normalization(coords, targets)


        n_samples = _get_n_samples(coords)
        train_idx, val_idx = _split_indices(n_samples, val_ratio, seed)

        train_coords = _index_dict(coords, train_idx)
        train_targets = _index_dict(targets, train_idx)
        val_coords = _index_dict(coords, val_idx) if val_idx is not None else None
        val_targets = _index_dict(targets, val_idx) if val_idx is not None else None




        self._model.train()
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
        criterion = nn.MSELoss()


        if patience is not None and val_ratio == 0.0:
            logger.warning(
                "patience=%d has no effect when val_ratio=0 "
                "(no validation data for early stopping)",
                patience,
            )


        best_val_loss = float("inf")
        best_epoch: int | None = None
        best_state: dict[str, Tensor] | None = None
        epochs_without_improvement = 0
        final_loss = float("inf")
        final_val_loss: float | None = None
        early_stopped = False
        epoch = 0
        track_best = restore_best and val_coords is not None
        has_val = val_coords is not None and val_targets is not None




        loss_history: list[float] = []
        val_loss_history: list[float] | None = [] if has_val else None

        for epoch in range(1, max_epochs + 1):
            final_loss = _train_step(
                self._model, optimizer, criterion, train_coords, train_targets
            )
            loss_history.append(final_loss)


            if val_coords is not None and val_targets is not None:
                final_val_loss = _eval_loss(
                    self._model, criterion, val_coords, val_targets
                )
                if val_loss_history is not None:
                    val_loss_history.append(final_val_loss)

                if final_val_loss < best_val_loss:
                    best_val_loss = final_val_loss
                    best_epoch = epoch
                    if track_best:


                        best_state = {
                            name: tensor.detach().clone()
                            for name, tensor in self._model.state_dict().items()
                        }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                    if patience is not None and epochs_without_improvement >= patience:
                        early_stopped = True
                        logger.debug(
                            "Early stop at epoch %d (patience=%d)",
                            epoch,
                            patience,
                        )
                        break


        best_restored = False
        if best_state is not None:
            self._model.load_state_dict(best_state)
            best_restored = True
        self._model.eval()
        return TrainingResult(
            final_loss=final_loss,
            epochs_run=epoch,
            early_stopped=early_stopped,
            val_loss=final_val_loss,
            best_val_loss=best_val_loss if best_epoch is not None else None,
            best_epoch=best_epoch,
            best_restored=best_restored,
            loss_history=loss_history,
            val_loss_history=val_loss_history,
        )

    def _set_normalization(
        self,
        coords: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> None:
        coord_stats: dict[str, tuple[Tensor, Tensor]] = {}
        for name in self._model.coord_names:
            data = coords[name]
            coord_stats[name] = (data.mean(), data.std())

        field_stats: dict[str, tuple[Tensor, Tensor]] = {}
        for name in self._model.field_names:
            data = targets[name]
            field_stats[name] = (data.mean(), data.std())

        self._model.set_normalization(coord_stats, field_stats)







def _reinit_parameters(model: nn.Module) -> None:
    for module in model.modules():
        reset_fn = getattr(module, "reset_parameters", None)
        if reset_fn is not None:
            reset_fn()


def _infer_dtype(coords: dict[str, Tensor], targets: dict[str, Tensor]) -> torch.dtype:
    for tensors in (coords, targets):
        for v in tensors.values():
            return v.dtype
    return torch.float32


def _validate_val_ratio(val_ratio: float) -> None:
    if val_ratio < 0.0 or val_ratio >= 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")


def _normalize_device(device: str | torch.device | None) -> torch.device | None:
    if device is None:
        return None
    try:
        return torch.device(device)
    except RuntimeError as exc:
        raise ValueError(
            f"Invalid training device {device!r}: expected a torch device string "
            "or torch.device"
        ) from exc


def _validate_device_available(device: torch.device | None) -> None:
    if device is None or device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise ValueError(
            f"Requested CUDA device {device!s}, but CUDA is not available. "
            "Choose an available training device."
        )
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested CUDA device {device!s}, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are available."
        )


def _move_tensors(data: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {k: v.to(device) for k, v in data.items()}


def _get_n_samples(data: dict[str, Tensor]) -> int:
    return next(iter(data.values())).shape[0]


def _split_indices(n: int, val_ratio: float, seed: int) -> tuple[Tensor, Tensor | None]:
    if val_ratio == 0.0:
        return torch.arange(n), None


    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(n, generator=gen)

    n_val = max(1, int(n * val_ratio))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def _index_dict(data: dict[str, Tensor], idx: Tensor) -> dict[str, Tensor]:
    return {k: v[idx] for k, v in data.items()}


def _train_step(
    model: FieldModel,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    coords: dict[str, Tensor],
    targets: dict[str, Tensor],
) -> float:
    model.train()
    optimizer.zero_grad()
    preds = model(**coords)
    loss = _compute_mse(criterion, preds, targets)
    loss.backward()
    optimizer.step()
    return loss.detach().item()


@torch.no_grad()
def _eval_loss(
    model: FieldModel,
    criterion: nn.Module,
    coords: dict[str, Tensor],
    targets: dict[str, Tensor],
) -> float:
    model.eval()
    preds = model(**coords)
    loss = _compute_mse(criterion, preds, targets)
    return loss.item()


def _compute_mse(
    criterion: nn.Module,
    preds: dict[str, Tensor],
    targets: dict[str, Tensor],
) -> Tensor:
    losses: list[Tensor] = []
    for name, pred in preds.items():
        losses.append(criterion(pred, targets[name]))
    return torch.stack(losses).mean()
