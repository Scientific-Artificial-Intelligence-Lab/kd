
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kd.models.trainer import TrainingResult
    from kd.search.recorder import VizRecorder








SURROGATE_EPOCH_KEY = "surrogate_epoch"
SURROGATE_TRAIN_LOSS_KEY = "surrogate_train_loss"
SURROGATE_VAL_LOSS_KEY = "surrogate_val_loss"
SURROGATE_BEST_EPOCH_KEY = "surrogate_best_epoch"
SURROGATE_EPOCHS_RUN_KEY = "surrogate_epochs_run"
SURROGATE_EARLY_STOPPED_KEY = "surrogate_early_stopped"
SURROGATE_METRICS: tuple[str, ...] = (
    SURROGATE_EPOCH_KEY,
    SURROGATE_TRAIN_LOSS_KEY,
    SURROGATE_VAL_LOSS_KEY,
    SURROGATE_BEST_EPOCH_KEY,
    SURROGATE_EPOCHS_RUN_KEY,
    SURROGATE_EARLY_STOPPED_KEY,
)





_SURROGATE_HISTORY_CAP = 1024



_EARLY_STOPPED_TRUE = 1.0
_EARLY_STOPPED_FALSE = 0.0


def log_surrogate_training(
    recorder: VizRecorder | None,
    training_result: TrainingResult | None,
) -> None:
    if recorder is None:
        return
    if training_result is None:
        return
    if recorder.get(SURROGATE_TRAIN_LOSS_KEY):
        return

    loss_history = training_result.loss_history
    if not loss_history:
        return
    val_history = training_result.val_loss_history



    if val_history is not None and len(val_history) != len(loss_history):
        raise ValueError(
            f"val_loss_history length {len(val_history)} != loss_history "
            f"length {len(loss_history)}"
        )

    keep = _downsample_indices(len(loss_history), _SURROGATE_HISTORY_CAP)


    epochs = [index + 1 for index in keep]
    recorder.log(SURROGATE_EPOCH_KEY, epochs)
    recorder.log(
        SURROGATE_TRAIN_LOSS_KEY,
        [float(loss_history[index]) for index in keep],
    )
    if val_history is not None:
        recorder.log(
            SURROGATE_VAL_LOSS_KEY,
            [float(val_history[index]) for index in keep],
        )

    if training_result.best_epoch is not None:
        recorder.log(SURROGATE_BEST_EPOCH_KEY, float(training_result.best_epoch))
    recorder.log(SURROGATE_EPOCHS_RUN_KEY, float(training_result.epochs_run))
    recorder.log(
        SURROGATE_EARLY_STOPPED_KEY,
        _EARLY_STOPPED_TRUE
        if training_result.early_stopped
        else _EARLY_STOPPED_FALSE,
    )


def _downsample_indices(n: int, cap: int) -> list[int]:
    if n <= 0:
        return []
    if n <= cap:
        return list(range(n))
    last = n - 1



    if cap < 2:
        return [0, last]
    indices = sorted({round(i * last / (cap - 1)) for i in range(cap)})
    return indices


__all__ = [
    "SURROGATE_BEST_EPOCH_KEY",
    "SURROGATE_EARLY_STOPPED_KEY",
    "SURROGATE_EPOCHS_RUN_KEY",
    "SURROGATE_EPOCH_KEY",
    "SURROGATE_METRICS",
    "SURROGATE_TRAIN_LOSS_KEY",
    "SURROGATE_VAL_LOSS_KEY",
    "log_surrogate_training",
]
