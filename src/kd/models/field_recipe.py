
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final, Literal

from kd.models._training_precision import (
    TrainingDtype,
    omit_default_training_dtype,
    validate_training_dtype,
)

__all__ = ["DLGA_NN_1", "FieldRecipe"]


@dataclass(frozen=True)
class FieldRecipe:

    name: str
    version: int
    hidden_sizes: tuple[int, ...]
    activation: Literal["tanh", "sin", "relu"]
    lr: float
    max_epochs: int
    patience: int | None
    val_ratio: float
    restore_best: bool
    training_dtype: TrainingDtype | None = None

    def __post_init__(self) -> None:
        validate_training_dtype(self.training_dtype)

    def to_dict(self) -> dict[str, Any]:
        record = {**asdict(self), "hidden_sizes": list(self.hidden_sizes)}
        return omit_default_training_dtype(record)


DLGA_NN_1: Final[FieldRecipe] = FieldRecipe(
    name="dlga_nn_1",
    version=1,
    hidden_sizes=(50, 50, 50, 50, 50),
    activation="sin",
    lr=1e-3,
    max_epochs=50000,
    patience=None,
    val_ratio=0.2,
    restore_best=True,
)
