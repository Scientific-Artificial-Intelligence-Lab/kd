
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Literal

import torch

TrainingDtype = Literal["float32", "float64"]

TRAINING_DTYPES: Final[dict[str, torch.dtype]] = {
    "float32": torch.float32,
    "float64": torch.float64,
}

TRAINING_DTYPE_KEY: Final[str] = "training_dtype"


def validate_training_dtype(value: object) -> None:
    if value is None or (isinstance(value, str) and value in TRAINING_DTYPES):
        return
    raise ValueError(
        f"training_dtype must be None, 'float32' or 'float64', got {value!r}"
    )


def resolve_training_dtype(
    training_dtype: TrainingDtype | None, data_dtype: torch.dtype
) -> torch.dtype:
    if training_dtype is None:
        return data_dtype
    return TRAINING_DTYPES[training_dtype]


def omit_default_training_dtype(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if not (key == TRAINING_DTYPE_KEY and value is None)
    }
