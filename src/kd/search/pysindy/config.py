
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

_DEFAULT_TERMS: tuple[str, ...] = ("u", "u_x", "u_xx", "mul(u, u_x)")




_TYPED_OPTIMIZER_FIELDS: tuple[str, ...] = (
    "threshold",
    "max_iter",
    "normalize_columns",
    "unbias",
)


@dataclass(frozen=True)
class PySINDyConfig:

    terms: tuple[str, ...] = _DEFAULT_TERMS
    threshold: float = 0.1
    max_iter: int = 20
    normalize_columns: bool = False
    unbias: bool = True
    seed: int = 0
    extra_optimizer_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("terms must be a non-empty tuple of candidate terms")
        if not (math.isfinite(self.threshold) and self.threshold >= 0):
            raise ValueError(f"threshold must be finite and >= 0, got {self.threshold}")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {self.max_iter}")
        if self.extra_optimizer_kwargs:
            collisions = sorted(
                set(self.extra_optimizer_kwargs) & set(_TYPED_OPTIMIZER_FIELDS)
            )
            if collisions:
                raise ValueError(
                    "extra_optimizer_kwargs must not override typed fields "
                    f"{collisions}; set the typed field instead"
                )


__all__ = ["PySINDyConfig"]
