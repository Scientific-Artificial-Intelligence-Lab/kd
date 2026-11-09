
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from kd.core.equation._sketch_terms import lower_term

_DEFAULT_TERMS: tuple[str, ...] = ("u", "u_x", "u_xx", "mul(u, u_x)")




_TYPED_OPTIMIZER_FIELDS: tuple[str, ...] = (
    "threshold",
    "max_iter",
    "normalize_columns",
    "unbias",
    "alpha",
)


@dataclass(frozen=True)
class PySINDyConfig:

    terms: tuple[str, ...] = _DEFAULT_TERMS
    threshold: float = 0.1
    max_iter: int = 20
    normalize_columns: bool = False
    relative_threshold: bool = False
    unbias: bool = True
    alpha: float = field(default=0.05, metadata={"legacy": 0.05})
    seed: int = 0
    extra_optimizer_kwargs: dict[str, Any] | None = None
    degree: int = 2

    target_lhs_order: int = field(default=1, metadata={"legacy": 1})

    def __post_init__(self) -> None:
        if type(self.target_lhs_order) is not int or self.target_lhs_order not in (
            1,
            2,
        ):
            raise ValueError("target_lhs_order must be the integer 1 or 2")
        if self.degree < 1:
            raise ValueError(f"degree must be >= 1, got {self.degree}")
        if not self.terms:
            raise ValueError("terms must be a non-empty tuple of candidate terms")


        object.__setattr__(self, "terms", tuple(map(lower_term, self.terms)))
        if not (math.isfinite(self.threshold) and self.threshold >= 0):
            raise ValueError(f"threshold must be finite and >= 0, got {self.threshold}")
        if not (math.isfinite(self.alpha) and self.alpha >= 0):
            raise ValueError(f"alpha must be finite and >= 0, got {self.alpha}")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {self.max_iter}")
        if self.relative_threshold and not self.normalize_columns:
            raise ValueError("relative_threshold requires normalize_columns=True")
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
