
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch


@dataclass(frozen=True)
class DerivativeReqs:

    provider_kind: Literal["finite_diff", "autograd"] = "finite_diff"
    max_atomic_order: int = 2
    lhs_order: int = 1
    needs_surrogate: bool = False
    surrogate_model: torch.nn.Module | None = None
    surrogate_train_kwargs: dict[str, Any] | None = None
    surrogate_arch_kwargs: dict[str, Any] | None = None
    """Optional override for the default surrogate model architecture.

    Forwarded to ``FieldModel(...)`` when ``surrogate_model is None`` and
    ``provider_kind="autograd"``. Common keys:

    - ``hidden_sizes: list[int]`` — layer widths (default: builder choice)
    - ``activation: str`` — activation name, e.g. ``"tanh"``, ``"sin"``

    Plugins that need a specific architecture (e.g. DLGA Stage I uses
    ``sin`` + 5×50 to align with Xu 2020 Table 1 baseline) declare it
    here. ``None`` lets the builder pick its default (``tanh`` + 5×64).
    """

    def __post_init__(self) -> None:
        if self.needs_surrogate and self.provider_kind != "autograd":
            raise ValueError(
                "needs_surrogate=True requires provider_kind='autograd' "
                f"(got '{self.provider_kind}'); SurrogateContext relies on "
                "AutogradProvider.get_field()"
            )
