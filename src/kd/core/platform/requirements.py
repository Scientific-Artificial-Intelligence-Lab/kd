
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

    def __post_init__(self) -> None:
        if self.needs_surrogate and self.provider_kind != "autograd":
            raise ValueError(
                "needs_surrogate=True requires provider_kind='autograd' "
                f"(got '{self.provider_kind}'); SurrogateContext relies on "
                "AutogradProvider.get_field()"
            )


def assert_lhs_order_supported(
    dataset_lhs_order: int,
    plugin_lhs_order: int,
    algorithm: str,
) -> None:
    if dataset_lhs_order == plugin_lhs_order:
        return
    raise NotImplementedError(
        f"Algorithm '{algorithm}' does not support a dataset with "
        f"lhs_order={dataset_lhs_order}: it targets lhs_order={plugin_lhs_order} "
        f"only. A higher-order LHS (e.g. u_tt, the wave/telegraph case) is not "
        f"yet discoverable end-to-end — reduce to a first-order system, or pass "
        f"lhs_order=1 data. Second-order LHS support is deferred to DATA-4."
    )
