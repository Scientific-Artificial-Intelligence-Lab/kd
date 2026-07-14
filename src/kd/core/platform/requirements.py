
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from kd.data.schema import DataTopology


@dataclass(frozen=True)
class DerivativeReqs:

    provider_kind: Literal["finite_diff", "autograd", "none"] = "finite_diff"
    max_atomic_order: int = 2
    lhs_order: int = 1
    needs_surrogate: bool = False
    surrogate_model: torch.nn.Module | None = None
    surrogate_train_kwargs: dict[str, Any] | None = None
    surrogate_arch_kwargs: dict[str, Any] | None = None
    supported_topologies: frozenset[DataTopology] = field(
        default_factory=lambda: frozenset({DataTopology.GRID})
    )

    def __post_init__(self) -> None:
        if self.needs_surrogate and self.provider_kind != "autograd":
            raise ValueError(
                "needs_surrogate=True requires provider_kind='autograd' "
                f"(got '{self.provider_kind}'); SurrogateContext relies on "
                "AutogradProvider.get_field()"
            )



        if self.provider_kind == "none":
            if self.needs_surrogate:
                raise ValueError(
                    "provider_kind='none' cannot set needs_surrogate=True: the "
                    "light bundle builds no context/provider to wrap a surrogate."
                )
            if self.surrogate_model is not None:
                raise ValueError(
                    "provider_kind='none' cannot carry a surrogate_model: the "
                    "light bundle builds no provider that would consume it."
                )
            if self.surrogate_train_kwargs is not None:
                raise ValueError(
                    "provider_kind='none' cannot carry surrogate_train_kwargs: the "
                    "light bundle builds no provider that would consume them."
                )
            if self.surrogate_arch_kwargs is not None:
                raise ValueError(
                    "provider_kind='none' cannot carry surrogate_arch_kwargs: the "
                    "light bundle builds no provider that would consume them."
                )


def assert_dataset_supported(
    dataset_lhs_order: int,
    dataset_topology: DataTopology,
    reqs: DerivativeReqs,
    algorithm: str,
) -> None:
    if dataset_topology not in reqs.supported_topologies:
        supported = ", ".join(sorted(t.value for t in reqs.supported_topologies))
        raise NotImplementedError(
            f"Algorithm '{algorithm}' does not support {dataset_topology.value} "
            f"data (topology): it supports [{supported}] only. Use a dataset with "
            f"a supported topology, or a plugin that declares "
            f"'{dataset_topology.value}' in its derivative_requirements "
            f"(arch041 step 3b / C-A)."
        )
    if dataset_lhs_order != reqs.lhs_order:
        raise NotImplementedError(
            f"Algorithm '{algorithm}' does not support a dataset with "
            f"lhs_order={dataset_lhs_order}: it targets lhs_order={reqs.lhs_order} "
            f"only (the single order whose evaluator it builds). Match the dataset "
            f"order to the plugin's target, or use a plugin that selects this order."
        )
