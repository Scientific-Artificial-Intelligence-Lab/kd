
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from kd._registry import Registry, registering_builtins
from kd.data.derivatives.autograd import AutogradProvider
from kd.data.derivatives.base import DerivativeProvider
from kd.data.derivatives.finite_diff import FiniteDiffProvider
from kd.data.schema import PDEDataset

if TYPE_CHECKING:
    from kd.core.platform.requirements import DerivativeReqs


@dataclass(frozen=True)
class ProviderRequest:

    dataset: PDEDataset
    reqs: DerivativeReqs
    device: torch.device | None
    coords: dict[str, torch.Tensor] | None
    surrogate_model: torch.nn.Module | None


@dataclass(frozen=True)
class DerivativeProviderEntry:

    build: Callable[[ProviderRequest], DerivativeProvider]
    surrogate: bool
    implementation: str


_providers: Registry[DerivativeProviderEntry] = Registry(
    "derivative_provider", implementation_of=lambda entry: entry.implementation
)


def register_derivative_provider(
    name: str,
    build: Callable[[ProviderRequest], DerivativeProvider],
    *,
    surrogate: bool,
    implementation: str,
) -> None:
    if name == "none":
        raise ValueError("provider name 'none' is reserved for no derivative service")
    if not implementation:
        raise ValueError("provider implementation must be non-empty")
    _providers.register(
        name,
        DerivativeProviderEntry(build, surrogate, implementation),
    )


def derivative_provider(name: str) -> DerivativeProviderEntry:
    return _providers.get(name)


def derivative_providers() -> tuple[str, ...]:
    return _providers.names()




_FINITE_DIFF_ACCURACY: dict[str, int] = {"finite_diff": 4, "fd2": 2}


def is_finite_diff_provider(name: str) -> bool:
    return name in _FINITE_DIFF_ACCURACY


def finite_diff_provider(
    name: str, dataset: PDEDataset, max_order: int
) -> FiniteDiffProvider:
    return FiniteDiffProvider(
        dataset, max_order=max_order, accuracy=_FINITE_DIFF_ACCURACY[name]
    )


def _finite_diff(request: ProviderRequest) -> DerivativeProvider:
    return finite_diff_provider(
        "finite_diff", request.dataset, request.reqs.max_atomic_order
    )


def _fd2(request: ProviderRequest) -> DerivativeProvider:
    return finite_diff_provider("fd2", request.dataset, request.reqs.max_atomic_order)


def _autograd(request: ProviderRequest) -> DerivativeProvider:

    return AutogradProvider(
        cast(torch.nn.Module, request.surrogate_model),
        cast(dict[str, torch.Tensor], request.coords),
        request.dataset,
        max_order=request.reqs.max_atomic_order,
    )


with registering_builtins():
    register_derivative_provider(
        "finite_diff",
        _finite_diff,
        surrogate=False,
        implementation="kd.data.derivatives.finite_diff:FiniteDiffProvider",
    )
    register_derivative_provider(
        "autograd",
        _autograd,
        surrogate=True,
        implementation="kd.data.derivatives.autograd:AutogradProvider",
    )
    register_derivative_provider(
        "fd2",
        _fd2,
        surrogate=False,
        implementation="kd.data.derivatives.finite_diff:FiniteDiffProvider(accuracy=2)",
    )
