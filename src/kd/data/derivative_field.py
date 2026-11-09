
from __future__ import annotations

import torch

from kd.data.derivatives.finite_diff import MAX_SUPPORTED_ORDER
from kd.data.derivatives.registry import (
    derivative_provider,
    finite_diff_provider,
    is_finite_diff_provider,
)
from kd.data.schema import DataTopology, PDEDataset
from kd.data.subsample import _require_grid

__all__ = ["derivative_field"]


def derivative_field(
    dataset: PDEDataset,
    field: str,
    axis: str,
    order: int = 1,
    *,
    derivatives: str = "finite_diff",
) -> torch.Tensor:
    if derivative_provider(derivatives).surrogate:
        raise ValueError(
            f"derivative_field: derivatives={derivatives!r} differentiates a "
            "field network that a fit trains, and this function trains nothing. "
            "For the derivative of a network-smoothed field, pass "
            "kd.network_smooth(dataset, recipe=...).dataset, or "
            "kd.surrogate_field(dataset, model) for a network already trained, "
            "with derivatives='finite_diff'."
        )
    if not is_finite_diff_provider(derivatives):
        raise ValueError(
            f"derivative_field: derivatives={derivatives!r} was registered by an "
            "extension; only kd's finite-difference providers 'finite_diff' and "
            "'fd2' are served here."
        )
    if dataset.topology is DataTopology.SCATTERED:
        raise ValueError(
            f"derivative_field: {dataset.name!r} holds scattered samples, which "
            "kd differentiates only through a field network that a fit trains, "
            "and this function trains nothing. Put the samples on a grid first "
            "with kd.data.resample_to_grid(dataset, axes=...) and pass the result."
        )
    _require_grid(dataset, what="derivative_field")
    if not 1 <= order <= MAX_SUPPORTED_ORDER:
        raise ValueError(
            f"derivative_field: order must be in [1, {MAX_SUPPORTED_ORDER}], "
            f"got {order}"
        )
    provider = finite_diff_provider(derivatives, dataset, order)
    return provider.get_derivative(field, axis, order)
