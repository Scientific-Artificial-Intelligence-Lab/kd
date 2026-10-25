
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import kd

if TYPE_CHECKING:
    from kd import PDEDataset, TabularDataset


@dataclass(frozen=True)
class PlatformEvaluation:

    nmse: float | None
    coefficients: list[float] | None
    error: str | None
    dropped: int | None
    unavailable_reason: str | None = None


def lhs_order_of(catalog_fit: dict[str, Any] | None) -> int | None:
    if catalog_fit is None:
        return None
    lhs_spec = catalog_fit["lhs_spec"]
    if lhs_spec is None:
        return 0
    order: int = lhs_spec["order"]
    return order


def platform_evaluation(
    dataset: PDEDataset | TabularDataset, support: Sequence[str], lhs_order: int | None
) -> PlatformEvaluation:
    if lhs_order == 0 or isinstance(dataset, kd.TabularDataset):
        return PlatformEvaluation(None, None, None, None, "unsupported_form")
    try:
        result = kd.refit_terms(
            dataset, list(support), max_order=3, lhs_order=lhs_order
        )
    except kd.SearchInterrupted:
        raise
    except kd.InvalidTermsError as exc:
        return PlatformEvaluation(
            None, None, f"{type(exc).__name__}: {exc}", len(exc.rejected)
        )
    except Exception as exc:
        return PlatformEvaluation(None, None, f"{type(exc).__name__}: {exc}", None)
    return PlatformEvaluation(result.nmse, list(result.coefficients), None, 0)
