
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import kd

if TYPE_CHECKING:
    from kd.data import PDEDataset


@dataclass(frozen=True)
class PlatformEvaluation:

    nmse: float | None
    coefficients: list[float] | None
    error: str | None
    dropped: int | None


def lhs_order_of(catalog_fit: dict[str, Any] | None) -> int | None:
    if catalog_fit is None:
        return None
    lhs_spec = catalog_fit["lhs_spec"]
    if lhs_spec is None:
        return 0
    order: int = lhs_spec["order"]
    return order


def platform_evaluation(
    dataset: PDEDataset, support: Sequence[str], lhs_order: int | None
) -> PlatformEvaluation:
    terms = list(support)
    try:
        result = kd.evaluate_terms(
            dataset, terms, max_order=3, skip_invalid=True, lhs_order=lhs_order
        )
    except kd.SearchInterrupted:
        raise
    except kd.InvalidTermsError as exc:
        return PlatformEvaluation(
            None, None, f"{type(exc).__name__}: {exc}", len(exc.rejected)
        )
    except Exception as exc:
        return PlatformEvaluation(None, None, f"{type(exc).__name__}: {exc}", None)
    dropped = len(terms) - len(result.terms or ())
    if dropped > 0:
        return PlatformEvaluation(
            None,
            None,
            f"DroppedTerms: evaluate_terms dropped {dropped} of {len(terms)} "
            "support terms",
            dropped,
        )
    coefficients = (
        None
        if result.coefficients is None
        else result.coefficients.detach().flatten().tolist()
    )
    return PlatformEvaluation(float(result.nmse), coefficients, None, 0)
