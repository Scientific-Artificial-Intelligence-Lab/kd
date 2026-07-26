
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from kd.core.evaluator import EvaluationResult


MAGNITUDE_FILTER_MIN = 5e-5

MAGNITUDE_FILTER_MAX = 1e4


def magnitude_reject_reason(
    coefficients: Tensor,
    selected_indices: Sequence[int] | None = None,
) -> str | None:
    rejection = magnitude_rejection(coefficients, selected_indices)
    return rejection[0] if rejection is not None else None


def magnitude_rejection(
    coefficients: Tensor,
    selected_indices: Sequence[int] | None,
) -> tuple[str, str] | None:
    if coefficients.numel() == 0:
        return None
    if selected_indices is not None:
        if len(selected_indices) == 0:
            return None
        index = torch.as_tensor(
            list(selected_indices),
            dtype=torch.long,
            device=coefficients.device,
        )
        if (
            int(index.max().item()) >= coefficients.numel()
            or int(index.min().item()) < 0
        ):
            return (
                "small_coe: selected_indices out of range for coefficient "
                f"vector of length {coefficients.numel()}",
                "structural_reject",
            )
        coefficients = coefficients.index_select(0, index)
    coef_abs = coefficients.abs()
    min_abs = float(coef_abs.min().item())
    max_abs = float(coef_abs.max().item())
    if not math.isfinite(min_abs) or not math.isfinite(max_abs):
        return (
            f"large_coe: non-finite coefficient magnitude ({max_abs})",
            "non_finite",
        )
    if min_abs < MAGNITUDE_FILTER_MIN:
        return (
            f"small_coe: coefficient magnitude below {MAGNITUDE_FILTER_MIN:.0e} "
            f"(min|w|={min_abs:.3e})",
            "structural_reject",
        )
    if max_abs > MAGNITUDE_FILTER_MAX:
        return (
            f"large_coe: coefficient magnitude above {MAGNITUDE_FILTER_MAX:.0e} "
            f"(max|w|={max_abs:.3e})",
            "structural_reject",
        )
    return None


def apply_magnitude_filter(result: EvaluationResult) -> EvaluationResult:
    if not result.is_valid:
        return result

    coefficients = result.coefficients
    if coefficients is None or coefficients.numel() == 0:
        return result

    rejection = magnitude_rejection(coefficients, result.selected_indices)
    if rejection is not None:
        detail, invalid_reason = rejection
        return _reject(result, detail, invalid_reason=invalid_reason)
    return result


def _reject(
    result: EvaluationResult,
    detail: str,
    *,
    invalid_reason: str,
) -> EvaluationResult:
    return EvaluationResult(
        mse=result.mse,
        nmse=result.nmse,
        r2=result.r2,
        score=result.score,
        complexity=result.complexity,
        coefficients=result.coefficients,
        is_valid=False,
        error_message=detail,
        invalid_reason=invalid_reason,
        selected_indices=result.selected_indices,
        residuals=result.residuals,
        terms=result.terms,
        expression=result.expression,
        lhs_name=result.lhs_name,
    )


__all__ = [
    "MAGNITUDE_FILTER_MAX",
    "MAGNITUDE_FILTER_MIN",
    "apply_magnitude_filter",
    "magnitude_rejection",
    "magnitude_reject_reason",
]
