
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from kd.core.evaluator import EvaluationResult
from kd.core.metrics import make_aic_scorer, nmse
from kd.search.term_utils import fold_add


def support_from_coefficients(xi: np.ndarray) -> list[int]:
    return [int(index) for index in np.flatnonzero(xi != 0.0)]


def render_best_expression(terms: list[str], support: list[int]) -> str:
    return fold_add([terms[index] for index in support])


def build_native_result(
    *,
    theta: Tensor,
    lhs_flat: Tensor,
    xi: np.ndarray,
    terms: list[str],
    expression: str,
) -> EvaluationResult:
    xi_t = torch.as_tensor(xi, dtype=theta.dtype, device=theta.device)
    support = support_from_coefficients(xi)



    cast_support = support_from_coefficients(xi_t.detach().cpu().numpy())
    if cast_support != support:
        raise RuntimeError(
            "Casting native PySINDy coefficients to the Theta dtype changed "
            "the nonzero support; refusing to record an inconsistent result"
        )
    y_pred = theta @ xi_t
    residuals = (y_pred - lhs_flat).detach()
    mse = float(residuals.square().mean().item())
    if not math.isfinite(mse):
        raise RuntimeError("PySINDy native fit produced a non-finite MSE")




    target_var = float(lhs_flat.detach().double().var(correction=0).item())
    if not math.isfinite(target_var):
        raise RuntimeError("PySINDy native fit target variance is non-finite")
    nmse_value = nmse(mse, target_var)



    r2_value = 1.0 - nmse_value
    complexity = len(support)
    score = make_aic_scorer(lhs_flat.numel())(mse, complexity)

    return EvaluationResult(
        mse=mse,
        nmse=nmse_value,
        r2=r2_value,
        score=score,
        complexity=complexity,
        coefficients=xi_t.detach(),
        is_valid=True,
        error_message="",
        selected_indices=support,
        residuals=residuals,
        terms=list(terms),
        expression=expression,
    )


__all__ = [
    "build_native_result",
    "render_best_expression",
    "support_from_coefficients",
]
