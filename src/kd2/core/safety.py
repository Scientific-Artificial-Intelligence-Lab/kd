"""Numerical safety utilities.

Protected operations that guarantee finite outputs (no NaN, no Inf).
All numerical code in kd2 must use these instead of raw arithmetic.
"""

import torch
from torch import Tensor


def safe_div(a: Tensor, b: Tensor, eps: float = 1e-10) -> Tensor:
    """Protected division: ``a / b`` that never produces NaN or Inf.

    When ``b`` is near zero, adds ``eps`` with the sign of ``b``
    to avoid division by zero while preserving sign semantics.

    Args:
        a: Numerator tensor.
        b: Denominator tensor.
        eps: Small constant added to denominator near zero.

    Returns:
        Result tensor with all finite values.
    """
    sign_b = torch.sign(b)
    sign_b = torch.where(sign_b == 0, torch.ones_like(sign_b), sign_b)
    return a / (b + eps * sign_b)


def safe_exp(x: Tensor, max_val: float = 50.0) -> Tensor:
    """Protected exponential: clamps input to prevent overflow.

    Args:
        x: Input tensor.
        max_val: Maximum value before clamping.

    Returns:
        ``exp(clamp(x, max=max_val))``, always finite.
    """
    return torch.exp(torch.clamp(x, max=max_val))


def safe_log(x: Tensor, eps: float = 1e-10) -> Tensor:
    """Protected logarithm: takes log of ``|x|`` clamped above eps.

    Handles zero, negative, and near-zero inputs safely.

    Args:
        x: Input tensor (any sign).
        eps: Minimum absolute value before taking log.

    Returns:
        ``log(clamp(|x|, min=eps))``, always finite.
    """
    return torch.log(torch.clamp(x.abs(), min=eps))
