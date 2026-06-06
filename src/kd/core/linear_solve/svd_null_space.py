
from __future__ import annotations

import math

import torch

from kd.core.linear_solve._helpers import (
    compute_r2,
    squared_residual,
    upcast_for_solve,
)
from kd.core.linear_solve.base import SolveResult, SparseSolver

_SELECT_EPS = 1e-14


class SVDNullSpaceSolver(SparseSolver):

    def __init__(self, eps: float = 1e-8) -> None:
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = eps

    def solve(self, theta: torch.Tensor, y: torch.Tensor) -> SolveResult:
        self._validate_shape(theta, y)
        y_1d = y.squeeze(-1) if y.dim() == 2 else y

        if not torch.isfinite(theta).all() or not torch.isfinite(y_1d).all():
            return self._invalid(theta, "SVD input must be finite")

        with torch.no_grad():
            theta_solve = upcast_for_solve(theta)
            y_solve = upcast_for_solve(y_1d)
            augmented = torch.column_stack((y_solve, theta_solve))
            try:
                _u, _s, vh = torch.linalg.svd(augmented, full_matrices=False)
            except RuntimeError as exc:
                return self._invalid(theta, f"SVD failed: {exc}")

            null_vector = vh[-1,:]
            denominator = null_vector[0]
            if abs(float(denominator.item())) < self.eps:
                return self._invalid(theta, "SVD null-space denominator is near zero")

            coefficients = (-null_vector[1:] / denominator).to(
                device=theta.device, dtype=theta.dtype
            )
            residual = squared_residual(theta, coefficients, y_1d)
            if not math.isfinite(residual):
                return self._invalid(theta, "SVD residual is not finite")

            selected_indices = (
                (coefficients.abs() > _SELECT_EPS).nonzero(as_tuple=True)[0].tolist()
            )
            return SolveResult(
                coefficients=coefficients.detach(),
                residual=residual,
                r2=compute_r2(theta, coefficients, y_1d),
                condition_number=self._condition_number(augmented),
                selected_indices=selected_indices,
            )

    @staticmethod
    def _validate_shape(theta: torch.Tensor, y: torch.Tensor) -> None:
        if theta.dim() != 2:
            raise ValueError(f"theta must be 2D, got {theta.dim()}D")
        if y.dim() < 1:
            raise ValueError(f"y must be at least 1D, got {y.dim()}D")
        if theta.shape[0] == 0 or theta.shape[1] == 0:
            raise ValueError("theta is empty (0 rows or 0 columns)")
        if theta.dtype != y.dtype:
            raise ValueError(
                f"dtype mismatch: theta is {theta.dtype}, y is {y.dtype}. "
                "Cast to same dtype before solving to avoid precision loss."
            )
        if y.dim() == 2 and y.shape[1] != 1:
            raise ValueError(f"y must be 1D or (n, 1), got shape {y.shape}")
        y_1d = y.squeeze(-1) if y.dim() == 2 else y
        if y_1d.shape[0] != theta.shape[0]:
            raise ValueError(
                f"dimension mismatch: theta has {theta.shape[0]} rows, "
                f"y has {y_1d.shape[0]} elements"
            )

    @staticmethod
    def _condition_number(matrix: torch.Tensor) -> float:
        if (matrix == 0).all():
            return float("inf")
        try:
            return float(torch.linalg.cond(matrix).item())
        except RuntimeError:
            return float("inf")

    @staticmethod
    def _invalid(theta: torch.Tensor, message: str) -> SolveResult:
        return SolveResult(
            coefficients=torch.zeros(
                theta.shape[1],
                dtype=theta.dtype,
                device=theta.device,
            ),
            residual=float("inf"),
            r2=-float("inf"),
            condition_number=float("inf"),
            selected_indices=[],
            is_valid=False,
            error_message=message,
        )
