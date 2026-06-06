
from __future__ import annotations

import torch

from kd.core.linear_solve._helpers import (
    compute_r2,
    squared_residual,
    upcast_for_solve,
)
from kd.core.linear_solve.base import SolveResult, SparseSolver


class LeastSquaresSolver(SparseSolver):

    def __init__(self, rcond: float | None = None) -> None:
        self.rcond = rcond

    def solve(
        self,
        theta: torch.Tensor,
        y: torch.Tensor,
    ) -> SolveResult:
        self._validate_inputs(theta, y)

        with torch.no_grad():
            y_1d = y.squeeze(-1) if y.dim() == 2 else y
            theta_solve = upcast_for_solve(theta)
            y_solve = upcast_for_solve(y_1d)

            condition_number = self._compute_condition_number(theta_solve)

            result = torch.linalg.lstsq(
                theta_solve,
                y_solve.unsqueeze(1),
                rcond=self.rcond,
            )
            coefficients = result.solution.squeeze()

            if coefficients.dim() == 0:
                coefficients = coefficients.unsqueeze(0)

            coefficients = coefficients.to(device=theta.device, dtype=theta.dtype)
            residual = squared_residual(theta, coefficients, y_1d)
            r2 = compute_r2(theta, coefficients, y_1d)

            return SolveResult(
                coefficients=coefficients.detach(),
                residual=residual,
                r2=r2,
                condition_number=condition_number,
                selected_indices=None,
            )

    def _validate_inputs(self, theta: torch.Tensor, y: torch.Tensor) -> None:

        if theta.dim() != 2:
            raise ValueError(f"theta must be 2D, got {theta.dim()}D")


        if y.dim() < 1:
            raise ValueError(f"y must be at least 1D, got {y.dim()}D (0D tensor)")


        if theta.shape[0] == 0 or theta.shape[1] == 0:
            raise ValueError("theta is empty (0 rows or 0 columns)")


        if theta.dtype != y.dtype:
            raise ValueError(
                f"dtype mismatch: theta is {theta.dtype}, y is {y.dtype}. "
                "Cast to same dtype before solving to avoid precision loss."
            )


        if torch.isnan(theta).any():
            raise ValueError("theta contains NaN values")
        if torch.isnan(y).any():
            raise ValueError("y contains NaN values")


        if torch.isinf(theta).any():
            raise ValueError("theta contains Inf values")
        if torch.isinf(y).any():
            raise ValueError("y contains Inf values")


        y_1d = y.squeeze(-1) if y.dim() == 2 else y


        if y.dim() == 2 and y.shape[1] != 1:
            raise ValueError(f"y must be 1D or (n, 1), got shape {y.shape}")


        n_samples = theta.shape[0]
        if y_1d.shape[0] != n_samples:
            raise ValueError(
                f"dimension mismatch: theta has {n_samples} rows, "
                f"y has {y_1d.shape[0]} elements"
            )

    def _compute_condition_number(self, theta: torch.Tensor) -> float:

        if (theta == 0).all():
            return float("inf")

        try:
            cond: float = float(torch.linalg.cond(theta).item())
            return cond
        except RuntimeError:

            return float("inf")
