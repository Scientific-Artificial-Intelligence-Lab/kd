
from __future__ import annotations

import logging

import torch

from kd.core.linear_solve._helpers import (
    SOLVE_DTYPE,
    compute_condition_number,
    r2_score,
    upcast_for_solve,
)
from kd.core.linear_solve.base import SolveResult, SparseSolver

logger = logging.getLogger(__name__)


_DEFAULT_TOL = 0.1
_DEFAULT_LAM = 0.0
_DEFAULT_MAX_ITER = 10
_DEFAULT_NORMALIZE = 2


_ZERO_COL_EPS = 1e-14
_COND_WARN_THRESHOLD = 1e10


class STRidgeSolver(SparseSolver):

    def __init__(
        self,
        tol: float = _DEFAULT_TOL,
        lam: float = _DEFAULT_LAM,
        max_iter: int = _DEFAULT_MAX_ITER,
        normalize: int = _DEFAULT_NORMALIZE,
        compute_condition_number: bool = False,
    ) -> None:
        self.tol = tol
        self.lam = lam
        self.max_iter = max_iter
        self.normalize = normalize
        self.compute_condition_number = compute_condition_number

    def solve(
        self,
        theta: torch.Tensor,
        y: torch.Tensor,
    ) -> SolveResult:
        return self.solve_with_tol(theta, y, tol=self.tol)

    def solve_with_tol(
        self,
        theta: torch.Tensor,
        y: torch.Tensor,
        tol: float,
    ) -> SolveResult:
        self._validate_inputs(theta, y)

        with torch.no_grad():
            y_1d = y.squeeze(-1) if y.dim() == 2 else y
            theta_solve = upcast_for_solve(theta)
            y_solve = upcast_for_solve(y_1d)
            _n, d = theta.shape

            condition_number = None
            if self.compute_condition_number:

                condition_number = compute_condition_number(theta_solve)
                if condition_number > _COND_WARN_THRESHOLD:
                    logger.warning(
                        "High condition number %.2e detected in theta matrix",
                        condition_number,
                    )


            zero_mask = _detect_zero_columns(theta_solve)
            nonzero_cols = (~zero_mask).nonzero(as_tuple=True)[0]

            if nonzero_cols.numel() == 0:

                return _build_result(
                    torch.zeros(d, dtype=theta.dtype, device=theta.device),
                    theta_solve,
                    y_solve,
                    condition_number,
                )

            x0 = theta_solve[:, nonzero_cols]
            d_reduced = x0.shape[1]


            x_norm, mreg = _normalize_columns(x0, self.normalize)


            w = _initial_solve(x_norm, y_solve, self.lam, d_reduced)


            w, biginds = _iterative_threshold(
                x_norm,
                y_solve,
                w,
                tol,
                self.lam,
                self.max_iter,
                d_reduced,
            )


            if len(biginds) > 0:
                w[biginds] = _lstsq(x_norm[:, biginds], y_solve)


            if self.normalize != 0:
                w = mreg * w




            full_w = torch.zeros(d, dtype=SOLVE_DTYPE, device=theta_solve.device)
            full_w[nonzero_cols] = w.squeeze(-1)
            full_w = full_w.to(device=theta.device, dtype=theta.dtype)

            return _build_result(full_w, theta_solve, y_solve, condition_number)

    def _validate_inputs(self, theta: torch.Tensor, y: torch.Tensor) -> None:
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

        if y_1d.shape[0] != theta.shape[0]:
            raise ValueError(
                f"dimension mismatch: theta has {theta.shape[0]} rows, "
                f"y has {y_1d.shape[0]} elements"
            )







def _detect_zero_columns(theta: torch.Tensor) -> torch.Tensor:
    col_norms = torch.linalg.norm(theta, dim=0)
    mask: torch.Tensor = col_norms < _ZERO_COL_EPS
    return mask


def _normalize_columns(
    x0: torch.Tensor, normalize: int
) -> tuple[torch.Tensor, torch.Tensor]:
    d = x0.shape[1]
    if normalize == 0:
        mreg = torch.ones(d, 1, dtype=x0.dtype, device=x0.device)
        return x0, mreg







    norms = torch.linalg.norm(x0, ord=normalize, dim=0)
    mreg = (1.0 / norms).unsqueeze(1)
    x_norm = x0 * mreg.squeeze(-1)
    return x_norm, mreg


def _lstsq(mat: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    b_2d = b.unsqueeze(1) if b.dim() == 1 else b
    sol: torch.Tensor = torch.linalg.lstsq(mat, b_2d).solution
    return sol


def _ridge_solve(x: torch.Tensor, y: torch.Tensor, lam: float, d: int) -> torch.Tensor:
    y_2d = y.unsqueeze(1) if y.dim() == 1 else y
    xtx = x.T @ x + lam * torch.eye(d, dtype=x.dtype, device=x.device)
    xty = x.T @ y_2d
    sol: torch.Tensor = torch.linalg.lstsq(xtx, xty).solution
    return sol


def _initial_solve(
    x: torch.Tensor, y: torch.Tensor, lam: float, d: int
) -> torch.Tensor:
    if lam != 0:
        return _ridge_solve(x, y, lam, d)
    return _lstsq(x, y)


def _iterative_threshold(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    tol: float,
    lam: float,
    max_iter: int,
    d: int,
) -> tuple[torch.Tensor, list[int]]:
    num_relevant = d
    biginds: list[int] = list(range(d))

    for j in range(max_iter):

        smallinds = (w.abs() < tol).squeeze(-1).nonzero(as_tuple=True)[0]
        smallinds_set = set(smallinds.tolist())


        new_biginds = [i for i in range(d) if i not in smallinds_set]


        if num_relevant == len(new_biginds):
            break
        num_relevant = len(new_biginds)


        if len(new_biginds) == 0:
            if j == 0:

                return w, biginds
            else:
                break

        biginds = new_biginds


        w[smallinds] = 0


        if lam != 0:
            w[biginds] = _ridge_solve(x[:, biginds], y, lam, len(biginds))
        else:
            w[biginds] = _lstsq(x[:, biginds], y)

    return w, biginds


def _build_result(
    full_w: torch.Tensor,
    theta_solve: torch.Tensor,
    y_solve: torch.Tensor,
    condition_number: float | None,
) -> SolveResult:
    coef_64 = upcast_for_solve(full_w)
    y_pred_64 = theta_solve @ coef_64
    residual = float(((y_solve - y_pred_64) ** 2).sum().item())
    r2 = r2_score(y_pred_64, y_solve)


    nonzero_mask = full_w.abs() > _ZERO_COL_EPS
    selected_indices = nonzero_mask.nonzero(as_tuple=True)[0].tolist()

    return SolveResult(
        coefficients=full_w.detach(),
        residual=residual,
        r2=r2,
        condition_number=condition_number,
        selected_indices=selected_indices,
    )
