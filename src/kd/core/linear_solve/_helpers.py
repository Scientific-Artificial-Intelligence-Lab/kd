
from __future__ import annotations

import math

import torch

SOLVE_DTYPE = torch.float64



R2_EPS_TOT = 1e-15
R2_EPS_RES = 1e-10


def upcast_for_solve(t: torch.Tensor) -> torch.Tensor:
    if t.device.type == "mps":
        return t.cpu().to(dtype=SOLVE_DTYPE)
    return t.to(dtype=SOLVE_DTYPE)


def compute_r2(theta: torch.Tensor, coef: torch.Tensor, lhs: torch.Tensor) -> float:
    lhs_1d = lhs.squeeze(-1) if lhs.dim() == 2 else lhs
    theta_64 = upcast_for_solve(theta)
    coef_64 = upcast_for_solve(coef)
    lhs_64 = upcast_for_solve(lhs_1d)

    y_pred = theta_64 @ coef_64
    ss_res = float(((lhs_64 - y_pred) ** 2).sum().item())
    if not math.isfinite(ss_res):
        return -float("inf")

    ss_tot = float(((lhs_64 - lhs_64.mean()) ** 2).sum().item())
    if ss_tot < R2_EPS_TOT:
        return 1.0 if ss_res < R2_EPS_RES else 0.0
    return 1.0 - ss_res / ss_tot


def squared_residual(
    theta: torch.Tensor,
    coef: torch.Tensor,
    lhs: torch.Tensor,
) -> float:
    lhs_1d = lhs.squeeze(-1) if lhs.dim() == 2 else lhs
    theta_64 = upcast_for_solve(theta)
    coef_64 = upcast_for_solve(coef)
    lhs_64 = upcast_for_solve(lhs_1d)
    return float(((lhs_64 - theta_64 @ coef_64) ** 2).sum().item())
