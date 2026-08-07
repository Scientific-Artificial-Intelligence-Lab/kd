
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


def r2_score(y_pred: torch.Tensor, lhs: torch.Tensor) -> float:
    y_pred_1d = y_pred.squeeze(-1) if y_pred.dim() == 2 else y_pred
    lhs_1d = lhs.squeeze(-1) if lhs.dim() == 2 else lhs
    if y_pred_1d.shape != lhs_1d.shape:
        raise ValueError(
            "r2_score shape mismatch: y_pred "
            f"{tuple(y_pred.shape)} vs lhs {tuple(lhs.shape)}"
        )
    y_pred_64 = upcast_for_solve(y_pred_1d)
    lhs_64 = upcast_for_solve(lhs_1d)

    ss_res = float(((lhs_64 - y_pred_64) ** 2).sum().item())
    if not math.isfinite(ss_res):
        return -float("inf")

    ss_tot = float(((lhs_64 - lhs_64.mean()) ** 2).sum().item())
    if ss_tot < R2_EPS_TOT:
        return 1.0 if ss_res < R2_EPS_RES else 0.0
    return 1.0 - ss_res / ss_tot


def compute_r2(theta: torch.Tensor, coef: torch.Tensor, lhs: torch.Tensor) -> float:
    theta_64 = upcast_for_solve(theta)
    coef_64 = upcast_for_solve(coef)
    return r2_score(theta_64 @ coef_64, lhs)


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






_CPU_ALLOC_FRAGMENT = "DefaultCPUAllocator"


def is_cpu_alloc_failure(exc: RuntimeError) -> bool:
    return _CPU_ALLOC_FRAGMENT in str(exc)


def compute_condition_number(matrix: torch.Tensor) -> float:
    if (matrix == 0).all():
        return float("inf")
    try:
        return float(torch.linalg.cond(matrix).item())
    except torch.cuda.OutOfMemoryError:





        raise
    except RuntimeError as exc:
        if is_cpu_alloc_failure(exc):




            raise

        return float("inf")
