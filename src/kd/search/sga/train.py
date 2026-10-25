
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from kd.core.linear_solve import deterministic_lstsq
from kd.core.metrics import aic_no_n
from kd.search.sga.config import SGAConfig
from kd.search.sga.evaluate import DiffContext, build_theta, prune_invalid_terms
from kd.search.sga.pde import PDE

_LOGGER = logging.getLogger(__name__)


_ZERO_COL_EPS = 1e-14

_TABULAR_ROUNDOFF_FACTOR = 64


@dataclass
class TrainResult:

    coefficients: Tensor
    selected_indices: list[int]
    aic_score: float
    mse: float
    best_tol: float


@dataclass
class CandidateResult:

    train_result: TrainResult
    pruned_pde: PDE
    valid_term_indices: list[int]
    fitted_terms: tuple[str, ...] | None = None



    @property
    def coefficients(self) -> Tensor:
        return self.train_result.coefficients

    @property
    def selected_indices(self) -> list[int]:
        return self.train_result.selected_indices

    @property
    def aic_score(self) -> float:
        return self.train_result.aic_score

    @property
    def mse(self) -> float:
        return self.train_result.mse

    @property
    def best_tol(self) -> float:
        return self.train_result.best_tol


def _invalid_result(n_terms: int, device: torch.device) -> TrainResult:
    return TrainResult(
        coefficients=torch.zeros(n_terms, device=device),
        selected_indices=[],
        aic_score=float("inf"),
        mse=float("inf"),
        best_tol=0.0,
    )


def _result_or_invalid(
    w: Tensor,
    aic_score: float,
    mse: float,
    best_tol: float,
    device: torch.device,
) -> TrainResult:
    if not _selected_indices(w):
        return _invalid_result(w.shape[0], device)
    return TrainResult(
        coefficients=w.clone(),
        selected_indices=_selected_indices(w),
        aic_score=aic_score,
        mse=mse,
        best_tol=best_tol,
    )


def _active_mask(w: Tensor) -> Tensor:
    return w.abs() > _ZERO_COL_EPS


def _count_active(w: Tensor) -> int:
    return int(_active_mask(w).sum().item())


def _selected_indices(w: Tensor) -> list[int]:
    return _active_mask(w).nonzero(as_tuple=True)[0].tolist()


def _compute_mse(theta: Tensor, y_1d: Tensor, w: Tensor, n_samples: int) -> float:
    residual = ((y_1d - theta @ w) ** 2).sum().item()
    return residual / n_samples


def _tabular_mse_floor(y_1d: Tensor) -> float:
    info = torch.finfo(torch.float64)
    tau = _TABULAR_ROUNDOFF_FACTOR * y_1d.shape[0] * info.eps
    mean_square = float(torch.mean(y_1d.to(torch.float64) ** 2).item())
    return max(info.tiny, tau**2 * mean_square)


def _tabular_aic(
    mse: float, k: int, n_samples: int, ratio: float, floor: float
) -> float:
    if not math.isfinite(mse) or k >= n_samples:
        return float("inf")
    return aic_no_n(max(mse, floor), k, ratio)


def _scorer(
    y_1d: Tensor, ratio: float, *, tabular_scoring: bool
) -> Callable[[float, int], float]:
    if not tabular_scoring:
        return lambda mse, k: aic_no_n(mse, k, ratio)
    floor = _tabular_mse_floor(y_1d)
    n_samples = y_1d.shape[0]
    return lambda mse, k: _tabular_aic(mse, k, n_samples, ratio, floor)


def _validate_inputs(theta: Tensor, y: Tensor) -> None:
    if theta.dim() != 2:
        raise ValueError(f"theta must be 2D, got {theta.dim()}D")

    if y.dim() < 1:
        raise ValueError(f"y must be at least 1D, got {y.dim()}D")

    y_1d = y.squeeze(-1) if y.dim() == 2 else y
    if y_1d.shape[0] != theta.shape[0]:
        raise ValueError(
            f"dimension mismatch: theta has {theta.shape[0]} rows, "
            f"y has {y_1d.shape[0]} elements"
        )

    if torch.isnan(theta).any():
        raise ValueError("theta contains NaN values")
    if torch.isnan(y).any():
        raise ValueError("y contains NaN values")

    if torch.isinf(theta).any():
        raise ValueError("theta contains Inf values")
    if torch.isinf(y).any():
        raise ValueError("y contains Inf values")


def _stridge_no_debias(
    theta: Tensor,
    y_1d: Tensor,
    lam: float,
    max_iter: int,
    tol: float,
    normalize: int,
) -> Tensor:
    n, d = theta.shape

    if d == 0:
        return torch.zeros(0, dtype=theta.dtype, device=theta.device)


    col_norms = torch.linalg.norm(theta, dim=0)
    nonzero_mask = col_norms > _ZERO_COL_EPS
    nonzero_cols = nonzero_mask.nonzero(as_tuple=True)[0]

    if nonzero_cols.numel() == 0:
        return torch.zeros(d, dtype=theta.dtype, device=theta.device)

    x0 = theta[:, nonzero_cols]
    d_reduced = x0.shape[1]


    if normalize != 0:
        norms = torch.linalg.norm(x0, ord=normalize, dim=0)
        mreg = (1.0 / norms).unsqueeze(1)
        x_norm = x0 * mreg.squeeze(-1)
    else:
        x_norm = x0
        mreg = torch.ones(d_reduced, 1, dtype=theta.dtype, device=theta.device)


    w = _solve(x_norm, y_1d, lam, d_reduced)


    num_relevant = d_reduced
    biginds: list[int] = list(range(d_reduced))

    for j in range(max_iter):
        smallinds = (w.abs() < tol).squeeze(-1).nonzero(as_tuple=True)[0]
        smallinds_set = set(smallinds.tolist())
        new_biginds = [i for i in range(d_reduced) if i not in smallinds_set]

        if num_relevant == len(new_biginds):
            break
        num_relevant = len(new_biginds)

        if len(new_biginds) == 0:
            if j == 0:




                return torch.zeros(d, dtype=theta.dtype, device=theta.device)
            else:
                break

        biginds = new_biginds
        w[smallinds] = 0
        w[biginds] = _solve(x_norm[:, biginds], y_1d, lam, len(biginds))





    if normalize != 0:
        w = mreg * w


    full_w = torch.zeros(d, dtype=theta.dtype, device=theta.device)
    full_w[nonzero_cols] = w.squeeze(-1)
    return full_w


def _solve(x: Tensor, y: Tensor, lam: float, d: int) -> Tensor:
    y_2d = y.unsqueeze(1) if y.dim() == 1 else y
    if lam != 0:
        xtx = x.T @ x + lam * torch.eye(d, dtype=x.dtype, device=x.device)
        xty = x.T @ y_2d
        return deterministic_lstsq(xtx, xty)
    return deterministic_lstsq(x, y_2d)


def train_sweep(
    theta: Tensor,
    y: Tensor,
    config: SGAConfig,
    *,
    active_support_only: bool = False,
    tabular_scoring: bool = False,
) -> TrainResult:
    _validate_inputs(theta, y)

    n_terms = theta.shape[1]
    device = theta.device

    if n_terms == 0:
        return _invalid_result(0, device)

    n_samples = theta.shape[0]
    y_1d = y.squeeze(-1) if y.dim() == 2 else y

    with torch.no_grad():
        return _train_sweep_impl(
            theta,
            y_1d,
            n_samples,
            n_terms,
            device,
            config,
            active_support_only=active_support_only,
            tabular_scoring=tabular_scoring,
        )


def _train_sweep_impl(
    theta: Tensor,
    y_1d: Tensor,
    n_samples: int,
    n_terms: int,
    device: torch.device,
    config: SGAConfig,
    *,
    active_support_only: bool,
    tabular_scoring: bool,
) -> TrainResult:
    score = _scorer(y_1d, config.aic_ratio, tabular_scoring=tabular_scoring)

    w_baseline = _stridge_no_debias(
        theta,
        y_1d,
        lam=0.0,
        max_iter=config.str_iters,
        tol=0.0,
        normalize=config.normalize,
    )
    if active_support_only:
        w_baseline = w_baseline.masked_fill(~_active_mask(w_baseline), 0.0)
    mse = _compute_mse(theta, y_1d, w_baseline, n_samples)
    k = _count_active(w_baseline)
    best_aic = score(mse, k)
    best = _result_or_invalid(w_baseline, best_aic, mse, 0.0, device)
    best_aic = best.aic_score


    tol = config.d_tol
    d_tol = config.d_tol

    for iter_idx in range(config.maxit):
        w = _stridge_no_debias(
            theta,
            y_1d,
            lam=config.lam,
            max_iter=config.str_iters,
            tol=tol,
            normalize=config.normalize,
        )
        if active_support_only:
            w = w.masked_fill(~_active_mask(w), 0.0)
        mse = _compute_mse(theta, y_1d, w, n_samples)
        k = _count_active(w)
        aic = score(mse, k)




        candidate = _result_or_invalid(w, aic, mse, tol, device)
        aic = candidate.aic_score

        if aic <= best_aic:

            best_aic = aic
            best = candidate
            tol += d_tol
        else:

            tol = max(0.0, tol - 2.0 * d_tol)
            remaining = config.maxit - iter_idx
            if remaining > 0:
                d_tol = 2.0 * d_tol / remaining
            tol += d_tol

    return best


def evaluate_candidate(
    pde: PDE,
    data_dict: dict[str, Tensor],
    default_terms: Tensor | None,
    y: Tensor,
    config: SGAConfig,
    diff_ctx: DiffContext | None = None,
    *,
    active_support_only: bool = False,
) -> CandidateResult:

    pruned_pde, valid_terms, valid_term_indices = prune_invalid_terms(
        pde,
        data_dict,
        diff_ctx=diff_ctx,
    )


    theta = build_theta(valid_terms, default_terms)


    if theta.shape[1] == 0:
        return CandidateResult(
            train_result=_invalid_result(0, theta.device),
            pruned_pde=pruned_pde,
            valid_term_indices=valid_term_indices,
        )


    train_result = train_sweep(
        theta, y, config, active_support_only=active_support_only
    )
    return CandidateResult(
        train_result=train_result,
        pruned_pde=pruned_pde,
        valid_term_indices=valid_term_indices,
    )
