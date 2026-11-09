
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
    n_protected: int,
) -> TrainResult:
    selected = _selected_indices(w, n_protected)
    if not selected:
        return _invalid_result(w.shape[0], device)
    return TrainResult(
        coefficients=w.clone(),
        selected_indices=selected,
        aic_score=aic_score,
        mse=mse,
        best_tol=best_tol,
    )


def _active_mask(w: Tensor) -> Tensor:
    return w.abs() > _ZERO_COL_EPS


def _count_active(w: Tensor) -> int:
    return int(_active_mask(w).sum().item())


def _selected_indices(w: Tensor, n_protected: int) -> list[int]:
    mask = _active_mask(w)
    mask[:n_protected] = True
    return mask.nonzero(as_tuple=True)[0].tolist()


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


class _STRidgeSystem:

    def __init__(
        self, theta: Tensor, y: Tensor, normalize: int, n_protected: int
    ) -> None:
        self.theta = theta
        self.y = y
        self.normalize = normalize
        self.n_protected = n_protected
        col_norms = torch.linalg.norm(theta, dim=0)
        nonzero_mask = col_norms > _ZERO_COL_EPS
        self.nonzero_cols = nonzero_mask.nonzero(as_tuple=True)[0]
        self.x_norm = theta[:, self.nonzero_cols]
        d_reduced = self.x_norm.shape[1]
        self.mreg = theta.new_ones(d_reduced, 1)
        if d_reduced:

            assert bool(nonzero_mask[:n_protected].all())
            if normalize != 0:
                norms = torch.linalg.norm(self.x_norm, ord=normalize, dim=0)
                self.mreg = (1.0 / norms).unsqueeze(1)
                self.x_norm = self.x_norm * self.mreg.squeeze(-1)
        self.solutions: dict[tuple[float, tuple[int, ...]], Tensor] = {}

    def solve(self, columns: tuple[int, ...], lam: float) -> Tensor:
        key = (lam, columns)
        if key not in self.solutions:
            x = (
                self.x_norm
                if len(columns) == self.x_norm.shape[1]
                else self.x_norm[:, list(columns)]
            )


            self.solutions[key] = _solve(
                x, self.y, lam, len(columns), self.n_protected
            ).clone()
        return self.solutions[key]

    def fit(self, lam: float, max_iter: int, tol: float) -> Tensor:
        d_reduced = self.x_norm.shape[1]
        if d_reduced == 0:
            return self.theta.new_zeros(self.theta.shape[1])


        w = self.solve(tuple(range(d_reduced)), lam).clone()
        num_relevant = d_reduced
        for j in range(max_iter):
            small = (w.abs() < tol).squeeze(-1)
            small[: self.n_protected] = False
            smallinds = small.nonzero(as_tuple=True)[0]
            smallinds_set = set(smallinds.tolist())
            biginds = [i for i in range(d_reduced) if i not in smallinds_set]
            if num_relevant == len(biginds):
                break
            num_relevant = len(biginds)
            if len(biginds) == 0:
                if j == 0:

                    return self.theta.new_zeros(self.theta.shape[1])
                break
            w[smallinds] = 0
            w[biginds] = self.solve(tuple(biginds), lam)

        if self.normalize != 0:
            w = self.mreg * w
        full_w = self.theta.new_zeros(self.theta.shape[1])
        full_w[self.nonzero_cols] = w.squeeze(-1)
        return full_w


def _stridge_no_debias(
    theta: Tensor,
    y_1d: Tensor,
    lam: float,
    max_iter: int,
    tol: float,
    normalize: int,
    n_protected: int = 0,
) -> Tensor:
    return _STRidgeSystem(theta, y_1d, normalize, n_protected).fit(lam, max_iter, tol)


def _solve(x: Tensor, y: Tensor, lam: float, d: int, n_protected: int = 0) -> Tensor:
    y_2d = y.unsqueeze(1) if y.dim() == 1 else y
    if lam != 0:
        penalty = lam * torch.eye(d, dtype=x.dtype, device=x.device)
        penalty.diagonal()[:n_protected] = 0.0
        xtx = x.T @ x + penalty
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
    n_protected: int = 0,
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
            n_protected=n_protected,
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
    n_protected: int,
) -> TrainResult:
    score = _scorer(y_1d, config.aic_ratio, tabular_scoring=tabular_scoring)
    system = _STRidgeSystem(theta, y_1d, config.normalize, n_protected)

    w_baseline = system.fit(
        lam=0.0,
        max_iter=config.str_iters,
        tol=0.0,
    )
    if active_support_only:
        w_baseline = w_baseline.masked_fill(~_active_mask(w_baseline), 0.0)
    mse = _compute_mse(theta, y_1d, w_baseline, n_samples)
    k = _count_active(w_baseline[n_protected:])
    best_aic = score(mse, k)
    best = _result_or_invalid(w_baseline, best_aic, mse, 0.0, device, n_protected)
    best_aic = best.aic_score


    tol = config.d_tol
    d_tol = config.d_tol

    for iter_idx in range(config.maxit):
        w = system.fit(
            lam=config.lam,
            max_iter=config.str_iters,
            tol=tol,
        )
        if active_support_only:
            w = w.masked_fill(~_active_mask(w), 0.0)
        mse = _compute_mse(theta, y_1d, w, n_samples)
        k = _count_active(w[n_protected:])
        aic = score(mse, k)




        candidate = _result_or_invalid(w, aic, mse, tol, device, n_protected)
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
    n_protected: int = 0,
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
        theta,
        y,
        config,
        active_support_only=active_support_only,
        n_protected=n_protected,
    )
    return CandidateResult(
        train_result=train_result,
        pruned_pde=pruned_pde,
        valid_term_indices=valid_term_indices,
    )
